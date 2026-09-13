"""CPU-only CTM and multi-factor similar-day point forecast.

This is an isolated extraction of the reference project's active CTM-Base v4
ideas.  It intentionally excludes Q-QRA, legacy LSTM, Stage3 experts and every
post-processing rule so the first P3 contract stays reviewable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.research.forecasting.contracts import FORECAST_ALGORITHM_VERSION, CTMTrainingConfig


class ForecastCancelled(RuntimeError):
    """The desktop user cancelled a running training job."""


class ForecastModelError(ValueError):
    """The fixed CTM contract could not train or predict."""


def _check_cancel(cancelled: Callable[[], bool] | None) -> None:
    if cancelled and cancelled():
        raise ForecastCancelled("预测已取消")


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def build_feature_frame(frame: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """Create causal history features and explicitly future-visible inputs."""

    data = frame.copy().sort_index()
    data.index = pd.to_datetime(data.index)
    result = pd.DataFrame(index=data.index)
    hour = data.index.hour + data.index.minute / 60.0
    slot = data.index.hour * 4 + data.index.minute // 15
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    result["slot_sin"] = np.sin(2 * np.pi * slot / 96.0)
    result["slot_cos"] = np.cos(2 * np.pi * slot / 96.0)
    result["dow_sin"] = np.sin(2 * np.pi * data.index.dayofweek / 7.0)
    result["dow_cos"] = np.cos(2 * np.pi * data.index.dayofweek / 7.0)
    result["weekend"] = (data.index.dayofweek >= 5).astype(float)

    visible_price = _numeric(data, "rt_price")
    visible_price.loc[visible_price.index > train_end] = np.nan
    for lag in (2, 4, 8, 96, 192, 672):
        result[f"price_lag_{lag}"] = visible_price.shift(lag)
    shifted_price = visible_price.shift(1)
    for window in (4, 16, 96, 672):
        rolling = shifted_price.rolling(window, min_periods=min(16, window))
        result[f"price_mean_{window}"] = rolling.mean()
        result[f"price_std_{window}"] = rolling.std()
    result["price_ramp_1"] = shifted_price.diff()
    result["price_ramp_4"] = shifted_price.diff(4)

    future_columns = [
        column
        for column in data
        if column.startswith(("fcst_", "wx_", "news_"))
    ]
    for column in future_columns:
        result[column] = _numeric(data, column)

    actual_columns = [
        column
        for column in data
        if column not in {"rt_price"}
        and not column.startswith("fcst_")
        and not column.startswith("wx_")
        and not column.startswith("news_")
        and pd.api.types.is_numeric_dtype(data[column])
    ]
    for column in actual_columns:
        values = _numeric(data, column)
        values.loc[values.index > train_end] = np.nan
        result[f"hist_{column}_lag96"] = values.shift(96)
        result[f"hist_{column}_lag672"] = values.shift(672)

    load = _numeric(result, "fcst_load_type_11")
    renewable = _numeric(result, "fcst_re_total")
    net_load = _numeric(result, "fcst_net_load")
    if not load.isna().all() and not renewable.isna().all():
        result["renewable_load_ratio"] = renewable / load.replace(0.0, np.nan)
    if not net_load.isna().all():
        result["net_load_ramp_4"] = net_load.diff(4)
    solar = _numeric(result, "wx_solar_radiation")
    cloud = _numeric(result, "wx_cloud_cover")
    pv = _numeric(result, "fcst_new_energy_pv_unified")
    if not solar.isna().all() and not pv.isna().all():
        result["solar_pv_interaction"] = solar * pv
    if not cloud.isna().all() and not pv.isna().all():
        result["cloud_pv_interaction"] = cloud * pv

    training = result.loc[:train_end]
    fill_values = {
        column: float(values.median()) if not values.empty else 0.0
        for column in result
        if not (values := pd.to_numeric(training[column], errors="coerce").dropna()).empty
    }
    for column in result:
        fill_values.setdefault(column, 0.0)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_values).fillna(0.0)


def _daily_factor_frame(frame: pd.DataFrame, *, include_price: bool) -> pd.DataFrame:
    factors: dict[str, pd.Series] = {}
    definitions = {
        "fcst_net_load": ("net_load", "mean"),
        "fcst_load_type_11": ("load", "mean"),
        "fcst_re_total": ("renewable", "mean"),
        "fcst_new_energy_pv_unified": ("solar_output", "max"),
        "wx_temperature": ("temperature", "mean"),
        "wx_wind_speed_eighty": ("wind", "mean"),
        "wx_cloud_cover": ("cloud", "mean"),
        "wx_solar_radiation": ("radiation", "max"),
        "wx_hour_precipitation": ("precipitation", "sum"),
    }
    for source, (name, method) in definitions.items():
        if source not in frame:
            continue
        series = _numeric(frame, source)
        grouped = series.resample("D")
        factors[name] = grouped.sum(min_count=1) if method == "sum" else getattr(grouped, method)()
    if include_price and "rt_price" in frame:
        price = _numeric(frame, "rt_price")
        factors["price_mean"] = price.resample("D").mean()
        factors["price_range"] = price.resample("D").max() - price.resample("D").min()
        factors["negative_ratio"] = price.lt(0).astype(float).resample("D").mean()
        factors["spike_ratio"] = price.ge(500).astype(float).resample("D").mean()
    return pd.DataFrame(factors).replace([np.inf, -np.inf], np.nan) if factors else pd.DataFrame()


def similar_day_prior(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    forecast_index: pd.DatetimeIndex,
    config: CTMTrainingConfig,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Build a seven-neighbour daily curve with causal calendar fallback."""

    history = frame.loc[:train_end].copy()
    future = frame.reindex(frame.index.union(forecast_index)).loc[forecast_index].copy()
    price = _numeric(history, "rt_price").dropna()
    if price.empty:
        raise ForecastModelError("相似日没有可见历史电价")
    history_factors = _daily_factor_frame(history, include_price=False)
    future_factors = _daily_factor_frame(future, include_price=False)
    target_day = forecast_index[0].normalize()
    target_factor = future_factors.loc[target_day] if target_day in future_factors.index else pd.Series(dtype=float)
    candidates = history_factors.loc[history_factors.index < target_day].copy()
    complete_days = price.groupby(price.index.normalize()).count()
    candidates = candidates.loc[candidates.index.isin(complete_days.loc[complete_days >= 90].index)]
    selected: list[pd.Timestamp] = []
    weights = pd.Series(dtype=float)
    used_features: list[str] = []
    if not candidates.empty and not target_factor.empty:
        score = pd.Series(0.0, index=candidates.index)
        total_weight = 0.0
        for column in candidates:
            target_value = target_factor.get(column)
            values = pd.to_numeric(candidates[column], errors="coerce")
            if pd.isna(target_value) or values.dropna().empty:
                continue
            scale = float(values.quantile(0.75) - values.quantile(0.25))
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = max(float(values.std()), 1.0)
            score += (values - float(target_value)).abs().fillna(scale * 2) / scale
            total_weight += 1.0
            used_features.append(column)
        if total_weight:
            score /= total_weight
            calendar_penalty = pd.Series(
                [0.0 if day.dayofweek == target_day.dayofweek else 0.35 for day in score.index],
                index=score.index,
            )
            score += calendar_penalty
            picked: list[pd.Timestamp] = []
            for days in config.similar_day_windows:
                window = score.loc[score.index >= target_day - pd.Timedelta(days=days)]
                if not window.empty:
                    picked.extend(window.nsmallest(max(1, config.similar_day_top_k // 3)).index)
            picked.extend(score.nsmallest(config.similar_day_top_k * 2).index)
            selected = list(dict.fromkeys(picked))[: config.similar_day_top_k]
            selected_score = score.loc[selected]
            scale = max(float(selected_score.median()), 0.7)
            weights = np.exp(-selected_score / scale)
            weights /= weights.sum()
    if not selected:
        available_days = sorted(day for day, count in complete_days.items() if count >= 90 and day < target_day)
        same_dow = [day for day in available_days if day.dayofweek == target_day.dayofweek]
        selected = list(reversed(same_dow[-config.similar_day_top_k :]))
        if not selected:
            selected = list(reversed(available_days[-config.similar_day_top_k :]))
        if not selected:
            raise ForecastModelError("找不到可用相似日或日历回退日")
        weights = pd.Series(1.0 / len(selected), index=selected)

    curves: list[np.ndarray] = []
    curve_weights: list[float] = []
    for day in selected:
        daily = price.loc[price.index.normalize() == day].sort_index()
        if daily.empty:
            continue
        x = np.linspace(0, 1, len(daily))
        curves.append(np.interp(np.linspace(0, 1, len(forecast_index)), x, daily.to_numpy(dtype=float)))
        curve_weights.append(float(weights.get(day, 0.0)))
    if not curves:
        raise ForecastModelError("相似日曲线为空")
    matrix = np.vstack(curves)
    normalized_weights = np.asarray(curve_weights, dtype=float)
    if normalized_weights.sum() <= 0:
        normalized_weights = np.ones(len(curves), dtype=float)
    normalized_weights /= normalized_weights.sum()
    prior = pd.Series(np.average(matrix, axis=0, weights=normalized_weights), index=forecast_index)
    spread = pd.Series(np.std(matrix, axis=0), index=forecast_index)
    return prior, spread, {
        "selected_days": [day.date().isoformat() for day in selected],
        "used_features": used_features,
    }


class LightweightCTM(nn.Module):
    """Small continuous-time-inspired recurrent multi-tick forecaster."""

    def __init__(self, input_size: int, config: CTMTrainingConfig) -> None:
        super().__init__()
        self.horizon = config.horizon_steps
        self.ticks = config.internal_ticks
        self.memory_length = config.memory_length
        hidden = config.hidden_size
        self.history = nn.GRU(input_size, hidden, batch_first=True)
        self.future = nn.Linear(input_size, hidden)
        self.memory = nn.GRUCell(hidden, hidden)
        self.dropout = nn.Dropout(config.dropout)
        self.tick_heads = nn.ModuleList(nn.Linear(hidden, 1) for _ in range(self.ticks))
        self.sync_left = nn.Linear(hidden, config.sync_pairs)
        self.sync_right = nn.Linear(hidden, config.sync_pairs)
        self.certainty = nn.Linear(config.sync_pairs, self.ticks)
        self.regimes = nn.Linear(hidden, 4)

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _, hidden = self.history(history)
        state = hidden[-1]
        tick_predictions = []
        regime_logits = []
        memory_states: list[torch.Tensor] = []
        future_encoded = torch.tanh(self.future(future))
        for step in range(self.horizon):
            state = self.memory(future_encoded[:, step, :], state)
            memory_states.append(state)
            context = torch.stack(memory_states[-self.memory_length :], dim=0).mean(dim=0)
            dropped = self.dropout(context)
            tick_predictions.append(torch.cat([head(dropped) for head in self.tick_heads], dim=1))
            regime_logits.append(self.regimes(dropped))
        ticks = torch.stack(tick_predictions, dim=1)
        synchronized = torch.tanh(self.sync_left(context)) * torch.tanh(self.sync_right(context))
        certainty = torch.softmax(self.certainty(synchronized), dim=1)
        prediction = (ticks * certainty[:, None, :]).sum(dim=2)
        return prediction, ticks, certainty, torch.stack(regime_logits, dim=1)


@dataclass
class CTMBundle:
    model: LightweightCTM
    scaler: StandardScaler
    feature_columns: list[str]
    target_mean: float
    target_scale: float
    train_end: pd.Timestamp
    config: CTMTrainingConfig


def _training_arrays(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    train_end: pd.Timestamp,
    config: CTMTrainingConfig,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = scaler.transform(features.to_numpy(dtype=float)).astype(np.float32)
    values = target.to_numpy(dtype=float)
    end_position = features.index.searchsorted(train_end, side="right")
    positions = []
    for start in range(config.lookback_steps, end_position - config.horizon_steps + 1, config.training_stride):
        future_target = values[start : start + config.horizon_steps]
        if np.isfinite(future_target).all():
            positions.append(start)
    if len(positions) > config.maximum_training_samples:
        indexes = np.linspace(0, len(positions) - 1, config.maximum_training_samples).round().astype(int)
        positions = [positions[index] for index in indexes]
    if not positions:
        raise ForecastModelError("CTM无法构造完整的历史训练窗口")
    histories = np.stack([scaled[pos - config.lookback_steps : pos] for pos in positions])
    futures = np.stack([scaled[pos : pos + config.horizon_steps] for pos in positions])
    labels = np.stack([values[pos : pos + config.horizon_steps] for pos in positions])
    return histories, futures, labels


def train_ctm(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    config: CTMTrainingConfig,
    progress: Callable[[int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> CTMBundle:
    """Train the fixed reference-sized CTM on CPU with deterministic ordering."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_end = pd.Timestamp(train_end).tz_localize(None) if pd.Timestamp(train_end).tzinfo else pd.Timestamp(train_end)
    target = _numeric(frame, "rt_price")
    if int(target.loc[:train_end].notna().sum()) < config.minimum_training_rows:
        raise ForecastModelError(
            f"CTM训练电价不足：{int(target.loc[:train_end].notna().sum())} < {config.minimum_training_rows}"
        )
    features = build_feature_frame(frame, train_end)
    scaler = StandardScaler().fit(features.loc[:train_end].to_numpy(dtype=float))
    history, future, labels = _training_arrays(
        features, target, train_end=train_end, config=config, scaler=scaler
    )
    target_mean = float(np.mean(labels))
    target_scale = max(float(np.std(labels)), 1.0)
    scaled_labels = ((labels - target_mean) / target_scale).astype(np.float32)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(history),
            torch.from_numpy(future),
            torch.from_numpy(scaled_labels),
            torch.from_numpy(labels.astype(np.float32)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = LightweightCTM(history.shape[2], config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = config.early_stopping_patience
    for epoch in range(1, config.max_epochs + 1):
        _check_cancel(cancelled)
        model.train()
        total = 0.0
        rows = 0
        for history_batch, future_batch, label_batch, raw_batch in loader:
            _check_cancel(cancelled)
            optimizer.zero_grad()
            prediction, _ticks, _certainty, regime_logits = model(history_batch, future_batch)
            point = torch.nn.functional.smooth_l1_loss(prediction, label_batch)
            raw_prediction = prediction * target_scale + target_mean
            negative_over = torch.relu(raw_prediction - raw_batch).masked_select(raw_batch < 0).mean()
            spike_under = torch.relu(raw_batch - raw_prediction).masked_select(raw_batch >= 500).mean()
            if torch.isnan(negative_over):
                negative_over = torch.tensor(0.0)
            if torch.isnan(spike_under):
                spike_under = torch.tensor(0.0)
            regime_targets = torch.stack(
                (raw_batch < 0, raw_batch >= 350, raw_batch >= 500, raw_batch >= 700), dim=2
            ).float()
            regime = torch.nn.functional.binary_cross_entropy_with_logits(regime_logits, regime_targets)
            loss = point + 0.0035 * negative_over + 0.0025 * spike_under + 0.18 * regime
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            total += float(loss.item()) * len(history_batch)
            rows += len(history_batch)
        epoch_loss = total / max(rows, 1)
        if epoch_loss + 1e-6 < best_loss:
            best_loss = epoch_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            patience = config.early_stopping_patience
        else:
            patience -= 1
        if progress:
            progress(epoch, f"CTM训练 {epoch}/{config.max_epochs}，loss={epoch_loss:.5f}")
        if patience <= 0:
            break
    if best_state is None:
        raise ForecastModelError("CTM训练没有产生有效模型")
    model.load_state_dict(best_state)
    model.eval()
    return CTMBundle(model, scaler, list(features.columns), target_mean, target_scale, train_end, config)


def predict_ctm(
    bundle: CTMBundle,
    frame: pd.DataFrame,
    forecast_index: pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    features = build_feature_frame(frame, bundle.train_end).reindex(columns=bundle.feature_columns)
    full = features.index.union(forecast_index).sort_values()
    features = features.reindex(full).fillna(0.0)
    scaled = pd.DataFrame(
        bundle.scaler.transform(features.to_numpy(dtype=float)),
        index=features.index,
    )
    history = (
        scaled.loc[:bundle.train_end]
        .tail(bundle.config.lookback_steps)
        .to_numpy(dtype=np.float32, copy=True)
    )
    if len(history) != bundle.config.lookback_steps:
        raise ForecastModelError("CTM推理缺少96点历史窗口")
    future = (
        scaled.reindex(forecast_index)
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=True)
    )
    if len(future) != bundle.config.horizon_steps:
        raise ForecastModelError("CTM最小版本只接受96点预测窗口")
    with torch.no_grad():
        prediction, ticks, certainty, regimes = bundle.model(
            torch.from_numpy(history[None, :, :]), torch.from_numpy(future[None, :, :])
        )
    prediction_values = prediction.squeeze(0).numpy() * bundle.target_scale + bundle.target_mean
    raw = pd.Series(prediction_values, index=forecast_index)
    tick_values = ticks.squeeze(0).numpy() * bundle.target_scale + bundle.target_mean
    certainty_value = float(certainty.squeeze(0).max().item())
    probabilities = torch.sigmoid(regimes.squeeze(0)).numpy()
    return raw.clip(bundle.config.price_floor, bundle.config.price_cap), {
        "ctm_raw": raw,
        "ctm_tick_std": pd.Series(tick_values.std(axis=1), index=forecast_index),
        "ctm_certainty": pd.Series(certainty_value, index=forecast_index),
        "ctm_p_negative": pd.Series(probabilities[:, 0], index=forecast_index),
        "ctm_p_spike": pd.Series(probabilities[:, 2], index=forecast_index),
    }


def forecast_ctm_similar_day(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    forecast_index: pd.DatetimeIndex,
    config: CTMTrainingConfig,
    progress: Callable[[int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    prior, spread, selection = similar_day_prior(
        frame, train_end=train_end, forecast_index=forecast_index, config=config
    )
    bundle = train_ctm(
        frame,
        train_end=train_end,
        config=config,
        progress=progress,
        cancelled=cancelled,
    )
    ctm, meta = predict_ctm(bundle, frame, forecast_index)
    divergence = (ctm - prior).abs() / spread.clip(lower=30.0)
    certainty = meta["ctm_certainty"].clip(0.0, 1.0)
    weight = (0.45 + 0.40 * certainty - 0.12 * divergence.clip(0.0, 2.0)).clip(0.25, 0.85)
    final = (weight * ctm + (1.0 - weight) * prior).clip(config.price_floor, config.price_cap)
    return pd.DataFrame(
        {
            "predicted_rt_price": final,
            "ctm_prediction": ctm,
            "similar_day_prior": prior,
            "similar_day_spread": spread,
            "ctm_weight": weight,
            "ctm_certainty": certainty,
            "ctm_tick_std": meta["ctm_tick_std"],
            "ctm_p_negative": meta["ctm_p_negative"],
            "ctm_p_spike": meta["ctm_p_spike"],
            "algorithm_version": FORECAST_ALGORITHM_VERSION,
            "similar_days": ",".join(selection["selected_days"]),
        },
        index=forecast_index,
    )
