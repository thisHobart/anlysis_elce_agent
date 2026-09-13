"""Deterministic local artifacts for one Shandong forecast run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from app.research.forecasting.contracts import ForecastFoldResult, ForecastMetricSet, ForecastPlan


def _points(
    values: pd.Series,
    *,
    width: int,
    height: int,
    bounds: tuple[float, float] | None = None,
    padding: int = 28,
) -> str:
    clean = pd.to_numeric(values, errors="coerce")
    if clean.dropna().empty:
        return ""
    minimum, maximum = bounds or (float(clean.min()), float(clean.max()))
    span = max(maximum - minimum, 1.0)
    xs = [padding + index * (width - 2 * padding) / max(len(clean) - 1, 1) for index in range(len(clean))]
    ys = [height - padding - (float(value) - minimum) * (height - 2 * padding) / span for value in clean]
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))


def write_forecast_figure(frame: pd.DataFrame, destination: Path) -> None:
    width, height = 920, 340
    combined = pd.concat([frame["predicted_rt_price"], frame["similar_day_prior"]])
    bounds = (float(combined.min()), float(combined.max()))
    predicted = _points(
        frame["predicted_rt_price"], width=width, height=height, bounds=bounds
    )
    prior = _points(frame["similar_day_prior"], width=width, height=height, bounds=bounds)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/><text x="28" y="22" font-size="16">山东次日实时电价预测</text>
<line x1="28" y1="312" x2="892" y2="312" stroke="#cbd5e1"/><polyline points="{prior}" fill="none" stroke="#94a3b8" stroke-width="2"/>
<polyline points="{predicted}" fill="none" stroke="#2563eb" stroke-width="2.5"/><text x="690" y="22" fill="#2563eb">CTM＋相似日</text><text x="805" y="22" fill="#64748b">相似日先验</text>
</svg>""",
        encoding="utf-8",
    )


def write_backtest_figure(metrics: dict[str, ForecastMetricSet], destination: Path) -> None:
    labels = ["model", "persistence", "day_naive", "week_naive"]
    colors = ["#2563eb", "#94a3b8", "#f59e0b", "#10b981"]
    maximum = max(metrics[name].mae for name in labels) or 1.0
    bars = []
    for index, (name, color) in enumerate(zip(labels, colors, strict=True)):
        value = metrics[name].mae
        width = 650 * value / maximum
        y = 55 + index * 58
        bars.append(
            f'<text x="24" y="{y + 18}" font-size="14">{name}</text>'
            f'<rect x="140" y="{y}" width="{width:.1f}" height="24" fill="{color}"/>'
            f'<text x="{150 + width:.1f}" y="{y + 18}" font-size="14">MAE {value:.2f}</text>'
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="310" viewBox="0 0 920 310">'
        '<rect width="100%" height="100%" fill="#fff"/><text x="24" y="28" font-size="16">三折回测聚合比较</text>'
        + "".join(bars)
        + "</svg>",
        encoding="utf-8",
    )


def write_report(
    *,
    plan: ForecastPlan,
    folds: list[ForecastFoldResult],
    aggregate: dict[str, ForecastMetricSet],
    warnings: list[str],
    destination: Path,
) -> None:
    enough_points = all(fold.common_observations >= 72 for fold in folds)
    verified = enough_points and aggregate["model"].mae < min(
        aggregate["day_naive"].mae,
        aggregate["week_naive"].mae,
    )
    lines = [
        "# 山东次日实时电价预测",
        "",
        "## 结论",
        "",
        (
            "三折样本外回测中，CTM＋相似日的聚合 MAE 优于日/周朴素基线。"
            if verified
            else "三折样本外回测尚未验证出相对日/周朴素基线的预测增益。未来曲线仅供实验参考。"
        ),
        "",
        f"预测窗口：{plan.forecast_start.isoformat()} — {plan.forecast_end.isoformat()}，共 96 个15分钟点。",
        "",
        "![三折回测](figures/backtest_comparison.svg)",
        "",
        "![次日预测](figures/forecast_curve.svg)",
        "",
        "## 聚合指标",
        "",
        "| 方法 | N | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|",
    ]
    display = {
        "model": "CTM＋相似日",
        "persistence": "持续法",
        "day_naive": "前一日同刻朴素法",
        "week_naive": "周前朴素法",
    }
    for name in ("model", "persistence", "day_naive", "week_naive"):
        metric = aggregate[name]
        lines.append(
            f"| {display[name]} | {metric.observations} | {metric.mae:.2f} | {metric.rmse:.2f} | {metric.bias:+.2f} |"
        )
    lines.extend(["", "## 回测锚点", ""])
    for fold in folds:
        lines.append(
            f"- {fold.anchor.isoformat()}：模型 MAE {fold.model.mae:.2f}；"
            f"共同有效点 {fold.common_observations}/96"
        )
    lines.extend(["", "## 新闻特征", ""])
    news_columns = sorted({column for fold in folds for column in fold.news_feature_columns})
    if news_columns:
        lines.append("本次模型实际读取的新闻特征：" + "、".join(f"`{name}`" for name in news_columns) + "。")
    else:
        lines.append("本次没有新闻特征通过选择与可获得性门禁。")
    lines.extend(["", "## 限制与警告", ""])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- 未发现额外运行警告。")
    lines.extend(
        [
            "",
            "本次运行只读取数据库和冻结快照，不写业务预测表；未包含Q-QRA概率区间、Stage3专家或后置规则。",
            "",
            f"算法版本：`{plan.algorithm_version}`；数据指纹：`{plan.data_fingerprint}`。",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics(
    folds: list[ForecastFoldResult],
    aggregate: dict[str, ForecastMetricSet],
    warnings: list[str],
    destination: Path,
) -> None:
    destination.write_text(
        json.dumps(
            {
                "folds": [item.model_dump(mode="json") for item in folds],
                "aggregate": {name: value.model_dump(mode="json") for name, value in aggregate.items()},
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def artifact_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
