"""Render a compact, reproducible set of static EDA figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from app.research.schemas.study import StudyConfig

INK = "#262A30"
GRID = "#D9DEE5"
BLUE = "#2F6B9A"
GOLD = "#C58A2B"
ORANGE = "#D66C35"
OLIVE = "#758542"
PINK = "#B75A7A"


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(INK)
    axis.tick_params(colors=INK, labelsize=8)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=144, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _price_time_series(frame: pd.DataFrame, config: StudyConfig, path: Path) -> None:
    values = frame[config.target.name]
    plotted = values.resample("1D").mean() if len(values) > 2000 else values
    grain = "daily mean" if len(values) > 2000 else config.study.frequency
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(plotted.index, plotted.values, color=BLUE, linewidth=1.15)
    axis.axhline(0, color=INK, linewidth=0.8, linestyle="--", alpha=0.7)
    axis.set_title(
        f"Real-time price over time\n{grain} | {config.target.unit} | {values.notna().sum():,} observations",
        loc="left",
        color=INK,
        fontsize=11,
    )
    axis.set_xlabel(f"Timestamp ({config.study.timezone})", color=INK)
    axis.set_ylabel(config.target.unit, color=INK)
    _style_axis(axis)
    _save(figure, path)


def _price_distribution(frame: pd.DataFrame, config: StudyConfig, path: Path) -> None:
    values = frame[config.target.name].dropna()
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.hist(values, bins=60, color=BLUE, edgecolor="#214A68", linewidth=0.45)
    axis.axvline(0, color=INK, linewidth=0.9, linestyle="--")
    axis.set_title(
        f"Real-time price distribution\n{config.target.unit} | N={len(values):,} | dashed line marks zero",
        loc="left",
        color=INK,
        fontsize=11,
    )
    axis.set_xlabel(config.target.unit, color=INK)
    axis.set_ylabel("Intervals", color=INK)
    _style_axis(axis)
    _save(figure, path)


def _seasonal_patterns(frame: pd.DataFrame, config: StudyConfig, path: Path) -> None:
    values = frame[config.target.name]
    hourly = values.groupby(values.index.hour).mean()
    weekday = values.groupby(values.index.dayofweek).mean().reindex(range(7))
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(hourly.index, hourly.values, color=BLUE, edgecolor="#214A68", linewidth=0.4)
    axes[0].axhline(0, color=INK, linewidth=0.8)
    axes[0].set_title("Mean price by hour\nCanonical local hour", loc="left", color=INK, fontsize=10)
    axes[0].set_xlabel("Hour", color=INK)
    axes[0].set_ylabel(config.target.unit, color=INK)
    axes[0].set_xticks(range(0, 24, 3))
    axes[1].bar(weekday_labels, weekday.values, color=GOLD, edgecolor="#8B611E", linewidth=0.4)
    axes[1].axhline(0, color=INK, linewidth=0.8)
    axes[1].set_title("Mean price by weekday\nAll available intervals", loc="left", color=INK, fontsize=10)
    axes[1].set_xlabel("Weekday", color=INK)
    axes[1].set_ylabel(config.target.unit, color=INK)
    for axis in axes:
        _style_axis(axis)
    figure.suptitle(
        f"Price seasonality | {config.study.timezone} | N={values.notna().sum():,}",
        x=0.07,
        ha="left",
        color=INK,
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    _save(figure, path)


def _correlation_matrix(
    frame: pd.DataFrame,
    config: StudyConfig,
    path: Path,
    exogenous_names: list[str] | None = None,
) -> None:
    selected = exogenous_names if exogenous_names is not None else [series.name for series in config.exogenous]
    names = [config.target.name, *selected]
    correlation = frame[names].corr(min_periods=config.analysis.min_relationship_observations)
    size = max(8.5, min(13.0, 5.5 + len(names) * 0.38))
    figure, axis = plt.subplots(figsize=(size, size * 0.86))
    image = axis.imshow(correlation.to_numpy(), cmap="PuOr_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(names)), labels=names, rotation=55, ha="right", fontsize=7)
    axis.set_yticks(range(len(names)), labels=names, fontsize=7)
    axis.set_title(
        "Contemporaneous Pearson correlation matrix\nPairwise-complete observations; correlation does not establish causation",
        loc="left",
        color=INK,
        fontsize=11,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label("Correlation", color=INK)
    axis.spines[:].set_visible(False)
    _save(figure, path)


def _lag_relationships(summary: dict[str, Any], config: StudyConfig, path: Path) -> None:
    relationships = summary["relationships"]["series"]
    ranked = sorted(
        relationships.items(),
        key=lambda item: abs((item[1].get("best_absolute_lag") or {}).get("correlation") or 0),
        reverse=True,
    )[:5]
    figure, axis = plt.subplots(figsize=(10, 5.2))
    colors = [BLUE, GOLD, ORANGE, OLIVE, PINK]
    line_styles = ["-", "--", "-.", ":", (0, (5, 2))]
    for (name, result), color, line_style in zip(
        ranked,
        colors[: len(ranked)],
        line_styles[: len(ranked)],
        strict=True,
    ):
        rows = [row for row in result.get("lag_profile", []) if row["correlation"] is not None]
        if rows:
            axis.plot(
                [row["lag"] for row in rows],
                [row["correlation"] for row in rows],
                label=name,
                color=color,
                linestyle=line_style,
                linewidth=1.4,
            )
    axis.axhline(0, color=INK, linewidth=0.8)
    axis.set_title(
        "Leading exogenous-variable correlations\nPositive lag compares feature[t-lag] with price[t]; top five by absolute peak",
        loc="left",
        color=INK,
        fontsize=11,
    )
    axis.set_xlabel(f"Lead in {config.study.frequency} intervals", color=INK)
    axis.set_ylabel("Pearson correlation", color=INK)
    axis.set_ylim(-1, 1)
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    _style_axis(axis)
    _save(figure, path)


def render_eda_charts(
    frame: pd.DataFrame,
    config: StudyConfig,
    summary: dict[str, Any],
    figure_directory: Path,
) -> dict[str, Path]:
    """Render the chart contract used by the Phase 1 Markdown report."""

    figure_directory.mkdir(parents=True, exist_ok=False)
    paths = {
        "price_timeseries": figure_directory / "price_timeseries.png",
        "price_distribution": figure_directory / "price_distribution.png",
        "seasonal_patterns": figure_directory / "seasonal_patterns.png",
        "correlation_matrix": figure_directory / "correlation_matrix.png",
        "lag_relationships": figure_directory / "lag_relationships.png",
    }
    _price_time_series(frame, config, paths["price_timeseries"])
    _price_distribution(frame, config, paths["price_distribution"])
    _seasonal_patterns(frame, config, paths["seasonal_patterns"])
    _correlation_matrix(frame, config, paths["correlation_matrix"])
    _lag_relationships(summary, config, paths["lag_relationships"])
    return paths


def render_agent_eda_charts(
    frame: pd.DataFrame,
    config: StudyConfig,
    summary: dict[str, Any],
    figure_directory: Path,
) -> dict[str, Path]:
    """Render only figures justified by the user-approved Agent plan."""

    paths: dict[str, Path] = {}
    if not any(section in summary for section in ("price", "exogenous", "relationships")):
        return paths
    figure_directory.mkdir(parents=True, exist_ok=False)
    if "price" in summary:
        price_methods = set(summary["price"].get("methods", []))
        paths["price_timeseries"] = figure_directory / "price_timeseries.png"
        _price_time_series(frame, config, paths["price_timeseries"])
        if not price_methods or price_methods.intersection({"distribution", "extremes"}):
            paths["price_distribution"] = figure_directory / "price_distribution.png"
            _price_distribution(frame, config, paths["price_distribution"])
        if not price_methods or "seasonality" in price_methods:
            paths["seasonal_patterns"] = figure_directory / "seasonal_patterns.png"
            _seasonal_patterns(frame, config, paths["seasonal_patterns"])
    exogenous_methods = set(summary.get("exogenous", {}).get("methods", []))
    relationship_methods = set(summary.get("relationships", {}).get("methods", []))
    wants_correlation = (
        ("exogenous" in summary and (not exogenous_methods or "collinearity" in exogenous_methods))
        or bool(relationship_methods.intersection({"pearson", "spearman"}))
    )
    if wants_correlation:
        paths["correlation_matrix"] = figure_directory / "correlation_matrix.png"
        selected = summary.get("selected_variables", [])
        _correlation_matrix(frame, config, paths["correlation_matrix"], selected)
    has_lag_rows = any(
        result.get("lag_profile") for result in summary.get("relationships", {}).get("series", {}).values()
    )
    if has_lag_rows:
        paths["lag_relationships"] = figure_directory / "lag_relationships.png"
        _lag_relationships(summary, config, paths["lag_relationships"])
    return paths
