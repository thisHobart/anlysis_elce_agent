"""Build a traceable Markdown narrative from structured EDA evidence."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig


def _fmt(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{decimals}f}"
    return str(value)


def _lag_duration(lag: int | None, frequency: str) -> str:
    if lag is None:
        return "—"
    offset = pd.tseries.frequencies.to_offset(frequency)
    seconds = offset.nanos / 1_000_000_000
    hours = lag * seconds / 3600
    return f"{lag} intervals ({hours:g} h)"


def _quality_table(quality: DataQualityReport) -> list[str]:
    lines = [
        "| Series | Raw rows | Coverage | Missing intervals | Outliers | Availability |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for report in quality.series.values():
        lines.append(
            f"| {report.name} | {report.raw_rows} | {report.aligned_coverage_rate:.2%} | {report.missing_interval_count:,} | {report.outlier_count:,} | {report.availability_type} |"
        )
    return lines


def _issues_table(quality: DataQualityReport) -> list[str]:
    if not quality.issues:
        return ["No automated quality issues were detected."]
    lines = ["| Severity | Series | Issue | Risk |", "|---|---|---|---|"]
    for issue in quality.issues:
        lines.append(f"| {issue.severity} | {issue.series or 'study'} | {issue.code} | {issue.message} |")
    return lines


def _relationship_table(summary: dict[str, Any], frequency: str) -> list[str]:
    rows = []
    for name, result in summary["relationships"]["series"].items():
        pearson = result["contemporaneous"]["pearson"]
        spearman = result["contemporaneous"]["spearman"]
        best = result.get("best_absolute_lag") or {}
        rows.append(
            {
                "name": name,
                "pearson": pearson["correlation"],
                "spearman": spearman["correlation"],
                "observations": pearson["observations"],
                "best_lag": best.get("lag"),
                "best_correlation": best.get("correlation"),
            }
        )
    rows.sort(key=lambda row: abs(row["pearson"] or 0), reverse=True)
    lines = [
        "| Variable | Pairwise N | Pearson | Spearman | Strongest lead | Lead correlation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {n:,} | {pearson} | {spearman} | {lag} | {best} |".format(
                name=row["name"],
                n=row["observations"],
                pearson=_fmt(row["pearson"]),
                spearman=_fmt(row["spearman"]),
                lag=_lag_duration(row["best_lag"], frequency),
                best=_fmt(row["best_correlation"]),
            )
        )
    return lines


def build_eda_report(config: StudyConfig, quality: DataQualityReport, summary: dict[str, Any]) -> str:
    """Create a concise report whose claims are traceable to JSON artifacts."""

    price = summary["price"]
    distribution = price["distribution"]
    signs = price["signs"]
    extremes = price["extremes"]
    hourly = price["seasonality"]["hour_of_day"]
    highest_hour = max(hourly, key=lambda row: row["mean"] if row["mean"] is not None else float("-inf"))
    lowest_hour = min(hourly, key=lambda row: row["mean"] if row["mean"] is not None else float("inf"))
    strong_pairs = summary["exogenous"]["strong_collinearity_pairs"]

    lines = [
        f"# {config.study.name}: Electricity Price and Exogenous Variables EDA",
        "",
        "> This is descriptive exploratory analysis. Correlation, lag correlation, and grouped differences do not establish causation.",
        "",
        "## Study scope",
        "",
        f"- Market: `{config.study.market}`",
        f"- Target: `{config.target.name}` ({config.target.unit})",
        f"- Window: `{quality.alignment.start_time}` to `{quality.alignment.end_time}`",
        f"- Canonical frequency/timezone: `{config.study.frequency}` / `{config.study.timezone}`",
        f"- Canonical rows: `{quality.alignment.expected_rows:,}`",
        (
            f"- Complete cases across all variables: `{quality.alignment.complete_case_rows:,}` "
            f"(`{quality.alignment.complete_case_rate:.2%}`)"
        ),
        "- Relationship calculations use pairwise-complete observations, not all-variable complete cases.",
        "",
        "## Data quality",
        "",
        f"EDA readiness: **{'usable' if quality.usable_for_eda else 'not usable'}**.",
        "",
        *_quality_table(quality),
        "",
        "### Quality risks",
        "",
        *_issues_table(quality),
        "",
        "## Price profile",
        "",
        f"- Observed intervals: **{price['observations']:,}**; coverage **{price['coverage_rate']:.2%}**.",
        f"- Mean / median: **{_fmt(distribution['mean'])} / {_fmt(distribution['median'])}** {config.target.unit}.",
        f"- Range: **{_fmt(distribution['min'])} to {_fmt(distribution['max'])}** {config.target.unit}.",
        f"- Negative-price intervals: **{signs['negative_count']:,} ({signs['negative_rate']:.2%})**.",
        (
            f"- Outer-fence high spikes: **{extremes['high_spike_count']:,} "
            f"({extremes['high_spike_rate']:.2%})**; threshold `{_fmt(extremes['upper_threshold'])}`."
        ),
        (
            f"- Highest mean hour: **{highest_hour['group']:02d}:00** at `{_fmt(highest_hour['mean'])}`; "
            f"lowest mean hour: **{lowest_hour['group']:02d}:00** at `{_fmt(lowest_hour['mean'])}`."
        ),
        "",
        "![Price over time](figures/price_timeseries.png)",
        "",
        "![Price distribution](figures/price_distribution.png)",
        "",
        "![Seasonal patterns](figures/seasonal_patterns.png)",
        "",
        "## Exogenous-variable relationships",
        "",
        *_relationship_table(summary, config.study.frequency),
        "",
        (
            "The p-values in the structured result are descriptive screening evidence only. Large time-series samples, serial "
            "correlation, multiple comparisons, changing regimes, and incomplete coverage can make conventional p-values look "
            "stronger than the evidence warrants."
        ),
        "",
        "![Correlation matrix](figures/correlation_matrix.png)",
        "",
        "![Lag relationships](figures/lag_relationships.png)",
        "",
        "### Strong exogenous collinearity",
        "",
    ]
    if strong_pairs:
        lines.extend(["| Left | Right | Correlation | Pairwise N |", "|---|---|---:|---:|"])
        for pair in strong_pairs:
            lines.append(
                f"| {pair['left']} | {pair['right']} | {_fmt(pair['correlation'])} | {pair['observations']:,} |"
            )
    else:
        lines.append("No exogenous pair exceeded the configured strong-correlation threshold.")

    lines.extend(
        [
            "",
            "## Interpretation limits and next checks",
            "",
            (
                "- Source units and market identity are marked as unknown in the inferred study context and must be confirmed "
                "before business interpretation."
            ),
            "- Variables marked `observed_only` are useful for descriptive diagnosis but may leak future information in forecasting.",
            "- Day-ahead and other forecast variables must be verified against their actual publication/availability timestamps.",
            (
                "- The next prediction phase should use rolling time splits and should evaluate feature transformations inside "
                "each training fold."
            ),
            "- Negative prices and spikes are retained as legitimate market outcomes; they are not silently removed as data errors.",
            "",
            "## Reproducibility artifacts",
            "",
            "- `study_context.json`: inferred data and time-axis context used by this run",
            "- `data_quality.json`: source and alignment quality evidence",
            "- `eda_summary.json`: complete structured EDA output",
            "- `aligned_data.parquet`: canonical aligned analysis frame",
            "- `manifest.json`: input/output hashes, Git state, and runtime versions",
            "",
        ]
    )
    return "\n".join(lines)
