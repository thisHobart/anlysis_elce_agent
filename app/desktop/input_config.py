"""Build an executable study config from optional, user-selected files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.desktop.session import InputRole, ResearchSession
from app.research.schemas.study import AnalysisSettings, SeriesSpec, StudyConfig, StudyDefinition, load_study_config
from app.runtime_paths import default_research_output_directory

ROLE_LABELS: dict[InputRole, str] = {
    "config": "研究配置",
    "target": "目标电价",
    "actuals": "实际外生变量",
    "forecasts": "预测外生变量",
}
FILE_FILTERS: dict[InputRole, str] = {
    "config": "YAML 配置 (*.yaml *.yml)",
    "target": "数据文件 (*.csv *.parquet *.pq)",
    "actuals": "数据文件 (*.csv *.parquet *.pq)",
    "forecasts": "数据文件 (*.csv *.parquet *.pq)",
}


def validate_input_path(role: InputRole, path: str | Path) -> Path:
    """Validate only existence and format; filenames are deliberately unrestricted."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"文件不存在：{resolved}")
    suffix = resolved.suffix.casefold()
    allowed = {".yaml", ".yml"} if role == "config" else {".csv", ".parquet", ".pq"}
    if suffix not in allowed:
        expected = "YAML" if role == "config" else "CSV/Parquet"
        raise ValueError(f"{ROLE_LABELS[role]}只支持 {expected} 文件")
    return resolved


def _role_paths(config: StudyConfig, role: InputRole) -> set[Path]:
    if role == "actuals":
        return {spec.path for spec in config.exogenous if spec.availability_type == "observed_only"}
    if role == "forecasts":
        return {spec.path for spec in config.exogenous if spec.availability_type == "forecast"}
    return set()


def discover_inputs_from_config(path: str | Path) -> dict[InputRole, Path]:
    """Populate optional file slots when a YAML references one source per role."""

    config_path = validate_input_path("config", path)
    config = load_study_config(config_path)
    discovered: dict[InputRole, Path] = {
        "config": config_path,
        "target": validate_input_path("target", config.target.path),
    }
    for role in ("actuals", "forecasts"):
        paths = _role_paths(config, role)
        if len(paths) == 1:
            discovered[role] = validate_input_path(role, next(iter(paths)))
    return discovered


def _read_sample(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, nrows=2500)
        return pd.read_parquet(path).head(2500)
    except Exception as exc:
        raise ValueError(f"无法读取 {path.name}：{exc}") from exc


def _timestamp_column(frame: pd.DataFrame, filename: str) -> str:
    preferred = ("datetime", "timestamp", "time", "date", "日期", "时间")
    by_lower = {str(column).casefold(): str(column) for column in frame.columns}
    for name in preferred:
        if name.casefold() in by_lower:
            return by_lower[name.casefold()]
    for column in frame.columns:
        values = frame[column].dropna()
        if values.empty or pd.api.types.is_numeric_dtype(values):
            continue
        parsed = pd.to_datetime(values, errors="coerce")
        if float(parsed.notna().mean()) >= 0.9:
            return str(column)
    raise ValueError(f"{filename} 中无法自动识别时间列，请提供 YAML 研究配置")


def _numeric_columns(frame: pd.DataFrame, timestamp_column: str, filename: str) -> list[str]:
    result: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name == timestamp_column:
            continue
        source = frame[column].dropna()
        if source.empty:
            continue
        converted = pd.to_numeric(source, errors="coerce")
        if float(converted.notna().mean()) >= 0.9:
            result.append(name)
    if not result:
        raise ValueError(f"{filename} 中没有可用数值列，请提供 YAML 研究配置")
    return result


def _target_value_column(columns: list[str]) -> str:
    priorities = ("rt_node_price_2b", "rt_price", "real_time_price", "price", "电价")
    lowered = {column.casefold(): column for column in columns}
    for name in priorities:
        if name in lowered:
            return lowered[name]
    contains_price = [column for column in columns if "price" in column.casefold() or "电价" in column]
    return contains_price[0] if contains_price else columns[0]


def _safe_name(value: str, *, prefix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-")
    if not normalized or not normalized[0].isalpha():
        normalized = f"{prefix}_{normalized or 'value'}"
    return normalized


def _frequency(frame: pd.DataFrame, timestamp_column: str) -> str:
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(timestamps) >= 3:
        try:
            inferred = pd.infer_freq(timestamps)
        except ValueError:
            inferred = None
        if inferred:
            return inferred
    differences = timestamps.diff().dropna().dt.total_seconds()
    if differences.empty:
        return "1h"
    seconds = max(1, round(float(differences.median())))
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def _auto_config(session: ResearchSession, *, output_directory: Path | None = None) -> StudyConfig:
    if not session.inputs["target"].path:
        raise ValueError("执行 EDA 前至少需要选择目标电价文件")
    target_path = validate_input_path("target", session.inputs["target"].path)
    target_frame = _read_sample(target_path)
    target_timestamp = _timestamp_column(target_frame, target_path.name)
    target_columns = _numeric_columns(target_frame, target_timestamp, target_path.name)
    target_value = _target_value_column(target_columns)

    target = SeriesSpec(
        name=_safe_name(target_value, prefix="price"),
        path=target_path,
        timestamp_column=target_timestamp,
        value_column=target_value,
        unit="unknown",
        availability_type="observed_only",
    )
    exogenous: list[SeriesSpec] = []
    used_names = {target.name}
    for role, availability, prefix in (
        ("actuals", "observed_only", "actual"),
        ("forecasts", "forecast", "forecast"),
    ):
        path_text = session.inputs[role].path
        if not path_text:
            continue
        source_path = validate_input_path(role, path_text)  # type: ignore[arg-type]
        frame = _read_sample(source_path)
        timestamp = _timestamp_column(frame, source_path.name)
        for column in _numeric_columns(frame, timestamp, source_path.name):
            name = _safe_name(column, prefix=prefix)
            if name in used_names:
                name = _safe_name(f"{prefix}_{name}", prefix=prefix)
            counter = 2
            candidate = name
            while candidate in used_names:
                candidate = f"{name}_{counter}"
                counter += 1
            used_names.add(candidate)
            exogenous.append(
                SeriesSpec(
                    name=candidate,
                    path=source_path,
                    timestamp_column=timestamp,
                    value_column=column,
                    unit="unknown",
                    availability_type=availability,
                )
            )
    return StudyConfig(
        study=StudyDefinition(
            name="desktop_auto_eda",
            market="unspecified",
            timezone="Asia/Shanghai",
            frequency=_frequency(target_frame, target_timestamp),
        ),
        target=target,
        exogenous=exogenous,
        analysis=AnalysisSettings(output_directory=output_directory or default_research_output_directory()),
    )


def build_session_study_config(
    session: ResearchSession,
    *,
    output_directory: Path | None = None,
) -> StudyConfig:
    """Use optional YAML metadata when present, otherwise infer a safe minimal config."""

    config_path = session.inputs["config"].path
    if not config_path:
        return _auto_config(session, output_directory=output_directory)

    base = load_study_config(validate_input_path("config", config_path))
    target_path = (
        validate_input_path("target", session.inputs["target"].path)
        if session.inputs["target"].path
        else validate_input_path("target", base.target.path)
    )
    target = base.target.model_copy(update={"path": target_path})
    exogenous: list[SeriesSpec] = []
    for role in ("actuals", "forecasts"):
        path_text = session.inputs[role].path
        if not path_text:
            continue
        selected_path = validate_input_path(role, path_text)  # type: ignore[arg-type]
        exogenous.extend(
            spec.model_copy(update={"path": selected_path})
            for spec in base.exogenous
            if spec.path in _role_paths(base, role)
        )
    analysis = base.analysis
    if output_directory is not None:
        analysis = analysis.model_copy(update={"output_directory": output_directory.resolve()})
    return base.model_copy(update={"target": target, "exogenous": exogenous, "analysis": analysis})
