"""Validate desktop data selections and build their runtime study context."""

from __future__ import annotations

from pathlib import Path

from app.desktop.session import InputRole, ResearchSession
from app.research.data.inference import infer_study_context
from app.research.schemas.study import StudyConfig

ROLE_LABELS: dict[InputRole, str] = {
    "target": "目标电价",
    "actuals": "实际外生变量",
    "forecasts": "预测外生变量",
}
FILE_FILTERS: dict[InputRole, str] = {
    "target": "数据文件 (*.csv *.parquet *.pq)",
    "actuals": "数据文件 (*.csv *.parquet *.pq)",
    "forecasts": "数据文件 (*.csv *.parquet *.pq)",
}


def validate_input_path(role: InputRole, path: str | Path) -> Path:
    """Validate only existence and format; filenames are deliberately unrestricted."""

    if role not in ROLE_LABELS:
        raise ValueError(f"不支持的数据文件角色：{role}")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"文件不存在：{resolved}")
    if resolved.suffix.casefold() not in {".csv", ".parquet", ".pq"}:
        raise ValueError(f"{ROLE_LABELS[role]}只支持 CSV/Parquet 文件")
    return resolved


def build_runtime_study(session: ResearchSession, *, output_directory: Path | None = None) -> StudyConfig:
    """Infer the internal execution contract from the three desktop data roles."""

    if not session.inputs["target"].path:
        raise ValueError("执行 EDA 前至少需要选择目标电价文件")
    target = validate_input_path("target", session.inputs["target"].path)
    actuals = (
        validate_input_path("actuals", session.inputs["actuals"].path)
        if session.inputs["actuals"].path
        else None
    )
    forecasts = (
        validate_input_path("forecasts", session.inputs["forecasts"].path)
        if session.inputs["forecasts"].path
        else None
    )
    return infer_study_context(
        target_path=target,
        actuals_path=actuals,
        forecasts_path=forecasts,
        output_directory=output_directory,
    )
