"""Prepare research data and ask the EDA Subagent for a constrained plan."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.research.agent.schemas import ConversationMessage, EDAPlan, ResearchDataProfile, ResearchProposal
from app.research.agent.subagents.eda import EDASubagent
from app.research.data.alignment import AlignmentResult, align_loaded_series
from app.research.data.loader import LoadedSeries, load_series
from app.research.data.quality import build_quality_report
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.schemas.feedback import FeedbackPacket
from app.research.schemas.results import DataQualityReport
from app.research.schemas.study import StudyConfig, load_study_config
from app.research.skills.contracts import SkillDefinition

ProgressCallback = Callable[[int, str], None]


def noop_progress(_value: int, _message: str) -> None:
    return None


@dataclass(frozen=True)
class PreparedResearchData:
    """Loaded and aligned data shared by planning and deterministic execution."""

    config: StudyConfig
    target: LoadedSeries
    exogenous: list[LoadedSeries]
    aligned: AlignmentResult
    quality: DataQualityReport


def prepare_research_data(config: StudyConfig) -> PreparedResearchData:
    """Load all configured series and create the canonical analysis frame."""

    options = {
        "study_timezone": config.study.timezone,
        "start_time": config.study.start_time,
        "end_time": config.study.end_time,
    }
    target = load_series(config.target, **options)
    exogenous = [load_series(spec, **options) for spec in config.exogenous]
    aligned = align_loaded_series(target, exogenous, config)
    quality = build_quality_report(target, exogenous, aligned, config)
    return PreparedResearchData(config=config, target=target, exogenous=exogenous, aligned=aligned, quality=quality)


def resolve_config(*, config_path: str | Path | None, study_config: StudyConfig | None) -> StudyConfig:
    if config_path is not None and study_config is not None:
        raise ValueError("provide either config_path or study_config, not both")
    if study_config is not None:
        return study_config
    if config_path is None:
        raise ValueError("a research configuration is required")
    return load_study_config(config_path)


def _data_profile(prepared: PreparedResearchData) -> ResearchDataProfile:
    severity = Counter(issue.severity for issue in prepared.quality.issues)
    target_report = prepared.quality.series[prepared.config.target.name]
    return ResearchDataProfile(
        target_name=prepared.config.target.name,
        target_unit=prepared.config.target.unit or "unknown",
        market=prepared.config.study.market,
        exogenous_names=[spec.name for spec in prepared.config.exogenous],
        aligned_rows=len(prepared.aligned.frame),
        start_time=prepared.quality.alignment.start_time,
        end_time=prepared.quality.alignment.end_time,
        frequency=prepared.config.study.frequency,
        timezone=prepared.config.study.timezone,
        target_coverage_rate=target_report.aligned_coverage_rate,
        issue_counts=dict(severity),
    )


def _proposal_message(plan: EDAPlan, profile: ResearchDataProfile) -> str:
    enabled = "、".join(step.title for step in plan.enabled_steps)
    metadata_note = ""
    if profile.target_unit.casefold() in {"", "unknown", "unspecified"} or "unspecified" in profile.market.casefold():
        metadata_note = " 当前市场或单位元数据尚不完整，业务解释前需要确认。"
    return (
        f"我已检查 {profile.aligned_rows:,} 个对齐时间点，目标 {profile.target_name} 的覆盖率为 "
        f"{profile.target_coverage_rate:.2%}。大模型建议执行：{enabled}。"
        "方案展示后会等待 30 秒；如需调整，请直接在对话中提出修改意见，由大模型生成修订版。"
        f"{metadata_note}"
    )


class EDAPlanningService:
    """Application service that supplies data evidence to the EDA Subagent."""

    def __init__(self, subagent: EDASubagent) -> None:
        self.subagent = subagent

    def propose(
        self,
        *,
        question: str,
        config_path: str | Path | None = None,
        study_config: StudyConfig | None = None,
        conversation: list[ConversationMessage | dict[str, Any]] | None = None,
        progress: ProgressCallback | None = None,
        skill: SkillDefinition | None = None,
        feedback: list[FeedbackPacket] | None = None,
    ) -> ResearchProposal:
        callback = progress or noop_progress
        callback(5, "读取研究配置")
        config = resolve_config(config_path=config_path, study_config=study_config)
        callback(20, "加载并对齐数据")
        prepared = prepare_research_data(config)
        callback(65, "EDA Subagent 分析问题与数据画像")
        history = [
            {
                "role": str(item.role if isinstance(item, ConversationMessage) else item.get("role", "user")),
                "content": str(item.content if isinstance(item, ConversationMessage) else item.get("content", "")),
            }
            for item in (conversation or [])
        ]
        plan = self.subagent.propose(
            question,
            config,
            prepared.quality,
            history=history,
            skill=skill,
            feedback=feedback,
        )
        inputs = input_file_manifest(config)
        plan = plan.model_copy(update={"data_fingerprint": study_fingerprint(config, inputs)})
        profile = _data_profile(prepared)
        callback(100, "分析方案已生成，等待用户反馈")
        return ResearchProposal(
            plan=plan,
            assistant_message=_proposal_message(plan, profile),
            data_profile=profile,
            quality_report=prepared.quality,
        )

    def revise_from_feedback(
        self,
        *,
        current_plan: EDAPlan,
        study_config: StudyConfig,
        skill: SkillDefinition,
        feedback: list[FeedbackPacket],
        conversation: list[ConversationMessage | dict[str, Any]] | None = None,
        source: str = "automatic_evaluation",
        progress: ProgressCallback | None = None,
    ) -> ResearchProposal:
        proposal = self.propose(
            question=current_plan.question,
            study_config=study_config,
            conversation=conversation,
            progress=progress,
            skill=skill,
            feedback=feedback,
        )
        reason = "；".join(item.message for item in feedback) or "根据结构化反馈自动修订。"
        revised = proposal.plan.model_copy(
            update={
                "parent_plan_id": current_plan.plan_id,
                "revision": current_plan.revision + 1,
                "revision_reason": reason,
                "revision_source": source,
                "planning_notes": [*proposal.plan.planning_notes, reason],
            }
        )
        return proposal.model_copy(update={"plan": revised})
