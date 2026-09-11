"""Prepare research data and ask the EDA Subagent for a constrained plan."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.research.agent.context import compact_episode_context
from app.research.agent.schemas import ConversationMessage, EDAPlan, ResearchDataProfile, ResearchProposal
from app.research.agent.subagents.eda import EDASubagent
from app.research.data.alignment import AlignmentResult, align_loaded_series
from app.research.data.loader import LoadedSeries, load_series
from app.research.data.quality import build_quality_report
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.data.sources.materialize import (
    MaterializedSnapshot,
    materialize_dataset,
    restore_dataset,
)
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


def assemble_research_data(
    config: StudyConfig,
    target: LoadedSeries,
    exogenous: list[LoadedSeries],
) -> PreparedResearchData:
    """Derive the analysis frame and quality evidence from already-read series.

    Alignment and quality are pure functions of the series and the configuration,
    so data read from a frozen snapshot produces exactly the same evidence as data
    read from the sources.
    """

    aligned = align_loaded_series(target, exogenous, config)
    quality = build_quality_report(target, exogenous, aligned, config)
    return PreparedResearchData(config=config, target=target, exogenous=exogenous, aligned=aligned, quality=quality)


def prepare_research_data(config: StudyConfig) -> PreparedResearchData:
    """Load all configured series and create the canonical analysis frame."""

    options = {
        "study_timezone": config.study.timezone,
        "start_time": config.study.start_time,
        "end_time": config.study.end_time,
    }
    target = load_series(config.target, **options)
    # A chat-selected short window may predate a recently introduced factor.
    # Preserve it as an all-missing quality finding instead of rejecting the
    # whole price study; deterministic screening will keep it out of analysis.
    exogenous = [load_series(spec, **options, allow_empty=True) for spec in config.exogenous]
    return assemble_research_data(config, target, exogenous)


def restore_prepared_data(fingerprint: str | None, config: StudyConfig) -> tuple[PreparedResearchData, MaterializedSnapshot] | None:
    """Rebuild the analysis inputs from a frozen dataset, or return None if there is none."""

    restored = restore_dataset(fingerprint, config=config)
    if restored is None:
        return None
    return assemble_research_data(config, restored.target, restored.exogenous), restored.snapshot


def freeze_research_data(
    prepared: PreparedResearchData,
    *,
    fingerprint: str,
    input_manifest: list[dict[str, Any]],
) -> MaterializedSnapshot | None:
    """Write this dataset down so the approved plan can no longer be moved under.

    Failing to write is not failing to research: the run falls back to reading the
    sources, which is what happened before snapshots existed.
    """

    try:
        return materialize_dataset(
            config=prepared.config,
            target=prepared.target,
            exogenous=prepared.exogenous,
            quality=prepared.quality,
            fingerprint=fingerprint,
            input_manifest=input_manifest,
        )
    except (OSError, ValueError, ImportError):
        return None


def resolve_config(*, config_path: str | Path | None, study_config: StudyConfig | None) -> StudyConfig:
    if config_path is not None and study_config is not None:
        raise ValueError("provide either config_path or study_config, not both")
    if study_config is not None:
        return study_config
    if config_path is None:
        raise ValueError("a runtime study context is required")
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
        "方案展示后会等待明确确认；如需调整，请直接在对话中提出修改意见，由大模型生成修订版。"
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
        revision_context: dict[str, Any] | None = None,
        episode_summaries: list[dict[str, Any]] | None = None,
    ) -> ResearchProposal:
        callback = progress or noop_progress
        callback(5, "解析数据字段与时间轴")
        config = resolve_config(config_path=config_path, study_config=study_config)
        callback(20, "加载并对齐数据")
        prepared = prepare_research_data(config)
        inputs = input_file_manifest(config)
        current_data_fingerprint = study_fingerprint(config, inputs)
        # Freeze before the plan exists, so what the analyst approves and what the
        # run reads are the same rows even if the sources move in between.
        snapshot = freeze_research_data(prepared, fingerprint=current_data_fingerprint, input_manifest=inputs)
        callback(65, "EDA Subagent 分析问题与数据画像")
        history = [
            {
                "message_id": str(
                    item.message_id if isinstance(item, ConversationMessage) else item.get("message_id", "")
                ),
                "turn_id": item.turn_id if isinstance(item, ConversationMessage) else item.get("turn_id"),
                "episode_id": item.episode_id if isinstance(item, ConversationMessage) else item.get("episode_id"),
                "role": str(item.role if isinstance(item, ConversationMessage) else item.get("role", "user")),
                "content": str(item.content if isinstance(item, ConversationMessage) else item.get("content", "")),
                "created_at": str(
                    item.created_at if isinstance(item, ConversationMessage) else item.get("created_at", "")
                ),
            }
            for item in (conversation or [])
        ]
        episode_memory = compact_episode_context(
            episode_summaries or [],
            data_fingerprint=current_data_fingerprint,
        )
        plan = self.subagent.propose(
            question,
            config,
            prepared.quality,
            history=history,
            skill=skill,
            feedback=feedback,
            revision_context=revision_context,
            episode_memory=episode_memory,
        )
        plan = plan.model_copy(update={"data_fingerprint": current_data_fingerprint})
        profile = _data_profile(prepared)
        callback(100, "分析方案已生成，等待用户反馈")
        return ResearchProposal(
            plan=plan,
            assistant_message=_proposal_message(plan, profile),
            data_profile=profile,
            quality_report=prepared.quality,
            data_summary=snapshot.summary if snapshot is not None else None,
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
        authorization_envelope: dict[str, Any] | None = None,
        episode_summaries: list[dict[str, Any]] | None = None,
    ) -> ResearchProposal:
        proposal = self.propose(
            question=current_plan.question,
            study_config=study_config,
            conversation=conversation,
            progress=progress,
            skill=skill,
            feedback=feedback,
            episode_summaries=episode_summaries,
            revision_context={
                "current_plan": current_plan.model_dump(mode="json"),
                "authorization_envelope": authorization_envelope,
                "allowed_changes": {
                    "selected_variables": "approved subset only",
                    "max_lag": "decrease only within the approved per-function maximum",
                    "enabled_functions": "remove approved functions only; do not add functions",
                },
            },
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
