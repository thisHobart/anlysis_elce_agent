"""Regression tests for the bounded model prompts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.llm.gateway import ModelMessage
from app.research.agent.orchestrator import DIALOGUE_PROMPT_VERSION, ModelResearchDialogue, compact_evidence
from app.research.agent.prompts import DIALOGUE_SYSTEM_PROMPT
from app.research.agent.schemas import EDAPlan
from app.research.agent.subagents.eda import AGENDA_FUNCTION_NAME, PLANNING_PROMPT_VERSION, ModelEDAPlanner
from app.research.application.planning import prepare_research_data
from app.research.reporting.capabilities import FIGURE_CATALOG
from app.research.schemas.study import load_study_config
from app.research.skills.loader import load_skill


class CaptureGateway:
    enabled = True
    model_name = "capture"

    def __init__(self) -> None:
        self.calls: list[list[ModelMessage]] = []

    def invoke_structured(self, *, messages, schema):
        self.calls.append(messages)
        raise RuntimeError(f"captured {schema.__name__}")

    def invoke_text(self, *, messages):
        self.calls.append(messages)
        raise RuntimeError("captured text")


class ToolCallCaptureGateway(CaptureGateway):
    """Capture the tool set a planning call would actually offer the model."""

    def __init__(self) -> None:
        super().__init__()
        self.tools: list[list[dict]] = []

    def invoke_tool_calls(self, *, messages, tools):
        self.calls.append(messages)
        self.tools.append(tools)
        raise RuntimeError("captured tool calls")


def _payload(messages: list[ModelMessage]) -> dict:
    return json.loads(next(message.content for message in messages if message.role == "user"))


def test_planning_prompt_states_local_eda_scope_and_version(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()

    with pytest.raises(RuntimeError, match="captured EDAPlanDraft"):
        ModelEDAPlanner(gateway=gateway).propose(
            "分析电价分布",
            config,
            prepared.quality,
            skill=skill,
        )

    messages = gateway.calls[0]
    assert "输入文件由本地程序只读加载" in messages[0].content
    assert "Function Calling" in messages[0].content
    payload = _payload(messages)
    assert payload["prompt_version"] == PLANNING_PROMPT_VERSION
    assert "task_scope" not in payload
    assert "output_schema" not in payload
    assert payload["output_capabilities"]["automatic_report_generation"] is True
    assert payload["output_capabilities"]["figure_format"] == "SVG"
    assert payload["constraints"]["executable_functions"] == "provided_function_names_only"
    assert payload["constraints"]["function_cardinality"] == "each_function_at_most_once_per_plan"
    assert "price_calendar_group_profile" in payload["allowed_functions"]
    assert payload["allowed_functions"]["price_segment_distribution_comparison"]["batch_parameter"] == "segments"
    assert payload["allowed_functions"]["price_lag_autocorrelation"]["batch_parameter"] == "max_lag"
    assert "reasoning_control" not in payload
    assert (
        payload["active_skill"]["research_protocol"]["protocol_id"]
        == "electricity-price-evidence-ladder"
    )
    assert "method_id" not in json.dumps(payload, ensure_ascii=False)
    assert "target_must_never_appear_in_selected_variables" not in payload["constraints"]


def test_planning_prompt_keeps_episode_memory_outside_trimmed_chat(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()
    history = [
        {
            "message_id": "old-user",
            "turn_id": "turn-old",
            "role": "user",
            "content": "actual_wind 的 max_lag 不超过 24",
        },
        {
            "message_id": "old-assistant",
            "turn_id": "turn-old",
            "role": "assistant",
            "content": "已记录这项历史偏好。",
        },
        *[
            {
                "message_id": f"m-{turn}-{role}",
                "turn_id": f"turn-{turn}",
                "role": role,
                "content": f"第 {turn} 轮普通天气记录",
            }
            for turn in range(1, 6)
            for role in ("user", "assistant")
        ],
        {
            "message_id": "current-user",
            "turn_id": "turn-current",
            "role": "user",
            "content": "继续分析 actual_wind 的滞后关系",
        },
    ]
    memory = [{"episode_id": "episode-kept", "summary": "已验证历史证据。"}]

    with pytest.raises(RuntimeError, match="captured EDAPlanDraft"):
        ModelEDAPlanner(gateway=gateway).propose(
            "继续分析 actual_wind 的滞后关系",
            config,
            prepared.quality,
            history=history,
            episode_memory=memory,
            skill=skill,
        )

    payload = _payload(gateway.calls[0])
    assert len(payload["conversation_history"]) == 8
    assert "继续分析 actual_wind 的滞后关系" not in {
        item["content"] for item in payload["conversation_history"]
    }
    assert [item["turn_id"] for item in payload["earlier_related_turns"]] == ["turn-old"]
    assert [item["role"] for item in payload["earlier_related_turns"][0]["messages"]] == [
        "user",
        "assistant",
    ]
    assert payload["episode_memory"] == memory


def test_dialogue_prompt_states_route_contract_and_version(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()

    with pytest.raises(RuntimeError, match="captured DialogueDecision"):
        ModelResearchDialogue(gateway=gateway).decide(
            question="分析电价结构",
            status="idle",
            config=config,
            plan=None,
            data_profile=None,
            quality_report=prepared.quality.model_dump(mode="json"),
            summary=None,
            evaluation=None,
            history=[],
            available_skills=[skill.prompt_context()],
            active_gate="result_limitations",
            episode_goal="分析电价与外生变量的周期性",
            episode_summaries=[
                {
                    "episode_id": "episode-1",
                    "episode_number": 1,
                    "goal": "分析历史电价分布",
                    "status": "accepted",
                    "run_id": "run-1",
                    "plan_id": "plan-1",
                    "evaluation_decision": "accept",
                    "summary": "历史结果已通过评估。",
                    "findings": ["历史均值稳定。"],
                    "warnings": [],
                    "report_path": "research/run-1/report.md",
                    "figure_count": 1,
                    "figure_keys": ["price_distribution"],
                }
            ],
            latest_run={
                "run_id": "run-current",
                "plan_id": "plan-current",
                "artifact_directory": "research/run-current",
                "report_path": "research/run-current/report.md",
                "figure_paths": {
                    "seasonal_patterns": "research/run-current/figures/seasonal_patterns.svg",
                    "correlation_matrix": "research/run-current/figures/correlation_matrix.svg",
                },
            },
        )

    messages = gateway.calls[0]
    assert "清楚区分“产品支持什么”与“当前回合已经生成什么”" in messages[0].content
    assert "DialogueDecision schema" in messages[0].content
    payload = _payload(messages)
    assert payload["prompt_version"] == DIALOGUE_PROMPT_VERSION
    assert "task_scope" not in payload
    assert "output_schema" not in payload
    assert payload["output_capabilities"]["desktop_report_reader"] is True
    assert payload["interaction_context"]["current_run"]["status"] == "available"
    assert payload["interaction_context"]["current_run"]["figure_count"] == 2
    assert [item["key"] for item in payload["interaction_context"]["current_run"]["figures"]] == [
        "seasonal_patterns",
        "correlation_matrix",
    ]
    assert (
        payload["revision_contract"]["function_selection"]
        == "non_null_enabled_functions_is_complete_replacement"
    )
    assert payload["revision_contract"]["agenda_selection"] == (
        "a_replaced_function_set_rebuilds_the_agenda_from_retained_functions"
    )
    assert "reasoning_control" not in payload
    assert payload["study"]["target"]["name"] == config.target.name
    assert payload["episode_memory"][0]["episode_id"] == "episode-1"
    assert payload["episode_memory"][0]["findings"] == ["历史均值稳定。"]
    assert payload["interaction_context"]["active_gate"] == "result_limitations"
    assert payload["interaction_context"]["episode_goal"] == "分析电价与外生变量的周期性"
    assert payload["episode_memory"][0]["figure_keys"] == ["price_distribution"]


def test_dialogue_prompt_does_not_treat_missing_current_artifacts_as_missing_capability(
    synthetic_study: Path,
):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()

    with pytest.raises(RuntimeError, match="captured DialogueDecision"):
        ModelResearchDialogue(gateway=gateway).decide(
            question="项目能生成周期性图表吗？",
            status="awaiting_approval",
            config=config,
            plan=None,
            data_profile=None,
            quality_report=prepared.quality.model_dump(mode="json"),
            summary=None,
            evaluation=None,
            history=[],
            available_skills=[skill.prompt_context()],
            latest_run=None,
        )

    messages = gateway.calls[0]
    payload = _payload(messages)
    assert payload["interaction_context"]["current_run"] == {
        "status": "not_generated_for_current_episode",
        "run_id": None,
        "report_path": None,
        "figure_count": 0,
        "figures": [],
    }
    figure_catalog = {
        item["key"]: item for item in payload["output_capabilities"]["figure_catalog"]
    }
    assert figure_catalog["seasonal_patterns"]["form"] == "柱状图"
    assert figure_catalog["price_month_profile"]["title"] == "分月份平均电价"
    assert figure_catalog["correlation_matrix"]["form"] == "热力图"
    assert set(figure_catalog) == set(FIGURE_CATALOG)
    assert "status 不是 available 时，只能说当前尚未生成" in messages[0].content


def test_compact_evidence_preserves_periodic_results_without_large_profiles():
    summary = {
        "study": {"target": "price"},
        "price": {
            "methods": ["price_calendar_group_profile", "price_seasonal_decomposition"],
            "seasonality": {"hour_of_day": [{"group": 0, "mean": 10.0}]},
            "decomposition": {"seasonal_strength": {"day": 0.7}},
            "autocorrelation": [
                {"lag": lag, "correlation": lag / 100}
                for lag in range(1, 41)
            ],
            "partial_autocorrelation": {
                "max_lag": 40,
                "series": [{"lag": lag, "partial_correlation": 0.1} for lag in range(1, 41)],
                "strongest_lags": [{"lag": 1, "partial_correlation": 0.4}],
            },
            "duration_curve": {
                "points": [{"exceedance_share": index / 100, "price": index} for index in range(101)],
                "share_above_zero": 0.9,
            },
        },
        "relationships": {
            "methods": ["relationship_pearson_by_hour", "relationship_pearson_by_month"],
            "series": {
                "load": {
                    "by_hour": [{"group": 0, "correlation": 0.2}],
                    "by_month": [{"group": 1, "correlation": -0.1}],
                }
            },
            "mutual_information": {
                "series": {
                    "load": {
                        "best_lag": {"lag": 2, "normalized_mutual_information": 0.12},
                        "profile": [{"lag": lag} for lag in range(40)],
                    }
                }
            },
            "rolling_stability": {
                "series": {
                    "load": {
                        "mean_correlation": 0.2,
                        "timeline": [{"timestamp": str(index)} for index in range(40)],
                    }
                }
            },
        },
    }

    compact = compact_evidence(summary, None)

    assert compact["price"]["seasonality"] == summary["price"]["seasonality"]
    assert compact["price"]["decomposition"] == summary["price"]["decomposition"]
    assert compact["price"]["autocorrelation"]["evaluated_lags"] == 40
    assert len(compact["price"]["autocorrelation"]["strongest_lags"]) == 8
    assert "series" not in compact["price"]["partial_autocorrelation"]
    assert compact["price"]["duration_curve"]["plotted_point_count"] == 101
    assert "points" not in compact["price"]["duration_curve"]
    assert compact["relationships"]["series"]["load"]["by_hour"][0]["correlation"] == 0.2
    assert compact["relationships"]["series"]["load"]["by_month"][0]["correlation"] == -0.1
    assert "profile" not in compact["relationships"]["mutual_information"]["series"]["load"]
    assert "timeline" not in compact["relationships"]["rolling_stability"]["series"]["load"]


def test_automatic_revision_prompt_contains_current_plan_and_approval_boundary(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()
    revision_context = {
        "current_plan": {"plan_id": "approved-plan", "selected_variables": ["load"]},
        "authorization_envelope": {"functions": ["price_descriptive_distribution"]},
        "allowed_changes": {"max_lag": "decrease only"},
    }

    with pytest.raises(RuntimeError, match="captured EDAPlanDraft"):
        ModelEDAPlanner(gateway=gateway).propose_with_context(
            "分析电价",
            config,
            prepared.quality,
            skill=skill,
            revision_context=revision_context,
        )

    messages = gateway.calls[0]
    assert "revision_context" in messages[0].content
    payload = _payload(messages)
    assert payload["revision_context"] == revision_context


def test_recorded_planning_prompt_version_cannot_drift_from_the_prompt_in_use():
    """The version written into research_plan.json is provenance, not a label."""

    assert EDAPlan.model_fields["planning_prompt_version"].default == PLANNING_PROMPT_VERSION


def test_capability_context_locates_methods_and_report_content(synthetic_study: Path):
    """The report stopped carrying methods and thresholds; the model must know where they went."""

    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()

    with pytest.raises(RuntimeError, match="captured DialogueDecision"):
        ModelResearchDialogue(gateway=gateway).decide(
            question="判读阈值写在哪里？",
            status="idle",
            config=config,
            plan=None,
            data_profile=None,
            quality_report=prepared.quality.model_dump(mode="json"),
            summary=None,
            evaluation=None,
            history=[],
            available_skills=[skill.prompt_context()],
        )

    capabilities = _payload(gateway.calls[0])["output_capabilities"]
    assert capabilities["methods_file"] == "methods.md"
    assert "判读阈值" in capabilities["methods_file_contains"]
    assert "读图" in capabilities["report_file_contains"]
    assert "methods.md" in DIALOGUE_SYSTEM_PROMPT


def test_dialogue_prompt_names_the_evidence_keys_the_payload_actually_carries(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    gateway = CaptureGateway()

    with pytest.raises(RuntimeError, match="captured DialogueDecision"):
        ModelResearchDialogue(gateway=gateway).decide(
            question="上一轮的结论是什么？",
            status="idle",
            config=config,
            plan=None,
            data_profile=None,
            quality_report=prepared.quality.model_dump(mode="json"),
            summary={"price": {"distribution": {"mean": 1.0}}},
            evaluation={"decision": "accept", "summary": "已收敛"},
            history=[],
            available_skills=[],
        )

    payload = _payload(gateway.calls[0])
    assert "evaluation" not in payload
    assert payload["evidence"]["evaluation"]["decision"] == "accept"
    assert "evidence.evaluation" in DIALOGUE_SYSTEM_PROMPT


def test_model_facing_domain_text_uses_the_analysis_vocabulary(synthetic_study: Path):
    """`门禁` is the architecture's approval vocabulary, not what an analyst is told."""

    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()

    with pytest.raises(RuntimeError, match="captured EDAPlanDraft"):
        ModelEDAPlanner(gateway=gateway).propose("分析电价分布", config, prepared.quality, skill=skill)

    payload = _payload(gateway.calls[0])
    assert "门禁" not in json.dumps(payload, ensure_ascii=False)


def test_a_data_only_skill_still_gets_the_agenda_function(synthetic_study: Path):
    """Without the agenda call there is no way to answer 'can this data be used at all'."""

    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    data_only = skill.model_copy(update={"allowed_functions": ["data_quality"]})
    gateway = ToolCallCaptureGateway()

    with pytest.raises(RuntimeError, match="captured tool calls"):
        ModelEDAPlanner(gateway=gateway).propose("这批数据能用吗", config, prepared.quality, skill=data_only)

    offered = [tool["function"]["name"] for tool in gateway.tools[0]]
    assert offered == [AGENDA_FUNCTION_NAME]
