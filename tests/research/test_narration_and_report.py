"""Checks for user-facing narration, the second core Skill, and the HTML report."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.research.application.execution import EDAExecutionService
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.graph.narration import (
    NODE_PROGRESS,
    SILENT_NODES,
    narrate_event,
    narrate_node,
    progress_message,
    split_progress_message,
)
from app.research.planning.compiler import EDAPlanCompiler
from app.research.planning.contracts import EDAPlanDraft
from app.research.schemas.study import load_study_config
from app.research.skills.registry import SkillRegistry
from app.research.tools.catalog import STAGE_TITLES, TOOL_CATALOG


def test_narration_turns_loop_events_into_plain_language():
    step = narrate_event(
        {
            "name": "准备执行函数：电价平稳性检验 [price_stationarity_tests]",
            "status": "running",
            "details": {"function": "price_stationarity_tests"},
        }
    )

    assert step.stage_label == "执行分析"
    assert step.title == "分析中 · 电价平稳性检验"
    assert step.function_name == "price_stationarity_tests"
    assert step.detail == TOOL_CATALOG["price_stationarity_tests"].description.rstrip("。")
    assert "price_stationarity_tests" not in step.title


def test_narration_keeps_identifiers_out_of_routing_and_evaluation_lines():
    routing = narrate_event({"name": "主 Agent 路由：new_plan", "status": "completed", "details": {"intent": "new_plan"}})
    evaluation = narrate_event(
        {"name": "评估运行：agent-123 → accept", "status": "completed", "details": {"decision": "accept"}}
    )
    locking = narrate_event({"name": "锁定函数队列：abc123 · 6 个调用", "status": "completed", "details": {"calls": 6}})

    assert routing.title == "识别处理方式"
    assert routing.detail == "生成新的分析方案"
    assert evaluation.detail == "证据充分，本轮结论已收敛"
    assert "abc123" not in locking.detail
    assert "共 6 项" in locking.detail


def test_failed_events_are_reported_as_problems():
    step = narrate_event(
        {
            "name": "函数执行失败：电价平稳性检验 [price_stationarity_tests] · 有效样本不足",
            "status": "failed",
            "details": {},
        }
    )

    assert step.stage_label == "处理异常"
    assert step.title == "分析执行失败 · 电价平稳性检验"
    assert step.status == "failed"
    assert step.detail == "有效样本不足"


def test_progress_messages_round_trip_through_the_desktop_encoding():
    raw_name = "函数执行完成：电价日历规律 [price_calendar_group_profile]"
    original = narrate_event({"name": raw_name, "status": "completed", "details": {"duration_ms": 1500.0}})
    source_event, decoded = split_progress_message(progress_message(original, raw_name))

    assert source_event == raw_name
    assert decoded.stage == original.stage
    assert decoded.title == original.title
    assert decoded.detail == original.detail
    assert decoded.function_name == "price_calendar_group_profile"


def test_progress_messages_without_a_source_event_still_decode():
    source_event, decoded = split_progress_message(progress_message(narrate_node("lock_plan")[1]))

    assert source_event == ""
    assert decoded.title == "锁定执行计划"
    assert decoded.stage_label == "生成方案"


def test_every_graph_node_has_a_staged_label_or_is_silent():
    from app.research.graph.narration import STAGE_LABELS

    for node_name, (_value, stage, title, _detail) in NODE_PROGRESS.items():
        assert stage in STAGE_LABELS, node_name
        assert title and not title.isascii(), node_name
    assert SILENT_NODES.isdisjoint(NODE_PROGRESS)
    assert narrate_node("unknown_node")[1].title == "研究进行中"


def test_two_core_skills_ship_with_the_application():
    registry = SkillRegistry.default(Settings(skill_paths=""))
    names = {item["name"] for item in registry.metadata()}

    assert names == {"price-exogenous-eda", "price-forecastability-audit"}
    audit = registry.get("price-forecastability-audit")
    assert audit.domain == "eda"
    assert audit.research_protocol is not None
    assert "data_quality" in audit.allowed_tools
    assert not any(name.startswith(("exogenous_", "relationship_")) for name in audit.allowed_tools)


def test_catalog_stages_match_the_builtin_protocol_stages():
    for skill in (
        SkillRegistry.default(Settings(skill_paths="")).get("price-exogenous-eda"),
    ):
        assert skill.research_protocol is not None
        for stage in skill.research_protocol.stages:
            for function_name in stage.functions:
                assert TOOL_CATALOG[function_name].stage == stage.stage_id
            assert stage.stage_id in STAGE_TITLES or not stage.functions


@pytest.fixture
def forecastability_result(synthetic_study: Path, tmp_path: Path):
    config = load_study_config(synthetic_study)
    skill = SkillRegistry.default(Settings(skill_paths="")).get("price-forecastability-audit")
    steps = [
        {"tool": name, "enabled": True, "rationale": "可预测性审计", "parameters": {"max_lag": 24} if TOOL_CATALOG[name].uses_max_lag else {}}
        for name in (
            "price_descriptive_distribution",
            "price_duration_curve",
            "price_spike_regime_profile",
            "price_stationarity_tests",
            "price_partial_autocorrelation",
            "price_variance_stabilization_check",
            "price_naive_baseline_benchmark",
        )
    ]
    plan = EDAPlanCompiler().compile(
        EDAPlanDraft(objective="判断这条电价序列有多可预测", selected_variables=[], steps=steps),
        question="这条电价序列好预测吗？",
        config=config,
        skill=skill,
        model_name="regression",
        prompt_version="regression",
    )
    plan = plan.model_copy(update={"data_fingerprint": study_fingerprint(config, input_file_manifest(config))})
    return EDAExecutionService().execute(
        plan=plan,
        study_config=config,
        output_directory=tmp_path / "artifacts",
        run_id="forecastability-audit",
    )


def test_target_only_skill_produces_a_report_without_driver_sections(forecastability_result):
    result = forecastability_result

    assert result.plan.skill_name == "price-forecastability-audit"
    assert result.plan.selected_variables == []
    assert "exogenous" not in result.eda_summary
    assert "relationships" not in result.eda_summary

    report = result.report_path.read_text(encoding="utf-8")
    assert result.report_path.name == "report.html"
    assert "影响因素质量" not in report
    assert "电价与影响因素的关系" not in report
    assert "可预测性与建模就绪" in report
    assert "<svg" in report
    assert "报告由确定性统计函数生成" in report


def test_report_package_keeps_every_figure_as_svg(forecastability_result):
    result = forecastability_result

    figures = sorted(path.name for path in (result.artifact_directory / "figures").iterdir())
    assert figures
    assert all(name.endswith(".svg") for name in figures)
    assert set(result.figure_paths) == {name.removesuffix(".svg") for name in figures}
    assert (result.artifact_directory / "report.md").is_file()
