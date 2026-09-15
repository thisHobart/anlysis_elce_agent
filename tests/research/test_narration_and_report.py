"""Checks for user-facing narration, the second core Skill, and the Markdown report."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from app.config import Settings
from app.research.application.execution import EDAExecutionService
from app.research.data.snapshot import input_file_manifest, study_fingerprint
from app.research.graph.narration import (
    HUMAN_GATE_NODES,
    NODE_PROGRESS,
    SILENT_NODES,
    narrate_event,
    narrate_node,
    progress_message,
    split_progress_message,
)
from app.research.planning.compiler import EDAPlanCompiler
from app.research.planning.contracts import EDAPlanDraft
from app.research.reporting.agent_report import build_agent_eda_report
from app.research.reporting.report_charts import build_report_figures
from app.research.schemas.study import load_study_config
from app.research.skills.registry import SkillRegistry
from app.research.tools.catalog import FUNCTION_CATALOG, STAGE_TITLES


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
    assert step.detail == FUNCTION_CATALOG["price_stationarity_tests"].description.rstrip("。")
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


def test_narration_distinguishes_consecutive_question_processing_stages():
    submitted = narrate_event({"name": "提交研究问题：分析电价", "status": "completed"})
    received = narrate_event({"name": "接收研究问题：分析电价", "status": "completed"})
    routed = narrate_event(
        {"name": "主 Agent 路由：new_plan", "status": "completed", "details": {"intent": "new_plan"}}
    )

    assert [submitted.stage_label, received.stage_label, routed.stage_label] == [
        "提交问题",
        "解析问题",
        "识别意图",
    ]


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
    assert HUMAN_GATE_NODES == {
        "prepare_approval",
        "approval_interrupt",
        "prepare_need_user",
        "user_interrupt",
        "result_interrupt",
    }
    assert narrate_node("unknown_node")[1].title == "研究进行中"


# What the Skill promises the model, and the constant the evaluator judges by.
SKILL_THRESHOLD_CONTRACT = (
    ("SEASONAL_STRENGTH_THRESHOLD", 0.3, "< 0.3 视为无稳定周期结构"),
    ("CALENDAR_EFFECT_THRESHOLD", 0.06, "≥ 6% 的电价方差"),
    ("DISTRIBUTION_SKEW_THRESHOLD", 0.5, "绝对值 ≥ 0.5"),
    ("DISTRIBUTION_KURTOSIS_THRESHOLD", 1.0, "（超额峰度）≥ 1.0"),
    ("VOLATILITY_REGIME_RATIO", 2.0, "比值 ≥ 2"),
    ("MINIMUM_DRIVER_COVERAGE", 0.9, "< 90% 的变量不足以支撑关系分析"),
)


def test_the_skill_states_the_thresholds_the_evaluator_actually_applies():
    """A model told one threshold and judged by another produces unexplainable verdicts."""

    from app.research.evaluation import eda as evaluator

    skill = SkillRegistry.default(Settings(skill_paths="")).get("price-exogenous-eda")
    for name, expected_value, promised_text in SKILL_THRESHOLD_CONTRACT:
        assert getattr(evaluator, name) == expected_value, name
        assert promised_text in skill.instructions, f"{name} 的判据没有写进 SKILL.md"


def test_both_skills_tell_the_model_a_hypothesis_must_be_verifiable():
    """Since unverifiable hypotheses now block a run, the Skill has to say so up front."""

    registry = SkillRegistry.default(Settings(skill_paths=""))
    for name in ("price-exogenous-eda", "price-forecastability-audit"):
        instructions = registry.get(name).instructions
        assert "必须能被**本轮选择的函数**判定" in instructions, name
        protocol = registry.get(name).research_protocol
        assert protocol is not None
        assert any("写不出对应函数的猜想不得列入议程" in item for item in protocol.invariants), name


def test_two_core_skills_ship_with_the_application():
    registry = SkillRegistry.default(Settings(skill_paths=""))
    names = {item["name"] for item in registry.metadata()}

    assert names == {"price-exogenous-eda", "price-forecastability-audit"}
    audit = registry.get("price-forecastability-audit")
    assert audit.domain == "eda"
    assert audit.research_protocol is not None
    assert "data_quality" in audit.allowed_functions
    assert not any(name.startswith(("exogenous_", "relationship_")) for name in audit.allowed_functions)


def test_catalog_stages_match_the_builtin_protocol_stages():
    for skill in (
        SkillRegistry.default(Settings(skill_paths="")).get("price-exogenous-eda"),
    ):
        assert skill.research_protocol is not None
        for stage in skill.research_protocol.stages:
            for function_name in stage.functions:
                assert FUNCTION_CATALOG[function_name].stage == stage.stage_id
            assert stage.stage_id in STAGE_TITLES or not stage.functions


@pytest.fixture
def forecastability_result(synthetic_study: Path, tmp_path: Path):
    config = load_study_config(synthetic_study)
    skill = SkillRegistry.default(Settings(skill_paths="")).get("price-forecastability-audit")
    steps = [
        {"function": name, "enabled": True, "rationale": "可预测性审计", "parameters": {"max_lag": 24} if FUNCTION_CATALOG[name].uses_max_lag else {}}
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
    assert result.report_path.name == "report.md"
    assert "影响因素质量" not in report
    assert "电价与影响因素的关系" not in report
    assert "可预测性基线" in report
    assert "](figures/" in report
    assert "报告由确定性统计函数生成" in report


def test_report_states_each_fact_once_and_leaves_verification_to_methods(forecastability_result):
    """The report reads evidence; verdicts and checks belong to methods.md."""

    result = forecastability_result
    report = result.report_path.read_text(encoding="utf-8")
    methods = (result.artifact_directory / "methods.md").read_text(encoding="utf-8")

    # Data risks are stated once, under the conclusion's limits, never as a second risk list.
    assert "高优先级数据风险" not in report
    for warning in result.evaluation.warnings:
        assert report.count(warning) <= 1, warning

    # Hypothesis verdicts and validity checks are verification records, not report sections.
    headings = re.findall(r"^#+ .+$", report, flags=re.MULTILINE)
    assert not [line for line in headings if any(word in line for word in ("假设验收", "有效性核验", "判据"))]
    assert "## 5 假设验收" in methods and "## 4 有效性核验" in methods
    for item in result.evaluation.hypothesis_assessments:
        assert item.hypothesis not in report, item.hypothesis
        assert item.hypothesis in methods, item.hypothesis
    for check in result.evaluation.checks:
        assert check.message not in report, check.message

    # Thresholds are quoted where a reading uses them, and defined once in methods.md.
    assert "## 3 判读阈值" in methods


def test_every_evidence_subsection_carries_a_reading(forecastability_result):
    """A figure without a reading is the failure mode this structure exists to prevent."""

    result = forecastability_result
    report = result.report_path.read_text(encoding="utf-8")

    subsections = re.split(r"^### (?:\d+\.\d+|A\.\d+) .+$", report, flags=re.MULTILINE)[1:]
    evidence_subsections = [block for block in subsections if "![" in block]
    assert evidence_subsections
    for block in evidence_subsections:
        assert "**读图**：" in block or "**读表**：" in block, block[:120]
    # A reading only claims to read a figure when the run actually produced one.
    for block in subsections:
        if "**读图**：" in block:
            assert "![" in block, block[:120]


def test_report_package_keeps_every_figure_as_svg(forecastability_result):
    result = forecastability_result

    figures = sorted(path.name for path in (result.artifact_directory / "figures").iterdir())
    assert figures
    assert all(name.endswith(".svg") for name in figures)
    assert set(result.figure_paths) == {name.removesuffix(".svg") for name in figures}
    assert (result.artifact_directory / "report.md").is_file()


def test_report_shows_every_figure_the_run_produced(forecastability_result):
    """A chart that is drawn but never referenced is invisible to the reader."""

    result = forecastability_result
    report = result.report_path.read_text(encoding="utf-8")

    referenced = set(re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\.svg\)", report))
    assert referenced == set(result.figure_paths)
    for name in referenced:
        assert (result.artifact_directory / "figures" / f"{name}.svg").is_file(), name


def test_desktop_package_is_grouped_by_review_action(forecastability_result):
    result = forecastability_result
    root = result.artifact_directory

    assert {path.name for path in root.iterdir() if path.is_file()} == {
        "report.md",
        "methods.md",
        "manifest.json",
    }
    assert {path.name for path in root.iterdir() if path.is_dir()} == {
        "figures",
        "evidence",
        "provenance",
        "data",
    }
    assert {path.name for path in (root / "evidence").iterdir()} == {
        "data_quality.json",
        "call_evidence.json",
        "eda_summary.json",
        "agent_evaluation.json",
    }
    assert {path.name for path in (root / "provenance").iterdir()} == {
        "conversation.json",
        "research_plan.json",
        "execution_trace.json",
        "study_context.json",
    }
    assert {path.name for path in (root / "data").iterdir()} == {"aligned_data.parquet"}

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    output_paths = {item["path"] for item in manifest["outputs"]}
    assert "methods.md" in output_paths
    assert "evidence/call_evidence.json" in output_paths
    assert "evidence/eda_summary.json" in output_paths
    assert "provenance/execution_trace.json" in output_paths
    assert "data/aligned_data.parquet" in output_paths

    ledger = json.loads((root / "evidence" / "call_evidence.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "evidence" / "eda_summary.json").read_text(encoding="utf-8"))
    assert ledger["call_order"] == [
        call_id
        for call_id, _item in sorted(
            ledger["calls"].items(),
            key=lambda pair: pair[1]["sequence"],
        )
    ]
    assert set(summary["call_evidence"]) == set(ledger["calls"])
    for call_id, projected in summary["call_evidence"].items():
        canonical = ledger["calls"][call_id]
        assert projected["arguments"] == canonical["arguments"]
        assert projected["value"] == canonical["value"]
        assert projected["output_hash"] == canonical["output_hash"]


def test_report_and_chart_consumers_require_call_level_evidence():
    assert "evidence" in inspect.signature(build_agent_eda_report).parameters
    assert "summary" not in inspect.signature(build_agent_eda_report).parameters
    assert "evidence" in inspect.signature(build_report_figures).parameters
    assert "summary" not in inspect.signature(build_report_figures).parameters


def test_methods_document_exactly_matches_executed_functions_and_evidence(forecastability_result):
    result = forecastability_result
    root = result.artifact_directory
    methods = (root / "methods.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#### \d+\. .+ · `([^`]+)` v", methods, flags=re.MULTILINE))
    enabled = {step.function for step in result.plan.enabled_steps}

    assert documented == enabled
    locations = re.findall(r"^- \*\*结果位置\*\*：`([^`]+)`", methods, flags=re.MULTILINE)
    assert len(locations) == len(enabled)
    for location in locations:
        relative_path, _, pointer = location.partition("#")
        evidence_path = root / relative_path
        assert evidence_path.is_file(), location
        if pointer:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            for token in pointer.removeprefix("/").split("/"):
                assert isinstance(evidence, dict) and token in evidence, location
                evidence = evidence[token]

    trace = json.loads((root / "provenance" / "execution_trace.json").read_text(encoding="utf-8"))
    assert {item["function"] for item in trace} == enabled
    assert all(item["call_id"] and item["work_id"] and item["output_hash"] for item in trace)

    report = result.report_path.read_text(encoding="utf-8")
    assert "[methods.md](methods.md)" in report
    # Method identifiers stay in methods.md; the report is a reading of the evidence.
    for function_name in enabled:
        assert f"`{function_name}`" not in report, function_name
    # Every back-link in methods.md must land on a heading the report really has.
    back_links = re.findall(r"^- \*\*报告位置\*\*：report\.md「([^」]+)」", methods, flags=re.MULTILINE)
    assert back_links
    for link in back_links:
        for part in link.split(" · "):
            assert re.search(rf"^#+ [\dA][\d.]* {re.escape(part)}$", report, flags=re.MULTILINE), link


INTERNAL_VOCABULARY = ("判据", "水平序列", "量纲")
RAW_ENUMS = (
    "trend_or_break_suspected",
    "unit_root",
    "first_difference",
    "asinh_median_mad",
    "candidate_support",
    "not_supported",
    "needs_restatement",
)


def test_reader_facing_text_avoids_raw_codes_and_broken_punctuation(forecastability_result):
    """Verdict codes are contracts between functions; a reader gets sentences."""

    report = forecastability_result.report_path.read_text(encoding="utf-8")
    evaluation = forecastability_result.evaluation

    for sentence in [evaluation.summary, *evaluation.findings, *evaluation.warnings]:
        assert "。；" not in sentence and "。、" not in sentence, sentence
        for code in RAW_ENUMS:
            assert code not in sentence, sentence
    for code in RAW_ENUMS:
        assert code not in report, code
    for word in INTERNAL_VOCABULARY:
        assert word not in report, word
    # Every reading explains its numbers instead of listing them.
    for reading in re.findall(r"\*\*读[图表]\*\*：(.+)", report):
        assert len(reading) > 30, reading
