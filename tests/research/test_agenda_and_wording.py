"""One question must get one verdict, in words a reader outside the codebase can follow."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.research.agent.schemas import EDAPlan
from app.research.application.planning import prepare_research_data
from app.research.evaluation.agenda import resolve_agenda_item
from app.research.evaluation.eda import evaluate_compatibility_summary as evaluate_agent_run
from app.research.planning.compiler import (
    FUNCTION_AGENDA_HYPOTHESES,
    FUNCTION_AGENDA_ITEM_IDS,
    EDAPlanCompiler,
)
from app.research.planning.contracts import EDAPlanDraft
from app.research.schemas.study import load_study_config
from app.research.skills.registry import SkillRegistry


def _compile(
    study: Path,
    *,
    hypotheses: list[str],
    functions: list[str],
    question: str = "这批数据能看出什么？",
) -> EDAPlan:
    config = load_study_config(study)
    skill = SkillRegistry.default(Settings(skill_paths="")).get("price-exogenous-eda")
    steps = [{"function": name, "enabled": True, "rationale": "回归测试"} for name in functions]
    return EDAPlanCompiler().compile(
        EDAPlanDraft(objective="检查议程", selected_variables=[], steps=steps, hypotheses=hypotheses),
        question=question,
        config=config,
        skill=skill,
        model_name="regression",
        prompt_version="regression",
    )


def test_every_registered_hypothesis_resolves_to_its_own_agenda_item():
    """The compiler's wording and the evaluator's rule must name the same item."""

    for function_name, wording in FUNCTION_AGENDA_HYPOTHESES.items():
        assert resolve_agenda_item(wording) == FUNCTION_AGENDA_ITEM_IDS[function_name], function_name


@pytest.mark.parametrize(
    ("wording", "item_id"),
    [
        ("目标电价的日历分组画像能够揭示小时、星期、工作日/周末或月份层面的结构差异。", "price.seasonality"),
        ("目标电价的滚动均值与标准差能够揭示波动是否随时间分阶段变化。", "price.volatility"),
    ],
)
def test_planner_wording_resolves_to_the_function_that_can_decide_it(wording: str, item_id: str):
    """These two wordings used to be reported as unverifiable while their own function ran."""

    assert resolve_agenda_item(wording) == item_id


def test_one_agenda_item_keeps_one_hypothesis(synthetic_study: Path):
    """The planner's phrasing replaces the registered twin instead of standing beside it."""

    plan = _compile(
        synthetic_study,
        hypotheses=["目标电价的日历分组画像能够揭示小时、星期、工作日/周末或月份层面的结构差异。"],
        functions=["price_calendar_group_profile"],
    )

    items = [resolve_agenda_item(item) for item in plan.hypotheses]
    assert len(items) == len(set(items))
    assert plan.hypotheses.count("电价可能存在日内或季节结构。") == 0
    assert plan.unverifiable_hypotheses == []


def test_a_hypothesis_no_function_can_decide_never_enters_the_agenda(synthetic_study: Path):
    """Parking it at plan time is what the skill asks for; the alternative is asking the
    user after execution to restate a question the evaluator could never answer."""

    guess = "碳配额成本可能传导到批发市场价格。"
    plan = _compile(synthetic_study, hypotheses=[guess], functions=["price_descriptive_distribution"])

    assert guess not in plan.hypotheses
    assert plan.unverifiable_hypotheses == [guess]
    assert any("没有对应的确定性检验" in note for note in plan.planning_notes)


def test_requested_statistics_are_deliverables_not_unverifiable_hypotheses(synthetic_study: Path):
    plan = _compile(
        synthetic_study,
        hypotheses=["电价总体均值、最低价和最高价可以由有效目标观测计算。"],
        functions=["price_descriptive_distribution"],
        question="给出实时电价均值、最低价和最高价",
    )

    assert plan.analysis_kind == "descriptive"
    assert plan.requested_statistics == ("mean", "min", "max")
    assert plan.unverifiable_hypotheses == []


def test_a_flat_day_is_volatility_evidence_rather_than_missing_data(synthetic_study: Path):
    """A price held at a cap or floor for a day makes the ratio undefined, not the data unusable."""

    plan = _compile(
        synthetic_study,
        hypotheses=[],
        functions=["price_rolling_mean_std"],
    )
    config = load_study_config(synthetic_study)
    summary = {
        "study": {"target": config.target.name},
        "price": {
            "volatility": {"first_difference_std": 1.0},
            "rolling_statistics": {"one_day": {"minimum_std": 0.0, "maximum_std": 88.0}},
        },
    }
    quality = prepare_research_data(config).quality
    evaluation = evaluate_agent_run(plan=plan, quality=quality, summary=summary)

    volatility = next(
        item for item in evaluation.hypothesis_assessments if item.item_id == "price.volatility"
    )
    assert volatility.status == "candidate_support"
    assert volatility.scope == "inherent"
    assert "补充" not in (volatility.remediation or "")
