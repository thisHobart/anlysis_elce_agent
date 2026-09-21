from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.llm.gateway import ModelMessage, ModelToolCall, ModelToolTurn
from app.research.agent.dynamic import build_tool_request
from app.research.agent.schemas import EDAResearchScope
from app.research.application.planning import prepare_research_data
from app.research.evaluation.tool_calling import ToolSelectionCase, score_tool_turn
from app.research.schemas.study import load_study_config
from app.research.skills.registry import SkillRegistry
from app.research.tools.catalog import FUNCTION_CATALOG
from app.research.tools.calling import ToolTurnProtocolError, compile_tool_turn
from app.research.tools.contracts import ToolCall, ToolContext
from app.research.tools.eda.functions import build_eda_tool_registry
from app.research.tools.executor import ToolExecutor, validate_tool_result
from app.research.tools.parameters import compile_function_parameters
from app.research.tools.policy import ToolPolicy
from app.research.tools.recipes import build_recipe_registry


def _scope(*, functions: list[str], variables: list[str] | None = None) -> EDAResearchScope:
    skill = SkillRegistry.default().get("price-exogenous-eda")
    return EDAResearchScope(
        scope_id="debug-scope",
        question="调试工具调用",
        objective="调试工具调用",
        study_name="synthetic",
        data_fingerprint="1" * 12,
        skill_name=skill.name,
        skill_version=skill.version,
        authorized_functions=functions,
        authorized_variables=variables or [],
        initial_strategy=["按问题选择最小工具集合"],
    )


def test_request_builder_is_pure_and_exposes_the_exact_model_schema():
    registry = build_eda_tool_registry()
    recipes = build_recipe_registry()
    scope = _scope(functions=["relationship_scipy_pearson_pairwise"], variables=["load"])
    skill = SkillRegistry.default().get(scope.skill_name)
    messages = [ModelMessage(role="user", content="检查负荷与电价的线性相关")]

    request = build_tool_request(
        scope=scope,
        skill=skill,
        messages=messages,
        registry=registry,
        recipes=recipes,
    )

    schema = request.tools[0]["function"]
    assert schema["name"] == "relationship_scipy_pearson_pairwise"
    assert "min_observations" not in schema["parameters"]["properties"]
    assert request.messages == tuple(messages)
    assert request.as_gateway_kwargs()["purpose"] == "eda_analysis"


def test_turn_compiler_expands_recipe_without_graph_or_coordinator(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    scope = _scope(
        functions=[
            "price_descriptive_distribution",
            "price_calendar_group_profile",
            "price_rolling_mean_std",
        ]
    )
    turn = ModelToolTurn(
        tool_calls=[ModelToolCall(name="price_quick_profile", arguments={}, call_id="provider-1")]
    )

    batch = compile_tool_turn(
        turn,
        scope=scope,
        study_config=config,
        registry=build_eda_tool_registry(),
        recipes=build_recipe_registry(),
        remaining_tool_calls=3,
    )

    assert [call.name for call in batch.calls] == [
        "price_descriptive_distribution",
        "price_calendar_group_profile",
        "price_rolling_mean_std",
    ]
    assert batch.groups["provider-1"].origin == "recipe"
    assert batch.proposals[0].requested_arguments == {}
    assert batch.exceeds_budget is False


def test_turn_compiler_isolates_invalid_proposal_and_keeps_valid_one(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    scope = _scope(functions=["price_descriptive_distribution"])
    turn = ModelToolTurn(
        tool_calls=[
            ModelToolCall(name="missing_function", arguments={}, call_id="bad"),
            ModelToolCall(name="price_descriptive_distribution", arguments={}, call_id="good"),
        ]
    )

    batch = compile_tool_turn(
        turn,
        scope=scope,
        study_config=config,
        registry=build_eda_tool_registry(),
        recipes=build_recipe_registry(),
    )

    assert [call.name for call in batch.calls] == ["price_descriptive_distribution"]
    assert batch.groups["bad"].status == "failed"
    assert batch.groups["good"].status == "pending"
    assert len(batch.failures) == 1


def test_turn_compiler_rejects_reused_provider_identity_before_compilation(synthetic_study: Path):
    turn = ModelToolTurn(
        tool_calls=[ModelToolCall(name="price_descriptive_distribution", arguments={}, call_id="used")]
    )
    with pytest.raises(ToolTurnProtocolError) as caught:
        compile_tool_turn(
            turn,
            scope=_scope(functions=["price_descriptive_distribution"]),
            study_config=load_study_config(synthetic_study),
            registry=build_eda_tool_registry(),
            recipes=build_recipe_registry(),
            existing_provider_call_ids={"used"},
        )
    assert caught.value.code == "reused_provider_call_id"


def test_twelve_reviewed_selection_cases_and_scorer_mutation_detection():
    path = Path(__file__).resolve().parents[1] / "fixtures" / "tool_calling" / "cases.json"
    cases = [ToolSelectionCase.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]
    assert len(cases) == 12
    for case in cases:
        calls = [
            ModelToolCall(
                name=name,
                arguments=case.expected_arguments.get(name, {}),
                call_id=f"call-{index}",
            )
            for index, name in enumerate(case.required_tools, start=1)
        ]
        turn = ModelToolTurn(tool_calls=calls) if calls else ModelToolTurn(content="无需继续调用工具。")
        assert score_tool_turn(case, turn).passed, case.case_id

    lead = next(case for case in cases if case.case_id == "lead_scan_24")
    wrong = ModelToolTurn(
        tool_calls=[
            ModelToolCall(
                name="relationship_pearson_positive_lead_scan",
                arguments={"variables": ["load"], "max_lag": 12},
                call_id="wrong-lag",
            )
        ]
    )
    score = score_tool_turn(lead, wrong)
    assert not score.passed
    assert "max_lag" in score.argument_errors[0]


def test_all_atomic_tools_have_named_handlers_and_execute_through_the_shared_boundary(
    synthetic_study: Path,
):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    registry = build_eda_tool_registry()
    executor = ToolExecutor(registry)
    policy = ToolPolicy(frozenset(registry.names))
    context = ToolContext(config=config, frame=prepared.aligned.frame, quality=prepared.quality)
    variables = [item.name for item in config.exogenous]
    segments = [
        {"segment_id": "first_half", "label": "前半日", "kind": "hours", "hours": list(range(12))},
        {"segment_id": "second_half", "label": "后半日", "kind": "hours", "hours": list(range(12, 24))},
    ]

    assert registry.names == set(FUNCTION_CATALOG)
    for sequence, name in enumerate(FUNCTION_CATALOG, start=1):
        spec = FUNCTION_CATALOG[name]
        supplied: dict[str, object] = {}
        if spec.uses_variables:
            supplied["variables"] = variables
        if spec.uses_max_lag:
            supplied["max_lag"] = 4
        if spec.uses_segments:
            supplied.update({"comparison_id": "debug_comparison", "segments": segments})
        arguments = compile_function_parameters(
            name,
            supplied,
            selected_variables=variables if spec.uses_variables else [],
            config=config,
            enabled=True,
        )
        arguments.pop("max_lag_limit", None)
        registered = registry.get(name)
        assert registered.handler.__name__ != "handler"
        assert registered.handler.__closure__ is None
        call = ToolCall(
            call_id=f"atomic-{sequence}",
            work_id=f"{sequence:024x}",
            step_id=f"S{sequence}",
            name=name,
            version=registered.version,
            arguments=arguments,
        )
        result = executor.execute(
            call,
            context=context,
            policy=policy,
            data_fingerprint="2" * 12,
        )
        validate_tool_result(
            call,
            result,
            registry=registry,
            expected_data_fingerprint="2" * 12,
        )
