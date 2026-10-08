"""Compile model-proposed atomic function calls into deterministic plans."""

from __future__ import annotations

from uuid import uuid4

from app.research.agent.errors import DuplicateResearchFunctionError, ResearchPlanValidationError
from app.research.agent.schemas import EDAPlan, EDAPlanStep
from app.research.evaluation.agenda import resolve_agenda_item
from app.research.planning.contracts import DraftStep, EDAPlanDraft
from app.research.schemas.study import StudyConfig
from app.research.skills.contracts import SkillDefinition
from app.research.tools.catalog import FUNCTION_CATALOG
from app.research.tools.parameters import compile_function_parameters, max_lag_limit

FUNCTION_AGENDA_HYPOTHESES: dict[str, str] = {
    "price_descriptive_distribution": "电价分布可能明显偏斜或存在厚尾。",
    "price_rolling_mean_std": "电价波动可能分阶段变化，而不是全期稳定。",
    "exogenous_descriptive_distribution": "所选外生变量的覆盖率可能不足以支撑关系分析。",
    "exogenous_linear_index_trend": "部分外生变量可能存在长期漂移。",
    "relationship_feature_quartile_response": "电价对变量水平的响应可能不是单调的。",
    "price_tukey_outer_fence": "电价可能存在尖峰或极端值。",
    "price_calendar_group_profile": "电价可能存在日内或季节结构。",
    "price_lag_autocorrelation": "电价可能存在自相关或持续性。",
    "exogenous_iqr_outliers": "所选外生变量可能存在异常观测。",
    "exogenous_pearson_collinearity": "所选外生变量之间可能存在共线性。",
    "relationship_scipy_pearson_pairwise": "所选变量与电价可能存在同期关系。",
    "relationship_scipy_spearman_pairwise": "所选变量与电价可能存在单调关系。",
    "relationship_pearson_positive_lead_scan": "所选变量可能存在领先滞后关系。",
    "relationship_pearson_by_hour": "变量与电价的关系可能存在时段差异。",
    "relationship_pearson_by_month": "变量与电价的关系可能存在月份结构。",
    "relationship_pearson_segment_comparison": "峰段、谷段或其他明确分段可能存在关系差异。",
    "price_segment_distribution_comparison": "峰段、谷段或其他明确分段可能存在电价差异。",
    "price_stationarity_tests": "电价可能不是围绕稳定水平波动，建模前需要差分或去趋势。",
    "price_seasonal_decomposition": "电价的可解释部分可能主要来自日内和周内季节成分。",
    "price_partial_autocorrelation": "电价可能存在有限阶的直接记忆结构。",
    "price_spike_regime_profile": "极端价格可能集中出现并具有聚集性。",
    "price_duration_curve": "高价时段可能只占很小比例但贡献主要价值。",
    "price_variance_stabilization_check": "电价厚尾可能需要方差稳定变换才能进入建模。",
    "price_naive_baseline_benchmark": "朴素基线可能已经提供较低误差，后续模型需超越该底线。",
    "exogenous_variance_inflation": "所选变量之间可能存在多重共线性冗余。",
    "exogenous_stationarity_tests": "部分驱动变量可能自身非平稳，与电价形成共同趋势。",
    "relationship_mutual_information_scan": "部分变量可能与电价存在线性相关无法捕捉的非线性依赖。",
    "relationship_granger_causality_scan": "部分变量的历史可能在样本内改善电价自回归拟合。",
    "relationship_rolling_correlation_stability": "电价与变量的关系可能随时间漂移甚至反号。",
}

FUNCTION_AGENDA_ITEM_IDS: dict[str, str] = {
    "price_descriptive_distribution": "price.distribution",
    "price_rolling_mean_std": "price.volatility",
    "exogenous_descriptive_distribution": "exogenous.coverage",
    "exogenous_linear_index_trend": "exogenous.trend",
    "relationship_feature_quartile_response": "relationships.quantile_response",
    "price_tukey_outer_fence": "price.extremes",
    "price_calendar_group_profile": "price.seasonality",
    "price_lag_autocorrelation": "price.autocorrelation",
    "exogenous_iqr_outliers": "exogenous.outliers",
    "exogenous_pearson_collinearity": "exogenous.collinearity",
    "relationship_scipy_pearson_pairwise": "relationships.contemporaneous",
    "relationship_scipy_spearman_pairwise": "relationships.monotonic",
    "relationship_pearson_positive_lead_scan": "relationships.lead_lag",
    "relationship_pearson_by_hour": "relationships.by_hour",
    "relationship_pearson_by_month": "relationships.by_month",
    "relationship_pearson_segment_comparison": "comparisons.relationship_segments",
    "price_segment_distribution_comparison": "comparisons.price_segments",
    "price_stationarity_tests": "price.stationarity",
    "price_seasonal_decomposition": "price.decomposition",
    "price_partial_autocorrelation": "price.partial_autocorrelation",
    "price_spike_regime_profile": "price.spike_regime",
    "price_duration_curve": "price.duration_curve",
    "price_variance_stabilization_check": "price.variance_stabilization",
    "price_naive_baseline_benchmark": "price.naive_baselines",
    "exogenous_variance_inflation": "exogenous.multicollinearity",
    "exogenous_stationarity_tests": "exogenous.stationarity",
    "relationship_mutual_information_scan": "relationships.mutual_information",
    "relationship_granger_causality_scan": "relationships.granger",
    "relationship_rolling_correlation_stability": "relationships.rolling_stability",
}


def _requested_statistics(question: str) -> tuple[str, ...]:
    normalized = question.casefold()
    mappings = (
        ("mean", ("均值", "平均值", "mean", "average")),
        ("min", ("最低", "最小", "minimum", " min")),
        ("max", ("最高", "最大", "maximum", " max")),
    )
    return tuple(name for name, terms in mappings if any(term in normalized for term in terms))


def _is_statistic_deliverable(hypothesis: str, requested: tuple[str, ...]) -> bool:
    terms = {
        "mean": ("均值", "平均值", "mean", "average"),
        "min": ("最低", "最小", "minimum"),
        "max": ("最高", "最大", "maximum"),
    }
    normalized = hypothesis.casefold()
    return bool(requested) and all(any(term in normalized for term in terms[name]) for name in requested)


def _agenda_hypotheses(
    draft: EDAPlanDraft,
    functions: list[str],
    *,
    requested_statistics: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """Build the agenda from the selected functions, keeping one hypothesis per item.

    The planner's own wording is preferred when it resolves to an item, because it
    is usually the more specific sentence.  A wording that no selected or catalog
    function can decide is parked instead of entering the agenda: letting it through
    would make the run ask the user to restate a question the evaluator was never
    able to answer.
    """

    generated = {
        FUNCTION_AGENDA_ITEM_IDS[name]: FUNCTION_AGENDA_HYPOTHESES[name]
        for name in functions
        if name in FUNCTION_AGENDA_HYPOTHESES
    }
    agenda: dict[str, str] = {}
    parked: list[str] = []
    for hypothesis in draft.hypotheses:
        item_id = resolve_agenda_item(hypothesis)
        if item_id is None:
            if _is_statistic_deliverable(hypothesis, requested_statistics):
                continue
            parked.append(hypothesis)
            continue
        agenda.setdefault(item_id, hypothesis)
    for item_id, hypothesis in generated.items():
        agenda.setdefault(item_id, hypothesis)
    return list(agenda.values()), list(dict.fromkeys(parked))


def compile_function_step(
    number: int,
    draft: DraftStep,
    *,
    selected_variables: list[str],
    config: StudyConfig,
) -> EDAPlanStep:
    catalog = FUNCTION_CATALOG[draft.function]
    parameters = compile_function_parameters(
        draft.function,
        draft.parameters,
        selected_variables=selected_variables,
        config=config,
        enabled=draft.enabled,
    )
    return EDAPlanStep(
        step_id=f"S{number}",
        function=draft.function,
        title=catalog.title,
        description=catalog.description,
        rationale=draft.rationale,
        enabled=draft.enabled,
        parameters=parameters,
        function_version=catalog.version,
    )


class EDAPlanCompiler:
    """Enforce Skill, function, variable, argument and version constraints."""

    def compile(
        self,
        draft: EDAPlanDraft,
        *,
        question: str,
        config: StudyConfig,
        skill: SkillDefinition,
        model_name: str | None,
        prompt_version: str,
    ) -> EDAPlan:
        valid_names = {spec.name for spec in config.exogenous}
        unknown_variables = sorted(set(draft.selected_variables).difference(valid_names))
        if unknown_variables:
            raise ResearchPlanValidationError(f"大模型选择了未知变量：{', '.join(unknown_variables)}")
        selected = list(dict.fromkeys(draft.selected_variables))
        functions = [step.function for step in draft.steps]
        if "data_quality" in functions:
            raise ResearchPlanValidationError("data_quality 由编译器添加，模型不应直接选择")
        disallowed = sorted(set(functions).difference(skill.allowed_functions))
        if disallowed:
            raise ResearchPlanValidationError(f"Skill {skill.name} 未授权函数：{', '.join(disallowed)}")
        disallowed_deferred = sorted(set(draft.deferred_functions).difference(skill.allowed_functions))
        if disallowed_deferred:
            raise ResearchPlanValidationError(
                f"Skill {skill.name} 未授权延后函数：{', '.join(disallowed_deferred)}"
            )
        if len(functions) != len(set(functions)):
            duplicate = next(name for name in functions if functions.count(name) > 1)
            spec = FUNCTION_CATALOG[duplicate]
            recommendation = spec.planning_guidance
            if spec.uses_max_lag:
                recommendation = "请合并为一次调用，并用最大的 max_lag 覆盖完整滞后范围。"
            elif spec.uses_segments:
                recommendation = "请把全部子样本合并到一次 segments 调用。"
            elif spec.uses_variables:
                recommendation = "请合并为一次调用，并在 variables 中列出全部变量。"
            raise DuplicateResearchFunctionError(duplicate, recommendation)
        ordered_names = skill.order_function_names(functions)
        draft_by_function = {step.function: step for step in draft.steps}
        ordered_draft_steps = [draft_by_function[name] for name in ordered_names]

        quality_catalog = FUNCTION_CATALOG["data_quality"]
        steps = [
            EDAPlanStep(
                step_id="S1",
                function="data_quality",
                title=quality_catalog.title,
                description=quality_catalog.description,
                rationale="任何研究结论都必须先通过确定性的数据质量与时间对齐核验。",
                enabled=True,
                required=True,
                function_version=quality_catalog.version,
            )
        ]
        steps.extend(
            compile_function_step(number, step, selected_variables=selected, config=config)
            for number, step in enumerate(ordered_draft_steps, start=2)
        )

        assumptions = list(
            dict.fromkeys(
                [
                    *draft.assumptions,
                    "所有统计结论均为描述性证据，不解释为因果关系。",
                    "关系分析使用成对有效样本，不进行隐式插补。",
                    "变量进入预测候选集前必须证明其在预测起点真实可获得。",
                ]
            )
        )
        if skill.research_protocol is None:
            notes = [
                (
                    f"API 模型 {model_name or 'configured-model'} 在 Skill 授权范围内提出原子研究函数，"
                    "确定性计划编译器完成函数版本、变量和参数校验。"
                )
            ]
        else:
            notes = [
                (
                    f"本地领域协议 {skill.research_protocol.protocol_id}@"
                    f"{skill.research_protocol.version} 固定研究阶段、函数顺序和停止边界。"
                ),
                (
                    f"API 模型 {model_name or 'configured-model'} 只提出具体 Function Call；"
                    "模型网关通过请求参数禁用 thinking，并丢弃响应中残留的思考内容。"
                ),
            ]
        if config.target.unit.casefold() in {"", "unknown", "unspecified"}:
            notes.append("目标电价单位未知，绝对数值和阈值解释前需要用户确认单位。")
        if "unspecified" in config.study.market.casefold():
            notes.append("市场范围尚未明确，当前不生成依赖具体市场规则的解释。")
        requested_statistics = _requested_statistics(question)
        agenda, parked = _agenda_hypotheses(
            draft,
            ordered_names,
            requested_statistics=requested_statistics,
        )
        if parked:
            notes.append(
                "以下说法没有对应的确定性检验，未列入本轮议程：" + "；".join(parked[:3])
            )
        return EDAPlan(
            plan_id=uuid4().hex[:12],
            question=question,
            objective=draft.objective,
            study_name=config.study.name,
            planner="llm",
            planning_model=model_name,
            planning_prompt_version=prompt_version,
            skill_name=skill.name,
            skill_version=skill.version,
            research_protocol_id=(
                skill.research_protocol.protocol_id if skill.research_protocol is not None else None
            ),
            research_protocol_version=(
                skill.research_protocol.version if skill.research_protocol is not None else None
            ),
            research_protocol_function_order=(
                list(skill.research_protocol.function_order)
                if skill.research_protocol is not None
                else []
            ),
            research_protocol_step_texts=(
                dict(skill.research_protocol.display_text_by_function)
                if skill.research_protocol is not None
                else {}
            ),
            hypotheses=agenda,
            unverifiable_hypotheses=parked,
            analysis_kind="descriptive" if requested_statistics else "research",
            requested_statistics=requested_statistics,
            selected_variables=selected,
            variable_selection_mode=draft.variable_selection_mode,
            variable_selection_stage=draft.variable_selection_stage,
            deferred_functions=draft.deferred_functions,
            variable_recommendation_limit=draft.variable_recommendation_limit,
            steps=steps,
            assumptions=assumptions,
            planning_notes=notes,
        )
