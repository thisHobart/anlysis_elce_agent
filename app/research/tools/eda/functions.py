"""Atomic EDA functions executed only after plan validation and approval."""

from __future__ import annotations

from collections.abc import Callable

from app.research.tools.catalog import FUNCTION_CATALOG, FUNCTION_TO_ANALYSIS_METHOD
from app.research.tools.contracts import (
    DataQualityArguments,
    ExogenousOutlierArguments,
    NoArguments,
    PriceAutocorrelationArguments,
    PriceExtremeArguments,
    RelationshipArguments,
    RelationshipLagArguments,
    RelationshipSegmentComparisonArguments,
    SegmentComparisonArguments,
    ToolArguments,
    ToolContext,
    ToolOutput,
    ToolSpec,
    VariablesArguments,
)
from app.research.tools.eda.dependence import (
    analyze_granger_precedence,
    analyze_mutual_information,
    analyze_rolling_stability,
)
from app.research.tools.eda.drivers import analyze_driver_stationarity, analyze_variance_inflation
from app.research.tools.eda.exogenous import analyze_exogenous
from app.research.tools.eda.price import analyze_price
from app.research.tools.eda.relationships import analyze_relationships
from app.research.tools.eda.segments import compare_price_segments, compare_relationship_segments
from app.research.tools.eda.structure import (
    analyze_duration_curve,
    analyze_naive_baselines,
    analyze_partial_autocorrelation,
    analyze_seasonal_decomposition,
    analyze_spike_regime,
    analyze_stationarity,
    analyze_variance_stabilization,
)
from app.research.tools.registry import ToolRegistry


def _data_quality(context: ToolContext, _arguments: ToolArguments) -> ToolOutput:
    return ToolOutput(result_key="data_quality", value=context.quality.model_dump(mode="json"))


def _run_price_analysis(function_name: str, context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    method = FUNCTION_TO_ANALYSIS_METHOD[function_name]
    config = context.config
    max_lag = config.analysis.max_lag
    spike_multiplier = config.analysis.spike_iqr_multiplier
    if function_name == "price_lag_autocorrelation":
        max_lag = PriceAutocorrelationArguments.model_validate(arguments).max_lag
    elif function_name == "price_tukey_outer_fence":
        spike_multiplier = PriceExtremeArguments.model_validate(arguments).spike_iqr_multiplier
    return ToolOutput(
        result_key="price",
        value=analyze_price(
            context.frame[config.target.name],
            unit=config.target.unit,
            frequency=config.study.frequency,
            max_lag=max_lag,
            spike_iqr_multiplier=spike_multiplier,
            methods={method},
        ),
    )


def _price_descriptive_distribution(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_price_analysis("price_descriptive_distribution", context, arguments)


def _price_rolling_mean_std(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_price_analysis("price_rolling_mean_std", context, arguments)


def _price_tukey_outer_fence(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_price_analysis("price_tukey_outer_fence", context, arguments)


def _price_lag_autocorrelation(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_price_analysis("price_lag_autocorrelation", context, arguments)


def _price_calendar_group_profile(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_price_analysis("price_calendar_group_profile", context, arguments)


def _run_exogenous_analysis(function_name: str, context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    method = FUNCTION_TO_ANALYSIS_METHOD[function_name]
    parsed = VariablesArguments.model_validate(arguments)
    multiplier = context.config.analysis.outlier_iqr_multiplier
    if function_name == "exogenous_iqr_outliers":
        multiplier = ExogenousOutlierArguments.model_validate(arguments).outlier_iqr_multiplier
    units = {spec.name: spec.unit for spec in context.config.exogenous}
    return ToolOutput(
        result_key="exogenous",
        value=analyze_exogenous(
            context.frame,
            names=parsed.variables,
            units=units,
            outlier_iqr_multiplier=multiplier,
            methods={method},
        ),
    )


def _exogenous_descriptive_distribution(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_exogenous_analysis("exogenous_descriptive_distribution", context, arguments)


def _exogenous_iqr_outliers(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_exogenous_analysis("exogenous_iqr_outliers", context, arguments)


def _exogenous_linear_index_trend(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_exogenous_analysis("exogenous_linear_index_trend", context, arguments)


def _exogenous_pearson_collinearity(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_exogenous_analysis("exogenous_pearson_collinearity", context, arguments)


def _run_relationship_analysis(function_name: str, context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    method = FUNCTION_TO_ANALYSIS_METHOD[function_name]
    parsed = RelationshipArguments.model_validate(arguments)
    max_lag = context.config.analysis.max_lag
    if function_name == "relationship_pearson_positive_lead_scan":
        max_lag = RelationshipLagArguments.model_validate(arguments).max_lag
    return ToolOutput(
        result_key="relationships",
        value=analyze_relationships(
            context.frame,
            target_name=context.config.target.name,
            exogenous_names=parsed.variables,
            max_lag=max_lag,
            min_observations=parsed.min_observations,
            methods={method},
        ),
    )


def _relationship_scipy_pearson_pairwise(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_relationship_analysis("relationship_scipy_pearson_pairwise", context, arguments)


def _relationship_scipy_spearman_pairwise(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_relationship_analysis("relationship_scipy_spearman_pairwise", context, arguments)


def _relationship_pearson_positive_lead_scan(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_relationship_analysis("relationship_pearson_positive_lead_scan", context, arguments)


def _relationship_pearson_by_hour(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_relationship_analysis("relationship_pearson_by_hour", context, arguments)


def _relationship_pearson_by_month(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_relationship_analysis("relationship_pearson_by_month", context, arguments)


def _relationship_feature_quartile_response(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    return _run_relationship_analysis("relationship_feature_quartile_response", context, arguments)


def _price_stationarity(context: ToolContext, _arguments: ToolArguments) -> ToolOutput:
    config = context.config
    return ToolOutput(
        result_key="price",
        value=analyze_stationarity(
            context.frame[config.target.name],
            unit=config.target.unit,
            frequency=config.study.frequency,
        ),
    )


def _price_decomposition(context: ToolContext, _arguments: ToolArguments) -> ToolOutput:
    config = context.config
    return ToolOutput(
        result_key="price",
        value=analyze_seasonal_decomposition(
            context.frame[config.target.name],
            unit=config.target.unit,
            frequency=config.study.frequency,
        ),
    )


def _price_partial_autocorrelation(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    config = context.config
    parsed = PriceAutocorrelationArguments.model_validate(arguments)
    return ToolOutput(
        result_key="price",
        value=analyze_partial_autocorrelation(
            context.frame[config.target.name],
            unit=config.target.unit,
            frequency=config.study.frequency,
            max_lag=parsed.max_lag,
        ),
    )


def _price_spike_regime(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    config = context.config
    parsed = PriceExtremeArguments.model_validate(arguments)
    return ToolOutput(
        result_key="price",
        value=analyze_spike_regime(
            context.frame[config.target.name],
            unit=config.target.unit,
            spike_iqr_multiplier=parsed.spike_iqr_multiplier,
        ),
    )


def _price_duration_curve(context: ToolContext, _arguments: ToolArguments) -> ToolOutput:
    config = context.config
    return ToolOutput(
        result_key="price",
        value=analyze_duration_curve(context.frame[config.target.name], unit=config.target.unit),
    )


def _price_variance_stabilization(context: ToolContext, _arguments: ToolArguments) -> ToolOutput:
    config = context.config
    return ToolOutput(
        result_key="price",
        value=analyze_variance_stabilization(context.frame[config.target.name], unit=config.target.unit),
    )


def _price_naive_baselines(context: ToolContext, _arguments: ToolArguments) -> ToolOutput:
    config = context.config
    return ToolOutput(
        result_key="price",
        value=analyze_naive_baselines(
            context.frame[config.target.name],
            unit=config.target.unit,
            frequency=config.study.frequency,
        ),
    )


def _exogenous_variance_inflation(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    parsed = VariablesArguments.model_validate(arguments)
    return ToolOutput(result_key="exogenous", value=analyze_variance_inflation(context.frame, names=parsed.variables))


def _exogenous_stationarity(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    parsed = VariablesArguments.model_validate(arguments)
    return ToolOutput(
        result_key="exogenous",
        value=analyze_driver_stationarity(
            context.frame,
            names=parsed.variables,
            frequency=context.config.study.frequency,
        ),
    )


def _relationship_mutual_information(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    parsed = RelationshipLagArguments.model_validate(arguments)
    return ToolOutput(
        result_key="relationships",
        value=analyze_mutual_information(
            context.frame,
            target_name=context.config.target.name,
            variables=parsed.variables,
            max_lag=parsed.max_lag,
            min_observations=parsed.min_observations,
        ),
    )


def _relationship_granger(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    parsed = RelationshipLagArguments.model_validate(arguments)
    return ToolOutput(
        result_key="relationships",
        value=analyze_granger_precedence(
            context.frame,
            target_name=context.config.target.name,
            variables=parsed.variables,
            max_lag=parsed.max_lag,
            frequency=context.config.study.frequency,
            min_observations=parsed.min_observations,
        ),
    )


def _relationship_rolling_stability(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    parsed = RelationshipArguments.model_validate(arguments)
    return ToolOutput(
        result_key="relationships",
        value=analyze_rolling_stability(
            context.frame,
            target_name=context.config.target.name,
            variables=parsed.variables,
            frequency=context.config.study.frequency,
            min_observations=parsed.min_observations,
        ),
    )


def _price_segment_comparison(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    parsed = SegmentComparisonArguments.model_validate(arguments)
    return ToolOutput(
        result_key="comparisons",
        value=compare_price_segments(
            context.frame[context.config.target.name],
            comparison_id=parsed.comparison_id,
            segments=parsed.segments,
            min_observations=parsed.min_observations,
        ),
    )


def _relationship_segment_comparison(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    parsed = RelationshipSegmentComparisonArguments.model_validate(arguments)
    return ToolOutput(
        result_key="comparisons",
        value=compare_relationship_segments(
            context.frame,
            target_name=context.config.target.name,
            variables=parsed.variables,
            comparison_id=parsed.comparison_id,
            segments=parsed.segments,
            min_observations=parsed.min_observations,
        ),
    )


def build_eda_tool_registry() -> ToolRegistry:
    """Register every model-visible research function as one deterministic process."""

    handlers: dict[str, tuple[type[ToolArguments], Callable[[ToolContext, ToolArguments], ToolOutput]]] = {
        "data_quality": (DataQualityArguments, _data_quality),
        "price_descriptive_distribution": (NoArguments, _price_descriptive_distribution),
        "price_rolling_mean_std": (NoArguments, _price_rolling_mean_std),
        "price_tukey_outer_fence": (PriceExtremeArguments, _price_tukey_outer_fence),
        "price_lag_autocorrelation": (
            PriceAutocorrelationArguments,
            _price_lag_autocorrelation,
        ),
        "price_calendar_group_profile": (NoArguments, _price_calendar_group_profile),
        "price_stationarity_tests": (NoArguments, _price_stationarity),
        "price_seasonal_decomposition": (NoArguments, _price_decomposition),
        "price_partial_autocorrelation": (PriceAutocorrelationArguments, _price_partial_autocorrelation),
        "price_spike_regime_profile": (PriceExtremeArguments, _price_spike_regime),
        "price_duration_curve": (NoArguments, _price_duration_curve),
        "price_variance_stabilization_check": (NoArguments, _price_variance_stabilization),
        "price_naive_baseline_benchmark": (NoArguments, _price_naive_baselines),
        "price_segment_distribution_comparison": (
            SegmentComparisonArguments,
            _price_segment_comparison,
        ),
        "exogenous_descriptive_distribution": (
            VariablesArguments,
            _exogenous_descriptive_distribution,
        ),
        "exogenous_iqr_outliers": (ExogenousOutlierArguments, _exogenous_iqr_outliers),
        "exogenous_linear_index_trend": (
            VariablesArguments,
            _exogenous_linear_index_trend,
        ),
        "exogenous_pearson_collinearity": (
            VariablesArguments,
            _exogenous_pearson_collinearity,
        ),
        "exogenous_variance_inflation": (VariablesArguments, _exogenous_variance_inflation),
        "exogenous_stationarity_tests": (VariablesArguments, _exogenous_stationarity),
        "relationship_scipy_pearson_pairwise": (
            RelationshipArguments,
            _relationship_scipy_pearson_pairwise,
        ),
        "relationship_scipy_spearman_pairwise": (
            RelationshipArguments,
            _relationship_scipy_spearman_pairwise,
        ),
        "relationship_pearson_positive_lead_scan": (
            RelationshipLagArguments,
            _relationship_pearson_positive_lead_scan,
        ),
        "relationship_pearson_by_hour": (
            RelationshipArguments,
            _relationship_pearson_by_hour,
        ),
        "relationship_pearson_by_month": (
            RelationshipArguments,
            _relationship_pearson_by_month,
        ),
        "relationship_feature_quartile_response": (
            RelationshipArguments,
            _relationship_feature_quartile_response,
        ),
        "relationship_mutual_information_scan": (RelationshipLagArguments, _relationship_mutual_information),
        "relationship_granger_causality_scan": (RelationshipLagArguments, _relationship_granger),
        "relationship_rolling_correlation_stability": (RelationshipArguments, _relationship_rolling_stability),
        "relationship_pearson_segment_comparison": (
            RelationshipSegmentComparisonArguments,
            _relationship_segment_comparison,
        ),
    }
    return ToolRegistry(
        [
            ToolSpec(
                name=name,
                version=FUNCTION_CATALOG[name].version,
                description=f"{FUNCTION_CATALOG[name].description} {FUNCTION_CATALOG[name].planning_guidance}",
                display_name=FUNCTION_CATALOG[name].title,
                result_key=FUNCTION_CATALOG[name].result_key,
                arguments_model=arguments_model,
                handler=handler,
            )
            for name, (arguments_model, handler) in handlers.items()
        ]
    )
