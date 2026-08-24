"""Registered EDA functions executed only after plan validation and approval."""

from __future__ import annotations

from app.research.tools.catalog import TOOL_CATALOG
from app.research.tools.contracts import (
    DataQualityArguments,
    ExogenousProfileArguments,
    PriceProfileArguments,
    RelationshipArguments,
    ToolArguments,
    ToolContext,
    ToolOutput,
    ToolSpec,
)
from app.research.tools.eda.exogenous import analyze_exogenous
from app.research.tools.eda.price import analyze_price
from app.research.tools.eda.relationships import analyze_relationships
from app.research.tools.registry import ToolRegistry


def _data_quality(context: ToolContext, _arguments: ToolArguments) -> ToolOutput:
    return ToolOutput(result_key="data_quality", value=context.quality.model_dump(mode="json"))


def _price_profile(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    parsed = PriceProfileArguments.model_validate(arguments)
    config = context.config
    return ToolOutput(
        result_key="price",
        value=analyze_price(
            context.frame[config.target.name],
            unit=config.target.unit,
            frequency=config.study.frequency,
            max_lag=parsed.max_lag,
            spike_iqr_multiplier=parsed.spike_iqr_multiplier,
            methods=set(parsed.methods),
        ),
    )


def _exogenous_profile(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    parsed = ExogenousProfileArguments.model_validate(arguments)
    units = {spec.name: spec.unit for spec in context.config.exogenous}
    return ToolOutput(
        result_key="exogenous",
        value=analyze_exogenous(
            context.frame,
            names=parsed.variables,
            units=units,
            outlier_iqr_multiplier=parsed.outlier_iqr_multiplier,
            methods=set(parsed.methods),
        ),
    )


def _relationship_analysis(context: ToolContext, arguments: ToolArguments) -> ToolOutput:
    parsed = RelationshipArguments.model_validate(arguments)
    return ToolOutput(
        result_key="relationships",
        value=analyze_relationships(
            context.frame,
            target_name=context.config.target.name,
            exogenous_names=parsed.variables,
            max_lag=parsed.max_lag,
            min_observations=parsed.min_observations,
            methods=set(parsed.methods),
        ),
    )


def build_eda_tool_registry() -> ToolRegistry:
    """Register the local functions exposed to the Phase 1 Agent."""

    handlers = {
        "data_quality": (DataQualityArguments, _data_quality),
        "price_profile": (PriceProfileArguments, _price_profile),
        "exogenous_profile": (ExogenousProfileArguments, _exogenous_profile),
        "relationship_analysis": (RelationshipArguments, _relationship_analysis),
    }
    return ToolRegistry(
        [
            ToolSpec(
                name=name,
                version=TOOL_CATALOG[name].version,
                description=TOOL_CATALOG[name].description,
                arguments_model=arguments_model,
                handler=handler,
            )
            for name, (arguments_model, handler) in handlers.items()
        ]
    )
