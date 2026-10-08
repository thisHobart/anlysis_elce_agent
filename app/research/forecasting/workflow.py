"""Four-node LangGraph orchestration for the recoverable P3 forecast run."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.research.forecasting.contracts import ForecastPlan, ForecastRunResult
from app.research.forecasting.service import run_forecast_plan


class ForecastGraphState(TypedDict, total=False):
    completed_work_item: str
    result: ForecastRunResult


def execute_forecast_workflow(
    plan: ForecastPlan,
    *,
    progress: Callable[[int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ForecastRunResult:
    """Run three fold nodes and one future node over immutable file checkpoints."""

    graph = StateGraph(ForecastGraphState)

    def fold_node(number: int) -> Callable[[ForecastGraphState], ForecastGraphState]:
        work_item = f"backtest-{number}"

        def execute(_state: ForecastGraphState) -> ForecastGraphState:
            run_forecast_plan(
                plan,
                progress=progress,
                cancelled=cancelled,
                stop_after_work_item=work_item,
            )
            return {"completed_work_item": work_item}

        return execute

    def future_node(_state: ForecastGraphState) -> ForecastGraphState:
        result = run_forecast_plan(plan, progress=progress, cancelled=cancelled)
        if result is None:
            raise RuntimeError("未来预测节点没有生成结果")
        return {"completed_work_item": "future", "result": result}

    for number in range(1, 4):
        graph.add_node(f"backtest_{number}", fold_node(number))
    graph.add_node("future", future_node)
    graph.add_edge(START, "backtest_1")
    graph.add_edge("backtest_1", "backtest_2")
    graph.add_edge("backtest_2", "backtest_3")
    graph.add_edge("backtest_3", "future")
    graph.add_edge("future", END)
    state = graph.compile().invoke({})
    result = state.get("result")
    if result is None:
        raise RuntimeError("预测Graph结束时缺少最终结果")
    return result
