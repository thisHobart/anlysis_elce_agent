from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.classify import classify
from app.graph.nodes.compose import compose
from app.graph.nodes.prepare_turn import prepare_turn
from app.graph.nodes.query_data import query_data
from app.graph.nodes.screen_action import screen_action
from app.graph.nodes.validate import validate
from app.graph.state import AgentState
from app.services.llm import AgentLLM, get_llm_service


def build_workflow(*, llm_service: AgentLLM | None = None, checkpointer: Any | None = None):
    service = llm_service or get_llm_service()

    def classify_node(state: AgentState) -> dict:
        return classify(state, llm_service=service)

    def compose_node(state: AgentState) -> dict:
        return compose(state, llm_service=service)

    graph = StateGraph(AgentState)
    graph.add_node("prepare_turn", prepare_turn)
    graph.add_node("classify", classify_node)
    graph.add_node("validate", validate)
    graph.add_node("query_data", query_data)
    graph.add_node("compose", compose_node)
    graph.add_node("screen_action", screen_action)
    graph.add_edge(START, "prepare_turn")
    graph.add_edge("prepare_turn", "classify")
    graph.add_edge("classify", "validate")
    graph.add_edge("validate", "query_data")
    graph.add_edge("query_data", "compose")
    graph.add_edge("compose", "screen_action")
    graph.add_edge("screen_action", END)
    return graph.compile(checkpointer=checkpointer)
