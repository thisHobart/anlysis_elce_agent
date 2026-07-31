from langgraph.graph import END, START, StateGraph

from app.graph.nodes.classify import classify
from app.graph.nodes.compose import compose
from app.graph.nodes.query_data import query_data
from app.graph.nodes.screen_action import screen_action
from app.graph.nodes.validate import validate
from app.graph.state import AgentState


def build_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("validate", validate)
    graph.add_node("query_data", query_data)
    graph.add_node("compose", compose)
    graph.add_node("screen_action", screen_action)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "validate")
    graph.add_edge("validate", "query_data")
    graph.add_edge("query_data", "compose")
    graph.add_edge("compose", "screen_action")
    graph.add_edge("screen_action", END)
    return graph.compile()
