"""Characterization and stress checks for the complete single-session memory path.

These tests deliberately distinguish durable desktop storage, Graph retention,
and model-visible prompt context, including semantic, reference-only, and long-
answer retrieval boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.desktop.session import ResearchSession, SessionMessage, SessionStore
from app.research.agent.context import MAX_PERSISTED_CONVERSATION_MESSAGES
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent, ModelResearchDialogue
from app.research.agent.retrieval import (
    DEFAULT_RECENT_TURN_CHARACTER_BUDGET,
    DEFAULT_RETRIEVAL_CHARACTER_BUDGET,
    retrieve_related,
    select_conversation_context,
)
from app.research.agent.schemas import ConversationMessage
from app.research.agent.subagents.eda import EDASubagent, ModelEDAPlanner
from app.research.application.coordinator import ResearchCoordinator
from app.research.planning.contracts import EDAPlanDraft
from app.research.schemas.study import load_study_config


def _message(index: int, content: str, *, role: str = "user", turn_id: str | None = None) -> ConversationMessage:
    return ConversationMessage(
        message_id=f"boundary-message-{index:04d}",
        turn_id=turn_id,
        role=role,
        content=content,
        created_at=f"2026-08-29T00:{index % 60:02d}:00+00:00",
    )


def _turn(index: int, question: str, answer: str) -> list[ConversationMessage]:
    turn_id = f"boundary-turn-{index:04d}"
    return [
        _message(index * 2 - 1, question, turn_id=turn_id),
        _message(index * 2, answer, role="assistant", turn_id=turn_id),
    ]


def _flatten_turns(turns) -> list:
    return [message for turn in turns for message in turn.messages]


@pytest.mark.parametrize("turn_count", [0, 1, 4, 5, 8, 16, 32, 64, 128, 256, 500])
def test_selector_stays_bounded_through_one_thousand_messages(turn_count: int) -> None:
    history = [
        message
        for index in range(1, turn_count + 1)
        for message in _turn(
            index,
            (
                "请记住压力测试目标 stress-anchor-0001"
                if index == 1
                else f"普通压力记录 filler-{index:04d}"
            ),
            "stress-anchor-0001 的值是 48" if index == 1 else f"普通回答 {index}",
        )
    ]

    recent, earlier = select_conversation_context(
        history,
        question="stress-anchor-0001 的值是什么",
        recent_turns=4,
        retrieved_turns=3,
    )

    assert len({item.turn_id for item in recent}) <= 4
    assert sum(len(item.content) for item in recent) <= DEFAULT_RECENT_TURN_CHARACTER_BUDGET
    assert len(earlier) <= 3
    assert sum(len(item.content) for item in _flatten_turns(earlier)) <= DEFAULT_RETRIEVAL_CHARACTER_BUDGET
    assert all([item.role for item in turn.messages] == ["user", "assistant"] for turn in earlier)
    if turn_count > 4:
        assert earlier and earlier[0].turn_id == "boundary-turn-0001"


def test_prompt_memory_character_budgets_are_hard_even_for_oversized_turns() -> None:
    history = [
        message
        for index in range(1, 9)
        for message in _turn(
            index,
            f"预算主题 budget-{index:04d} " + "问" * 10_000,
            f"预算主题 budget-{index:04d} " + "答" * 10_000,
        )
    ]

    recent, earlier = select_conversation_context(
        history,
        question="budget-0001 的预算主题是什么",
        recent_turns=4,
        retrieved_turns=3,
    )

    assert 0 < sum(len(item.content) for item in recent) <= DEFAULT_RECENT_TURN_CHARACTER_BUDGET
    assert sum(len(item.content) for item in _flatten_turns(earlier)) <= DEFAULT_RETRIEVAL_CHARACTER_BUDGET
    assert len({item.turn_id for item in recent}) == 1
    assert earlier[0].turn_id == "boundary-turn-0001"


def test_session_store_round_trips_one_thousand_messages_without_loss(tmp_path: Path) -> None:
    session = ResearchSession(
        title="一千条消息压力测试",
        messages=[
            SessionMessage(
                message_id=f"desktop-{index:04d}",
                turn_id=f"desktop-turn-{index // 2:04d}",
                role="user" if index % 2 == 0 else "assistant",
                content=f"桌面存档消息 {index} archive-anchor-{index:04d}",
            )
            for index in range(1_000)
        ],
    )
    store = SessionStore(tmp_path / "sessions.json")

    store.save([session])
    restored = store.load()

    assert len(restored) == 1
    assert len(restored[0].messages) == 1_000
    assert restored[0].messages[0].message_id == "desktop-0000"
    assert restored[0].messages[-1].content.endswith("archive-anchor-0999")


class _BoundaryDialogueGateway:
    enabled = True
    model_name = "boundary-capture"

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def invoke_structured(self, *, messages, schema):
        del schema
        payload = json.loads(next(item.content for item in messages if item.role == "user"))
        self.payloads.append(payload)
        answer = (
            "边界代号 graph-anchor-0001 的答案是 48。"
            if payload["question"] == "请记住边界代号 graph-anchor-0001"
            else "普通模拟回答。"
        )
        return DialogueDecision(intent="discussion", response=answer)


class _UnusedPlanner:
    enabled = True
    model_name = "unused-boundary-planner"

    def propose(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("planner should not run in the dialogue boundary scenario")


class _NewPlanDialogueGateway:
    enabled = True
    model_name = "new-plan-capture"

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def invoke_structured(self, *, messages, schema):
        del schema
        payload = json.loads(next(item.content for item in messages if item.role == "user"))
        self.payloads.append(payload)
        return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")


class _CapturingPlannerGateway:
    enabled = True
    model_name = "planner-memory-capture"

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def invoke_structured(self, *, messages, schema):
        del schema
        payload = json.loads(next(item.content for item in messages if item.role == "user"))
        self.payloads.append(payload)
        return EDAPlanDraft(objective=payload["question"])


def test_graph_planning_path_receives_recent_and_retrieved_memory(
    synthetic_study: Path,
    tmp_path: Path,
) -> None:
    config = load_study_config(synthetic_study)
    settings = Settings(_env_file=None, llm_history_messages=8, llm_retrieved_turns=3)
    dialogue_gateway = _NewPlanDialogueGateway()
    planner_gateway = _CapturingPlannerGateway()
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(
            model_dialogue=ModelResearchDialogue(settings=settings, gateway=dialogue_gateway),
        ),
        eda_subagent=EDASubagent(
            model_planner=ModelEDAPlanner(settings=settings, gateway=planner_gateway),
        ),
        checkpoint_path=tmp_path / "planner-graph.sqlite3",
    )
    history = [
        *_turn(1, "wind 的 max_lag 不超过 24", "已经记录历史参数。"),
        *[
            item
            for index in range(2, 7)
            for item in _turn(index, f"普通天气记录 {index}", "普通回答。")
        ],
    ]
    try:
        snapshot = coordinator.submit_user_message(
            session_id="planner-memory-session",
            message="继续按 max_lag 分析 wind 的滞后关系",
            message_id="planner-current-message",
            turn_id="planner-current-turn",
            study_config=config,
            conversation=history,
        )
    finally:
        coordinator.close()

    assert snapshot.interrupt is not None and snapshot.interrupt.kind == "plan_approval"
    assert len(dialogue_gateway.payloads) == 1
    assert len(planner_gateway.payloads) == 1
    for payload in (dialogue_gateway.payloads[0], planner_gateway.payloads[0]):
        assert len({item["turn_id"] for item in payload["conversation_history"]}) == 4
        assert payload["earlier_related_turns"][0]["turn_id"] == "boundary-turn-0001"
        assert [
            item["role"] for item in payload["earlier_related_turns"][0]["messages"]
        ] == ["user", "assistant"]
        assert "继续按 max_lag 分析 wind 的滞后关系" not in {
            item["content"] for item in payload["conversation_history"]
        }


@pytest.fixture(scope="module")
def graph_boundary_observation(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("single-session-graph-boundary")
    checkpoint_path = root / "graph.sqlite3"
    settings = Settings(_env_file=None, llm_history_messages=8, llm_retrieved_turns=3)
    gateway = _BoundaryDialogueGateway()
    coordinator = ResearchCoordinator(
        main_agent=MainResearchAgent(
            model_dialogue=ModelResearchDialogue(settings=settings, gateway=gateway),
        ),
        eda_subagent=EDASubagent(model_planner=_UnusedPlanner()),
        checkpoint_path=checkpoint_path,
    )
    session_id = "graph-boundary-session"
    try:
        coordinator.submit_user_message(
            session_id=session_id,
            message="请记住边界代号 graph-anchor-0001",
            message_id="graph-user-0001",
            turn_id="graph-turn-0001",
        )
        for index in range(2, 61):
            coordinator.submit_user_message(
                session_id=session_id,
                message=f"普通对话 filler-{index:04d}",
                message_id=f"graph-user-{index:04d}",
                turn_id=f"graph-turn-{index:04d}",
            )
        boundary = coordinator.submit_user_message(
            session_id=session_id,
            message="graph-anchor-0001 对应的答案是什么",
            message_id="graph-user-0061",
            turn_id="graph-turn-0061",
        )
        boundary_payload = gateway.payloads[-1]
        completed_messages = list(boundary.values["messages"])
    finally:
        coordinator.close()

    restored = ResearchCoordinator(
        main_agent=MainResearchAgent(
            model_dialogue=ModelResearchDialogue(settings=settings, gateway=_BoundaryDialogueGateway()),
        ),
        eda_subagent=EDASubagent(model_planner=_UnusedPlanner()),
        checkpoint_path=checkpoint_path,
    )
    try:
        restored_messages = list(restored.get_snapshot(session_id).values["messages"])
    finally:
        restored.close()
    return {
        "payload": boundary_payload,
        "completed_messages": completed_messages,
        "restored_messages": restored_messages,
    }


def test_graph_retention_is_exactly_120_messages_and_survives_restart(graph_boundary_observation: dict) -> None:
    completed = graph_boundary_observation["completed_messages"]
    restored = graph_boundary_observation["restored_messages"]

    assert MAX_PERSISTED_CONVERSATION_MESSAGES == 120
    assert len(completed) == MAX_PERSISTED_CONVERSATION_MESSAGES
    assert restored == completed
    assert completed[0]["turn_id"] == "graph-turn-0001"
    assert completed[-1]["turn_id"] == "graph-turn-0061"
    assert "graph-turn-0002" not in {item["turn_id"] for item in completed}
    assert all(
        [item["role"] for item in completed if item.get("turn_id") == turn_id] == ["user", "assistant"]
        for turn_id in dict.fromkeys(item.get("turn_id") for item in completed)
    )


def test_graph_never_injects_an_incomplete_retrieved_turn(graph_boundary_observation: dict) -> None:
    target = next(
        turn
        for turn in graph_boundary_observation["payload"]["earlier_related_turns"]
        if turn["turn_id"] == "graph-turn-0001"
    )

    assert [item["role"] for item in target["messages"]] == ["user", "assistant"]


def test_semantic_paraphrase_without_lexical_overlap_is_recalled() -> None:
    history = [
        *_turn(1, "把图表导出到报告目录", "输出保存为 HTML。"),
        *[
            item
            for index in range(2, 8)
            for item in _turn(index, f"普通天气记录 {index}", "普通回答。")
        ],
    ]

    retrieved = retrieve_related(history, question="将可视化写入产物文件夹", exclude=history[-8:])

    assert [turn.turn_id for turn in retrieved] == ["boundary-turn-0001"]


def test_pronoun_only_reference_beyond_recent_window_is_recalled() -> None:
    history = [
        *_turn(1, "蓝鲸调度窗口设为 48 小时", "已经记录这个参数。"),
        *[
            item
            for index in range(2, 7)
            for item in _turn(index, f"普通天气记录 {index}", "普通回答。")
        ],
    ]

    _recent, retrieved = select_conversation_context(history, question="继续按刚才那个做")

    assert [turn.turn_id for turn in retrieved] == ["boundary-turn-0001"]


def test_long_paired_answer_keeps_its_tail_conclusion() -> None:
    history = _turn(
        1,
        "海豚备用容量怎么设置？",
        "海豚备用容量分析如下。" + "背景说明。" * 1_000 + "最终结论：备用容量设为百分之十五。",
    )

    retrieved = retrieve_related(
        history,
        question="海豚备用容量设置结论",
        max_characters=600,
    )

    combined = "\n".join(item.content for item in retrieved[0].messages)
    assert "最终结论：备用容量设为百分之十五" in combined
