"""Headless desktop-to-Graph integration checks for long-session retrieval."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.config import Settings
from app.desktop.main_window import MainWindow
from app.desktop.session import ResearchSession, SessionMessage, SessionStore
from app.desktop.workspace import ResearchWorkspace
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent, ModelResearchDialogue
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator


class CapturingDialogueGateway:
    enabled = True
    model_name = "memory-integration-capture"

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def invoke_structured(self, *, messages, schema):
        del schema
        payload = json.loads(next(message.content for message in messages if message.role == "user"))
        self.payloads.append(payload)
        question = payload["question"]
        if question == "蓝鲸调度采用哪种方案？":
            response = "窗口设成48小时。"
        elif "海豚备用容量" in question and len(question) > 1_200:
            response = "已记录这项长文本约束。"
        else:
            response = "这是模拟应用产生的普通助手回答。"
        return DialogueDecision(intent="discussion", response=response)


class UnusedPlanner:
    enabled = True
    model_name = "unused-memory-integration-planner"

    def propose(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("planner should not run in a dialogue-only integration scenario")


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _coordinator(gateway: CapturingDialogueGateway, checkpoint_path: Path) -> ResearchCoordinator:
    settings = Settings(_env_file=None, llm_history_messages=8, llm_retrieved_turns=3)
    return ResearchCoordinator(
        main_agent=MainResearchAgent(
            model_dialogue=ModelResearchDialogue(settings=settings, gateway=gateway),
        ),
        eda_subagent=EDASubagent(model_planner=UnusedPlanner()),
        checkpoint_path=checkpoint_path,
    )


def _wait_until(qt_app: QApplication, predicate, timeout_seconds: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("desktop worker did not finish before timeout")


def _ask(
    qt_app: QApplication,
    window: MainWindow,
    gateway: CapturingDialogueGateway,
    question: str,
) -> dict:
    calls_before = len(gateway.payloads)
    window.workspace.submit_question(question)
    _wait_until(
        qt_app,
        lambda: not window.workspace.is_busy and len(gateway.payloads) == calls_before + 1,
    )
    return gateway.payloads[-1]


def _retrieved_text(payload: dict) -> str:
    return "\n".join(
        message["content"]
        for turn in payload["earlier_related_turns"]
        for message in turn["messages"]
    )


@pytest.fixture(scope="module")
def desktop_boundary_observation(
    qt_app: QApplication,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    root = tmp_path_factory.mktemp("desktop-memory-boundary")
    store = SessionStore(root / "sessions.json")
    checkpoint_path = root / "graph.sqlite3"
    gateway = CapturingDialogueGateway()
    coordinator = _coordinator(gateway, checkpoint_path)
    window = MainWindow(agent=coordinator, session_store=store)
    session_id = window.workspace.current_session.session_id
    try:
        _ask(qt_app, window, gateway, "蓝鲸调度采用哪种方案？")
        for index in range(2, 61):
            _ask(qt_app, window, gateway, f"桌面压力对话 {index}")
        boundary_payload = _ask(qt_app, window, gateway, "蓝鲸调度参数是什么？")
        graph_messages = list(coordinator.get_snapshot(session_id).values["messages"])
        desktop_messages = [item.model_dump(mode="json") for item in window.workspace.current_session.messages]
    finally:
        window.close()
        coordinator.close()
    restored = store.load()
    return {
        "payload": boundary_payload,
        "graph_messages": graph_messages,
        "desktop_messages": desktop_messages,
        "restored_messages": [item.model_dump(mode="json") for item in restored[0].messages],
    }


def test_real_desktop_survives_sixty_one_turns_without_losing_its_archive(
    desktop_boundary_observation: dict,
) -> None:
    desktop_messages = desktop_boundary_observation["desktop_messages"]

    assert len(desktop_boundary_observation["graph_messages"]) == 120
    assert desktop_boundary_observation["restored_messages"] == desktop_messages
    assert sum(item["role"] == "user" and item["kind"] == "text" for item in desktop_messages) == 61
    assert sum(item["role"] == "assistant" and item["kind"] == "text" for item in desktop_messages) == 62


@pytest.mark.xfail(
    strict=True,
    reason="The desktop archive is complete, but Graph retrieval cannot reach the first turn after its 120-message cap.",
)
def test_real_desktop_can_recall_the_first_turn_after_sixty_one_turns(
    desktop_boundary_observation: dict,
) -> None:
    target = next(
        turn
        for turn in desktop_boundary_observation["payload"]["earlier_related_turns"]
        if turn["turn_id"]
    )

    assert [item["role"] for item in target["messages"]] == ["user", "assistant"]
    assert "窗口设成48小时" in "\n".join(item["content"] for item in target["messages"])


@pytest.mark.xfail(
    strict=True,
    reason="Desktop-to-Graph bootstrap applies a message-level character budget and can keep only half of a long turn.",
)
def test_desktop_bootstrap_never_splits_an_oversized_complete_turn() -> None:
    session = ResearchSession(
        messages=[
            SessionMessage(
                role="user",
                kind="text",
                turn_id="oversized-bootstrap-turn",
                content="长问题" * 5_000,
            ),
            SessionMessage(
                role="assistant",
                kind="text",
                turn_id="oversized-bootstrap-turn",
                content="长回答" * 5_000,
            ),
        ]
    )

    projected = ResearchWorkspace._agent_conversation(None, session)  # type: ignore[arg-type]

    assert [item.role for item in projected] == ["user", "assistant"]


def test_real_desktop_flow_retrieves_complete_precise_turns_after_restart(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.json")
    checkpoint_path = tmp_path / "graph.sqlite3"
    first_gateway = CapturingDialogueGateway()
    first_coordinator = _coordinator(first_gateway, checkpoint_path)
    first = MainWindow(agent=first_coordinator, session_store=store)
    session_id = first.workspace.current_session.session_id

    try:
        _ask(qt_app, first, first_gateway, "蓝鲸调度采用哪种方案？")
        long_question = "背景资料。" * 400 + "最后约束是海豚备用容量必须保留百分之十五。"
        _ask(qt_app, first, first_gateway, long_question)
        for index in range(1, 7):
            _ask(qt_app, first, first_gateway, f"继续做电价分析的普通步骤 {index}")

        paired = _ask(qt_app, first, first_gateway, "蓝鲸调度参数是什么？")
        assert "蓝鲸调度参数是什么？" not in {
            message["content"] for message in paired["conversation_history"]
        }
        assert len({message["turn_id"] for message in paired["conversation_history"]}) == 4
        assert len(paired["earlier_related_turns"]) == 1
        assert [
            message["role"] for message in paired["earlier_related_turns"][0]["messages"]
        ] == ["user", "assistant"]
        assert "窗口设成48小时" in _retrieved_text(paired)

        weak = _ask(qt_app, first, first_gateway, "电价方面一个全新的问题")
        assert weak["earlier_related_turns"] == []

        long_payload = _ask(qt_app, first, first_gateway, "海豚备用容量约束是什么？")
        assert "海豚备用容量" in _retrieved_text(long_payload)
        assert "百分之十五" in _retrieved_text(long_payload)
        assert "…[前文已截断]" in _retrieved_text(long_payload)
    finally:
        first.close()
        first_coordinator.close()

    restored_gateway = CapturingDialogueGateway()
    restored_coordinator = _coordinator(restored_gateway, checkpoint_path)
    restored = MainWindow(agent=restored_coordinator, session_store=store)
    try:
        restored.workspace.select_session(session_id)
        restored_payload = _ask(qt_app, restored, restored_gateway, "蓝鲸调度参数是什么？")
        assert "窗口设成48小时" in _retrieved_text(restored_payload)
    finally:
        restored.close()
        restored_coordinator.close()
