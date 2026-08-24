"""Regression tests for cooperative desktop cancellation and checkpoint cleanup."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from time import sleep

from PySide6.QtWidgets import QApplication

from app.desktop.worker import FunctionWorker
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator
from app.research.schemas.study import load_study_config


class CancelDialogue:
    enabled = True
    model_name = "cancel-test-model"

    def decide(self, **kwargs):
        if kwargs.get("config") is None:
            return DialogueDecision(intent="discussion", response="先准备数据。")
        return DialogueDecision(intent="new_plan", skill_name="price-exogenous-eda")


class CancelPlanner:
    enabled = True
    model_name = "cancel-test-model"

    def propose(self, _question, _config, _quality, history=None, skill=None, feedback=None):
        del history, feedback
        assert skill is not None
        return {
            "objective": "验证取消路径",
            "hypotheses": ["取消应在进度边界生效。"],
            "selected_variables": [],
            "steps": [
                {
                    "tool": "price_profile",
                    "enabled": True,
                    "rationale": "提供一个可审批的本地分析步骤。",
                    "parameters": {"methods": ["distribution"]},
                }
            ],
        }


def test_worker_cancel_stops_at_next_progress_boundary():
    app = QApplication.instance() or QApplication([])
    started = Event()
    cancelled: list[bool] = []
    completed: list[bool] = []

    def operation(progress):
        started.set()
        while True:
            progress(50, "safe boundary")
            sleep(0.005)

    worker = FunctionWorker(operation)
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.completed.connect(lambda _result: completed.append(True))
    thread = Thread(target=worker.run)
    thread.start()
    assert started.wait(2)
    worker.cancel()
    thread.join(2)
    app.processEvents()

    assert not thread.is_alive()
    assert cancelled == [True]
    assert completed == []


def test_cancel_removes_checkpoint_so_restart_has_no_resumable_work(tmp_path: Path):
    database = tmp_path / "cancel.sqlite3"
    config = load_study_config(Path("configs/research/price_exogenous_eda.yaml"))
    first = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=CancelDialogue()),
        eda_subagent=EDASubagent(model_planner=CancelPlanner()),
        checkpoint_path=database,
    )
    session_id = "desktop-cancel-restart"
    approval = first.submit_user_message(
        session_id=session_id,
        message="分析电价结构",
        study_config=config,
    )
    assert approval.interrupt and approval.interrupt.kind == "plan_approval"
    first.cancel(session_id)
    assert not first.has_thread(session_id)
    first.close()

    restored = ResearchCoordinator(
        main_agent=MainResearchAgent(model_dialogue=CancelDialogue()),
        eda_subagent=EDASubagent(model_planner=CancelPlanner()),
        checkpoint_path=database,
    )
    try:
        assert not restored.has_thread(session_id)
    finally:
        restored.close()
