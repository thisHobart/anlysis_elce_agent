"""Measure the current single-session memory layers without calling a live model.

Run from the repository root:
    python scripts/benchmark_single_session_memory.py

The command prints one JSON document so results can be archived or compared
between revisions without coupling the benchmark to a particular report format.
"""

from __future__ import annotations

import json
import platform
import statistics
import tempfile
import time
from math import ceil
from pathlib import Path
from typing import Any

from app.config import Settings
from app.desktop.session import ResearchSession, SessionMessage, SessionStore
from app.research.agent.context import MAX_PERSISTED_CONVERSATION_MESSAGES
from app.research.agent.orchestrator import DialogueDecision, MainResearchAgent, ModelResearchDialogue
from app.research.agent.retrieval import (
    DEFAULT_RECENT_TURN_CHARACTER_BUDGET,
    DEFAULT_RETRIEVAL_CHARACTER_BUDGET,
    MAX_RETRIEVED_MESSAGE_CHARACTERS,
    retrieve_related,
    select_conversation_context,
)
from app.research.agent.schemas import ConversationMessage
from app.research.agent.subagents.eda import EDASubagent
from app.research.application.coordinator import ResearchCoordinator

MESSAGE_COUNTS = (8, 16, 32, 64, 128, 256, 512, 1_000, 2_000, 5_000, 10_000)


def percentile(samples: list[float], value: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * value
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def timing(samples: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(statistics.median(samples) * 1_000, 3),
        "p95_ms": round(percentile(samples, 0.95) * 1_000, 3),
        "max_ms": round(max(samples, default=0.0) * 1_000, 3),
    }


def estimated_tokens(text: str) -> tuple[int, str]:
    """Use the current OpenAI tokenizer when available, with a conservative fallback."""

    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text)), "o200k_base"
    except (ImportError, KeyError):
        return ceil(len(text) / 2), "character_fallback_2_to_1"


def message(index: int, content: str, *, role: str = "user", turn_id: str | None = None) -> ConversationMessage:
    return ConversationMessage(
        message_id=f"bench-message-{index:05d}",
        turn_id=turn_id,
        role=role,
        content=content,
        created_at=f"2026-08-29T00:{index % 60:02d}:00+00:00",
    )


def turn(index: int, question: str, answer: str) -> list[ConversationMessage]:
    turn_id = f"bench-turn-{index:05d}"
    return [
        message(index * 2 - 1, question, turn_id=turn_id),
        message(index * 2, answer, role="assistant", turn_id=turn_id),
    ]


def stress_history(message_count: int) -> list[ConversationMessage]:
    turn_count = message_count // 2
    return [
        item
        for index in range(1, turn_count + 1)
        for item in turn(
            index,
            "压力锚点 stress-anchor-0001" if index == 1 else f"普通压力记录 filler-{index:05d}",
            "stress-anchor-0001 对应 48" if index == 1 else f"普通回答 {index}",
        )
    ]


def benchmark_retrieval_scale() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message_count in MESSAGE_COUNTS:
        history = stress_history(message_count)
        samples: list[float] = []
        recent = []
        selected = []
        for _ in range(40):
            started = time.perf_counter()
            recent, selected = select_conversation_context(
                history,
                question="stress-anchor-0001 的值是什么",
                recent_turns=4,
                retrieved_turns=3,
            )
            samples.append(time.perf_counter() - started)
        target_turn = "bench-turn-00001"
        in_recent = any(item.turn_id == target_turn for item in recent)
        in_retrieval = any(item.turn_id == target_turn for item in selected)
        rows.append(
            {
                "messages": message_count,
                "turns": message_count // 2,
                "target_accessible": in_recent or in_retrieval,
                "target_location": "recent" if in_recent else ("retrieval" if in_retrieval else "missing"),
                "retrieved_turns": len(selected),
                "retrieved_characters": sum(
                    len(item.content) for selected_turn in selected for item in selected_turn.messages
                ),
                **timing(samples),
            }
        )
    return rows


def benchmark_retrieval_quality() -> dict[str, Any]:
    filler = [
        item
        for index in range(2, 8)
        for item in turn(index, f"普通天气记录 {index}", "普通回答。")
    ]
    cases = [
        (
            "exact_identifier",
            [*turn(1, "把 max_lag 调整为多少", "建议设为 48"), *filler],
            "max_lag 的值是什么",
            {"bench-turn-00001"},
        ),
        (
            "chinese_lexical_overlap",
            [*turn(1, "检查电价日内周期性", "需要按小时比较"), *filler],
            "电价日内周期性的结论",
            {"bench-turn-00001"},
        ),
        (
            "unrelated",
            [*turn(1, "检查电价日内周期性", "需要按小时比较"), *filler],
            "max_lag 应该设成多少",
            set(),
        ),
        (
            "synonym_only",
            [*turn(1, "把图表导出到报告目录", "输出保存为 HTML"), *filler],
            "将可视化写入产物文件夹",
            {"bench-turn-00001"},
        ),
        (
            "pronoun_only",
            [*turn(1, "蓝鲸调度窗口设为 48 小时", "已经记录参数"), *filler],
            "继续按刚才那个做",
            {"bench-turn-00001"},
        ),
    ]
    rows: list[dict[str, Any]] = []
    true_positives = 0
    retrieved_total = 0
    gold_total = 0
    for name, history, question, gold in cases:
        selected = retrieve_related(history, question=question, exclude=history[-8:])
        actual = {item.turn_id for item in selected}
        hits = actual.intersection(gold)
        true_positives += len(hits)
        retrieved_total += len(actual)
        gold_total += len(gold)
        rows.append(
            {
                "case": name,
                "gold": sorted(gold),
                "actual": sorted(str(item) for item in actual),
                "precision": round(len(hits) / len(actual), 4) if actual else (1.0 if not gold else 0.0),
                "recall": round(len(hits) / len(gold), 4) if gold else 1.0,
            }
        )
    return {
        "cases": rows,
        "micro_precision": round(true_positives / retrieved_total, 4) if retrieved_total else 0.0,
        "micro_recall": round(true_positives / gold_total, 4) if gold_total else 0.0,
    }


def benchmark_prompt_budgets() -> dict[str, Any]:
    history = [
        item
        for index in range(1, 9)
        for item in turn(
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
    earlier_messages = [item for selected_turn in earlier for item in selected_turn.messages]
    recent_text = "\n".join(item.content for item in recent)
    retrieved_text = "\n".join(item.content for item in earlier_messages)
    recent_tokens, tokenizer = estimated_tokens(recent_text)
    retrieved_tokens, _tokenizer = estimated_tokens(retrieved_text)
    return {
        "configured_recent_turns": 4,
        "configured_retrieved_turns": 3,
        "recent_turns_observed": len({item.turn_id for item in recent}),
        "retrieved_turns_observed": len(earlier),
        "recent_characters_observed": sum(len(item.content) for item in recent),
        "retrieved_characters_observed": sum(len(item.content) for item in earlier_messages),
        "recent_tokens_estimated": recent_tokens,
        "retrieved_tokens_estimated": retrieved_tokens,
        "combined_tokens_estimated": recent_tokens + retrieved_tokens,
        "tokenizer": tokenizer,
        "recent_character_hard_cap": DEFAULT_RECENT_TURN_CHARACTER_BUDGET,
        "retrieved_character_hard_cap": DEFAULT_RETRIEVAL_CHARACTER_BUDGET,
        "retrieved_per_message_hard_cap": MAX_RETRIEVED_MESSAGE_CHARACTERS,
    }


def benchmark_session_store(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message_count in MESSAGE_COUNTS:
        path = root / f"sessions-{message_count}.json"
        store = SessionStore(path)
        session = ResearchSession(
            title=f"{message_count} messages",
            messages=[
                SessionMessage(
                    message_id=f"store-{index:05d}",
                    turn_id=f"store-turn-{index // 2:05d}",
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"存档压力消息 {index} archive-anchor-{index:05d}",
                )
                for index in range(message_count)
            ],
        )
        save_samples: list[float] = []
        load_samples: list[float] = []
        restored: list[ResearchSession] = []
        for _ in range(7):
            started = time.perf_counter()
            store.save([session])
            save_samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            restored = store.load()
            load_samples.append(time.perf_counter() - started)
        rows.append(
            {
                "messages": message_count,
                "round_trip_exact": bool(restored and len(restored[0].messages) == message_count),
                "file_bytes": path.stat().st_size,
                "save": timing(save_samples),
                "load": timing(load_samples),
            }
        )
    return rows


class BoundaryGateway:
    enabled = True
    model_name = "benchmark-boundary-capture"

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def invoke_structured(self, *, messages, schema):
        del schema
        payload = json.loads(next(item.content for item in messages if item.role == "user"))
        self.payloads.append(payload)
        answer = (
            "图状态锚点 graph-anchor-0001 的值是 48"
            if payload["question"] == "记录图状态锚点 graph-anchor-0001"
            else "普通模拟回答"
        )
        return DialogueDecision(intent="discussion", response=answer)


class UnusedPlanner:
    enabled = True
    model_name = "unused-benchmark-planner"

    def propose(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("planner was not expected")


def graph_coordinator(gateway: BoundaryGateway, path: Path) -> ResearchCoordinator:
    settings = Settings(_env_file=None, llm_history_messages=8, llm_retrieved_turns=3)
    return ResearchCoordinator(
        main_agent=MainResearchAgent(
            model_dialogue=ModelResearchDialogue(settings=settings, gateway=gateway),
        ),
        eda_subagent=EDASubagent(model_planner=UnusedPlanner()),
        checkpoint_path=path,
    )


def benchmark_graph_boundary(root: Path) -> dict[str, Any]:
    path = root / "graph-boundary.sqlite3"
    gateway = BoundaryGateway()
    coordinator = graph_coordinator(gateway, path)
    session_id = "benchmark-boundary-session"
    samples: list[float] = []
    try:
        for index in range(1, 61):
            question = "记录图状态锚点 graph-anchor-0001" if index == 1 else f"普通对话 filler-{index:04d}"
            started = time.perf_counter()
            coordinator.submit_user_message(
                session_id=session_id,
                message=question,
                message_id=f"graph-user-{index:04d}",
                turn_id=f"graph-turn-{index:04d}",
            )
            samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        snapshot = coordinator.submit_user_message(
            session_id=session_id,
            message="graph-anchor-0001 的值是什么",
            message_id="graph-user-0061",
            turn_id="graph-turn-0061",
        )
        samples.append(time.perf_counter() - started)
        payload = gateway.payloads[-1]
        target = next(
            (item for item in payload["earlier_related_turns"] if item["turn_id"] == "graph-turn-0001"),
            None,
        )
        complete_messages = list(snapshot.values["messages"])
    finally:
        coordinator.close()

    started = time.perf_counter()
    restored = graph_coordinator(BoundaryGateway(), path)
    try:
        restored_messages = list(restored.get_snapshot(session_id).values["messages"])
    finally:
        restored.close()
    restore_seconds = time.perf_counter() - started
    checkpoint_bytes = sum(
        item.stat().st_size
        for item in root.glob(f"{path.name}*")
        if item.is_file()
    )
    return {
        "submitted_turns": 61,
        "persisted_messages": len(complete_messages),
        "restored_messages": len(restored_messages),
        "earliest_persisted_turn": complete_messages[0].get("turn_id"),
        "latest_persisted_turn": complete_messages[-1].get("turn_id"),
        "boundary_retrieval_roles": [item["role"] for item in (target or {}).get("messages", [])],
        "checkpoint_bytes": checkpoint_bytes,
        "restart_load_ms": round(restore_seconds * 1_000, 3),
        "turn_latency": timing(samples),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="vpp-memory-benchmark-") as temporary:
        root = Path(temporary)
        settings = Settings(_env_file=None)
        result = {
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "effective_defaults": {
                "history_messages": settings.llm_history_messages,
                "recent_turns": max(1, settings.llm_history_messages // 2),
                "retrieved_turns": settings.llm_retrieved_turns,
                "graph_messages": MAX_PERSISTED_CONVERSATION_MESSAGES,
                "recent_characters": DEFAULT_RECENT_TURN_CHARACTER_BUDGET,
                "retrieved_characters": DEFAULT_RETRIEVAL_CHARACTER_BUDGET,
            },
            "retrieval_scale": benchmark_retrieval_scale(),
            "retrieval_quality": benchmark_retrieval_quality(),
            "prompt_budgets": benchmark_prompt_budgets(),
            "session_store": benchmark_session_store(root),
            "graph_boundary": benchmark_graph_boundary(root),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
