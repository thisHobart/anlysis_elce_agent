"""Retrieval that lets one long session reach complete turns past its recent window."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.llm.gateway import ModelMessage
from app.research.agent.context import bounded_recent_history
from app.research.agent.orchestrator import ModelResearchDialogue
from app.research.agent.retrieval import retrieve_related, tokenize
from app.research.agent.schemas import ConversationMessage
from app.research.agent.subagents.eda import EDASubagent, ModelEDAPlanner
from app.research.application.planning import EDAPlanningService, prepare_research_data
from app.research.schemas.study import load_study_config
from app.research.skills.loader import load_skill


class CaptureGateway:
    enabled = True
    model_name = "capture"

    def __init__(self) -> None:
        self.calls: list[list[ModelMessage]] = []

    def invoke_structured(self, *, messages, schema):
        self.calls.append(messages)
        raise RuntimeError(f"captured {schema.__name__}")


def _payload(messages: list[ModelMessage]) -> dict:
    return json.loads(next(message.content for message in messages if message.role == "user"))


def _message(
    index: int,
    content: str,
    role: str = "user",
    *,
    turn_id: str | None = None,
) -> ConversationMessage:
    return ConversationMessage(
        message_id=f"m{index:03d}",
        turn_id=turn_id,
        role=role,
        content=content,
        created_at=f"2026-08-28T09:{index:02d}:00+00:00",
    )


def _turn(index: int, question: str, answer: str) -> list[ConversationMessage]:
    turn_id = f"turn-{index}"
    return [
        _message(index * 2 - 1, question, turn_id=turn_id),
        _message(index * 2, answer, role="assistant", turn_id=turn_id),
    ]


def test_chinese_text_is_tokenized_without_a_segmentation_dictionary():
    assert tokenize("电价周期") == ["电价", "价周", "周期"]
    assert tokenize("max_lag 48") == ["max_lag", "48"]


def test_retrieval_reaches_a_turn_the_recent_window_dropped():
    history = [
        *_turn(1, "先看电价的日内周期性", "周期分析需要按小时分组。"),
        *[
            message
            for index in range(2, 10)
            for message in _turn(index, f"无关的天气记录 {index}", "已记录。")
        ],
    ]
    window = bounded_recent_history(history, max_messages=8)

    retrieved = retrieve_related(history, question="电价的日内周期性做了吗", exclude=window)

    assert [item.turn_id for item in retrieved] == ["turn-1"]
    assert [message.role for message in retrieved[0].messages] == ["user", "assistant"]
    assert "周期" in retrieved[0].matched_terms


def test_matching_question_brings_the_answer_from_the_same_turn():
    history = [
        *_turn(1, "蓝鲸调度采用哪种方案？", "窗口设成48小时。"),
        *[
            message
            for index in range(2, 7)
            for message in _turn(index, f"普通天气记录 {index}", "普通回答。")
        ],
    ]

    retrieved = retrieve_related(
        history,
        question="蓝鲸调度参数是什么？",
        exclude=history[-8:],
    )

    assert len(retrieved) == 1
    assert [message.content for message in retrieved[0].messages] == [
        "蓝鲸调度采用哪种方案？",
        "窗口设成48小时。",
    ]


def test_retrieval_excludes_a_whole_turn_when_one_message_is_recent():
    old_turn = _turn(1, "电价周期性怎么分析？", "按小时和月份分别检查。")

    retrieved = retrieve_related(old_turn, question="电价周期性", exclude=[old_turn[-1]])

    assert retrieved == []


def test_domain_wide_common_terms_do_not_fill_the_retrieval_budget():
    history = [
        message
        for index in range(1, 9)
        for message in _turn(index, f"电价分析的普通记录 {index}", "继续进行电价分析。")
    ]

    assert retrieve_related(history, question="电价方面一个全新的问题", exclude=[]) == []


def test_unrelated_history_costs_nothing():
    history = [_message(index, f"完全无关的第 {index} 条") for index in range(1, 12)]

    assert retrieve_related(history, question="max_lag 应该设成多少", exclude=[]) == []


def test_exact_identifier_is_a_strong_match_by_itself():
    history = [
        *_turn(1, "把 max_lag 调整为多少？", "建议设为 48。"),
        *[
            message
            for index in range(2, 8)
            for message in _turn(index, f"普通记录 {index}", "已记录。")
        ],
    ]

    retrieved = retrieve_related(history, question="max_lag 的值是什么", exclude=history[-8:])

    assert [item.turn_id for item in retrieved] == ["turn-1"]


def test_selection_is_bounded_and_returned_in_conversation_order():
    history = [
        message
        for index in range(1, 10)
        for message in _turn(index, f"专题{index}讨论海豚备用容量", f"专题{index}结论。")
    ]

    retrieved = retrieve_related(history, question="海豚备用容量", exclude=[], limit=3)

    assert len(retrieved) == 3
    assert [item.turn_id for item in retrieved] == sorted(item.turn_id for item in retrieved)


def test_character_budget_keeps_the_match_near_the_end_of_a_long_message():
    history = _turn(
        1,
        "背景资料。" * 1_000 + "最后约束是海豚备用容量必须保留百分之十五。",
        "已记录这项约束。",
    )

    retrieved = retrieve_related(history, question="海豚备用容量约束是什么", exclude=[], max_characters=600)

    assert len(retrieved) == 1
    combined = "\n".join(message.content for message in retrieved[0].messages)
    assert len(combined) <= 600
    assert "海豚备用容量" in combined
    assert "百分之十五" in combined
    assert "…[前文已截断]" in combined


def test_legacy_messages_without_turn_ids_are_paired_by_adjacency():
    history = [
        _message(1, "蓝鲸调度采用哪种方案？"),
        _message(2, "窗口设成48小时。", role="assistant"),
    ]

    retrieved = retrieve_related(history, question="蓝鲸调度参数是什么？", exclude=[])

    assert len(retrieved) == 1
    assert [message.role for message in retrieved[0].messages] == ["user", "assistant"]


def test_dialogue_prompt_carries_retrieved_turns_beside_the_window(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    prepared = prepare_research_data(config)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()
    history = [
        *_turn(1, "重点关注电价的日内周期性", "需要比较每个小时。"),
        *[
            message
            for index in range(2, 10)
            for message in _turn(index, f"第 {index} 条天气记录", "已记录天气信息。")
        ],
    ]

    with pytest.raises(RuntimeError, match="captured DialogueDecision"):
        ModelResearchDialogue(gateway=gateway).decide(
            question="日内周期性的结论是什么",
            status="awaiting_user",
            config=config,
            plan=None,
            data_profile=None,
            quality_report=prepared.quality.model_dump(mode="json"),
            summary=None,
            evaluation=None,
            history=history,
            available_skills=[skill.prompt_context()],
        )

    payload = _payload(gateway.calls[0])
    assert len(payload["conversation_history"]) == 8
    assert [item["turn_id"] for item in payload["earlier_related_turns"]] == ["turn-1"]
    assert [message["role"] for message in payload["earlier_related_turns"][0]["messages"]] == [
        "user",
        "assistant",
    ]
    window_ids = {item["message_id"] for item in payload["conversation_history"]}
    retrieved_ids = {
        message["message_id"]
        for turn in payload["earlier_related_turns"]
        for message in turn["messages"]
    }
    assert not retrieved_ids.intersection(window_ids)


def test_planning_service_passes_complete_retrieved_turns_to_the_planner(synthetic_study: Path):
    config = load_study_config(synthetic_study)
    skill = load_skill(Path("app/research/skills/price-exogenous-eda/SKILL.md"), source="builtin")
    gateway = CaptureGateway()
    service = EDAPlanningService(
        EDASubagent(model_planner=ModelEDAPlanner(gateway=gateway)),
    )
    conversation = [
        *_turn(1, "wind 的 max_lag 不超过 24", "已记录这项历史偏好。"),
        *[
            message
            for index in range(2, 7)
            for message in _turn(index, f"第 {index} 轮普通天气记录", "已记录天气信息。")
        ],
        _message(
            13,
            "继续按 max_lag 分析 wind 的滞后关系",
            turn_id="turn-current",
        ),
    ]

    with pytest.raises(RuntimeError, match="captured EDAPlanDraft"):
        service.propose(
            question="继续按 max_lag 分析 wind 的滞后关系",
            study_config=config,
            conversation=conversation,
            skill=skill,
        )

    payload = _payload(gateway.calls[0])
    assert "继续按 max_lag 分析 wind 的滞后关系" not in {
        item["content"] for item in payload["conversation_history"]
    }
    assert [item["turn_id"] for item in payload["earlier_related_turns"]] == ["turn-1"]
    assert "max_lag 不超过 24" in payload["earlier_related_turns"][0]["messages"][0]["content"]
