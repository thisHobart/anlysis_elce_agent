"""FAQ 匹配服务。

加载 ``app.knowledge.faqs`` 的结构化 FAQ，按触发词做确定性匹配。

这是 P2 阶段的**确定性**匹配器，作为 P3 引入 LLM 意图识别之前的过渡实现：
基于触发词的双向子串包含打分。P3 落地后，classify 节点可用 LLM 产出候选，
再用本服务的 ``get`` 取回条目元数据（模板 / 数据需求 / 大屏联动）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.knowledge.faqs import FAQS


@dataclass(frozen=True)
class FaqMatch:
    """一次 FAQ 匹配结果，携带回答所需的全部元数据。"""

    faq_id: str
    category: str
    template: str
    data_requirements: list[dict[str, Any]]
    screen_action: str
    score: int
    confidence: float
    matched_triggers: tuple[str, ...] = field(default_factory=tuple)


def _normalize(text: str) -> str:
    """去空白 + casefold（统一英文大小写，如 VPP/SOC），中文不受影响。"""
    return "".join(text.split()).casefold()


def _trigger_score(question_norm: str, trigger_norm: str) -> int:
    """单个触发词对问题的匹配得分。

    - 正向（触发词 ⊆ 问题）：强信号，权重 = 触发词长度 × 2
    - 反向（问题 ⊆ 触发词，且问题 ≥ 3 字）：弱信号，权重 = 问题长度
      —— 覆盖"用户只打了个短关键词"的情况（如"发电功率"）
    """
    if not trigger_norm:
        return 0
    if trigger_norm in question_norm:
        return len(trigger_norm) * 2
    if len(question_norm) >= 3 and question_norm in trigger_norm:
        return len(question_norm)
    return 0


class FAQService:
    """结构化 FAQ 的加载与匹配。"""

    def __init__(self, faqs: list[dict[str, Any]] | None = None) -> None:
        self._faqs: list[dict[str, Any]] = faqs if faqs is not None else FAQS
        self._by_id: dict[str, dict[str, Any]] = {f["faq_id"]: f for f in self._faqs}

    def all(self) -> list[dict[str, Any]]:
        return list(self._faqs)

    def get(self, faq_id: str) -> dict[str, Any] | None:
        return self._by_id.get(faq_id)

    def classification_catalog(self) -> list[dict[str, Any]]:
        """Return the compact, non-sensitive FAQ fields needed by the classifier."""
        return [
            {
                "faq_id": faq["faq_id"],
                "category": faq["category"],
                "triggers": list(faq.get("triggers", [])),
            }
            for faq in self._faqs
        ]

    def match(self, question: str) -> FaqMatch | None:
        """返回得分最高的 FAQ；无任何触发词命中时返回 None（走知识/兜底路由）。

        打分：每个 FAQ 取其触发词的**最高**单项得分作为 FAQ 得分。
        并列时优先"命中触发词更长"，再按 FAQ 出现顺序（编号靠前）稳定选择。
        """
        if not question or not question.strip():
            return None
        q_norm = _normalize(question)

        best: dict[str, Any] | None = None
        best_score = 0
        best_matched: list[str] = []
        best_max_trigger_len = 0

        for faq in self._faqs:
            score = 0
            matched: list[str] = []
            max_trigger_len = 0
            for trigger in faq.get("triggers", []):
                t_norm = _normalize(trigger)
                s = _trigger_score(q_norm, t_norm)
                if s > 0:
                    matched.append(trigger)
                    score = max(score, s)
                    max_trigger_len = max(max_trigger_len, len(t_norm))
            if score == 0:
                continue
            # 并列打破：更高分 > 命中触发词更长 > 保持出现顺序（不覆盖已选）
            if score > best_score or (score == best_score and max_trigger_len > best_max_trigger_len):
                best, best_score, best_matched, best_max_trigger_len = faq, score, matched, max_trigger_len

        if best is None:
            return None

        confidence = min(1.0, best_score / max(len(q_norm), 1))
        return FaqMatch(
            faq_id=best["faq_id"],
            category=best["category"],
            template=best["template"],
            data_requirements=best.get("data_requirements", []),
            screen_action=best.get("screen_action", ""),
            score=best_score,
            confidence=round(confidence, 3),
            matched_triggers=tuple(best_matched),
        )
