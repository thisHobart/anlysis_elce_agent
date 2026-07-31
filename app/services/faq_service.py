from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class FAQMatch:
    answer: str
    confidence: float


class FAQService:
    """Simple deterministic FAQ matcher; replace the in-memory list with your FAQ source."""

    entries: ClassVar[dict[str, str]] = {
        "什么是虚拟电厂": "虚拟电厂（VPP）通过平台聚合分布式电源、储能和可调负荷，并协调参与电力市场。",
        "如何接入": "请准备站点、设备和通信资料，并由平台管理员完成接入配置与验收。",
        "调度规则": "调度以授权范围、设备可用状态和市场/电网指令为准。",
    }

    def match(self, question: str) -> FAQMatch | None:
        normalized = "".join(question.lower().split())
        best: tuple[str, str] | None = None
        for key, answer in self.entries.items():
            compact_key = "".join(key.lower().split())
            if (compact_key in normalized or normalized in compact_key) and (best is None or len(compact_key) > len(best[0])):
                best = (compact_key, answer)
        return FAQMatch(answer=best[1], confidence=1.0) if best else None
