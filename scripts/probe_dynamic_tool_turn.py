"""Live acceptance probe for optional native tool turns and prompt_json compatibility.

The probe never executes the proposed function.  It sends a deterministic fake
tool result back with the same provider call ID, then verifies that a second
assistant turn can be decoded through the same gateway.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings, get_settings
from app.llm.budget import ModelRequestPurpose
from app.llm.factory import build_model_gateway
from app.llm.gateway import ModelMessage

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "price_descriptive_distribution",
            "description": "协议探测函数。仅在用户要求验证工具协议时调用；本脚本不会实际执行。",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
]


CASES = ("openai-native", "gemini-native", "prompt-json")


def _turn_payload(turn) -> dict:
    return {
        "content_present": bool(turn.content.strip()),
        "content_length": len(turn.content),
        "finish_reason": turn.finish_reason,
        "tool_calls": [
            {"name": call.name, "call_id": call.call_id, "argument_keys": sorted(call.arguments)}
            for call in turn.tool_calls
        ],
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _validate_case(case: str, settings: Settings) -> None:
    if case == "gemini-native" and settings.llm_provider != "gemini":
        raise ValueError("gemini-native 要求 VPP_LLM_PROVIDER=gemini")
    if case == "openai-native" and (
        settings.llm_provider == "gemini" or settings.llm_structured_output_method != "function_calling"
    ):
        raise ValueError("openai-native 要求 OpenAI-compatible provider 与 function_calling 模式")
    if case == "prompt-json" and (
        settings.llm_provider == "gemini" or settings.llm_structured_output_method != "prompt_json"
    ):
        raise ValueError("prompt-json 要求 OpenAI-compatible provider 与 prompt_json 模式")


def _write_report(path: Path | None, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if path is None:
        print(encoded)
        return
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + "\n", encoding="utf-8")
    print(f"验收报告已写入：{path}")


def main() -> int:
    args = _arguments()
    settings = Settings(_env_file=args.env_file) if args.env_file else get_settings()
    report = {
        "schema_version": 1,
        "case": args.case,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "structured_output_method": settings.llm_structured_output_method,
        "executed_at": datetime.now(UTC).isoformat(),
        "status": "failed",
        "turns": [],
    }
    try:
        _validate_case(args.case, settings)
    except ValueError as exc:
        report["error"] = str(exc)
        _write_report(args.output, report)
        return 2
    gateway = build_model_gateway(settings)
    if not gateway.enabled:
        report["error"] = "模型未配置完整，无法执行真实工具协议验收。"
        _write_report(args.output, report)
        return 2
    messages = [
        ModelMessage(
            role="system",
            content="你正在进行工具协议验收。先调用提供的唯一函数；收到结果后用一句话结束。",
        ),
        ModelMessage(role="user", content="请验证一次电价分析工具调用协议。"),
    ]
    try:
        first = gateway.invoke_tool_turn(
            messages=messages,
            tools=TOOLS,
            purpose=ModelRequestPurpose.EDA_ANALYSIS,
        )
        report["turns"].append(_turn_payload(first))
        if not first.tool_calls:
            raise ValueError("第一回合未返回工具调用")
        call_ids = [str(call.call_id or "") for call in first.tool_calls]
        if any(not call_id for call_id in call_ids) or len(call_ids) != len(set(call_ids)):
            raise ValueError("第一回合 provider call ID 缺失或重复")
        messages.append(
            ModelMessage(
                role="assistant",
                content=first.content,
                tool_calls=first.tool_calls,
            )
        )
        messages.extend(
            ModelMessage(
                role="tool",
                tool_call_id=str(call.call_id),
                content=json.dumps(
                    {
                        "status": "completed",
                        "protocol_probe": True,
                        "function": call.name,
                    },
                    ensure_ascii=False,
                ),
            )
            for call in first.tool_calls
        )
        second = gateway.invoke_tool_turn(
            messages=messages,
            tools=TOOLS,
            purpose=ModelRequestPurpose.EDA_ANALYSIS,
        )
        report["turns"].append(_turn_payload(second))
        if second.tool_calls:
            raise ValueError("第二回合未结束，模型仍返回了工具调用")
        if not second.content.strip():
            raise ValueError("第二回合缺少完成正文")
    except Exception as exc:  # noqa: BLE001 - command-line diagnostics
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(args.output, report)
        return 1
    report["status"] = "passed"
    _write_report(args.output, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
