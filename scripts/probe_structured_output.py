"""Verify that the configured provider actually enforces a supplied response schema.

The old probe used fixed field names that a model could guess from the prompt. This probe
creates a new schema with unpredictable field names on every attempt. Native modes verify
provider-side schema transport; ``prompt_json`` verifies prompt adherence plus local validation.

Usage (configuration is read from .env):

    python scripts/probe_structured_output.py
    python scripts/probe_structured_output.py --attempts 5 --show-raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, create_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import get_settings
from app.llm.factory import build_model_gateway
from app.llm.gateway import ModelGatewayError, ModelMessage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证模型端点是否真正执行原生结构化输出 schema。"
    )
    parser.add_argument("--attempts", type=int, default=3, choices=range(1, 11))
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="显示每次调用的模型原始响应与本地解析错误。",
    )
    return parser


def _probe_contract() -> tuple[type[BaseModel], dict[str, object]]:
    suffix = uuid4().hex[:8]
    record_field = f"record_{suffix}"
    code_field = f"grid_code_{suffix}"
    direction_field = f"polarity_{suffix}"
    interval_field = f"minutes_{suffix}"

    record_type = create_model(
        f"ProbeRecord_{suffix}",
        __config__=ConfigDict(extra="forbid"),
        **{
            code_field: (str, ...),
            direction_field: (str, ...),
            interval_field: (int, ...),
        },
    )
    envelope_type = create_model(
        f"ProbeEnvelope_{suffix}",
        __config__=ConfigDict(extra="forbid"),
        **{record_field: (record_type, ...)},
    )
    expected = {
        record_field: {
            code_field: "NEM_SCHEMA_7",
            direction_field: "UP_ONLY",
            interval_field: 5,
        }
    }
    return envelope_type, expected


def _messages() -> list[ModelMessage]:
    return [
        ModelMessage(
            role="system",
            content=(
                "这是结构化输出协议测试。只使用 API 随请求提供的 response schema 决定"
                "字段名称和层级，不自行设计 JSON 格式。"
            ),
        ),
        ModelMessage(
            role="user",
            content=(
                "请把以下三项事实填入服务端提供的 schema：代码 NEM_SCHEMA_7；"
                "方向 UP_ONLY；间隔 5 分钟。"
            ),
        ),
    ]


def _print_trace(trace: dict[str, object] | None) -> None:
    if trace is None:
        print("    原始响应：调用在 observer 之前失败，未取得 raw")
        return
    raw_content = trace.get("raw_content")
    calls = trace.get("tool_calls") or []
    if raw_content:
        print("    原始响应文本：")
        for line in str(raw_content).splitlines():
            print(f"      {line}")
    elif calls:
        print("    原始函数参数：")
        print(json.dumps(calls, ensure_ascii=False, indent=6, default=str))
    else:
        print("    原始响应：无文本且无函数参数")
    if error := trace.get("parsing_error"):
        print(f"    LangChain 解析错误：{error}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()

    settings = get_settings()
    traces: list[dict[str, object]] = []
    gateway = build_model_gateway(
        settings,
        structured_output_observer=traces.append,
    )
    if not gateway.enabled:
        print("模型未配置完整。Gemini 需要模型名和 Google 认证；其他接口还需要 Base URL。")
        return 2

    method = (
        "json_schema（Gemini 原生 API）"
        if settings.llm_provider == "gemini"
        else (
            "prompt_json（Cherry 提示约束 + 本地校验）"
            if settings.llm_structured_output_method == "prompt_json"
            else f"{settings.llm_structured_output_method}（OpenAI 兼容 API）"
        )
    )
    print("结构化输出协议验证")
    print(f"  provider：{settings.llm_provider}")
    print(f"  model：   {settings.llm_model}")
    print(f"  method：  {method}")
    print(f"  attempts：{args.attempts}")

    failures = 0
    for index in range(1, args.attempts + 1):
        schema, expected = _probe_contract()
        before = len(traces)
        try:
            result = gateway.invoke_structured(messages=_messages(), schema=schema)
            actual = result.model_dump(mode="json")
            if actual != expected:
                failures += 1
                print(f"[失败 {index}/{args.attempts}] schema 可解析，但字段值不正确")
                print(f"    预期：{json.dumps(expected, ensure_ascii=False)}")
                print(f"    实际：{json.dumps(actual, ensure_ascii=False)}")
            else:
                print(f"[通过 {index}/{args.attempts}] 随机 schema 和字段值均正确")
        except ModelGatewayError as exc:
            failures += 1
            print(f"[失败 {index}/{args.attempts}] {type(exc).__name__}: {exc}")
        trace = traces[-1] if len(traces) > before else None
        if args.show_raw or trace and trace.get("parsing_error"):
            _print_trace(trace)

    print()
    if failures:
        enforcement = (
            "本地 schema 校验"
            if settings.llm_structured_output_method == "prompt_json"
            else "服务端 schema 执行"
        )
        print(
            f"结论：未通过（{failures}/{args.attempts} 次失败）。"
            f"当前链路未能稳定通过{enforcement}，不应运行 P2 模型抽取基准。"
        )
        if (
            settings.llm_provider != "gemini"
            and settings.llm_structured_output_method != "prompt_json"
        ):
            print(
                "若必须通过 Cherry 调用 Gemini，请设置 "
                "VPP_LLM_STRUCTURED_OUTPUT_METHOD=prompt_json。"
            )
        return 1

    if settings.llm_structured_output_method == "prompt_json":
        print(
            f"结论：通过（{args.attempts}/{args.attempts}）。"
            "模型在提示中读取 schema，结果连续通过本地严格校验；这不表示 Cherry 提供了"
            "服务端原生 schema 约束。"
        )
    else:
        print(
            f"结论：通过（{args.attempts}/{args.attempts}）。"
            "随机字段名未出现在业务提示词中，当前原生链路能稳定返回所提供的 schema。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
