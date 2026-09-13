"""Official MCP SDK client with allow-list enforcement and hashed audit records."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Self

from mcp import Client, StdioServerParameters
from pydantic import BaseModel, ConfigDict, Field

from app.integrations.mcp.config import MCPServerConfig
from app.runtime_paths import application_data_directory


class MCPClientError(RuntimeError):
    """An MCP transport, discovery, policy, or remote tool call failed."""


class MCPToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    title: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    structured_content: Any = None
    text: tuple[str, ...] = ()
    raw_content: tuple[dict[str, Any], ...] = ()


class MCPAuditStore:
    """Append metadata and hashes without storing credentials or page bodies."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or application_data_directory() / "mcp" / "audit.jsonl").resolve()

    def append(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class MCPConnection:
    """One negotiated MCP session; browser calls sharing state use this context."""

    def __init__(self, config: MCPServerConfig, audit: MCPAuditStore) -> None:
        self.config = config
        self.audit = audit
        self._client: Client | None = None

    async def __aenter__(self) -> Self:
        if not self.config.enabled:
            raise MCPClientError(f"MCP {self.config.server_id} 未启用")
        server: StdioServerParameters | str
        if self.config.transport == "stdio":
            server = StdioServerParameters(
                command=self.config.resolved_command(),
                args=list(self.config.args),
                env=self.config.resolved_environment(),
            )
        else:
            server = str(self.config.url)
        self._client = Client(server, read_timeout_seconds=self.config.read_timeout_seconds)
        try:
            await self._client.__aenter__()
        except Exception as exc:
            self._client = None
            raise MCPClientError(f"MCP {self.config.server_id} 连接失败：{exc}") from exc
        self.audit.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "event": "connected",
                "server_id": self.config.server_id,
                "transport": self.config.transport,
                "protocol_version": str(self._client.protocol_version),
            }
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.__aexit__(exc_type, exc, traceback)

    def _require_client(self) -> Client:
        if self._client is None:
            raise MCPClientError("MCP连接尚未打开")
        return self._client

    def _authorize(self, name: str) -> None:
        if name not in self.config.allowed_tools:
            raise MCPClientError(f"MCP工具未获准：{self.config.server_id}.{name}")

    async def list_tools(self) -> tuple[MCPToolDefinition, ...]:
        client = self._require_client()
        tools: list[MCPToolDefinition] = []
        cursor: str | None = None
        while True:
            result = await client.list_tools(cursor=cursor)
            tools.extend(
                MCPToolDefinition(
                    name=item.name,
                    title=item.title,
                    description=item.description,
                    input_schema=item.input_schema,
                )
                for item in result.tools
                if item.name in self.config.allowed_tools
            )
            cursor = result.next_cursor
            if not cursor:
                break
        return tuple(tools)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        self._authorize(name)
        client = self._require_client()
        started = perf_counter()
        try:
            result = await client.call_tool(
                name,
                arguments or {},
                read_timeout_seconds=self.config.read_timeout_seconds,
            )
        except Exception as exc:
            raise MCPClientError(f"MCP工具 {name} 调用失败：{exc}") from exc
        raw = tuple(item.model_dump(mode="json", by_alias=True) for item in result.content)
        text = tuple(str(item.get("text")) for item in raw if item.get("type") == "text")
        audit = {
            "at": datetime.now(UTC).isoformat(),
            "event": "tool_call",
            "server_id": self.config.server_id,
            "tool": name,
            "arguments_sha256": _hash_json(arguments or {}),
            "result_sha256": _hash_json({"content": raw, "structured": result.structured_content}),
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "is_error": bool(result.is_error),
        }
        self.audit.append(audit)
        if result.is_error:
            detail = "\n".join(text).strip() or "remote tool returned isError"
            raise MCPClientError(f"MCP工具 {name} 返回错误：{detail}")
        return MCPToolResult(
            tool_name=name,
            structured_content=result.structured_content,
            text=text,
            raw_content=raw,
        )


class MCPClient:
    """Create connected sessions for one validated server configuration."""

    def __init__(self, config: MCPServerConfig, *, audit_path: str | Path | None = None) -> None:
        self.config = config
        self.audit = MCPAuditStore(audit_path)

    def connect(self) -> MCPConnection:
        return MCPConnection(self.config, self.audit)
