"""Validated non-secret configuration for optional MCP servers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import runtime_env_file
from app.runtime_paths import source_worktree

MCP_CATALOG_FILENAME = "mcp_servers.yaml"
MCP_CATALOG_ENVIRONMENT_VARIABLE = "PRICE_RESEARCH_MCP_SERVERS"


class MCPConfigError(RuntimeError):
    """MCP configuration is missing, unsafe, or cannot start its transport."""


class MCPServerConfig(BaseModel):
    """One MCP connection without embedded credential values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    server_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    label: str = Field(min_length=1, max_length=128)
    enabled: bool = False
    transport: Literal["stdio", "streamable_http"]
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    environment_from: dict[str, str] = Field(default_factory=dict)
    read_timeout_seconds: float = Field(default=60, gt=0, le=600)
    allowed_tools: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        if self.transport == "stdio":
            if not self.command or self.url is not None:
                raise ValueError("stdio MCP requires command and forbids url")
        elif not self.url or self.command is not None or self.args:
            raise ValueError("streamable_http MCP requires url and forbids command/args")
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("allowed_tools contains duplicates")
        return self

    def resolved_command(self) -> str:
        if self.transport != "stdio" or not self.command:
            raise MCPConfigError(f"MCP {self.server_id} is not a stdio server")
        resolved = shutil.which(self.command)
        if not resolved:
            raise MCPConfigError(f"找不到MCP启动命令：{self.command}")
        return resolved

    def resolved_environment(self) -> dict[str, str] | None:
        if not self.environment_from:
            return None
        file_values = dotenv_values(runtime_env_file()) if runtime_env_file().is_file() else {}
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for child_name, source_name in self.environment_from.items():
            value = os.environ.get(source_name)
            if value is None:
                value = file_values.get(source_name)
            if value is None or not str(value).strip():
                missing.append(source_name)
            else:
                resolved[child_name] = str(value)
        if missing:
            raise MCPConfigError(f"MCP {self.server_id} 缺少环境变量：{'、'.join(sorted(missing))}")
        return resolved


def mcp_catalog_path(path: str | Path | None = None) -> Path | None:
    if path is not None:
        candidate = Path(path).expanduser().resolve()
        return candidate if candidate.is_file() else None
    env_path = runtime_env_file()
    configured = os.environ.get(MCP_CATALOG_ENVIRONMENT_VARIABLE)
    if configured is None and env_path.is_file():
        configured = dotenv_values(env_path).get(MCP_CATALOG_ENVIRONMENT_VARIABLE)
    if configured:
        candidate = Path(str(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = env_path.parent / candidate
        return candidate.resolve() if candidate.is_file() else None
    roots = [Path(__file__).resolve().parent]
    worktree = source_worktree()
    if worktree is not None:
        roots.append(worktree / "configs")
    for root in roots:
        candidate = root / MCP_CATALOG_FILENAME
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_mcp_server_catalog(path: str | Path | None = None) -> dict[str, MCPServerConfig]:
    source = mcp_catalog_path(path)
    if source is None:
        raise MCPConfigError("找不到MCP服务器配置")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MCPConfigError(f"MCP服务器配置无法读取：{exc}") from exc
    rows = payload.get("servers") if isinstance(payload, dict) and payload.get("version") == 1 else None
    if not isinstance(rows, list):
        raise MCPConfigError("MCP服务器配置必须包含 version: 1 和 servers 列表")
    servers: dict[str, MCPServerConfig] = {}
    try:
        for row in rows:
            server = MCPServerConfig.model_validate(row)
            if server.server_id in servers:
                raise ValueError(f"duplicate server_id: {server.server_id}")
            servers[server.server_id] = server
    except (TypeError, ValueError) as exc:
        raise MCPConfigError(f"MCP服务器配置无效：{exc}") from exc
    return servers
