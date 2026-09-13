"""MCP configuration and the read-only P2 browser collection boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.integrations.mcp.browser_news import (
    BrowserMCPNewsCollector,
    BrowserNewsCollectionError,
    BrowserNewsTarget,
    freeze_collected_news,
)
from app.integrations.mcp.client import MCPToolDefinition, MCPToolResult
from app.integrations.mcp.config import MCPConfigError, MCPServerConfig, load_mcp_server_catalog
from app.integrations.mcp.tool_adapter import browser_news_tool_spec
from app.research.news.adapters import JsonlCollectedNewsAdapter


class _Connection:
    def __init__(self, page: dict[str, object]) -> None:
        self.page = page
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def list_tools(self):
        return tuple(
            MCPToolDefinition(name=name, input_schema={"type": "object"})
            for name in ("browser_navigate", "browser_evaluate", "browser_close")
        )

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        if name == "browser_evaluate":
            return MCPToolResult(
                tool_name=name,
                text=(f"### Result\n{json.dumps(self.page)}\n### Ran Playwright code\n```js\nfixed\n```",),
            )
        return MCPToolResult(tool_name=name)


class _Client:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection
        self.config = SimpleNamespace(server_id="browser")

    def connect(self):
        return self.connection


def _page() -> dict[str, object]:
    return {
        "url": "https://news.example/article-1",
        "title": "山东电网测试新闻",
        "body": "山东电网发布测试新闻正文。" * 20,
        "publishedAt": "2026-09-12T08:00:00+08:00",
        "sourceName": "测试来源",
    }


def test_default_browser_mcp_catalog_is_enabled_and_read_only():
    browser = load_mcp_server_catalog()["browser"]

    assert browser.enabled
    assert browser.transport == "stdio"
    assert browser.command == "npx"
    assert {"browser_navigate", "browser_evaluate"}.issubset(browser.allowed_tools)
    assert "browser_run_code_unsafe" not in browser.allowed_tools
    assert "browser_file_upload" not in browser.allowed_tools


def test_missing_stdio_command_fails_closed(monkeypatch: pytest.MonkeyPatch):
    browser = load_mcp_server_catalog()["browser"]
    monkeypatch.setattr("app.integrations.mcp.config.shutil.which", lambda _command: None)

    with pytest.raises(MCPConfigError, match="npx"):
        browser.resolved_command()


def test_stdio_config_does_not_embed_environment_secrets():
    server = MCPServerConfig(
        server_id="test",
        label="test",
        enabled=True,
        transport="stdio",
        command="python",
        environment_from={"TOKEN": "VPP_TEST_MCP_TOKEN"},
    )
    assert server.model_dump()["environment_from"] == {"TOKEN": "VPP_TEST_MCP_TOKEN"}


def test_browser_collector_freezes_auditable_p2_records(tmp_path: Path):
    connection = _Connection(_page())
    collector = BrowserMCPNewsCollector(
        _Client(connection),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 9, 13, 1, tzinfo=UTC),
    )
    records = collector.collect(
        [BrowserNewsTarget(url="https://news.example/article-1", query_source="山东 电网 新闻")]
    )
    snapshot, manifest = freeze_collected_news(records, tmp_path / "news.jsonl", server_id="browser")
    loaded = JsonlCollectedNewsAdapter(snapshot).load()
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert len(loaded) == 1
    assert loaded[0].source_ref == "https://news.example/article-1"
    assert loaded[0].metadata["collected_via"] == "mcp"
    assert loaded[0].metadata["content_sha256"] == hashlib.sha256(loaded[0].body.encode()).hexdigest()
    assert payload["source_file_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert [name for name, _ in connection.calls] == ["browser_navigate", "browser_evaluate", "browser_close"]


def test_browser_collection_requires_real_publication_time():
    page = _page() | {"publishedAt": None}
    collector = BrowserMCPNewsCollector(
        _Client(_Connection(page)),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 9, 13, 1, tzinfo=UTC),
    )

    with pytest.raises(BrowserNewsCollectionError, match="发布时间"):
        collector.collect([BrowserNewsTarget(url="https://news.example/article-1", query_source="test")])


def test_browser_tool_spec_uses_mcp_provider_without_exposing_raw_browser_tools():
    collector = BrowserMCPNewsCollector(_Client(_Connection(_page())))  # type: ignore[arg-type]
    spec = browser_news_tool_spec(collector)

    assert spec.name == "browser_collect_news_page"
    assert spec.provider == "mcp:browser"
    assert spec.function_schema()["function"]["parameters"]["additionalProperties"] is False
