"""Read-only Playwright MCP page collection and P2 JSONL freezing."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.integrations.mcp.client import MCPClient, MCPConnection, MCPToolResult
from app.research.news.contracts import CollectedNewsRecord

PAGE_EXTRACTION_FUNCTION = r"""() => {
  const first = (...values) => values.find(value => typeof value === 'string' && value.trim())?.trim() || null;
  const meta = name => document.querySelector(`meta[property="${name}"],meta[name="${name}"]`)?.content || null;
  let jsonDate = null;
  for (const node of document.querySelectorAll('script[type="application/ld+json"]')) {
    try {
      const parsed = JSON.parse(node.textContent || 'null');
      const queue = Array.isArray(parsed) ? parsed : [parsed];
      for (const item of queue) {
        if (item && typeof item === 'object') {
          jsonDate = first(item.datePublished, item.dateCreated, jsonDate);
        }
      }
    } catch (_) {}
  }
  return {
    url: location.href,
    title: first(meta('og:title'), document.querySelector('h1')?.innerText, document.title),
    body: first(document.querySelector('article')?.innerText, document.querySelector('main')?.innerText, document.body?.innerText),
    publishedAt: first(meta('article:published_time'), meta('datePublished'), meta('pubdate'),
      document.querySelector('time[datetime]')?.getAttribute('datetime'), jsonDate),
    sourceName: first(meta('og:site_name'), location.hostname)
  };
}"""


class BrowserNewsCollectionError(RuntimeError):
    """A page could not be collected without inventing source metadata."""


class BrowserNewsTarget(BaseModel):
    """One URL approved for collection before P2 extraction starts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    url: str = Field(min_length=1, max_length=2048)
    source_name: str | None = Field(default=None, max_length=128)
    published_at: datetime | None = None
    query_source: str = Field(min_length=1, max_length=1024)
    market_tags: tuple[str, ...] = ("CN-SHANDONG",)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("browser news target must be an HTTP(S) URL")
        return value

    @field_validator("published_at")
    @classmethod
    def validate_published_at(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("published_at must include a timezone")
        return value


def _json_from_playwright_result(result: MCPToolResult) -> dict[str, Any]:
    if isinstance(result.structured_content, dict):
        return result.structured_content
    for block in result.text:
        marker = "### Result"
        if marker not in block:
            continue
        payload = block.split(marker, 1)[1].split("\n### ", 1)[0].strip()
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    raise BrowserNewsCollectionError("Playwright MCP没有返回可解析的页面JSON")


def _publication_time(value: object, *, fallback: datetime | None) -> datetime:
    if value is None or not str(value).strip():
        if fallback is None:
            raise BrowserNewsCollectionError("页面没有可核验发布时间；请在采集目标中提供published_at")
        return fallback
    raw = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise BrowserNewsCollectionError(f"页面发布时间无法解析：{value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return parsed


class BrowserMCPNewsCollector:
    """Collect approved URLs in one browser session using fixed read-only scripts."""

    REQUIRED_TOOLS = frozenset({"browser_navigate", "browser_evaluate"})

    def __init__(
        self,
        client: MCPClient,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.client = client
        self.clock = clock or (lambda: datetime.now(UTC))

    async def collect_async(self, targets: list[BrowserNewsTarget]) -> tuple[CollectedNewsRecord, ...]:
        if not targets:
            return ()
        if len(targets) > 20:
            raise BrowserNewsCollectionError("一次P2补充最多采集20篇正文")
        records: list[CollectedNewsRecord] = []
        async with self.client.connect() as connection:
            tools = {item.name for item in await connection.list_tools()}
            missing = self.REQUIRED_TOOLS.difference(tools)
            if missing:
                raise BrowserNewsCollectionError(f"浏览器MCP缺少工具：{'、'.join(sorted(missing))}")
            for target in targets:
                records.append(await self._collect_one(connection, target))
            if "browser_close" in tools:
                await connection.call_tool("browser_close")
        return tuple(records)

    async def _collect_one(
        self,
        connection: MCPConnection,
        target: BrowserNewsTarget,
    ) -> CollectedNewsRecord:
        await connection.call_tool("browser_navigate", {"url": target.url})
        result = await connection.call_tool(
            "browser_evaluate",
            {"function": PAGE_EXTRACTION_FUNCTION},
        )
        page = _json_from_playwright_result(result)
        final_url = str(page.get("url") or target.url).strip()
        title = str(page.get("title") or "").strip()
        body = str(page.get("body") or "").strip()
        if not title:
            raise BrowserNewsCollectionError(f"网页缺少标题：{final_url}")
        if len(body) < 100:
            raise BrowserNewsCollectionError(f"网页正文不足100字符：{final_url}")
        collected_at = self.clock().astimezone(UTC)
        published_at = _publication_time(page.get("publishedAt"), fallback=target.published_at)
        source_name = target.source_name or str(page.get("sourceName") or urlsplit(final_url).netloc)
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return CollectedNewsRecord(
            source_name=source_name,
            source_document_id=hashlib.sha256(final_url.encode("utf-8")).hexdigest()[:32],
            source_ref=final_url,
            version=1,
            title=title,
            body=body,
            published_at=published_at,
            collected_at=collected_at,
            language="zh-CN",
            market_tags=target.market_tags,
            metadata={
                "query_source": target.query_source,
                "content_scope": "full_text",
                "content_sha256": body_hash,
                "retrieved_at": collected_at.isoformat(),
                "collected_via": "mcp",
                "mcp_server_id": self.client.config.server_id,
                "requested_url": target.url,
            },
        )

    def collect(self, targets: list[BrowserNewsTarget]) -> tuple[CollectedNewsRecord, ...]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.collect_async(targets))
        raise BrowserNewsCollectionError("同步采集不能在运行中的asyncio事件循环内调用")


def freeze_collected_news(
    records: tuple[CollectedNewsRecord, ...],
    destination: str | Path,
    *,
    server_id: str,
) -> tuple[Path, Path]:
    """Atomically freeze a P2-compatible JSONL plus its provenance manifest."""

    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_file": path.name,
                "source_file_sha256": digest,
                "record_count": len(records),
                "mcp_server_id": server_id,
                "frozen_at": datetime.now(UTC).isoformat(),
                "sources": [
                    {
                        "source_ref": item.source_ref,
                        "title": item.title,
                        "published_at": item.published_at.isoformat(),
                        "collected_at": item.collected_at.isoformat(),
                        "content_sha256": item.metadata["content_sha256"],
                        "query_source": item.metadata["query_source"],
                    }
                    for item in records
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path, manifest_path
