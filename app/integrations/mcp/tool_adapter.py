"""Bridge the P2 browser collector into the existing approved ToolSpec boundary."""

from __future__ import annotations

from datetime import datetime

from app.integrations.mcp.browser_news import BrowserMCPNewsCollector, BrowserNewsTarget
from app.research.tools.contracts import ToolArguments, ToolContext, ToolOutput, ToolSpec


class BrowserCollectNewsArguments(ToolArguments):
    url: str
    source_name: str | None = None
    published_at: datetime | None = None
    query_source: str
    market_tags: tuple[str, ...] = ("CN-SHANDONG",)


def browser_news_tool_spec(collector: BrowserMCPNewsCollector) -> ToolSpec:
    """Expose one high-level read-only call rather than arbitrary browser mutation tools."""

    def collect(_context: ToolContext, arguments: ToolArguments) -> ToolOutput:
        parsed = BrowserCollectNewsArguments.model_validate(arguments)
        record = collector.collect(
            [
                BrowserNewsTarget(
                    url=parsed.url,
                    source_name=parsed.source_name,
                    published_at=parsed.published_at,
                    query_source=parsed.query_source,
                    market_tags=parsed.market_tags,
                )
            ]
        )[0]
        return ToolOutput(result_key="news_collection", value={"record": record.model_dump(mode="json")})

    return ToolSpec(
        name="browser_collect_news_page",
        version="1.0.0",
        description="通过获准的浏览器MCP读取一个新闻网页并返回可冻结的P2新闻记录。",
        arguments_model=BrowserCollectNewsArguments,
        handler=collect,
        provider=f"mcp:{collector.client.config.server_id}",
        display_name="浏览器采集新闻正文",
        result_key="news_collection",
    )
