"""MCP transports and the bounded browser adapter used by P2."""

from app.integrations.mcp.browser_news import (
    BrowserMCPNewsCollector,
    BrowserNewsCollectionError,
    BrowserNewsTarget,
    freeze_collected_news,
)
from app.integrations.mcp.client import MCPClient, MCPClientError, MCPToolDefinition, MCPToolResult
from app.integrations.mcp.config import (
    MCPConfigError,
    MCPServerConfig,
    load_mcp_server_catalog,
    mcp_catalog_path,
)

__all__ = [
    "BrowserMCPNewsCollector",
    "BrowserNewsCollectionError",
    "BrowserNewsTarget",
    "MCPClient",
    "MCPClientError",
    "MCPConfigError",
    "MCPServerConfig",
    "MCPToolDefinition",
    "MCPToolResult",
    "freeze_collected_news",
    "load_mcp_server_catalog",
    "mcp_catalog_path",
]
