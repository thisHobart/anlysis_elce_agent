"""Connect to the configured browser MCP and verify its read-only P2 tool set."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.integrations.mcp import MCPClient, load_mcp_server_catalog


async def _validate(server_id: str, catalog: Path | None) -> dict[str, object]:
    servers = load_mcp_server_catalog(catalog)
    if server_id not in servers:
        available = "、".join(sorted(servers)) or "无"
        raise ValueError(f"MCP服务器不存在：{server_id}；可用服务器：{available}")
    config = servers[server_id]
    client = MCPClient(config)
    async with client.connect() as connection:
        tools = await connection.list_tools()
    names = sorted(item.name for item in tools)
    required = {"browser_navigate", "browser_evaluate"}
    missing = sorted(required.difference(names))
    return {
        "status": "passed" if not missing else "failed",
        "server_id": server_id,
        "transport": config.transport,
        "tools": names,
        "missing_required_tools": missing,
        "unsafe_tool_exposed": any(name in names for name in ("browser_run_code_unsafe", "browser_file_upload")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="browser")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(_validate(args.server, args.catalog))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" and not result["unsafe_tool_exposed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
