"""Collect approved news URLs through browser MCP and freeze a P2 JSONL snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.integrations.mcp import (
    BrowserMCPNewsCollector,
    BrowserNewsTarget,
    MCPClient,
    freeze_collected_news,
    load_mcp_server_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True, help="JSON数组，每项包含url、query_source等字段")
    parser.add_argument("--output", type=Path, required=True, help="要冻结的P2 JSONL路径")
    parser.add_argument("--server", default="browser")
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    servers = load_mcp_server_catalog(args.catalog)
    if args.server not in servers:
        parser.error(f"MCP服务器不存在：{args.server}")
    payload = json.loads(args.targets.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        parser.error("--targets 必须是JSON数组")
    targets = [BrowserNewsTarget.model_validate(item) for item in payload]
    collector = BrowserMCPNewsCollector(MCPClient(servers[args.server]))
    records = collector.collect(targets)
    snapshot, manifest = freeze_collected_news(records, args.output, server_id=args.server)
    print(
        json.dumps(
            {"status": "frozen", "records": len(records), "snapshot": str(snapshot), "manifest": str(manifest)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
