# P2 浏览器 MCP 接入

## 当前状态

项目已接入官方 Python MCP SDK v2，并配置微软 Playwright MCP 作为可选的 P2 新闻采集器。MCP 只负责读取已经批准的网页并冻结来源；P2 继续从本地不可变 JSONL 执行正文抽取、版本管理、复核、事件分析和特征导出。

默认配置在 `configs/mcp_servers.yaml`，当前固定为 `@playwright/mcp@0.0.80`，避免每次启动自动切换实现版本。运行时只允许：

- `browser_navigate`
- `browser_evaluate`
- `browser_snapshot`
- `browser_wait_for`
- `browser_tabs`
- `browser_close`

项目不会把 `browser_run_code_unsafe`、文件上传、表单填写或点击工具注册给研究 Agent。对页面的 `browser_evaluate` 只由采集适配器使用固定脚本调用。

## 环境要求

- Node.js 18或更高版本；
- `npx` 可以从进程 `PATH` 找到；
- 安装项目 Python 依赖后包含 `mcp>=2,<3`；
- 若使用自定义 MCP 配置，把 `PRICE_RESEARCH_MCP_SERVERS` 指向对应 YAML。

当前默认使用无界面的隔离浏览器，不复用登录态。需要登录的网站应改成持久用户目录、Playwright 扩展或独立 HTTP MCP 服务，并在新方案确认后再启用。

## 连接验证

```powershell
.\venv\Scripts\python.exe -m scripts.validate_browser_mcp `
  --output artifacts/acceptance/browser-mcp/discovery.json
```

验证会启动 MCP、协商协议并检查获准工具，不会访问新闻网页。

## 冻结新闻

先准备目标文件，例如 `targets.json`：

```json
[
  {
    "url": "https://example.org/news/1",
    "source_name": "来源名称",
    "published_at": "2026-09-12T08:00:00+08:00",
    "query_source": "山东 机组检修",
    "market_tags": ["CN-SHANDONG"]
  }
]
```

如果网页元数据包含可解析的发布时间，可以省略 `published_at`。网页和目标文件都没有发布时间时采集会失败，不会用取得时间冒充发布时间。一次最多20篇。

```powershell
.\venv\Scripts\python.exe -m scripts.collect_p2_news_mcp `
  --targets targets.json `
  --output data/news/p2-browser-news.jsonl
```

输出包括 JSONL 和相邻 manifest。每条记录保存最终 URL、查询来源、实际取得时间、正文哈希和 MCP server ID，可以直接传给现有 P2 工作台。

## 完整流程入口

```powershell
.\venv\Scripts\python.exe -m scripts.validate_p1_p2_p3 `
  --mcp-targets targets.json
```

该入口先完成浏览器采集和冻结，再把冻结文件交给 P2。未传 `--mcp-targets` 时仍使用 `--news` 指定的本地新闻，不会启动 MCP。

MCP 调用审计默认写入应用数据目录下的 `mcp/audit.jsonl`。审计只保存参数和结果哈希、工具名、耗时及错误状态，不保存凭据或网页正文。
