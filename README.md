# VPP LangGraph Agent

面向虚拟电厂（VPP）的 FastAPI + LangGraph 服务。当前已完成 P2 FAQ 结构化与确定性匹配，以及 P3 可配置 LLM、多轮会话、结构化意图分类和回答组织。模型默认关闭；未配置本地模型时自动回退到 P2 匹配与模板渲染。

## 目录

```text
app/
  graph/nodes/       # 回合准备、分类、校验、查询、回答和联动节点
  prompts/           # LLM 分类与回答系统提示词
  services/          # LLM、FAQ、VPP 工具适配器和回答模板
  knowledge/         # 结构化 FAQ 知识库
  security/policy.py # 工具、SQL、参数和大屏动作白名单
  main.py            # FastAPI 入口
tests/               # 工作流、策略与 FAQ 测试
```

## 快速开始

需要 Python 3.11+。

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -e ".[dev]"
copy .env.example .env
uvicorn app.main:app --reload
```

访问 `http://127.0.0.1:8000/docs` 查看 OpenAPI 文档。

## 调用示例

```bash
curl -X POST http://127.0.0.1:8000/v1/query ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo-001\",\"question\":\"什么是虚拟电厂\",\"role\":\"viewer\"}"
```

后续请求复用同一个 `session_id` 即可读取当前进程内的多轮消息和实体上下文。当前使用 `InMemorySaver`，服务重启后会话会丢失。

## 配置本地模型

P3 假设本地模型提供 OpenAI 兼容接口。复制 `.env.example` 后填写：

```dotenv
VPP_LLM_ENABLED=true
VPP_LLM_BASE_URL=http://127.0.0.1:xxxx/v1
VPP_LLM_API_KEY=local-placeholder
VPP_LLM_MODEL=your-model-name
VPP_LLM_STRUCTURED_MODE=json_prompt
```

`json_prompt` 对本地模型兼容性较好；模型原生支持结构化输出时可改为 `native`。LLM 分类或回答失败时，工作流会自动使用 P2 规则或 FAQ 模板。

当前路由包括：`faq`、`knowledge`、`data`、`direct`、`screen_action`、`chitchat` 和 `out_of_scope`。其中真实 API、数据库、RAG 和大屏执行尚未接入，相关路由只返回明确的占位或降级回答，不会编造数据。

## 接入现有系统

- 在 `app/services/vpp_tools.py` 或后续独立 service 中接入真实 VPP API/DB 工具。
- 在 `app/services/faq_service.py` 连接 FAQ 存储或搜索服务。
- 在 `app/graph/nodes/query_data.py` 的 `knowledge` 分支接入现有 RAG 检索器。
- 根据真实 RBAC 策略维护 `app/security/policy.py` 的工具与大屏动作白名单。

## 验证

```bash
pytest
ruff check .
```

默认测试使用 Fake LLM，不依赖网络、模型服务或 Token。
