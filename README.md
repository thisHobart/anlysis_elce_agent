# VPP LangGraph Agent

面向虚拟电厂（VPP）的 FastAPI + LangGraph 服务骨架。它将 FAQ、知识库入口、现有数据工具、回答模板和大屏联动拆分为独立节点，并在调用工具和生成联动前执行白名单与权限校验。

## 目录

```text
app/
  graph/nodes/       # 分类、校验、查询、回答和联动节点
  services/          # FAQ、VPP 工具适配器、回答模板
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
  -d "{\"question\":\"查询电站状态\",\"role\":\"operator\",\"params\":{\"station_id\":\"station-01\"},\"request_action\":true}"
```

## 接入现有系统

- 在 `app/services/vpp_tools.py` 将 3 个示例方法替换为现有 Python 工具/API 调用。
- 在 `app/services/faq_service.py` 连接 FAQ 存储或搜索服务。
- 在 `app/graph/nodes/query_data.py` 的 `knowledge` 分支接入现有 RAG 检索器。
- 根据真实 RBAC 策略维护 `app/security/policy.py` 的工具与大屏动作白名单。

## 验证

```bash
pytest
ruff check .
```
