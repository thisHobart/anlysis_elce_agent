# P0 安全边界修复完成报告

> 日期：2026-08-05  
> 优先级：P0（最高）  
> 状态：✅ 已完成

## 问题描述

在 P3 阶段，发现一个严重的安全问题：当知识库或数据源未接入时，LLM 仍可能基于其通用知识生成看似合理的业务说明或数值，导致：

- **知识库未接入**：LLM 可能基于通用知识回答"什么是 VPP"，但内容与聊城虚拟电厂的实际情况不符
- **数据源未接入**：LLM 可能编造"当前发电功率 XX 兆瓦"等数值
- **FAQ 数据未获取**：LLM 可能填充不存在的实时数据

对于 VPP 这种对数据准确性要求极高的生产系统，这是**不可接受的行为**。

## 解决方案

### 1. 修改 `compose` 节点的强制兜底逻辑

**文件**：`app/graph/nodes/compose.py`

**核心改动**：

1. 定义"数据源未连接"的标记集合：
```python
NOT_CONNECTED_REASONS = {
    "knowledge_not_connected",
    "data_query_not_connected",
    "faq_data_not_connected",
}
```

2. 定义必须使用固定消息的路由（安全边界）：
```python
FIXED_MESSAGE_ROUTES = {"out_of_scope"}
```

3. 在 `compose` 函数中强制判断：
```python
must_use_fixed = fallback_reason in NOT_CONNECTED_REASONS or route in FIXED_MESSAGE_ROUTES

if must_use_fixed or not service.enabled:
    answer, source = _fallback_answer(state, entry)
    trace_marker = f"composed:{source}:forced" if must_use_fixed else f"composed:{source}"
else:
    # 只有在安全的情况下才调用 LLM
    answer = service.compose(payload=payload)
```

4. 增强 `_fallback_answer` 函数，针对不同未连接场景返回明确消息：
```python
if fallback_reason == "faq_data_not_connected" and entry:
    faq_id = state.get("faq_id", "")
    return f"FAQ {faq_id} 需要实时数据，但数据源暂未接入。请稍后再试。", "fixed"
```

### 2. 创建 P0 安全测试套件

**文件**：`tests/test_p0_safety.py`

**测试覆盖**：

1. ✅ `test_knowledge_not_connected_blocks_llm`：知识库未接入时，返回固定消息而非 LLM 生成内容
2. ✅ `test_data_query_not_connected_blocks_llm`：数据查询未接入时，返回固定消息而非编造数值
3. ✅ `test_faq_needing_data_not_connected_blocks_llm`：FAQ 需要实时数据但未获取时，明确说明数据源未接入
4. ✅ `test_static_faq_still_uses_llm`：静态 FAQ（无数据需求）仍可使用 LLM 进行自然语言组织
5. ✅ `test_chitchat_uses_llm_normally`：闲聊路由正常使用 LLM（不涉及业务数据）
6. ✅ `test_out_of_scope_returns_fixed_message`：越界请求返回固定拒绝消息

**关键验证点**：
- LLM 的 `compose` 方法在 `NOT_CONNECTED_REASONS` 场景下**绝不会被调用**
- 返回的 `answer` 中不包含 `[HALLUCINATION]` 标记（测试中用于检测意外调用）
- `compose_source` 正确标记为 `"fixed"`
- `trace` 包含 `"composed:fixed:forced"` 标记

## 测试结果

```bash
$ pytest tests/test_p0_safety.py -v
============================== 6 passed in 0.29s ===============================

$ pytest tests/ -v
============================== 56 passed, 1 warning in 0.63s ====================
```

✅ **所有测试通过**，包括：
- 6 个新增的 P0 安全测试
- 50 个现有功能测试（确保未破坏原有功能）

## 行为变化

### Before (P3 初版)

```yaml
用户: "请解释一下 VPP 的工作原理"
分类: route=knowledge, fallback_reason=knowledge_not_connected
组合: compose_source=llm
回答: "虚拟电厂是一种创新的电力管理模式..." ❌ LLM 编造
```

### After (P0 修复)

```yaml
用户: "请解释一下 VPP 的工作原理"
分类: route=knowledge, fallback_reason=knowledge_not_connected
组合: compose_source=fixed (forced)
回答: "知识库服务暂未接入，请稍后再试。" ✅ 明确告知
```

### 其他场景

| 路由 | 未接入原因 | compose 行为 | 回答示例 |
|---|---|---|---|
| `knowledge` | `knowledge_not_connected` | 强制 fixed | "知识库服务暂未接入，请稍后再试。" |
| `data` | `data_query_not_connected` | 强制 fixed | "实时数据查询暂未接入，目前无法提供准确数值。" |
| `faq` (FAQ-06) | `faq_data_not_connected` | 强制 fixed | "FAQ FAQ-06 需要实时数据，但数据源暂未接入。请稍后再试。" |
| `faq` (FAQ-01) | 无（静态 FAQ） | 正常 LLM | LLM 自然语言组织 ✅ |
| `chitchat` | 无 | 正常 LLM | LLM 回答 ✅ |
| `out_of_scope` | 无 | 强制 fixed | "抱歉，我只能协助处理虚拟电厂相关问题。" |

## 安全保障

### 数据准确性

✅ **不会编造业务事实**：LLM 无法在数据源未接入时生成看似合理的虚假内容  
✅ **明确告知用户**：用户清楚知道当前功能状态，而非得到误导性信息  
✅ **可追溯性**：通过 `trace` 字段可以看到 `composed:fixed:forced` 标记

### 降级策略

- **静态 FAQ**（如 FAQ-01）：无数据依赖，仍可使用 LLM 优化表达 ✅
- **闲聊**：不涉及业务数据，可以使用 LLM ✅
- **越界请求**：固定拒绝消息，明确安全边界 ✅

### 向后兼容

✅ 现有 50 个测试全部通过  
✅ P2 规则匹配、P3 LLM 分类、多轮会话等功能不受影响  
✅ `trace` 字段扩展了新标记，但不破坏现有解析逻辑

## 下一步工作

### P1：完善测试集（建议优先）

创建 LLM 验收问题集，覆盖更多边界场景：

```python
test_cases = [
    ("请解释 VPP 是什么", "faq", "FAQ-01"),
    ("你好", "chitchat", None),
    ("今天天气怎么样", "out_of_scope", None),
    ("当前总发电功率是多少", "faq/data_query", "FAQ-06"),
    ("切换到能源运行管理视图", "screen_action", None),
]
```

每个测试验证：
- `intent` 正确
- `route` 正确
- `faq_id` 匹配（如适用）
- `classification_source`（llm/rules）
- `compose_source`（llm/fixed/template）
- `fallback_reason`（如适用）
- `answer` 不编造事实

### P2：接入第一条真实 API FAQ

选择 FAQ-06"当前总发电功率是多少"：

1. 迁移 `vpp_api_call.py` → `app/services/vpp_api.py`
2. 配置白名单：`ALLOWED_API_COMMANDS = {"realtime", ...}`
3. 修改 `query_data` 节点执行真实 API 调用
4. 验证端到端：问题 → API → 数据 → LLM 回答

### P3：重建真实命令白名单

根据 [docs/agent-migration-plan.md](agent-migration-plan.md) 第 7 节，更新 `app/security/policy.py`：

- `ALLOWED_API_COMMANDS`：13 个真实命令
- `ALLOWED_DB_COMMANDS`：22+ 个真实命令
- SQL 安全校验：`_validate_sql_safe`（禁多语句/DML/危险函数）

### P4+：扩展完整能力

- 接入 MySQL 数据库
- 接入 Qdrant 知识库
- 开发前端界面（优先级较低）

## 关键文件清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `app/graph/nodes/compose.py` | 修改 | 强制兜底逻辑、NOT_CONNECTED_REASONS、FIXED_MESSAGE_ROUTES |
| `tests/test_p0_safety.py` | 新增 | P0 安全测试套件（6 个测试） |
| `docs/p0-safety-completion.md` | 新增 | 本文档 |

## 总结

✅ **P0 问题已解决**：LLM 不会在数据源未接入时编造业务事实  
✅ **测试覆盖完整**：6 个新测试 + 50 个回归测试全部通过  
✅ **向后兼容**：现有功能不受影响  
✅ **安全边界清晰**：知识库/数据/越界请求都有明确的固定消息

**当前系统状态**：安全的"可运行骨架"，准备好接入真实数据源。
