# 单会话记忆集成、缺陷与压力测试报告

测试日期：2026-08-29  
测试环境：Windows 11、Python 3.14.0  
范围：仅测试、诊断和报告；未修改生产记忆算法。

## 结论

当前方案足以支持中短会话、主题词或技术标识明确的 Agent 工作，但不能称为可靠的长会话记忆。

- 最近连续上下文可靠范围是默认 4 个完整 turn。
- 更早内容只有在仍位于 Graph 最近 120 条消息内且与当前问题存在足够词面重合时，才可能被召回。
- Graph 完成一轮后最多保留 120 条消息，通常等于 60 个完整 turn；处理下一条用户消息时，安全可用的是 59 个完整历史 turn，最老 turn 会先变成只有 assistant 的半个 turn。
- 默认每次请求最多注入 4 个最近 turn 和 3 个早期相关 turn，总对话正文硬预算为 24,000 + 6,000 = 30,000 字符；早期召回单条消息最多 1,200 字符。
- 桌面 JSON 存档没有应用层消息数量上限，10,000 条测试无损，但完整存档没有成为 Graph 召回的数据源。因此“存下来了”不等于“模型还能记得”。
- 一次真实模型冒烟测试成功使用了第 1 个召回 turn 的 `KAPPA-7319`，并忽略了历史里的 `WRONG-0000` 伪指令；这证明成功召回后模型可以正确使用该字段，但不能弥补召回缺失。

## 实际记忆链路

```text
ResearchSession.messages
  │  SessionStore.save(): 全量 JSON，未按消息数截断
  ▼
桌面会话存档
  │  新建/重建 Graph 时：只投影 text 类型，最多 120 条、24,000 字符
  ▼
LangGraph state + SQLite checkpoint
  │  每次写入按单条消息 messages[-120:] 截断
  ▼
select_conversation_context()
  ├─ 最近：默认 4 个完整 turn，最多 24,000 字符
  └─ 更早：BM25 + 中文二字组合，默认最多 3 个 turn、6,000 字符
  ▼
dialogue / planner JSON payload
  ├─ conversation_history
  └─ earlier_related_turns
```

相关实现：

- 桌面投影与保存：`app/desktop/session.py`、`app/desktop/workspace.py`
- Graph 保留和 checkpoint：`app/research/graph/workflow.py`、`app/research/graph/checkpoint.py`
- 最近窗口与 Episode 记忆：`app/research/agent/context.py`
- BM25 turn 召回：`app/research/agent/retrieval.py`
- dialogue 注入：`app/research/agent/orchestrator.py`
- planner 注入：`app/research/agent/subagents/eda.py`

当前 plan、current run 和 Episode 摘要不完全依赖聊天召回：dialogue 还会收到 `current_plan`、`interaction_context.current_run` 和最多 8 个 Episode 摘要；Graph 最多保留 48 个 Episode 摘要。这对研究结果连续性有帮助，但不能恢复被截掉的普通对话事实或用户偏好。

## 分层边界

| 层 | 代码硬边界 | 本次实测 | 实际含义 |
|---|---:|---:|---|
| 桌面 JSON 存档 | 未设置消息数/字符数上限 | 10,000 条无损，3,285,340 bytes | 可以保存很长，但每次保存会重写完整 JSON |
| Desktop → Graph 首次投影 | 120 条且 24,000 字符 | 长 turn 可能只保留 assistant 半边 | checkpoint 丢失或重建时不能从完整桌面历史恢复全部记忆 |
| Graph 对话状态 | 120 条消息 | 61 turn 后只保留 turn 2～61 | 完成态约 60 个完整 turn，是当前绝对原文召回边界 |
| 最近窗口 | 默认 4 turn，24,000 字符 | 两条各约 10,000 字的消息时只保留 1 turn，共 20,034 字符 | turn 原子性优先，字符预算可能有未利用空间 |
| 早期召回 | 默认 3 turn，6,000 字符 | 长 turn 召回 2,384 字符 | 每条召回消息另有 1,200 字符上限 |
| 默认模型对话正文 | 最多 7 turn、30,000 字符 | 长文本样本为 22,379 tokens（`o200k_base`） | 不含 system prompt、工具 schema、Episode memory 等额外内容 |

配置允许 `VPP_LLM_HISTORY_MESSAGES` 为 1～100、`VPP_LLM_RETRIEVED_TURNS` 为 0～10。前者实际换算为 `max(1, history_messages // 2)` 个最近 turn；即使提高数量，24,000/6,000 字符硬预算仍不改变。

长文本压力样本中，最近正文 20,034 字符约 20,015 tokens，召回正文 2,384 字符约 2,364 tokens，合计约 22,379 tokens（`o200k_base`）。这不是 30,000 字符硬上限的固定 token 换算；中文、代码和标识符可能接近一字符一 token，最终请求还要加上 system prompt、JSON 元数据、工具 schema 和 Episode memory。

## 压力测试

### BM25 + turn 选择

这是对独立检索层的测试；超过 120 条的行用于观察算法扩展性，不代表当前 Graph 能提供这么多候选。

| 消息数 | turn 数 | 目标位置 | p50 | p95 |
|---:|---:|---|---:|---:|
| 8 | 4 | 最近窗口 | 0.024 ms | 0.031 ms |
| 16 | 8 | 早期召回 | 0.077 ms | 0.091 ms |
| 128 | 64 | 早期召回 | 0.613 ms | 0.766 ms |
| 1,000 | 500 | 早期召回 | 4.948 ms | 5.816 ms |
| 2,000 | 1,000 | 早期召回 | 10.201 ms | 12.086 ms |
| 5,000 | 2,500 | 早期召回 | 29.396 ms | 54.825 ms |
| 10,000 | 5,000 | 早期召回 | 86.693 ms | 150.157 ms |

检索在 1,000 条仍很轻，性能不是当前主要瓶颈；主要瓶颈是 Graph 在 120 条处已经删除候选。

### SessionStore

| 消息数 | 文件大小 | 保存 p95 | 加载 p95 | 无损恢复 |
|---:|---:|---:|---:|---|
| 128 | 43,258 bytes | 4.306 ms | 19.452 ms | 是 |
| 1,000 | 328,839 bytes | 6.575 ms | 20.392 ms | 是 |
| 5,000 | 1,642,839 bytes | 22.092 ms | 43.589 ms | 是 |
| 10,000 | 3,285,340 bytes | 43.280 ms | 67.583 ms | 是 |

这些数据只测持久化与模型校验，没有测 10,000 个 Qt 消息组件同时渲染；因此不能据此声明桌面 UI 的实用上限也是 10,000 条。

### Graph + SQLite checkpoint

- 连续提交 61 个 turn 后，完成态仍严格保持 120 条消息。
- 最早保留 `graph-turn-0002`，最新为 `graph-turn-0061`。
- 重启后恢复 120/120 条，checkpoint 相关文件合计约 307,200 bytes。
- 重启读取约 119.312 ms。
- fake gateway 下整轮 Graph p50 24.580 ms、p95 30.421 ms；不含真实模型网络时间。
- 第 61 次模型请求期间，若查询第 1 个 turn，召回结果只有 assistant，证明按单条消息截断破坏了 turn 原子性。

## 召回质量

小型标注诊断集结果：micro precision = 1.00，micro recall = 0.50。样本量很小，只用于定位行为，不是生产质量统计。

| 场景 | 结果 |
|---|---|
| 精确技术标识 `max_lag` | 通过 |
| 有明显中文词面重合 | 通过 |
| 完全无关问题 | 正确返回空 |
| 领域常用词 | 现有回归测试正确抑制 |
| 无词面重合的同义改写 | 失败 |
| 超过最近 4 turn 的“继续按刚才那个” | 失败 |
| 长用户消息关键字在尾部 | 通过 |
| 长 assistant 回答结论在尾部、匹配词在开头 | 失败，尾部结论被截断 |

## 已确认缺陷

### M1：完整桌面历史无法参与长期召回（高，边界缺口）

复现：在实际桌面链路完成 61 个 turn，再询问第 1 个 turn 的明确主题。  
预期：桌面存档仍有完整问答，模型可召回。  
实际：桌面记录完整，Graph 已删除第 1 个 turn，`earlier_related_turns` 无目标。  
影响：第 60 个历史 turn 之外的普通对话事实、偏好和约束不可访问。

### M2：Graph 120 条临界点破坏完整 turn（高，正确性缺陷）

复现：Graph 已有 60 个完整 turn 时提交第 61 个问题，并让问题匹配第 1 个 assistant 回答。  
预期：召回完整的 user + assistant。  
实际：新 user 先占用一个位置，`messages[-120:]` 删除第 1 个 user；模型请求可能收到只有 assistant 的旧 turn。  
影响：每个超过边界的新请求都会短暂产生一个最老的孤立 assistant，不只发生一次。

### M3：Desktop → Graph 重建会按消息字符截断半个 turn（高，恢复缺陷）

复现：桌面存档末尾一个 turn 的 user 和 assistant 各约 15,000 字符，然后在 Graph 不存在时投影。  
预期：保留完整 turn，或按 turn 内规则同时截断两边。  
实际：只保留最新 assistant。  
影响：重建 Graph 时可能把没有问题上下文的回答作为历史导入。

### M4：跨窗口纯指代无法召回（中，常用交互缺陷）

复现：目标 turn 位于最近 4 turn 之外，询问“继续按刚才那个做”。  
实际：没有共同二字词或标识符，召回为空。  
影响：长会话中常见的“继续、上次那个、按前面方案”不可靠。

### M5：纯同义改写无法召回（中，算法边界）

复现：历史为“把图表导出到报告目录”，当前问题为“将可视化写入产物文件夹”。  
实际：BM25 无共同词面，召回为空。  
影响：用户自然改写用词时，即使语义一致也可能失忆。

### M6：长回答尾部结论可能消失（中，截取缺陷）

复现：长 assistant 回答在开头出现主题词、在尾部给最终参数。  
实际：摘录围绕开头匹配词，尾部结论不在 1,200 字符片段中。  
影响：召回了正确 turn，但模型仍看不到真正答案。

### M7：全量 JSON 重写随历史线性增长（低，未来性能风险）

10,000 条时单次保存 p95 约 43 ms，当前仍可接受；由于桌面每次追加多种消息都会保存全部会话，更多会话、更长正文或慢磁盘上会逐渐影响交互。

## 已通过的重要场景

- 最近 4 个 turn 与当前问题分离，当前问题不重复注入。
- 召回结果不与最近窗口重复。
- 精确标识符、中文词面匹配、常用词抑制和无关问题空召回。
- dialogue 和 planner 通过真实 Graph 调用链收到相同记忆结构。
- 桌面创建、连续对话、保存、关闭、SQLite restart 和继续对话。
- 61-turn 真实离屏桌面压力链路保持完整桌面存档。
- 1,000 条永久回归测试和 10,000 条独立压力基准。
- 当前 plan、run、result 与 Episode memory 的现有回归测试。
- 历史 Prompt 注入防护：系统提示规定旧要求不是当前指令；真实模型冒烟验证成功。

## 配置与架构建议

### 当前阶段

保持默认 `4 recent turns + 3 retrieved turns`。增加召回条数不能解决同义改写、纯指代或 120 条来源上限，反而会增加无关上下文。

优先修复顺序：

1. 将 Graph 和 Desktop → Graph 的裁剪改为完整 turn 原子裁剪。
2. 让召回读取完整会话存档，而不是只读取 Graph 最近 120 条。
3. 改进长 turn 摘录：对 assistant 优先保留结论段、数字/标识符附近片段，或同时保留头尾。
4. 为“继续、刚才、上次那个”等指代建立最近主题/当前任务的结构化引用，不要只依赖 BM25。
5. 在固定标注集上再评估 hybrid lexical + semantic retrieval。

### 什么时候使用 SQLite

当前项目已经用 SQLite 保存 LangGraph checkpoint，但它只保存被裁剪后的 Graph 状态，不能扩展记忆。

当产品要求可靠访问 60 turn 之前的原文，或单会话达到数千条且全量 JSON 重写开始影响 UI 时，应增加增量式消息存储/索引。可以先使用 SQLite 消息表和 FTS/预切分词索引，保留桌面投影作为 UI 数据，而不是简单调高 Graph 上限。

### 什么时候增加摘要或 embedding

- 需要长期保持目标、偏好、关键参数和“刚才那个”指代时，优先增加结构化 session facts / 滚动摘要。
- 需要召回同义表达和无共同词面的相关历史时，再增加 embedding，并与精确标识符/BM25 组成 hybrid retrieval。
- 本次同义改写和纯指代各为 0 recall，已经证明存在语义召回需求；但在修复 M1～M3 前直接上 embedding，会给被截断的数据源增加复杂度，不能解决根因。

## 可复现命令

```powershell
.\.venv\Scripts\python.exe -m pytest tests/research/test_memory_retrieval.py tests/desktop/test_memory_retrieval_integration.py tests/research/test_single_session_memory_boundaries.py -q -rxX
.\.venv\Scripts\python.exe scripts/benchmark_single_session_memory.py
.\.venv\Scripts\python.exe scripts/smoke_single_session_memory_live.py --confirm-live
.\.venv\Scripts\python.exe -m pytest -q -rxX
```

最终全量回归结果：303 passed、6 xfailed，耗时 78.18 秒。6 个 xfail 均对应本报告 M1～M6 的已确认、尚未修复行为。
