# 单会话记忆集成、缺陷与压力测试报告

测试日期：2026-08-29  
测试环境：Windows 11、Python 3.14.0  
范围：测试、诊断，以及 M1～M6 的生产记忆修复与复验。

## 结论

修复复验：M1～M3 与 M6 已修复；M4 对单一显著旧主题增加了保守的引用焦点回退，歧义时仍返回空；M5 对研究工作台常见同义表达增加了可审计概念归一化。开放域同义词仍不等同于通用 embedding 召回。

- 最近连续上下文可靠范围是默认 4 个完整 turn。
- 桌面提问会以完整桌面文本历史为候选源，通过 BM25、领域概念词和引用焦点召回旧完整 turn，再以近期完整 turn 填充 Graph 的有界上下文。
- Graph 完成一轮后最多保留 120 条消息，处理下一条用户消息前为当前 user/assistant 预留两个位置；边界不会产生只有 assistant 的半个 turn。
- 默认每次请求最多注入 4 个最近 turn 和 3 个早期相关 turn，总对话正文硬预算为 24,000 + 6,000 = 30,000 字符；早期召回单条消息最多 1,200 字符。
- 桌面 JSON 存档没有应用层消息数量上限，10,000 条测试无损；完整文本存档现在是 Desktop 提问的召回候选源。
- 一次真实模型冒烟测试成功使用了第 1 个召回 turn 的 `KAPPA-7319`，并忽略了历史里的 `WRONG-0000` 伪指令；这证明成功召回后模型可以正确使用该字段，但不能弥补召回缺失。

## 实际记忆链路

```text
ResearchSession.messages
  │  SessionStore.save(): 全量 JSON，未按消息数截断
  ▼
桌面会话存档
  │  每次提问：完整 text 历史参与 hybrid turn 召回
  ▼
LangGraph state + SQLite checkpoint
  │  相关旧 turn 优先 + 近期 turn 填充；最多 120 条且 turn 原子
  ▼
select_conversation_context()
  ├─ 最近：默认 4 个完整 turn，最多 24,000 字符
  └─ 更早：BM25 + 中文二字组合 + 领域概念/引用焦点，默认最多 3 个 turn、6,000 字符
  ▼
dialogue / planner JSON payload
  ├─ conversation_history
  └─ earlier_related_turns
```

相关实现：

- 桌面投影与保存：`app/desktop/session.py`、`app/desktop/workspace.py`
- Graph 保留和 checkpoint：`app/research/graph/workflow.py`、`app/research/graph/checkpoint.py`
- 最近窗口与 Episode 记忆：`app/research/agent/context.py`
- hybrid turn 召回与双端摘录：`app/research/agent/retrieval.py`
- dialogue 注入：`app/research/agent/orchestrator.py`
- planner 注入：`app/research/agent/subagents/eda.py`

当前 plan、current run 和 Episode 摘要不完全依赖聊天召回：dialogue 还会收到 `current_plan`、`interaction_context.current_run` 和最多 8 个 Episode 摘要；Graph 最多保留 48 个 Episode 摘要。这对研究结果连续性有帮助，但不能恢复被截掉的普通对话事实或用户偏好。

## 分层边界

| 层 | 代码硬边界 | 本次实测 | 实际含义 |
|---|---:|---:|---|
| 桌面 JSON 存档 | 未设置消息数/字符数上限 | 10,000 条无损，3,285,340 bytes | 可以保存很长，但每次保存会重写完整 JSON |
| Desktop → Graph 投影 | 完整桌面 text 历史作为候选，Graph 前置选择最多 118 条 | 长 turn 的 user/assistant 同时保留 | 为当前 user/assistant 预留两个位置 |
| Graph 对话状态 | 最多 120 条消息，完整 turn 原子 | 61 turn 边界仍为 120 条，相关 turn 1 完整保留并淘汰无关 turn 2 | checkpoint 有界但不会形成孤立 assistant |
| 最近窗口 | 默认 4 turn，24,000 字符 | 两条各约 10,000 字的消息时只保留 1 turn，共 20,034 字符 | turn 原子性优先，字符预算可能有未利用空间 |
| 早期召回 | 默认 3 turn，6,000 字符 | 长 turn 召回 2,400 字符 | 每条召回消息另有 1,200 字符上限 |
| 默认模型对话正文 | 最多 7 turn、30,000 字符 | 长文本样本为 22,379 tokens（`o200k_base`） | 不含 system prompt、工具 schema、Episode memory 等额外内容 |

配置允许 `VPP_LLM_HISTORY_MESSAGES` 为 1～100、`VPP_LLM_RETRIEVED_TURNS` 为 0～10。前者实际换算为 `max(1, history_messages // 2)` 个最近 turn；即使提高数量，24,000/6,000 字符硬预算仍不改变。

长文本压力样本中，最近正文 20,034 字符约 20,015 tokens，召回正文 2,400 字符约 2,381 tokens，合计约 22,396 tokens（`o200k_base`）。这不是 30,000 字符硬上限的固定 token 换算；中文、代码和标识符可能接近一字符一 token，最终请求还要加上 system prompt、JSON 元数据、工具 schema 和 Episode memory。

## 压力测试

### Hybrid retrieval + turn 选择

这是对独立检索层的测试；Desktop 现在会让超过 120 条的完整文本历史参与候选选择，但最终 Graph checkpoint 仍只保存最多 120 条。

| 消息数 | turn 数 | 目标位置 | p50 | p95 |
|---:|---:|---|---:|---:|
| 8 | 4 | 最近窗口 | 0.028 ms | 0.034 ms |
| 16 | 8 | 早期召回 | 0.098 ms | 0.153 ms |
| 128 | 64 | 早期召回 | 0.870 ms | 1.819 ms |
| 1,000 | 500 | 早期召回 | 7.084 ms | 9.044 ms |
| 2,000 | 1,000 | 早期召回 | 15.763 ms | 22.721 ms |
| 5,000 | 2,500 | 早期召回 | 42.518 ms | 71.505 ms |
| 10,000 | 5,000 | 早期召回 | 103.094 ms | 142.552 ms |

检索在 1,000 条仍很轻；当前完整桌面候选会在进入 Graph 前完成相关 turn 选择，Graph 的 120 条上限不再是 Desktop 长会话的绝对召回边界。

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
- 边界查询命中第 1 turn 时，最早保留 `graph-turn-0001`，无关的 `graph-turn-0002` 被淘汰，最新为 `graph-turn-0061`。
- 重启后恢复 120/120 条，checkpoint 相关文件合计约 307,200 bytes。
- 本次修复复验重启读取约 68.710 ms。
- fake gateway 下整轮 Graph p50 19.193 ms、p95 22.034 ms；不含真实模型网络时间。
- 第 61 次模型请求查询第 1 个 turn 时，召回结果同时包含 user 与 assistant。

## 召回质量

小型标注诊断集结果：micro precision = 1.00，micro recall = 1.00。样本量很小，只用于定位行为，不是生产质量统计。

| 场景 | 结果 |
|---|---|
| 精确技术标识 `max_lag` | 通过 |
| 有明显中文词面重合 | 通过 |
| 完全无关问题 | 正确返回空 |
| 领域常用词 | 现有回归测试正确抑制 |
| 无词面重合、但命中领域概念词典的同义改写 | 通过 |
| 超过最近 4 turn、且只有一个显著旧主题的“继续按刚才那个” | 通过 |
| 两个旧主题同样显著的纯指代 | 正确返回空，交由对话层澄清 |
| 长用户消息关键字在尾部 | 通过 |
| 长 assistant 回答结论在尾部、匹配词在开头 | 通过，同时保留匹配片段和尾部 |

## 已确认缺陷

### M1：完整桌面历史无法参与长期召回（高，已修复）

复现：在实际桌面链路完成 61 个 turn，再询问第 1 个 turn 的明确主题。  
预期：桌面存档仍有完整问答，模型可召回。  
修复：Desktop 每次提问传入完整文本历史；协调层用当前问题检索相关旧 turn，并将有界结果同步到新建、普通继续和 interrupt/resume Graph 路径。

### M2：Graph 120 条临界点破坏完整 turn（高，已修复）

复现：Graph 已有 60 个完整 turn 时提交第 61 个问题，并让问题匹配第 1 个 assistant 回答。  
预期：召回完整的 user + assistant。  
修复：Graph 写入统一按完整 turn 原子裁剪，并在写入当前 user 前用当前问题保留相关旧 turn、预留当前 user/assistant 的两个位置。

### M3：Desktop → Graph 重建会按消息字符截断半个 turn（高，已修复）

复现：桌面存档末尾一个 turn 的 user 和 assistant 各约 15,000 字符，然后在 Graph 不存在时投影。  
预期：保留完整 turn，或按 turn 内规则同时截断两边。  
修复：Desktop 不再在排除当前消息前执行单消息字符裁剪；重建选择保持完整 turn，长 user/assistant 会同时进入 Graph，并在模型最近窗口内按双方公平预算截取。

### M4：跨窗口纯指代无法召回（中，已修复单一显著主题场景）

复现：目标 turn 位于最近 4 turn 之外，询问“继续按刚才那个做”。  
修复：词法召回为空且问题含明确引用线索时，根据完整候选历史中的稀有主题词和任务设定语言生成 turn 级焦点分数；只有第一候选明显领先时才召回，否则保持为空以避免误指代。

### M5：纯同义改写无法召回（中，已修复常见领域表达）

复现：历史为“把图表导出到报告目录”，当前问题为“将可视化写入产物文件夹”。  
修复：在 BM25 词项之外加入小型、可审计的领域概念词典；至少两个概念一致才可绕过纯词面门槛，单个“保存”等宽泛概念不会触发召回。开放域同义表达仍需 embedding 或其他语义模型。

### M6：长回答尾部结论可能消失（中，已修复）

复现：长 assistant 回答在开头出现主题词、在尾部给最终参数。  
修复：超过单条预算时同时保留词面匹配附近片段和消息尾部；没有直接匹配的配对消息保留头尾，并继续满足每条 1,200 字符硬上限。

### M7：全量 JSON 重写随历史线性增长（低，未来性能风险）

10,000 条时单次保存 p95 约 43 ms，当前仍可接受；由于桌面每次追加多种消息都会保存全部会话，更多会话、更长正文或慢磁盘上会逐渐影响交互。

## 已通过的重要场景

- 最近 4 个 turn 与当前问题分离，当前问题不重复注入。
- 召回结果不与最近窗口重复。
- 精确标识符、中文词面匹配、领域同义概念、保守纯指代、常用词抑制和无关问题空召回。
- 长回答的匹配上下文与尾部结论同时保留。
- dialogue 和 planner 通过真实 Graph 调用链收到相同记忆结构。
- 桌面创建、连续对话、保存、关闭、SQLite restart 和继续对话。
- 61-turn 真实离屏桌面压力链路保持完整桌面存档。
- 1,000 条永久回归测试和 10,000 条独立压力基准。
- 当前 plan、run、result 与 Episode memory 的现有回归测试。
- 历史 Prompt 注入防护：系统提示规定旧要求不是当前指令；真实模型冒烟验证成功。

## 配置与架构建议

### 当前阶段

保持默认 `4 recent turns + 3 retrieved turns`。增加召回条数不能解决开放域同义词或歧义指代，反而会增加无关上下文。

优先修复顺序：

1. ~~将 Graph 和 Desktop → Graph 的裁剪改为完整 turn 原子裁剪。~~ 已完成。
2. ~~让召回读取完整会话存档，而不是只读取 Graph 最近 120 条。~~ 已完成。
3. ~~改进长 turn 摘录，同时保留匹配片段与尾部。~~ 已完成。
4. ~~为明确的纯指代建立保守 turn 级焦点回退。~~ 已完成；多主题歧义仍需澄清。
5. ~~增加领域概念词与词法检索的 hybrid 召回。~~ 已完成；扩大到开放域前需固定标注集和 embedding 评估。

### 什么时候使用 SQLite

当前项目已经用 SQLite 保存 LangGraph checkpoint，但它只保存被裁剪后的 Graph 状态，不能扩展记忆。

当产品要求可靠访问 60 turn 之前的原文，或单会话达到数千条且全量 JSON 重写开始影响 UI 时，应增加增量式消息存储/索引。可以先使用 SQLite 消息表和 FTS/预切分词索引，保留桌面投影作为 UI 数据，而不是简单调高 Graph 上限。

### 什么时候增加摘要或 embedding

- 当前焦点回退只处理一个明显旧主题；需要跨多个活跃任务可靠维持“刚才那个”时，仍应增加持久化的结构化 session facts / task pointer。
- 当前概念词典覆盖可审计的产品与研究术语；需要开放域同义召回时，再增加 embedding，并与精确标识符/BM25 组成 hybrid retrieval。

## 可复现命令

```powershell
.\.venv\Scripts\python.exe -m pytest tests/research/test_memory_retrieval.py tests/desktop/test_memory_retrieval_integration.py tests/research/test_single_session_memory_boundaries.py -q
.\.venv\Scripts\python.exe scripts/benchmark_single_session_memory.py
.\.venv\Scripts\python.exe scripts/smoke_single_session_memory_live.py --confirm-live
.\.venv\Scripts\python.exe -m pytest -q
```

本轮修复后全量回归结果：349 passed，耗时 121.52 秒；M4～M6 的原 strict xfail 已转为普通通过用例。
