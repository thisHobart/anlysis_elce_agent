# 持久化研究循环

桌面会话只维护一个持久化 `ResearchLoopGraph`，但不再把整个 `session_id/thread_id` 当作一次有限循环。循环状态明确分为四层：

```text
Session：长期对话与研究历史，不设置“总研究轮次”
└─ Episode：一个独立研究目标，拥有独立迭代计数和终止状态
   └─ Iteration：计划 → 执行 → 验证 → 评估；只有完成评估才计数
      └─ Stage attempt：`plan_attempt_number` 与 `call_attempt_number` 分开记录
```

主流程：

```text
用户提出新研究目标
→ begin_episode：生成 episode_id，重置已评估 Iteration 计数
→ 主 Agent / Skill / Subagent 形成候选计划
→ 确定性计划校验
→ interrupt 等待审批
→ 锁定原子函数队列
→ 单函数执行与结果校验
→ 确定性评估：检查项 + 假设议程 + scope
├─ accept → 保存 accepted 终态 → result interrupt
├─ revise → 同一 Episode 的下一 Iteration
├─ plan_error → 重试同一 Iteration，或由用户发起新 Episode
├─ result_limitations → 保存当前最佳证据并等待用户
└─ reject → result_rejected interrupt → 用户修改或确认停止
```

评估器把检查项和假设议程统一归类为 `within_envelope`、`needs_approval`、`needs_data` 或 `inherent`。只有仍有 `within_envelope` 待办时才自动修订；议程不再变化时立即停止，方法论限制和普通 `warning` 只保留在检查、警告和后续建议中。

## 计数语义

- `evaluated_iterations` 只在一次执行完成并产出独立评估后增加。
- 规划失败、参数校验失败、瞬时函数错误和崩溃恢复都只是 stage attempt，不消耗已评估迭代数。
- 评估要求修订时创建新的 `iteration_id`，并重置该 Iteration 的规划尝试计数。
- 用户明确“重试”会在同一 Episode 创建新的 Iteration；已经完成的证据和失败记录仍可审计。
- 主 Agent 判断为 `new_plan` 时创建新的 `episode_id`。新 Episode 继承会话记忆和历史产物，但已评估 Iteration 从零开始。
- `episode_history` 保存之前 Episode 的目标、终态、循环计数、方案和 run ID。

## 终止规则

循环不再以“整个会话总共两轮”为完成条件。每个 Episode 根据以下条件独立终止：

- 验收通过 `accepted`；
- 评估拒绝后用户确认停止；
- 用户停止 `stopped`；
- 相同计划、相同失败或相同证据表明没有进展；
- 需要扩大审批范围或需要用户修正输入。

`max_evaluated_iterations` 只作为异常情况下的安全熔断，不是正常的收敛条件。正常停止由议程收敛、议程无进展、数据阻断和人类审批决定。

## 人机交互

- 方案审批通过 `interrupt()` 暂停，通过 `Command(resume=...)` 恢复。
- 默认必须明确确认方案；只有宿主显式启用自动执行时，前台反馈窗口到期才发送 `timeout_accept`。
- `timeout_accept` 只能由前台倒计时在 deadline 到达后携带内部标记触发；提前调用或普通用户恢复请求会被 Graph 拒绝。
- 应用关闭期间不执行任务；重启发现审批过期时必须用户明确确认。
- 评估自动修订只允许收缩原审批函数、变量和参数范围；扩大范围转 `result_limitations`。
- 审批事实由 `approval_state` 中的方案 ID、方案指纹和审批时间共同确定；`plan_origin` 或模型 `execute_plan` 意图不能直接授权执行。
- 每个 interrupt 只接受 payload `choices` 中列出的 action；越界 action 保持原 checkpoint，不会默认结束或拒绝。
- `plan_error` 只提供重试、修改要求或停止，不允许接受/执行不存在的方案。
- `result_limitations` 只在已有研究结果时出现，用户可以说明下一步研究要求或结束本轮研究。

## 自校验与恢复

- 所有错误转换为结构化 `FeedbackPacket`。
- `call_id` 标识当前计划中的执行实例；`work_id` 只由数据指纹、函数版本和规范化参数生成，用于跨修订复用。
- 工具结果复用缓存最多64项；完整结果原子写入 checkpoint 同级的结果存储，Graph 只保存 `work_id/output_hash/storage_key` 引用；复用时重新加载、验哈希并绑定当前 `call_id/step_id`，保证证据溯源不串线。
- `ToolResult` 保存真实 `started_at/finished_at`；研究包执行轨迹不再用 finalize 时间伪造函数时间。
- 节点和外部副作用保持幂等；checkpoint 恢复只重跑未提交的函数 attempt，已完成结果直接复用。
- 方案校验时加载并固定一次执行快照，工具队列和 finalize 复用该快照；缓存最多保留两份并在终态释放。
- 最终合并、评估或报告落盘失败时进入持久化 `failed`，保留工具记录和结果，不再让 Graph 裸异常退出。
- 相同规划失败指纹在同一 Iteration 第二次出现即停止自动重试。
- 证据收敛使用所有历史工具证据指纹成员判断，可检测 A→B→A 振荡；评价文案和反馈 ID 不进入证据指纹。
- 计划、函数结果和最终评估分别验证；生成者不负责单独批准自己的结果。
- Episode 游标、Iteration 游标、规划/函数 attempt、反馈、循环计数、正负结果和终止原因均持久化。
- `feedback_packets` 只保存当前 Iteration 尚未解决的反馈；已被新候选计划消费的反馈进入有界 `feedback_history`，不再重复喂给 planner。
- Graph 内事件最多保留160条并使用单调 sequence；完整桌面轨迹由会话投影保存。Graph 消息最多120条且模型上下文执行硬字符上限，历史 Episode 按数据指纹过滤后作为独立上下文传入，运行历史只保存产物引用，不复制完整 summary。
- 每次 Graph 到达审批、结果或其他用户交互点后，SQLite 只保留每个 namespace 的最新完整 checkpoint 及其 pending writes；当前 Graph 不使用 `DeltaChannel`，桌面也不提供 time-travel。完整审计由会话投影、循环记录和研究包承担。

循环事件是界面“研究过程”的唯一来源：`app/research/graph/narration.py` 把事件名、状态和 details 翻译成分阶段的中文说明，供对话卡片和运行记录复用；同一函数的准备/完成/校验三条事件在对话卡中合并成一行，完整三条保留在运行记录里。叙述层不引入任何新的事实，也不接触模型私有推理。

路由只读取 `control`，`phase` 仅用于桌面展示。Episode 的 `episode_goal` 保持不变，当前消息写入 `latest_turn`，结果总结指令写入 `explanation_request`。

## 默认循环控制

| 控制项 | 默认值 |
|---|---:|
| 单 Iteration 规划尝试 | 3 次（含首次） |
| 单函数调用尝试 | 2 次（含首次） |
| 已完成评估的 Iteration 安全熔断 | 4 次 |

该上限只防止异常状态导致无限循环；正常收敛不依赖它，也不把它作为用户可见的研究完成条件。当前不设置 Episode 函数调用总量或活动处理时间预算。

SQLite checkpoint 是运行状态权威来源；`research_sessions.json` 是界面和跨 Episode 历史投影。Graph schema 为 11；旧 schema 在解析嵌套契约前进入只读安全投影，原始 checkpoint 和此前报告保持不变，界面提示用户新建对话，不在旧会话中迁移或重建。
