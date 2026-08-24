# 持久化研究循环

桌面端只运行一个 `ResearchLoopGraph`。同一 `session_id` 同时作为 LangGraph `thread_id`，SQLite checkpointer 在每个节点后保存状态。

```text
用户消息
→ 主 Agent 路由
→ Skill 校验
→ Subagent 规划
→ 计划校验与最多2次反馈修复
→ interrupt 等待审批
→ 锁定工具队列
→ 单工具执行与结果校验
→ 确定性评估
├─ accept → 模型解释 → result interrupt
├─ revise → 原审批范围内自动修订和第二轮执行
├─ need_user / 预算耗尽 → user interrupt
└─ reject → 保存负结果
```

## 人机交互

- 方案审批通过 `interrupt()` 暂停，通过 `Command(resume=...)` 恢复。
- 应用保持打开时，30秒无反馈由桌面发送 `timeout_accept`。
- 应用关闭期间不执行任务；重启发现审批过期时必须用户明确确认。
- 用户修改生成父子计划并重新审批。
- 评估自动修订只允许收缩原审批工具、变量、方法和参数范围；扩大范围转 `need_user`。
- 结果和 `need_user` 也是 interrupt，用户可以追问、接受限制、修改或停止。

## 自校验与恢复

- 所有错误转换为 `FeedbackPacket`，记录来源、代码、严重性、观测、期望、建议、可重试性和是否需要用户。
- 工具调用使用由计划、步骤、参数、版本和数据指纹生成的稳定 `call_id`。
- 工具执行前先checkpoint为 `running`；崩溃恢复时只重跑未提交的当前调用一次，已完成结果直接复用。
- 工具结果校验调用身份、输出Schema、结果键、工具版本、数据指纹和输出哈希。
- 相同计划、重复调用、相同证据或预算耗尽都会停止自动循环。
- 正结果、负结果、反馈、工具记录、预算和停止原因均写入研究制品。

## 默认预算

| 预算 | 默认值 |
|---|---:|
| 自动计划修复 | 2次 |
| 单工具重试 | 1次 |
| 研究执行轮次 | 2轮 |
| 工具调用 | 8次 |
| 活动处理时间 | 10分钟，不含interrupt等待 |

SQLite文件与 `research_sessions.json` 位于同一应用数据目录。JSON会话文件是界面投影，LangGraph checkpoint是研究运行状态的权威来源。
