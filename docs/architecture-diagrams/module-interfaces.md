# 模块接口清单

该清单定义总览图中七个大模块的公开边界。模块内部类和函数可以调整，但跨模块连接应优先经过这里列出的入口和出口。

## M01 桌面交互

- 代码：`app/desktop`
- 入口：用户问题、文件选择、方案操作、结果追问；`ResearchLoopSnapshot`、Graph 事件和报告路径。
- 出口：`submit_user_message`、`resume`、`continue_thread`、`cancel` 等用户命令；会话展示投影。
- 不负责：模型调用、统计计算、Graph 业务判断。

## M02 应用协调

- 代码：`app/research/application`，核心门面为 `ResearchCoordinator`。
- 入口：会话 ID、用户消息或恢复动作、`StudyConfig`、对话投影和进度回调。
- 出口：`ResearchLoopSnapshot`、`InterruptPayload`、进度叙述和结果引用。
- 调用：M03；同时为 M03 组装 M04、M05、M06 的服务能力。
- 不负责：维护第二套跨重启状态机。

## M03 持久化研究循环

- 代码：`app/research/graph`。
- 入口：初始 `ResearchLoopState` 或 `Command(resume=...)`。
- 出口：Graph 快照、interrupt、事件、最终或停止状态。
- 调用：M04 的决策与规划，M05 的工具执行，M06 的评估与最终化，M07 的 checkpoint。
- 权威数据：`ResearchLoopState` 与 SQLite checkpoint。
- 不负责：统计实现、UI 渲染和模型供应商连接。

## M04 决策、Skill 与计划

- 代码：`app/research/agent`、`app/llm`、`app/research/skills`、`app/research/planning`。
- 入口：有界对话、研究问题、数据画像、质量报告、当前计划、评估反馈、允许的函数 schema。
- 出口：`DialogueDecision`、`EDAPlanDraft`、经编译的 `EDAPlan`、文字解释。
- 外部依赖：配置的 OpenAI 兼容模型端点。
- 不负责：读取原始数据、执行任意代码、拥有 Graph 权威状态。

## M05 数据与确定性执行

- 代码：`app/research/data`、`app/research/tools`、`EDAExecutionService`。
- 入口：`StudyConfig`、已锁定 `EDAPlan`、`ToolCall`、Skill 许可和数据指纹。
- 出口：`PreparedExecution`、`DataQualityReport`、`ToolResult`、结构化证据和输出哈希。
- 计算实现：numpy、pandas、SciPy 和 statsmodels 下的注册式原子函数。
- 不负责：解释用户意图、决定 Graph 路由或评价研究是否完成。

## M06 评估与交付

- 代码：`app/research/evaluation`、`app/research/reporting`。
- 入口：已校验工具证据、质量报告、计划、Skill/协议和运行引用。
- 出口：`AgentEvaluation`、`FeedbackPacket`、`ArtifactBundle`、报告路径和 manifest。
- 决策：`accept`、`revise`、`need_user`、`reject`。
- 不负责：让模型主观决定状态，或补造统计事实和因果结论。

## M07 状态与契约基础

- 代码：`app/research/schemas`、`app/research/graph/contracts.py`、`checkpoint.py`、`migrations.py`、`app/desktop/session.py`、`app/runtime_paths.py`。
- 入口：Pydantic 数据、Graph 状态更新、会话投影和研究资产写入请求。
- 出口：校验后的契约、恢复状态、迁移结果、会话备份、研究包路径和文件哈希。
- 数据分层：checkpoint 是运行权威；session 是 UI 投影；研究包是可复核证据资产。
- 不负责：把三类数据互相替代。
