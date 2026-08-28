# 架构图完成审计

## 需求与证据

| 目标要求 | 当前证据 | 结论 |
|---|---|---|
| 基于实际代码与运行行为 | 图集来源列出了 Desktop、Coordinator、Graph、Agent、Tool、Evaluation 和 Reporting 的实际代码入口；接口清单使用当前 Pydantic 和 LangGraph 契约名称 | 已满足 |
| 先验证模块边界、职责、入口和出口 | `module-interfaces.md` 为 M01–M07 分别定义代码位置、入口、出口、调用关系和不负责事项 | 已满足 |
| 系统上下文图 | `A00-system-context` 展示用户、系统、模型端点、本地文件和本地持久化边界 | 已满足 |
| 项目模块总览图 | `A01-module-overview` 只展示 M01–M07 及模块级关系 | 已满足 |
| 每个核心模块的内部流程 | M01–M07 各有独立 Mermaid、SVG 和 PNG；每张图显式标注入口与出口 | 已满足 |
| 关键业务场景时序 | S01 问题到方案、S02 审批到报告、S03 有界修订、S04 重启恢复、S05 结果追问 | 已满足 |
| 研究会话状态迁移 | ST01 展示 `LoopPhase` 全部主状态、审批分支、用户中断恢复和三种终态 | 已满足 |
| ResearchLoopState 数据契约 | D01 展示权威状态的字段分组、核心嵌套契约、快照、中断和恢复载荷 | 已满足 |
| 避免跨层连线和巨型流程图 | 总览不出现函数节点；模块图只展开一个模块；M03 将 27 个 LangGraph 节点聚合为五个可追踪阶段 | 已满足 |
| 统一编号、命名和视觉规范 | A、M、ST、D、S 五类稳定编号；统一 Mermaid 默认主题和白色背景 | 已满足 |
| 可维护 Mermaid 源文件 | `sources/*.mmd` 共 16 个，独立文件、可单独修改 | 已满足 |
| SVG/PNG 交付 | `rendered/` 中各 16 个 SVG 和 PNG，PNG 使用 2 倍比例 | 已满足 |
| 可重复渲染与验证 | `render.ps1` 固定 Mermaid CLI 版本；`validate.ps1` 检查图集、链接、文件数量与尺寸 | 已满足 |
| 配套说明和图索引 | `README.md` 提供阅读顺序、图索引、代码覆盖矩阵、约束和渲染方法 | 已满足 |

## 代码事实抽查

- `app/desktop/workspace.py` 的 `submit_question`、`run_plan`、`_resume_graph` 和 `_loop_completed` 支撑 M01、S01、S02。
- `app/research/application/coordinator.py` 的 `submit_user_message`、`resume`、`continue_thread` 和 `get_snapshot` 支撑 M02。
- `app/research/graph/workflow.py` 的节点注册及 `add_edge` / `add_conditional_edges` 支撑 M03、S01–S05。
- `app/research/agent/orchestrator.py`、`agent/subagents/eda.py`、`app/llm`、SkillRegistry 和 EDAPlanCompiler 支撑 M04。
- `EDAExecutionService.prepare`、`compile_tool_queue`、`execute_call`、`validate_tool_result` 和 `finalize` 支撑 M05、S02、S03。
- `evaluate_agent_run` 和 `write_agent_research_package` 支撑 M06。
- `CheckpointerHandle`、Graph migrations、SessionStore、ArtifactLayout 和 runtime paths 支撑 M07、S04。
- `ResearchLoopState`、`LoopPhase`、`InterruptPayload`、`ResumePayload` 和 `ResearchLoopSnapshot` 支撑 ST01、D01。

## 可读性检查

- 全部 Mermaid 源文件已由固定版本 CLI 成功解析。
- 全部 PNG 为白色背景，最小宽度和高度通过自动检查。
- A00、A01、M01–M07、ST01、D01 和 S01–S05 均完成视觉抽查。
- 初版过宽的 M02、M04、M05 已改为纵向布局。
- 初版回路线较多的 M03 已从节点级图改为五阶段模块内流程，节点名保留在阶段框中。
