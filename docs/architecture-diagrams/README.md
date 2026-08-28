# 项目分层架构图

本目录用一组小图代替单张巨型流程图。每张图只回答一个问题，并通过稳定编号建立下钻关系。

## 阅读顺序

1. `A00`：系统与用户、模型端点和本地存储的边界。
2. `A01`：项目七个核心模块及其依赖方向。
3. `M01`–`M07`：逐个展开模块内部流程、入口和出口。
4. `ST01`：查看研究会话主状态及中断恢复路径。
5. `D01`：查看 `ResearchLoopState` 字段分组和嵌套契约。
6. `S01`–`S05`：按具体业务场景查看跨模块交互。
7. [模块接口清单](module-interfaces.md)：核对每个模块的输入、输出和边界约束。
8. [完成审计](verification.md)：查看目标要求、代码事实和交付证据的逐项核对。

## 图索引

| 编号 | 图 | 回答的问题 | Mermaid 源文件 | SVG | PNG |
|---|---|---|---|---|---|
| A00 | 系统上下文图 | 系统服务谁，与哪些外部实体交互？ | [A00](sources/A00-system-context.mmd) | [SVG](rendered/A00-system-context.svg) | [PNG](rendered/A00-system-context.png) |
| A01 | 项目模块总览图 | 当前代码由哪些核心模块组成？ | [A01](sources/A01-module-overview.mmd) | [SVG](rendered/A01-module-overview.svg) | [PNG](rendered/A01-module-overview.png) |
| M01 | 桌面交互模块 | 用户动作怎样转换为研究命令和界面投影？ | [M01](sources/M01-desktop-flow.mmd) | [SVG](rendered/M01-desktop-flow.svg) | [PNG](rendered/M01-desktop-flow.png) |
| M02 | 应用协调模块 | Coordinator 如何驱动 Graph 并返回快照？ | [M02](sources/M02-application-flow.mmd) | [SVG](rendered/M02-application-flow.svg) | [PNG](rendered/M02-application-flow.png) |
| M03 | 持久化研究循环 | Graph 内部怎样规划、审批、执行和评估？ | [M03](sources/M03-graph-flow.mmd) | [SVG](rendered/M03-graph-flow.svg) | [PNG](rendered/M03-graph-flow.png) |
| M04 | 决策、Skill 与计划模块 | 模型提议怎样变成受约束计划？ | [M04](sources/M04-decision-planning-flow.mmd) | [SVG](rendered/M04-decision-planning-flow.svg) | [PNG](rendered/M04-decision-planning-flow.png) |
| M05 | 数据与确定性执行模块 | 文件怎样变成已校验的工具证据？ | [M05](sources/M05-data-tools-flow.mmd) | [SVG](rendered/M05-data-tools-flow.svg) | [PNG](rendered/M05-data-tools-flow.png) |
| M06 | 评估与交付模块 | 证据怎样形成四态决策和研究包？ | [M06](sources/M06-evaluation-reporting-flow.mmd) | [SVG](rendered/M06-evaluation-reporting-flow.svg) | [PNG](rendered/M06-evaluation-reporting-flow.png) |
| M07 | 状态与契约基础模块 | 权威状态、展示投影和研究资产怎样保存？ | [M07](sources/M07-state-contracts-flow.mmd) | [SVG](rendered/M07-state-contracts-flow.svg) | [PNG](rendered/M07-state-contracts-flow.png) |
| ST01 | 研究会话状态图 | `phase` 如何迁移，Graph interrupt 如何暂停和恢复？ | [ST01](sources/ST01-research-session-state.mmd) | [SVG](rendered/ST01-research-session-state.svg) | [PNG](rendered/ST01-research-session-state.png) |
| D01 | ResearchLoopState 数据契约图 | 权威状态包含哪些字段组，并引用哪些核心契约？ | [D01](sources/D01-research-loop-state-contract.mmd) | [SVG](rendered/D01-research-loop-state-contract.svg) | [PNG](rendered/D01-research-loop-state-contract.png) |
| S01 | 问题到方案 | 用户问题怎样变成待审批计划？ | [S01](sources/S01-question-to-plan.mmd) | [SVG](rendered/S01-question-to-plan.svg) | [PNG](rendered/S01-question-to-plan.png) |
| S02 | 审批到报告 | 已审批计划怎样执行并产出报告？ | [S02](sources/S02-approval-to-report.mmd) | [SVG](rendered/S02-approval-to-report.svg) | [PNG](rendered/S02-approval-to-report.png) |
| S03 | 有界自动修订 | 评估反馈怎样触发授权范围内的下一轮？ | [S03](sources/S03-bounded-revision.mmd) | [SVG](rendered/S03-bounded-revision.svg) | [PNG](rendered/S03-bounded-revision.png) |
| S04 | 重启恢复 | 应用重启后怎样恢复 Graph 权威状态？ | [S04](sources/S04-restart-recovery.mmd) | [SVG](rendered/S04-restart-recovery.svg) | [PNG](rendered/S04-restart-recovery.png) |
| S05 | 结果追问 | 完成研究后怎样解释既有证据？ | [S05](sources/S05-result-followup.mmd) | [SVG](rendered/S05-result-followup.svg) | [PNG](rendered/S05-result-followup.png) |

## 当前运行时代码覆盖

| 代码区域 | 归属模块 |
|---|---|
| `app/desktop` | M01 桌面交互 |
| `app/research/application` | M02 应用协调 |
| `app/research/graph` | M03 持久化研究循环；其中 checkpoint、contracts 和 migrations 同时属于 M07 基础能力 |
| `app/research/agent` | M04 决策、Skill 与计划 |
| `app/llm` | M04 决策、Skill 与计划 |
| `app/research/skills` | M04 决策、Skill 与计划 |
| `app/research/planning` | M04 决策、Skill 与计划 |
| `app/research/data` | M05 数据与确定性执行 |
| `app/research/tools` | M05 数据与确定性执行 |
| `app/research/evaluation` | M06 评估与交付 |
| `app/research/reporting` | M06 评估与交付 |
| `app/research/schemas` | M07 状态与契约基础 |
| `app/config.py`、`app/runtime_paths.py` | M07 跨模块配置与文件边界 |

`tests`、`scripts`、构建配置和设计文档属于开发交付支撑，不参与运行时主流程，因此不作为业务模块节点绘制。

## 绘图约束

- 总览图只连接大模块，不出现函数级节点。
- 子流程图只展开一个模块；外部连接必须落在命名明确的入口或出口。
- 时序图一次只描述一个业务场景。
- 状态图只表达 `LoopPhase` 主状态迁移；episode、iteration 和 call 进度留在 `LoopCursor` 数据契约中。
- 数据契约图展示字段分组和重要嵌套类型，不替代源码中的完整类型定义。
- 实线表示控制或数据调用，虚线表示返回、投影或持久化关系。
- `M01`–`M07` 是稳定模块编号；类名和内部函数变化时优先保持跨模块接口稳定。

## 重新渲染

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File docs/architecture-diagrams/render.ps1
```

脚本使用固定版本的 Mermaid CLI，将 `sources/*.mmd` 全部渲染为白色背景的 SVG 和 2 倍 PNG。

渲染后运行完整性检查：

```powershell
powershell -ExecutionPolicy Bypass -File docs/architecture-diagrams/validate.ps1
```

检查会验证固定图集、索引链接、Mermaid 源文件、SVG/PNG 数量以及 PNG 的最小可读尺寸。

## 代码事实来源

- 桌面入口和界面投影：`app/desktop/main.py`、`main_window.py`、`workspace.py`。
- 应用门面：`app/research/application/coordinator.py`。
- Graph 节点与边：`app/research/graph/workflow.py`。
- 模型、Skill 和计划：`app/research/agent`、`app/llm`、`app/research/skills`、`app/research/planning`。
- 数据与工具：`app/research/data`、`app/research/tools`、`app/research/application/execution.py`。
- 评估与研究包：`app/research/evaluation`、`app/research/reporting`。
- 状态与契约：`app/research/schemas`、`app/research/graph/checkpoint.py`、`app/desktop/session.py`。
