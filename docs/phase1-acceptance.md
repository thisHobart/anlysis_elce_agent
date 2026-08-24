# 第一阶段验收记录

验收日期：2026-08-24

## 验收结论

第一阶段按照“真实模型、真实数据、真实桌面流程、确定性工具和可复现研究包”进行验收。自动测试替身只用于边界回归，不作为真实 Agent 完成证明。

当前自动化回归为 `63 passed`，其中持久化研究循环专项覆盖12个核心场景。

## 完成条件证据

| 条件 | 主要证据 |
|---|---|
| 统一连续对话 | PySide6 真实验收完成讨论、规划、修订、执行和追问 |
| 所有研究判断调用模型 | `ModelResearchDialogue` 和 `ModelEDAPlanner` 无本地规划回退；模型中断测试明确停止 |
| Skill 选择 | LangGraph `resolve_skill`；真实方案记录 `price-exogenous-eda@1.0.0` |
| 外部 Skill | `skills/`、`VPP_SKILL_PATHS`、外部 Skill 集成测试；外部脚本不执行 |
| Skill 工具权限 | Function Schema 取 Skill 许可交集；编译器和 `ToolPolicy` 双重拒绝越权 |
| 完整计划契约 | `research_plan.json` 保存 Skill、工具、方法、变量、参数和版本 |
| 30 秒反馈 | 前台超时自动resume；SQLite重启发现过期时必须明确确认 |
| 受控工具执行 | `ToolRegistry → ToolPolicy → ToolExecutor`；执行轨迹记录 provider 和版本 |
| 四类 EDA 工具 | 数据质量、电价画像、外生变量画像、关系分析均有确定性实现与测试 |
| LangGraph 编排 | 单一持久化Graph包含Skill、规划、interrupt审批、逐工具执行、校验、评估回流和结果interrupt |
| 真实数据 | 使用根目录 `data/` 和默认 YAML 完成真实运行 |
| 可复现性 | 同一数据和锁定计划重复执行的 summary、quality、evaluation 完全一致 |
| 版本与数据门禁 | 数据指纹、Skill、工具和方法版本变化测试均拒绝执行 |
| 完整研究包 | 配置、计划、反馈、工具记录、预算、质量、摘要、评估、轨迹、图表、报告、清单及哈希 |
| 会话恢复 | SQLite恢复审批、工具崩溃节点和结果追问；真实桌面关闭重启后继续调用模型 |
| 明确失败 | 模型缺失/中断、无效 Skill、工具越权、稀疏数据和版本错误均停止并提示 |
| 源码与 EXE | 源码真实桌面验收通过；单文件 EXE 启动通过且包含内置 Skill |

## 真实模型与数据运行

- 模型：`gpt-5.6-luna`
- 结构化模式：`native`，实际使用 Function Calling
- 数据配置：`configs/research/price_exogenous_eda.yaml`
- 真实持久化循环：`agent-20260824T000748386127Z-c79e3779ae0c`
- 真实桌面循环：`agent-20260824T001319972586Z-8d4d83e1bf2d`
- 最新真实持久化循环：`phase1-real-cb0e83d02c1b`，规划提示词 `eda-plan-v5`，对话提示词 `research-dialogue-v2`
- 最新真实桌面关闭/重启投影：`b560fc2ca02549688a85a6cc9b82b59b`
- 真实数据重复运行：`phase1-real-repeat`，结构化摘要、质量报告和评估结果均完全一致
- 方案修订：用户修订后由评估自动生成`v3`
- 所选变量：`actual_load`、`forecast_total_generation`
- 执行工具：`data_quality`、`price_profile`、`relationship_analysis`
- 自动研究轮次：2轮；第二轮仍有数据可获得性和统计风险，按预算转`need_user`，用户接受限制后生成解释。

真实研究证据保存在对应 `artifacts/research/<run_id>/`，桌面验收状态保存在 `artifacts/acceptance/`。

## Windows EXE

- 文件：`dist/PriceResearchAgent.exe`
- 大小：`163,044,875` 字节
- SHA-256：`6CB8FBBBC153FEAD2A26A7083A619F6DDBF8FEEF335300406961F775CDC06CF8`
- 内置资源：`price-exogenous-eda/SKILL.md`、`references/methodology.md`
- 持久化模块：`langgraph.checkpoint.sqlite`
- 外部 Skill 占位：`dist/skills/README.md`
- 冒烟结果：程序启动后持续运行，无提前退出；验收后主动结束测试进程。

## 可重复执行命令

```powershell
.\venv\Scripts\python.exe scripts\validate_phase1.py
.\venv\Scripts\python.exe scripts\validate_desktop_phase1.py
.\venv\Scripts\python.exe -m pytest -q
.\build_exe.bat --onefile
```
