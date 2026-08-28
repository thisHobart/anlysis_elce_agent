# 第一阶段验收记录

## 当前状态：待重新验收

上一次真实模型 + 真实数据验收于 **2026-08-24** 完成（记录见下方“历史验收”）。此后第一阶段进行了一次功能升级，**真实模型、真实数据和 EXE 验收需要重新执行**：

| 变更 | 影响 |
|---|---|
| 研究函数从 18 个扩展到 30 个（新增平稳性、趋势季节分解、偏自相关、尖峰状态、持续曲线、方差稳定、朴素基线、VIF、驱动平稳性、互信息、Granger 前置性、滚动稳定性） | 计划、执行、评估和报告全部涉及；新增 statsmodels 运行依赖 |
| 内置 Skill 从 1 个变为 2 个；`price-exogenous-eda` 升到 `3.2.0`，协议升到 `electricity-price-evidence-ladder@2.2.0`；新增 `price-forecastability-audit@1.2.0` | 旧方案的 Skill/协议版本不再匹配，必须重新规划 |
| 主报告统一为 `report.md`，桌面端新增内置 Markdown/SVG 阅读器 | 产物清单、`report_path` 与“查看完整报告”入口变化 |
| 研究包按复核动作分为 `evidence/`、`provenance/`、`data/`，新增自动生成的 `methods.md`；删除旧 CLI 聚合报告链 | 产物路径、方法说明、manifest 和桌面报告验收变化 |
| 对话中新增可展开的研究过程卡；循环事件统一经 `narration.py` 翻译为用户语言 | 桌面时间线、会话投影与运行记录展示变化 |
| 桌面文案整体改写为面向非开发用户的表述 | 界面文本断言与用户验收话术变化 |
| PyInstaller 现在打包 `app/research/skills/` 下全部内置 Skill，并显式收集 statsmodels 数据 | EXE 体积与内置资源清单变化 |

自动化回归当前为 **`184 passed`**，覆盖持久化恢复、审批真实性、非法 action、异常终态、单次执行快照、跨修订结果复用、A→B→A 收敛、修订议程重建、混合维度分段比较、Iteration 上限、Episode 隔离、旧 schema 隔离、事件有界化、用户目录、真实函数时间场景，以及 30 函数全量执行、结构诊断逐项断言、叙述层翻译、双 Skill 发现、Markdown/SVG 报告、研究过程可见性、分层研究包、方法与证据映射、三类数据文件直入和重启恢复。

自动测试替身只用于边界回归，不作为真实 Agent 完成证明。

## 重新验收清单

```powershell
.\venv\Scripts\python.exe -m ruff check app tests
$env:QT_QPA_PLATFORM="offscreen"
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe scripts\validate_phase1.py
.\venv\Scripts\python.exe scripts\validate_desktop_phase1.py
.\build_exe.bat --onefile
```

真实验收还需要人工确认：

1. 两个内置 Skill 都能被模型选中，并各自跑通一次真实数据研究。
2. 研究过程卡在执行期间实时增长，结束后折叠，重启后可展开回看。
3. `report.md` 在桌面内置阅读器中排版正常，SVG 图表随窗口缩放且可打开外部文件。
4. 单文件 EXE 启动后能加载两个内置 Skill，并完成一次含平稳性与朴素基线的分析。

## 完成条件证据

| 条件 | 主要证据 |
|---|---|
| 统一连续对话 | PySide6 真实验收完成讨论、规划、修订、执行和追问 |
| 所有研究判断调用模型 | `ModelResearchDialogue` 和 `ModelEDAPlanner` 无本地规划回退；模型中断测试明确停止 |
| Skill 选择 | LangGraph `resolve_skill`；内置 `price-exogenous-eda@3.2.0` 与 `price-forecastability-audit@1.2.0` |
| 领域研究协议 | 本地 `electricity-price-evidence-ladder@2.2.0` 与 `price-forecastability-ladder@1.2.0`；计划和 manifest 记录协议版本，编译器按协议排序 |
| 研究函数覆盖度 | 30 个原子函数按五个阶段分组，全量执行回归通过 |
| 统计实现可信 | ADF/KPSS/MSTL/PACF/Ljung-Box 直接调用 statsmodels，不自行重写 |
| 分段比较 | 峰谷、季节和时间范围通过单次 `segments` 调用进入同一 summary；小时、月份、时间范围只在各自维度内生成差值 |
| 审批真实性 | 方案 ID、方案指纹和审批时间共同生成 authorization envelope；模型意图不可绕过审批 |
| 自动修订边界 | 研究问题、步骤数、全部函数参数、变量、滞后和分段定义均 fail-closed |
| 结果复用 | `call_id` 保留执行溯源，`work_id` 支持同数据同函数同参数的跨修订复用 |
| 路由状态 | `control` 是唯一下一跳信号；`phase` 只服务桌面展示；Graph schema 为 11，旧 SQLite 状态保持只读并提示新建对话 |
| 外部 Skill | `skills/`、`VPP_SKILL_PATHS`、外部 Skill 集成测试；外部脚本不执行 |
| Skill 工具权限 | Function Schema 取 Skill 许可交集；编译器和 `ToolPolicy` 双重拒绝越权 |
| 完整计划契约 | `provenance/research_plan.json` 保存 Skill、原子函数、变量、参数和版本 |
| 显式审批 | 默认无操作不执行；宿主显式启用自动模式时，deadline 到达后才允许内部超时 resume |
| 受控函数执行 | `ToolRegistry → ToolPolicy → ToolExecutor`；执行轨迹记录 provider、版本和真实起止时间 |
| 可见的研究过程 | `narration.py` 只翻译已发生事件；请求参数禁用模型 thinking，残留思考内容被网关丢弃且不落库 |
| 报告可读性 | 桌面内置 `report.md` 阅读器 + SVG；报告按“图 → 读图 → 关键数字”组织证据，方法、阈值、假设验收与有效性核验集中在 `methods.md` |
| LangGraph 编排 | 单一持久化 Graph 包含 Skill、规划、interrupt 审批、逐函数执行、校验、评估回流和结果 interrupt |
| Loop 分层 | Session → Episode → Iteration → stage attempt；预算按 Episode 隔离，失败重试不消耗已评估迭代 |
| 可复现性 | 同一数据和锁定计划重复执行的 summary、quality、evaluation 完全一致 |
| 版本与数据门禁 | 数据指纹、Skill 和函数版本变化测试均拒绝执行 |
| 会话恢复 | SQLite 恢复审批、函数崩溃节点和结果交互；真实桌面关闭重启后继续调用模型 |
| 明确失败 | 模型缺失/中断、无效 Skill、函数越权、稀疏数据和版本错误均停止并提示 |

## 历史验收（2026-08-24，升级前）

- 模型：`gpt-5.6-luna`
- 结构化输出：固定使用 Function Calling
- 数据输入：目标电价、实际外生变量、预测外生变量三类 CSV/Parquet 文件
- 真实持久化循环：`agent-20260824T000748386127Z-c79e3779ae0c`
- 真实桌面循环：`agent-20260824T001319972586Z-8d4d83e1bf2d`
- 更早的真实持久化循环：`phase1-real-cb0e83d02c1b`，规划提示词 `eda-plan-v5`，对话提示词 `research-dialogue-v2`
- 真实桌面关闭/重启投影：`b560fc2ca02549688a85a6cc9b82b59b`
- 真实数据重复运行：`phase1-real-repeat`，结构化摘要、质量报告和评估结果均完全一致
- 方案修订：用户修订后由评估自动生成 `v3`
- 所选变量：`actual_load`、`forecast_total_generation`
- 当时的 Windows EXE：`dist/PriceResearchAgent.exe`，`163,044,875` 字节，
  SHA-256 `6CB8FBBBC153FEAD2A26A7083A619F6DDBF8FEEF335300406961F775CDC06CF8`

历史验收证据保存在对应 `artifacts/research/<run_id>/`，桌面验收状态保存在 `artifacts/acceptance/`。正式桌面默认把新研究包写入用户 `Documents/PriceResearchAgent/research/`；内部循环记录、会话和 SQLite 位于用户应用数据目录，不再写 EXE 安装或临时解压目录。
