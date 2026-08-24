# 第一阶段：电价与外生变量 EDA Agent

## 阶段目标

建立一个可直接运行的桌面研究 Agent。用户通过对话提出电价与外生变量问题，大模型读取数据画像后推荐 EDA 方案；用户通过对话提出修改意见，大模型生成修订版；方案锁定后由确定性 Python 工具执行，评估器检查结果并生成可复现研究包。

核心原则：

> 分析工具是确定的，分析过程由 Agent 根据问题、数据和用户选择动态确定。

现有统计函数不是固定工作流，而是 Agent 的可信工具箱。Agent 不能自行计算统计量，也不能调用工具白名单以外的代码。

## 交互闭环

```text
聊天入口
→ 用户提出研究问题
→ 进入统一研究对话
→ 加载、对齐并检查数据
→ Agent 推荐假设、变量、步骤和参数
→ 只读展示模型方案并等待用户反馈 30 秒
→ 有反馈：原方案和用户意见再次交给大模型，生成新版本并重新等待
→ 无反馈：锁定方案并自动执行确定性工具
→ 逐工具结果校验
→ 评估器输出 accept / revise / need_user / reject
→ revise 在原审批范围内自动进入下一轮，最多两轮
→ Agent 基于完整结构化证据回答追问或等待用户决策
→ 用户确认下一轮研究或继续修订
→ 保存对话、计划版本、运行血缘、执行轨迹、结果与哈希
```

Agent 只有一条模型规划路径：已配置的 OpenAI 兼容模型根据研究问题、工具目录、数据画像和对话历史生成结构化方案。模型未配置、调用失败或结构化输出无效时，研究暂停并提示重试，不能由关键词规则替代模型决策。

桌面的每个研究回合都进入同一个持久化 LangGraph。方案审批和结果决策使用 `interrupt/resume`；工具按稳定调用ID逐个执行和校验；评估反馈在预算内回到Subagent。SQLite checkpoint支持在审批、工具执行和结果追问位置恢复。

主 Agent 创建新方案时必须从 Skill 注册表选择一个 Skill。第一阶段内置 `price-exogenous-eda@1.0.0`，并可从程序同级 `skills/` 或 `VPP_SKILL_PATHS` 加载外部专业 Skill。外部 Skill 不能注册或执行代码，只能使用应用已注册且经策略授权的函数。

工具采用 OpenAI 兼容 Function Schema。模型负责选择 Skill、工具、方法和参数；计划编译器验证后，锁定方案由 `ToolExecutor` 通过 `ToolRegistry` 调用本地确定性函数。工具实现位于 `app/research/tools/eda`，不允许模型提供函数路径或任意 Python 代码。

模型输出不能直接执行，必须经过确定性计划编译器。编译器检查工具、具体方法实现、方法版本、变量、参数边界和数据指纹。统一对话支持概念讨论、生成新方案、修订当前方案、解释已有结果和立即执行五类动作；每类动作均由模型理解，结果解释只能引用确定性工具证据。

## 最简桌面界面

第一阶段不建设仪表盘或复杂导航，只提供一个三栏 Qt Widgets 研究工作区：

1. **左侧历史会话**：新建、搜索、选择、重命名和删除本地会话。
2. **中间连续对话**：用户消息、Agent 回复、数据检查、只读模型计划、30 秒反馈倒计时、工具状态和研究结果处于同一时间线。
3. **右侧会话上下文**：用户按需选择文件输入，查看当前方案和可独立滚动的 Agent 运行轨迹。

文件输入按研究配置、目标电价、实际外生变量和预测外生变量四种角色管理，但不限制文件名，也不要求全部填写。YAML、实际变量和预测变量均可省略；没有 YAML 时自动识别 CSV/Parquet 的时间列、数值列和频率。执行 EDA 时至少需要目标电价序列。用户更换文件后，旧数据画像和计划立即失效。

启动方式：

```powershell
.\venv\Scripts\python.exe -m app.desktop
```

Windows 非开发用户也可以双击仓库根目录的 `start_desktop.bat`。

## Agent 可调用工具

| 工具 | 作用 | 是否固定执行 |
|---|---|---|
| `data_quality` | 时间对齐、覆盖率、缺失、重复、异常和可获得性风险 | 必选安全门禁 |
| `price_profile` | 可分别选择分布、波动、极端值、自相关和季节性 | Agent 推荐/用户选择 |
| `exogenous_profile` | 可分别选择分布、异常、趋势和共线性 | Agent 推荐/用户选择 |
| `relationship_analysis` | 可分别选择 Pearson、Spearman、领先滞后、分时段及分位数组关系 | Agent 推荐/用户选择 |

大模型可以启用或停用后三项工具，进一步选择工具内部的方法、变量并设置最大滞后。用户修改不直接改写可执行计划，而是作为反馈再次交给模型。执行器只返回最终锁定的方法结果。最大滞后按数据频率换算，安全上限为 31 天。

方法目录同时记录模型可读的方法键和确定性实现 ID，例如 `relationship.scipy_pearson_pairwise`、`relationship.pearson_positive_lead_scan`、`price.calendar_group_profile`。每个实现都有独立版本；同一计划在方法版本不匹配时拒绝执行，避免“相关性分析”名称相同而实际算法变化。

每个计划保存父计划、版本、修订来源和数据指纹。执行前重新计算输入文件及配置指纹；如果文件在方案生成后被原地修改，旧方案会被拒绝并要求重新规划。

## 当前数据配置

默认配置为 `configs/research/price_exogenous_eda.yaml`，使用根目录 `data/` 下的数据：

| 文件 | 用途 |
|---|---|
| `target_rt_price.csv` | 实时节点电价目标 |
| `feature_actuals.csv` | 实际负荷、新能源、发电和备用变量 |
| `feature_forecasts.csv` | 总发电、非市场机组、新能源和水电预测变量 |

当前数据没有附带完整市场名称、单位字典和发布时间字段，因此业务解释和预测实验前仍需补齐这些元数据。

## 可复现研究包

每次 Agent 执行创建独立目录：

```text
artifacts/research/agent-{timestamp}-{fingerprint}/
  conversation.json
  research_plan.json
  execution_trace.json
  study_config.json
  data_quality.json
  eda_summary.json
  agent_evaluation.json
  aligned_data.parquet
  report.md
  figures/
  manifest.json
```

其中 `research_plan.json` 保存模型、提示词版本、用户反馈形成的修订关系、工具、方法实现 ID、方法版本、变量和参数；`manifest.json` 保存输入输出 SHA-256、Git 状态和运行环境。相同数据、代码、配置和锁定计划可以重新执行并核对结构化证据。

原有确定性入口仍可用于算法工具回归验证：

```powershell
.\venv\Scripts\python.exe -m app.research.cli --config configs/research/price_exogenous_eda.yaml
```

它不是桌面 Agent 的主要用户入口。

## 验证

```powershell
.\venv\Scripts\python.exe -m ruff check app/research app/desktop tests/research tests/desktop
$env:QT_QPA_PLATFORM="offscreen"
.\venv\Scripts\python.exe -m pytest -q tests/research tests/desktop
```

测试应覆盖：

- 不同问题产生不同的启用步骤、变量和参数。
- 不同问题产生不同的具体分析方法，而不是固定执行整个 EDA 清单。
- 无数据时仍由模型讨论研究方法；已有方案只能通过“用户反馈 → 模型修订”修改。
- 模型不可用或返回无效方案时明确失败，不能生成本地规则方案。
- 计划卡不提供本地规划开关或直接编辑入口，30 秒无反馈时自动执行。
- 执行计划记录并校验具体方法实现 ID 和版本。
- 结果追问使用完整 `eda_summary` 和评估证据，不只复述结论模板。
- 用户调整后的计划控制真实执行内容。
- 方案生成后原地修改输入文件会被数据指纹门禁拒绝。
- 研究包包含对话、计划、执行轨迹、评价和有效哈希。
- PySide6 主窗口只有一个三栏连续研究会话，不再存在聊天/研究双页签。
- 会话历史可以在应用重启后恢复。
- 右侧可选文件由用户选择，计划和轨迹与当前会话同步。
- 项目提供桌面运行与 Windows EXE 打包。
- 真实 `data/` 数据可以完成 Agent 计划执行。

## 第一阶段边界

本阶段只处理电价与外生变量 EDA，不包含新闻事件抽取、预测模型、特征晋级、长期实验数据库或无限自主循环。自动研究严格限制为两轮、每工具一次重试、最多八次工具调用和十分钟活动处理时间。
