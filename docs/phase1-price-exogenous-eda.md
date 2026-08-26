# 第一阶段：电价与外生变量 EDA Agent

## 阶段目标

建立一个可直接运行的桌面研究 Agent。用户通过对话提出电价与外生变量问题，大模型读取数据画像后推荐分析方案；用户通过对话提出修改意见，大模型生成修订版；方案锁定后由确定性 Python 函数执行，评估器检查结果并生成可复现研究包。

核心原则：

> 分析函数是确定的，领域研究阶段与门禁由本地 Skill 协议确定，阶段内的最小函数集合由 Agent 根据问题、数据和用户选择。

现有统计函数是 Agent 的可信工具箱。领域协议固定证据顺序，但不会强制执行无关函数。Agent 不能自行计算统计量，也不能调用工具白名单以外的代码。

## 交互闭环

```text
聊天入口
→ 用户提出研究问题
→ 进入统一研究对话
→ 加载、对齐并检查数据
→ Agent 推荐假设、变量、步骤和参数
→ 只读展示模型方案并等待用户反馈 30 秒
→ 有反馈：原方案和用户意见再次交给大模型，生成新版本并重新等待
→ 无反馈：锁定方案并自动执行确定性函数
→ 逐函数结果校验
→ 评估器输出 accept / revise / need_user / reject
→ revise 在原审批范围内进入同一 Episode 的下一 Iteration
→ reject 必须经过用户修改或停止决策，不直接结束线程
→ Agent 基于完整结构化证据回答追问或等待用户决策
→ 用户确认下一轮研究或继续修订
→ 保存对话、计划版本、运行血缘、执行轨迹、结果与哈希
```

Agent 只有一条模型函数提议路径：已配置的 OpenAI 兼容模型根据研究问题、授权函数、数据画像、领域协议和对话历史返回具体 Function Call。模型未配置、调用失败或结构化输出无效时，研究暂停并提示重试，不能由关键词规则替代模型决策。DeepSeek 全部模型与自定义 OpenAI 兼容接口均通过请求参数关闭 thinking；返回非空思考字段或标签时研究立即失败。可审计研究链只来自本地版本化协议、函数调用、确定性证据和门禁结果。

桌面的每个研究回合都进入同一个持久化 LangGraph。方案审批和结果决策使用 `interrupt/resume`；函数按稳定调用 ID 逐个执行和校验；评估反馈在预算内回到 Subagent。SQLite checkpoint 支持在审批、函数执行和结果交互位置恢复。

## 内置 Skill

主 Agent 创建新方案时必须从 Skill 注册表选择一个 Skill。第一阶段内置两个核心 Skill：

| Skill | 领域协议 | 授权函数 | 适用场景 |
|---|---|---:|---|
| `price-exogenous-eda@3.0.0` | `electricity-price-evidence-ladder@2.0.0` | 30 | 电价与外生变量联合研究 |
| `price-forecastability-audit@1.0.0` | `price-forecastability-ladder@1.0.0` | 14 | 只有目标电价数据，或需要先判断序列本身有多可预测 |

两个 Skill 都可以从程序同级 `skills/` 或 `VPP_SKILL_PATHS` 之外加载外部专业 Skill。Skill 通过受限相对路径声明本地 YAML 研究协议；加载器禁止协议引用越出 Skill 目录。外部 Skill 不能注册或执行代码，只能使用应用已注册且经策略授权的函数。协议函数集合必须与 Skill 授权集合完全一致，否则加载失败。

工具采用 OpenAI 兼容 Function Schema。模型负责选择 Skill 并提出具体函数与公开参数；计划编译器按领域协议排序并校验后，锁定方案由 `ToolExecutor` 通过 `ToolRegistry` 调用本地确定性函数。工具实现位于 `app/research/tools/eda`，不允许模型提供函数路径或任意 Python 代码。

模型输出不能直接执行，必须经过确定性计划编译器。编译器检查工具、具体方法实现、方法版本、变量、参数边界和数据指纹。统一对话支持概念讨论、生成新方案、修订当前方案、解释已有结果和立即执行五类动作。

## 最简桌面界面

第一阶段不建设仪表盘或复杂导航，只提供一个三栏 Qt Widgets 研究工作区：

1. **左侧历史研究**：新建、搜索、选择、重命名和删除本地会话。
2. **中间连续对话**：用户消息、助手回复、可展开的研究过程、只读方案卡、30 秒反馈倒计时和结果卡处于同一时间线。
3. **右侧会话上下文**：用户按需选择数据文件，并查看可独立滚动的完整研究过程记录。

**研究过程的展示口径**：界面显示的每一步都来自已经发生的循环事件（接收问题、路由意图、激活 Skill、生成方案、通过校验、锁定队列、逐函数执行与耗时、评估结论），由 `app/research/graph/narration.py` 统一翻译成中文。它不是模型的思考链——模型网关禁用 thinking，系统也不保存任何隐藏推理。

叙述遵循同一套审计日志文风：动词开头、句尾不加标点、不使用第二人称；进行中的分析写作 `分析中 · <函数名>` 并附该函数的注册技术说明，完成后写作 `已完成 · <函数名>` 并只附耗时；批量锁定按研究阶段报告构成（例如“数据体检 1 项、电价自身规律 2 项……共 8 项”）。同一函数的“准备执行 / 执行完成 / 结果校验通过”三条事件在界面上折叠为两行，完整三条仍保留在会话文件与研究包中。

数据文件按研究配置、目标电价、实际外生变量和预测外生变量四种角色管理，但不限制文件名，也不要求全部填写。执行分析时至少需要目标电价序列。用户更换文件后，旧数据画像和计划立即失效。

启动方式：

```powershell
.\venv\Scripts\python.exe -m app.desktop
```

Windows 非开发用户也可以双击仓库根目录的 `start_desktop.bat`。

## Agent 可调用 Research Function

30 个原子函数按研究阶段分组，`stage` 同时用于协议排序、界面分组和报告章节：

| 阶段 | 函数 |
|---|---|
| `data_readiness` | `data_quality`（编译器强制加入） |
| `target_structure` | `price_descriptive_distribution`、`price_duration_curve`、`price_tukey_outer_fence`、`price_spike_regime_profile`、`price_calendar_group_profile`、`price_rolling_mean_std`、`price_stationarity_tests`、`price_seasonal_decomposition`、`price_lag_autocorrelation`、`price_partial_autocorrelation`、`price_segment_distribution_comparison` |
| `driver_readiness` | `exogenous_descriptive_distribution`、`exogenous_iqr_outliers`、`exogenous_linear_index_trend`、`exogenous_stationarity_tests`、`exogenous_pearson_collinearity`、`exogenous_variance_inflation` |
| `relationship_evidence` | `relationship_scipy_pearson_pairwise`、`relationship_scipy_spearman_pairwise`、`relationship_mutual_information_scan`、`relationship_pearson_positive_lead_scan`、`relationship_granger_causality_scan`、`relationship_pearson_by_hour`、`relationship_pearson_by_month`、`relationship_feature_quartile_response`、`relationship_rolling_correlation_stability`、`relationship_pearson_segment_comparison` |
| `forecast_readiness` | `price_variance_stabilization_check`、`price_naive_baseline_benchmark` |

大模型通过原生 Function Calling 提出一个或多个具体函数调用。Graph 先拦截并编译为候选计划，用户批准前不会执行。用户修改不直接改写可执行队列，而是作为反馈再次交给模型；最大滞后按数据频率换算，安全上限为 31 天。

同一函数每个计划最多出现一次。变量合并到 `variables`，滞后合并到一个最大 `max_lag`，峰谷、季节和单一事件前后子样本合并到一次 `segments` 调用。`exogenous_variance_inflation` 至少需要两个变量，编译期即校验。

### 需要连续序列的方法

ADF/KPSS、MSTL/STL 分解、PACF 和 Ljung-Box 无法接受缺口。这些函数内部只为自身做时间插值，并在结构化结果中报告 `interpolated_share_for_tests` / `interpolated_share_for_decomposition`；其余统计量一律使用成对有效样本，不做隐式插补。长序列在检验和分解前按固定阶梯（1h → 2h → 3h → 6h → 12h → 1D）降采样，实际粒度写入 `analysis_resolution`。

自动修订必须保持研究问题哈希不变，启用步骤不能超过原审批数量；变量只能收缩、`max_lag` 只能减小、`segments` 必须完全一致，其余函数参数必须与审批值一致。无数据指纹方案在锁定前直接拒绝。

模型、计划、注册表、checkpoint 和审计记录使用同一个函数名。中文展示名称、"这一步回答什么问题"和所属阶段单独存储在 `app/research/tools/catalog.py`；算法变化通过函数版本管理。

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

每次执行创建独立目录：

```text
<output_directory>/agent-{timestamp}-{fingerprint}/
  report.html            # 主报告：自包含 HTML，内联 SVG 图表
  report.md              # 同一内容的 Markdown 版本
  conversation.json
  research_plan.json
  execution_trace.json
  study_config.json
  data_quality.json
  eda_summary.json
  agent_evaluation.json
  research_loop.json
  aligned_data.parquet
  figures/*.svg
  manifest.json
```

`report.html` 是用户和评审看的主入口：概览卡片、结论与限制、分析步骤、数据体检、电价规律、影响因素、关系证据、可预测性八个章节，只渲染本轮真的执行过的证据；方案编号、数据指纹、函数版本和参数收在末尾的折叠区。`research_plan.json` 保存模型、提示词版本、Skill、领域协议、用户反馈形成的修订关系、原子函数、函数版本、变量和参数；`manifest.json` 保存输入输出 SHA-256、Git 状态和运行环境（含 statsmodels 版本）。相同数据、代码、配置和锁定计划可以重新执行并核对结构化证据。

原有确定性入口仍可用于算法回归验证，输出 PNG 图表与 `report.md`：

```powershell
.\venv\Scripts\python.exe -m app.research.cli --config configs/research/price_exogenous_eda.yaml
```

它不是桌面 Agent 的主要用户入口。

## 验证

```powershell
.\venv\Scripts\python.exe -m ruff check app tests
$env:QT_QPA_PLATFORM="offscreen"
.\venv\Scripts\python.exe -m pytest -q
```

测试应覆盖：

- 不同问题产生不同的启用步骤、变量和参数。
- 不同问题产生不同的具体分析方法，而不是固定执行整个清单。
- 全部 30 个函数可在同一计划中执行，并各自写入自己的证据字段。
- 平稳性、季节分解、偏自相关、尖峰状态、持续曲线、朴素基线、方差稳定、VIF、驱动平稳性、互信息、前置性和滚动稳定性各有独立断言。
- 无数据时仍由模型讨论研究方法；已有方案只能通过“用户反馈 → 模型修订”修改。
- 模型不可用或返回无效方案时明确失败，不能生成本地规则方案。
- 计划卡只读，30 秒无反馈时自动执行。
- 执行计划记录并校验具体方法实现 ID 和版本。
- 研究过程在对话中分阶段可见、同一函数合并为一行、结束后折叠，并能在重启后恢复。
- 循环事件到中文说明的翻译不泄漏内部标识符、不使用第二人称，失败事件归入“处理异常”。
- 运行记录中同一函数只出现“分析中”和“已完成”两行，重复叙述被折叠。
- 结果交互使用完整 `eda_summary` 和评估证据，不只复述结论模板。
- 方案生成后原地修改输入文件会被数据指纹门禁拒绝。
- 研究包包含 HTML 报告、SVG 图表、对话、计划、执行轨迹、评价和有效哈希。
- 两个内置 Skill 均可发现；仅目标序列的 Skill 不会产生外生变量与关系章节。
- PySide6 主窗口只有一个三栏连续研究会话。
- 会话历史可以在应用重启后恢复。
- 项目提供桌面运行与 Windows EXE 打包。

## 第一阶段边界

本阶段只处理电价与外生变量 EDA，不包含新闻事件抽取、预测模型、特征晋级或长期实验数据库。朴素基线只作为误差底线的量级参考，不是模型评估。每个研究目标是独立 Episode；默认最多完成 4 次评估，规划与单函数保留局部防死循环次数。当前不设置 Episode 函数调用总量或活动时间预算。新研究目标创建新 Episode，已评估 Iteration 从零开始。
