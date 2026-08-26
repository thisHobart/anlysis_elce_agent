# 电价研究 Agent

面向电价预测的研究工作台。最终目标是电价预测；当前阶段先把**外生变量与新闻研究**这一前置研发环节做扎实：确定性算法是计算核心，Agent 是研究闭环的控制器。

用户在一个连续会话里提出电价研究问题，Agent 结合可用数据给出可调整的分析方案，确定性统计函数负责计算，评估器检查证据，最后生成一份可分享、可复现的研究报告。

## 桌面界面

```text
历史研究 | 与研究助手的连续对话 | 数据文件、研究过程
```

- 每次启动进入一个新的研究会话，历史记录保留在左侧；空白会话自动清理。
- 对话时间线包含你的提问、助手回复、**可展开的研究过程**、方案卡片和结果卡片。
- 研究过程按“准备 → 解析问题 → 生成方案 → 等待确认 → 执行分析 → 评估结果 → 生成结论”分阶段显示。运行中的分析写明所用统计方法，完成后只留名称与耗时；执行结束自动折叠成一行，点开可回看。
- 右侧“研究过程”面板是完整可审计的运行记录，可筛选、可放大、可回到最新。
- 数据文件按角色管理，文件名不受限制：目标电价、实际外生变量、预测外生变量。
- 实际变量和预测变量可选；程序直接从 CSV/Parquet 自动识别时间列、数值列和频率。
- 真正执行分析时至少需要目标电价数据；没有文件时仍可讨论研究方法。
- 方案由大模型生成、只读展示；想改就直接用自然语言说，助手会给出修订版。
- 只想确认数据能不能用时，方案可以只包含数据体检一步。
- 每版方案等待 30 秒，没有反馈就自动开始；开始输入修改意见时倒计时暂停。
- 会话保存最新证据和多轮运行血缘，重启后仍可继续追问已有结果。
- 单条会话记录损坏只影响它自己：其余会话照常载入，坏记录隔离到同目录的 `*.damaged-*.json`；整个文件无法解析时原文件改名保留，程序不会覆盖它。每次保存留一份 `.bak`。

## 分析能力

内置 **30 个原子研究函数**，每个函数对应一种固定的统计过程，分属五个研究阶段：

| 阶段 | 覆盖内容 |
|---|---|
| 数据体检 | 时间对齐、覆盖率、缺口、重复、数值有效性、预测时点可获得性 |
| 电价自身规律 | 水平与分布、持续曲线、极端值与尖峰/负价状态、日历规律、滚动波动、ADF/KPSS 平稳性、MSTL 趋势季节分解、自相关与偏自相关、分段对比 |
| 影响因素质量 | 分布画像、IQR 异常值、长期漂移、两两相关、方差膨胀因子、各变量自身平稳性 |
| 电价与因素的关系 | Pearson / Spearman、领先滞后扫描、互信息非线性依赖、Granger 样本内前置性、分小时/分月份差异、分位响应、滚动相关稳定性、分段关系对比 |
| 可预测性判断 | 方差稳定变换检查、持续法/日naive/周naive 的朴素基线误差底线 |

统计实现基于 numpy / pandas / scipy / statsmodels，不自行重写标准检验。方法依据与取舍记录在
[app/research/skills/price-exogenous-eda/references/methodology.md](app/research/skills/price-exogenous-eda/references/methodology.md)。

## 研究方法包（Skill）

内置两个核心 Skill：

- `price-exogenous-eda@3.2.0`：电价 + 外生变量联合研究，加载本地版本化协议 `electricity-price-evidence-ladder@2.2.0`，按“市场时钟 → 数据体检 → 电价自身规律 → 影响因素质量 → 关系证据 → 可预测性判断”组织研究，可使用全部 30 个函数。
- `price-forecastability-audit@1.2.0`：只用目标电价序列判断“这条序列有多可预测、后续模型的误差底线在哪”，加载 `price-forecastability-ladder@1.2.0`，只授权 14 个电价侧函数。没有外生变量数据时用它。

外部专业 Skill 可以放在项目或 EXE 同级的 `skills/<skill-name>/SKILL.md`，也可以通过 `VPP_SKILL_PATHS` 添加搜索目录。外部 Skill 只提供专业流程和工具许可，不会自动执行其 `scripts/`。

## 研究报告

每次分析生成一个独立结果文件夹，主报告是自包含的 `report.html`：

- 概览卡片、结论与限制、分析步骤、数据体检、电价规律、影响因素、关系证据、可预测性，共八个可跳转章节。
- 图表全部是内联 SVG（时序、分布、持续曲线、日历画像、自相关/偏自相关、成分占比、相关排序、相关矩阵、领先滞后曲线、互信息、前置性、滚动稳定性、分段对比），无外部依赖、可缩放、可直接打印成 PDF。
- 面向复核人员的编号、指纹、函数版本和参数收在最后的折叠区，正文只讲结论和证据。
- 同一内容另存一份 `report.md`；`figures/` 保留每张图的 SVG 源文件。

## Agent 架构

桌面端通过 `ResearchCoordinator` 驱动唯一的持久化 `ResearchLoopGraph`。主研究 Agent、Skill、EDA Subagent、计划校验、用户审批、逐工具执行、结果校验、评估回流和结果追问都处于同一个有界循环。

```text
PySide6 Desktop
→ ResearchCoordinator
→ app/research/graph
→ MainResearchAgent
→ SkillRegistry / resolve_skill
→ EDASubagent
→ 受约束 Function Calls
→ ToolRegistry / ToolExecutor
→ 工具结果校验
→ 确定性评估器
→ accept / 自动 revise / plan_error / result_limitations / reject 人类门禁
→ SQLite checkpoint / interrupt / resume
```

主 Agent 和 Subagent 共用 `app/llm` 中唯一的模型网关；只有该基础设施层可以创建 `ChatOpenAI`。不同 Agent 的区别是职责、提示词和结构化契约，而不是各自维护模型连接。

界面上显示的“研究过程”不是模型的私有思考链。它由 `app/research/graph/narration.py` 把已经发生的、可审计的循环事件（阶段、函数、参数门禁、评估结论）翻译成中文句子；模型网关始终禁用 thinking，也不保存任何隐藏推理。

大模型每次回复都要调用一次 `declare_research_agenda` 声明本轮目标、假设和前提；它不执行任何计算，只把议程记录进方案。评估器为每个研究函数都准备了确定性验收规则：假设要么得到明确结论，要么被标记为需要补数据、需要扩大审批范围或需要改写，不会被静默忽略。关系样本不足或某个变量全程没有可用相关时，评估器在原审批范围内提出收缩 `max_lag` 或移除该变量的自动修订。

最终证据合并或报告生成失败时，工具结果保留，用户可以选择重试合并或停止；两种结果都写入终态审计记录。

方案校验、函数错误和评估结果通过结构化 `FeedbackPacket` 回流。当前 Iteration 只接收尚未解决的反馈，历史反馈单独审计。会话下的每个研究目标拥有独立 Episode；规划或函数重试不计为已完成迭代。应用关闭期间不后台执行，过期审批在重启后必须明确确认。

命名上，**function** 指研究函数——Skill 授权的、领域协议排序的、方案步骤命名的那个东西；**tool** 只指 OpenAI tool-call 协议边界，即 `ToolCall`/`ToolOutput`/`ToolResult` 及承载它们的注册表、权限策略、执行器和 `tool_*` 循环状态。

EDA 工具位于 `app/research/tools/eda`。模型只能看到注册工具的 Function Schema；最终调用必须经过计划编译、用户反馈窗口、Skill 权限策略和确定性 `ToolExecutor`。

## 运行

需要 Python 3.11+。

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -e ".[dev]"
.\venv\Scripts\python.exe -m app.desktop
```

也可以双击：

```text
start_desktop.bat
```

## 数据文件

右侧有三个数据角色槽位：

| 角色 | 支持格式 | 必需性 |
|---|---|---|
| 目标电价 | CSV / Parquet | 执行分析时必需 |
| 实际外生变量 | CSV / Parquet | 可选 |
| 预测外生变量 | CSV / Parquet | 可选 |

可以点击选择，也可以把文件拖到对应角色槽位。更换文件后，旧数据画像和分析方案会自动失效。

## LLM 配置

复制 `.env.example` 为 `.env`，填写 OpenAI 兼容模型：

```dotenv
VPP_LLM_BASE_URL=http://127.0.0.1:xxxx/v1
VPP_LLM_API_KEY=local-placeholder
VPP_LLM_MODEL=your-model-name
VPP_SKILL_PATHS=
```

研究对话、函数提议和方案修订使用大模型。模型未配置、调用失败或返回无效方案时，任务会停止并提示重试，不存在本地关键词规划回退。大模型只承担受限路由和 Function Call 提议；研究阶段、函数顺序、门禁和停止条件来自本地领域协议。DeepSeek 全部模型固定发送 `enable_thinking=false`；自定义 OpenAI 兼容接口同时发送 `enable_thinking=false` 与 `chat_template_kwargs.enable_thinking=false`。任一路径返回非空 `reasoning_content` 或 `<think>...</think>` 时立即失败，不进入 LangGraph 状态。

结构化输出固定使用模型 Function Calling；模型或 OpenAI 兼容接口需要支持 Function Calling。

大模型只能从 Skill 授权的原子研究函数中选择具体调用。每个函数名唯一对应一种确定性统计过程；本地编译器再按领域协议重排并校验。执行计划保存 Skill、领域协议、函数名、函数版本、参数、输入数据指纹和代码环境，版本不匹配时拒绝执行。

每个原子函数每计划最多调用一次；多变量、多滞后和多子样本分别通过 `variables`、`max_lag` 和 `segments` 批处理。方案审批同时锁定方案 ID 与内容指纹，模型意图不能绕过审批。执行阶段只加载一次并固定数据快照；所有工具与最终评估复用该快照。

工具结果校验除了核对调用身份、结果键、函数版本和数据指纹，还会重算 `output_hash` 并检查该函数声明的证据字段是否存在；复用缓存和从 checkpoint 恢复的结果同样要通过这一关。

函数执行实例使用 `call_id` 记录计划与步骤溯源，可复用工作使用 `work_id` 标识“同一数据、函数版本和参数”。自动修订可以复用相同 `work_id` 的确定性结果；缓存最多 64 项。循环收敛直接比较工具 `work_id + output_hash`，可识别 A→B→A 振荡。

Graph checkpoint 只保留最近 160 条带 sequence 的事件、80 条对话上下文和紧凑运行引用；完整界面轨迹保存在桌面会话投影中。Graph schema 升级前会安全识别旧 checkpoint，SQLite 原始状态先备份再重建。

正式桌面默认把研究结果写入用户 `Documents/PriceResearchAgent/research/`，可用 `PRICE_RESEARCH_OUTPUT_DIRECTORY` 覆盖；内部 SQLite、会话和循环审计写入用户应用数据目录，可用 `PRICE_RESEARCH_APP_DATA_DIRECTORY` 覆盖。打包程序不会向安装目录或 PyInstaller 临时目录写研究数据。

源码运行时读取仓库根目录的 `.env`；打包后的程序读取 `PriceResearchAgent.exe` 同目录的 `.env`。

## 生成 Windows EXE

PyInstaller 必须在 Windows 上构建 Windows 程序。第一次建议使用默认的单目录模式，便于排查依赖；确认无误后可加 `--onefile` 生成单文件版本。

```powershell
.\build_exe.bat
```

单文件版本：

```powershell
.\build_exe.bat --onefile
```

输出位于 `dist/PriceResearchAgent/`，或单文件模式下的 `dist/PriceResearchAgent.exe`。打包会自动包含 `app/research/skills/` 下的全部内置 Skill。

## 确定性命令行

桌面端是主要入口。算法回归仍可直接运行：

```powershell
.\venv\Scripts\python.exe -m app.research.cli `
  --target data/target_rt_price.csv `
  --actuals data/feature_actuals.csv `
  --forecasts data/feature_forecasts.csv
```

## 测试

```powershell
.\venv\Scripts\python.exe -m ruff check app tests
$env:QT_QPA_PLATFORM="offscreen"
.\venv\Scripts\python.exe -m pytest -q
```

第一阶段说明见 [docs/phase1-price-exogenous-eda.md](docs/phase1-price-exogenous-eda.md)，长期路线见 [docs/research-agent-development.md](docs/research-agent-development.md)。

循环设计见 [docs/research-loop.md](docs/research-loop.md)；验收记录见 [docs/phase1-acceptance.md](docs/phase1-acceptance.md)。
