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
- 每版方案默认等待明确确认；可以继续询问、自然语言修改或拒绝，不会因用户暂时离开而自动执行。
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

统计实现基于 numpy / pandas / scipy / statsmodels，不自行重写标准检验。研究顺序与方法取舍记录在
[Skill 说明](app/research/skills/price-exogenous-eda/SKILL.md)和[研究协议](app/research/skills/price-exogenous-eda/references/research-protocol.yaml)中。

## 研究方法包（Skill）

内置两个核心 Skill：

- `price-exogenous-eda@3.2.1`：电价 + 外生变量联合研究，加载本地版本化协议 `electricity-price-evidence-ladder@2.2.1`，按“市场时钟 → 数据体检 → 电价自身规律 → 影响因素质量 → 关系证据 → 可预测性判断”组织研究，可使用全部 30 个函数。
- `price-forecastability-audit@1.2.1`：只用目标电价序列判断“这条序列有多可预测、后续模型的误差底线在哪”，加载 `price-forecastability-ladder@1.2.1`，只授权 14 个电价侧函数。没有外生变量数据时用它。

外部专业 Skill 可以放在项目或 EXE 同级的 `skills/<skill-name>/SKILL.md`，也可以通过 `VPP_SKILL_PATHS` 添加搜索目录。外部 Skill 只提供专业流程和工具许可，不会自动执行其 `scripts/`。

## 研究报告

每次分析生成一个独立结果文件夹，主报告是 `report.md`，由桌面端内置报告阅读器直接展示：

- `report.md` 只回答“证据说明了什么”：结论、数据与口径，然后每个分析步骤一节，按“图 → 读图 → 关键数字”排列；读图文字由确定性模板按判读阈值生成。
- 图表保存为 SVG（时序、分布、持续曲线、日历画像、自相关/偏自相关、成分占比、相关排序、相关矩阵、领先滞后曲线、互信息、前置性、滚动稳定性、分段对比），桌面阅读器按窗口宽度清晰缩放。
- `methods.md` 是方法与验收台账：研究设计、逐函数参数与判读口径、判读阈值、有效性核验和假设验收，并回链到报告中对应的小节。同一件事只在一处说明，报告与桌面卡片不再重复。
- 方案修订后的报告标题和议程只反映本轮实际执行函数；旧轮反馈进入历史，不会伪装成本轮限制。

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

主 Agent 和 Subagent 共用 `app/llm` 中的 `ModelGateway`（统一模型接口）。工厂按服务商选择协议适配器：DeepSeek、Qwen 和自定义 OpenAI 兼容端点使用 `ChatOpenAI`，Gemini 使用 `ChatGoogleGenerativeAI` 直接调用 Google 原生 API。不同 Agent 只定义职责、提示词和结构化契约，不各自维护模型连接。

界面上显示的“研究过程”不是模型的私有思考链。它由 `app/research/graph/narration.py` 把已经发生的、可审计的循环事件（阶段、函数、参数门禁、评估结论）翻译成中文句子；模型网关不把服务商返回的隐藏推理写入应用状态，严格策略下会拒绝含推理内容的响应。

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

复制 `.env.example` 为 `.env`。OpenAI 兼容端点配置示例：

```dotenv
VPP_LLM_PROVIDER=custom
VPP_LLM_API_STYLE=chat
VPP_LLM_BASE_URL=http://127.0.0.1:xxxx/v1
VPP_LLM_API_KEY=local-placeholder
VPP_LLM_MODEL=your-model-name
# 后台可选；桌面配置不会改写该值
VPP_LLM_REASONING_EFFORT=
VPP_SKILL_PATHS=
```

Gemini Developer API 配置示例（`Base URL` 留空，不经过 Cherry Studio 等 OpenAI 代理）：

```dotenv
VPP_LLM_PROVIDER=gemini
VPP_LLM_BASE_URL=
VPP_LLM_API_KEY=your-google-api-key
VPP_LLM_MODEL=gemini-2.5-flash
```

Vertex AI 使用应用默认凭据（ADC，即 Google SDK 读取的本机/工作负载身份），并额外设置 `VPP_LLM_GOOGLE_VERTEXAI=true`、`VPP_LLM_GOOGLE_PROJECT` 和 `VPP_LLM_GOOGLE_LOCATION`。模型名只填写 `gemini-*`，不能使用 Cherry Studio 的 `vertexai:` 前缀。

桌面端可以选择 `DeepSeek`、`Qwen`、`Gemini（原生 API）` 或 `Custom`。旧配置按 `custom + chat` 读取，不根据 URL 猜供应商。Gemini 固定使用 Google 原生 `json_schema`（服务商在生成时按数据结构约束输出）；其他端点按配置使用 Chat Completions 或 Responses。

如果 Gemini 只能通过 Cherry Studio API Server 调用，服务商保持 `Custom`，接口填写 Cherry 的 `/v1` 地址，并在“结构化输出”选择“Cherry 兼容 JSON（本地校验）”。该模式把 JSON Schema 放入系统消息，模型响应必须通过本地 Pydantic 校验；失败结果不会进入研究分析。它用于解决 Cherry 未转发原生 schema 的限制，不等同于服务商原生结构化输出。

研究对话、函数提议和方案修订使用大模型。模型未配置、调用失败或返回无效方案时，任务会停止并提示重试，不存在本地关键词规划回退。大模型只承担受限路由和 Function Call 提议；研究阶段、函数顺序、门禁和停止条件来自本地领域协议。响应中的 `reasoning_content`、reasoning 内容块或 `<think>...</think>` 不会进入 LangGraph 状态；默认丢弃，严格策略下拒绝本次响应。

消息由 Agent 内部的类型化角色统一表达，再由所选服务商适配器编码。Gemini 的结构化结果使用原生 JSON Schema，研究函数选择使用 Gemini 原生 Function Calling；OpenAI 兼容端点使用其明确配置的原生结构化协议。若端点拒绝必要协议，系统会报告不兼容，不会从普通文本中猜测或补救 JSON。

大模型只能从 Skill 授权的原子研究函数中选择具体调用。每个函数名唯一对应一种确定性统计过程；本地编译器再按领域协议重排并校验。执行计划保存 Skill、领域协议、函数名、函数版本、参数、输入数据指纹和代码环境，版本不匹配时拒绝执行。

每个原子函数每计划最多调用一次；多变量、多滞后和多子样本分别通过 `variables`、`max_lag` 和 `segments` 批处理。方案审批同时锁定方案 ID 与内容指纹，模型意图不能绕过审批。执行阶段只加载一次并固定数据快照；所有工具与最终评估复用该快照。

工具结果校验除了核对调用身份、结果键、函数版本和数据指纹，还会重算 `output_hash` 并检查该函数声明的证据字段是否存在；复用缓存和从 checkpoint 恢复的结果同样要通过这一关。

函数执行实例使用 `call_id` 记录计划与步骤溯源，可复用工作使用 `work_id` 标识“同一数据、函数版本和参数”。自动修订可以复用相同 `work_id` 的确定性结果；缓存最多 64 项。循环收敛直接比较工具 `work_id + output_hash`，可识别 A→B→A 振荡。

Graph checkpoint 只保留最近 160 条带 sequence 的事件、按完整 turn 原子选择的最多 120 条对话上下文、与当前数据指纹匹配的结构化 Episode 摘要和紧凑运行引用。完整桌面文本会话是每次提问的召回候选源；协调层先保留与当前问题相关的旧 turn，再用近期完整 turn 填充有界 Graph 上下文，并为当前 user/assistant 预留两个位置。完整工具结果写入 checkpoint 同级的结果存储，Graph 中只留经过哈希校验的引用。每次运行到达用户交互点后仅保留最新可恢复 checkpoint，完整界面轨迹仍保存在桌面会话投影中。Graph schema 升级后，旧 checkpoint 只读保留并提示新建对话，不在旧会话中迁移或重建。

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
构建结束前会在隔离目录中实际启动刚生成的程序，构造桌面窗口，并校验两个内置 Skill、30 个研究函数和 Skill 加载错误；探针失败时构建命令返回非零退出码。

## 测试

```powershell
.\venv\Scripts\python.exe -m ruff check app tests
$env:QT_QPA_PLATFORM="offscreen"
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe scripts\validate_phase1.py
.\venv\Scripts\python.exe scripts\validate_desktop_phase1.py
```

两个验证脚本使用真实模型和真实数据；为避免较大的 Function Calling 请求被过短的本地配置误杀，验证进程使用 120 秒请求超时下限，但不会改写 `.env`。`validate_phase1.py` 会分别验证仅目标电价和包含外生变量的两个 Skill 场景。

完整文档入口见 [docs/README.md](docs/README.md)。当前实现说明见
[docs/phase1-price-exogenous-eda.md](docs/phase1-price-exogenous-eda.md)，循环设计见
[docs/research-loop.md](docs/research-loop.md)，验收状态见
[docs/phase1-acceptance.md](docs/phase1-acceptance.md)。P2 的测试先行候选方案见
[docs/phase2-news-price-analysis-proposal.md](docs/phase2-news-price-analysis-proposal.md)；模拟新闻规范化和明显事件抽取已有独立测试原型，尚未接入桌面流程或生产新闻接口。
