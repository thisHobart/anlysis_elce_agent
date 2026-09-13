# 电价研究 Agent

P2 新增本地新闻工作台入口，支持批量导入、事件复核、持久化缓存和结果导出，运行方式见 [P2 工作台说明](docs/p2-workbench.md)。可选的Playwright浏览器MCP可以读取已批准URL并冻结为同一P2输入契约，见 [P2浏览器MCP说明](docs/browser-mcp-p2.md)。分析器已改用完整、匹配且不重叠的对照窗口；旧合成数据的显著性结论不应沿用。

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
- 右侧数据面板提供会话级地区按钮；选择地区后只读获取该地区的实际电价、实际影响因素和预测影响因素。本轮会话始终绑定所选地区，不会影响其他会话。
- 首次取数保存到本地后，连续提问和分析使用本地数据；重复选择当前地区也直接复用，重新打开该历史会话仍然有效。只有点击“取最新的”/“再试一次”、选择其他地区或当前本地数据文件缺失时才重新取数；新会话独立取数。
- 地区配置用 `history_start_date` 限定首次历史起点，用 `refresh_lookback_days` 控制增量刷新回溯范围；山东从 `2025-11-01` 开始，刷新时重读最近 30 天并合并新增或修订记录。每次成功刷新发布一个完整本地版本，失败时继续保留上一版。
- 数据库索引诊断可运行 `venv\Scripts\python.exe scripts\diagnose_region_queries.py shandong`。脚本只执行 `EXPLAIN`，输出查询计划和候选索引列，不创建或修改索引。
- 当前目录只启用山东测试数据源：小时电价将研究轴补至 2025-11-01，并读取负荷、总发电、新能源、风电、水电、光伏、备用及相应预测和气象变量；后续地区通过 `configs/region_databases.yaml` 增加非敏感表映射，并在 `.env` 增加对应凭据组，无需改动界面。
- 取数后可以直接在聊天框限定当前会话的研究区间，例如“仅分析 2026-01-01 到 2026-03-31”；回复“恢复全部时间范围”可取消限定。区间采用所选市场时区，未写时刻时包含完整的开始日和结束日。
- 当前会话已有P1结果后，可以直接输入“根据已经分析的数据，开始新闻分析最后电价预测”。桌面会复用P1证据，读取冻结且可追溯的山东新闻快照，依次完成P2新闻分析和P1新闻特征综合，再显示需要单独确认的P3预测方案卡。可用 `VPP_P2_NEWS_PATH` 指定其他符合新闻JSONL契约的冻结快照。
- 数据获取过程明确显示“正在找数据 / 已就绪 / 取不到”；失败后可以重试或改用本地文件。
- 本地数据文件仍按角色管理，文件名不受限制：目标电价、实际外生变量、预测外生变量。
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

规划器只返回一个紧凑的 `EDAPlanIntent`：变量使用稳定 ID，函数只提交唯一选择和必要覆盖参数；目标、假设和前提随同意图一次返回。本地适配器再按确定性目录补齐理由、默认参数、受信阈值、函数顺序和编译器议程。评估器为每个研究函数准备确定性验收规则：假设要么得到明确结论，要么被标记为需要补数据、需要扩大审批范围或需要改写，不会被静默忽略。关系样本不足或某个变量全程没有可用相关时，评估器在原审批范围内提出收缩 `max_lag` 或移除该变量的自动修订。

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
VPP_LLM_BASE_URL=http://127.0.0.1:xxxx/v1
VPP_LLM_API_KEY=local-placeholder
VPP_LLM_MODEL=your-model-name
VPP_SKILL_PATHS=
```

`.env` 只保存 Provider、Base URL、API Key 和模型名四项 LLM 连接身份。API 形式、结构化输出能力、
上下文窗口、最大输出、超时、重试、推理和历史预算由桌面配置写入用户应用数据目录中的
`settings.json` 与 `model_profiles.json`。无法匹配内置画像的调用路由会标记为“模型能力未验证”，
使用短上下文和紧凑结构化协议，用户可在模型配置中一次性补全画像。

Gemini Developer API 配置示例（`Base URL` 留空，不经过 Cherry Studio 等 OpenAI 代理）：

```dotenv
VPP_LLM_PROVIDER=gemini
VPP_LLM_BASE_URL=
VPP_LLM_API_KEY=your-google-api-key
VPP_LLM_MODEL=gemini-2.5-flash
```

Vertex AI 使用应用默认凭据（ADC，即 Google SDK 读取的本机/工作负载身份）；Vertex 开关、项目和地区在桌面的运行设置中保存。模型名只填写 `gemini-*`，不能使用 Cherry Studio 的 `vertexai:` 前缀。

桌面端可以选择 `DeepSeek`、`Qwen`、`Gemini（原生 API）` 或 `Custom`。旧配置按 `custom + chat` 读取，不根据 URL 猜供应商。Gemini 固定使用 Google 原生 `json_schema`（服务商在生成时按数据结构约束输出）；其他端点按配置使用 Chat Completions 或 Responses。

如果 Gemini 只能通过 Cherry Studio API Server 调用，服务商保持 `Custom`，接口填写 Cherry 的 `/v1` 地址，并在“结构化输出”选择“Cherry 兼容 JSON/函数选择（本地校验）”。该模式把 JSON Schema 和研究函数白名单放入系统消息，模型响应必须通过本地 Pydantic、函数名白名单和计划编译器校验；失败结果不会进入研究分析。它用于解决 Cherry 未转发原生 schema 或 `tools` 的限制，不等同于服务商原生结构化输出。模型只提出候选调用，用户批准前不会执行。

研究对话、函数提议和方案修订使用大模型。模型未配置、调用失败或返回无效方案时，任务会停止并提示重试，不存在本地关键词规划回退。大模型只承担受限路由和 Function Call 提议；研究阶段、函数顺序、门禁和停止条件来自本地领域协议。响应中的 `reasoning_content`、reasoning 内容块或 `<think>...</think>` 不会进入 LangGraph 状态；默认丢弃，严格策略下拒绝本次响应。

消息由 Agent 内部的类型化角色统一表达，再由所选服务商适配器编码。Gemini 的结构化结果使用原生 JSON Schema，研究函数选择使用 Gemini 原生 Function Calling；OpenAI 兼容端点默认使用明确配置的原生协议。只有显式选择 `prompt_json` 时，系统才接受由提示生成、经本地 Schema 与函数白名单校验的兼容结果；其他模式不会从普通文本猜测或补救 JSON。

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
构建结束前会在隔离目录中实际启动刚生成的程序，构造桌面窗口，并校验两个内置 Skill、30 个研究函数、P3 PyTorch/scikit-learn CPU 运行时和 Skill 加载错误；探针失败时构建命令返回非零退出码。

## P3 山东次日实时电价预测

选择山东并完成取数后，可在对话中提出“预测山东明天实时电价”。桌面会先固定3个历史回测日和1份次日输入，并显示必须手动确认的预测卡；确认后在 CPU 后台完成三折回测和次日96点预测。结果包含 CSV、逐点回测 Parquet、指标、SVG、报告、版本和文件哈希，全程不写业务数据库。实现边界和验收说明见 [P3 山东实时电价最小预测](docs/phase3-shandong-realtime-price-forecast.md)。

应用层还提供一次有界的 P1—P2—P3 编排：P1 生成新闻请求，P2 交付带时点的数值特征，P1 综合后由用户单独确认 P3；未达标或不可评估时只反馈分析一次。运行记录见 [一轮流程验证](docs/p1-p2-p3-one-round-validation-2026-09-12.md)。

## 测试

```powershell
.\venv\Scripts\python.exe -m ruff check app tests
$env:QT_QPA_PLATFORM="offscreen"
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m pytest -q -m integration
.\venv\Scripts\python.exe -m pytest -q -m functional
.\venv\Scripts\python.exe -m pytest -q -m e2e
.\venv\Scripts\python.exe scripts\validate_phase1.py
.\venv\Scripts\python.exe scripts\validate_desktop_phase1.py
.\venv\Scripts\python.exe -m scripts.validate_p1_p2_p3 --synthetic
```

真实验证脚本使用真实模型和真实数据；为避免较大的 Function Calling 请求被过短的本地配置误杀，验证进程使用 120 秒请求超时下限，但不会改写 `.env`。`validate_p1_p2_p3 --synthetic` 使用明确标注的合成数据，并通过桌面 P3 确认卡触发预测。`validate_phase1.py` 会分别验证仅目标电价和包含外生变量的两个 Skill 场景。

完整文档入口见 [docs/README.md](docs/README.md)。当前实现说明见
[docs/phase1-price-exogenous-eda.md](docs/phase1-price-exogenous-eda.md)，循环设计见
[docs/research-loop.md](docs/research-loop.md)，验收状态见
[docs/phase1-acceptance.md](docs/phase1-acceptance.md)。P2 当前状态见
[docs/phase2-status.md](docs/phase2-status.md)，目标与门禁见
[docs/phase2-news-price-analysis-proposal.md](docs/phase2-news-price-analysis-proposal.md)，长文本三方案实现、功能测试与 MVP 选型见
[docs/p2-long-context-strategy-evaluation.md](docs/p2-long-context-strategy-evaluation.md)；新闻研究链路已有独立实现和真实来源语料，但尚未通过真实模型生产准入，也未接入桌面流程或生产新闻接口。
