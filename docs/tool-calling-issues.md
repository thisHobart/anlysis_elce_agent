# 工具调用与工具暴露问题总结

状态：持续修订。第一部分（调用第三方 API）已按当前代码事实整理；后续部分追加到本文末尾。
更新日期：2026-09-15。

## 0. 本文范围与用词

本文汇总"把工具暴露给大模型、再按模型提议调用工具"这条链路上实际发生过的问题。

用词沿用根目录 `README.md` 的约定：

- **function**：研究函数，Skill 授权的、领域协议排序的、方案步骤命名的那一层；
- **tool**：OpenAI tool-call 协议边界，即 `ToolCall` / `ToolOutput` / `ToolResult` 及注册表、权限策略、执行器。

链路有三个外部依赖方，问题形态完全不同，因此分开记录：

1. **模型端点**：DeepSeek、Qwen、Gemini 原生/Vertex，以及 Cherry Studio 之类的 OpenAI 兼容代理；
2. **工具提供方**：MCP 服务器（当前为 Playwright 浏览器）；
3. **数据来源**：山东 MySQL 库、外部新闻 API。

第一类边界是本文档的第一部分，也是问题最集中的地方。

---

## 第一部分：调用第三方 API

### 1.1 三条第三方边界总览

| 边界 | 具体对象 | 主要失败形态 | 代码位置 |
|---|---|---|---|
| LLM 端点 | OpenAI 兼容代理、DeepSeek、Qwen、Gemini 原生/Vertex | 协议不兼容、能力被代理丢弃、异常语义混淆 | `app/llm/` |
| MCP 工具服务 | Playwright MCP（stdio / streamable_http） | 版本漂移、环境缺失、超时、远端报错 | `app/integrations/mcp/` |
| 外部数据源 | 山东 MySQL、外部新闻 API | 凭据缺失、只读边界、接口契约未取得 | `app/research/news/preparation.py`、`app/research/data/sources/regions.py` |

### 1.2 LLM 端点：问题最集中的地方

模型侧的问题可以分成六组。它们的共同根源是：**端点只承诺"OpenAI 兼容"，但兼容到什么程度完全不确定**，而研究 Agent 依赖的是其中最难被兼容的部分——原生结构化输出和原生 Function Calling。

#### 1.2.1 可选请求参数被端点拒绝

现象：同一个请求，在 A 端点正常、在 B 端点直接返回 400。被拒绝的通常是这几类参数：

- `temperature`（部分推理模型不允许固定为 0）；
- `parallel_tool_calls`（只存在于绑定研究函数时的请求上）；
- 推理/思考相关开关，服务商措辞各不相同：`enable_thinking`、`chat_template_kwargs`、`reasoning_effort`、`"thinking"`、`"reasoning"`。

处置：`app/llm/compat.py` 用 `UNSUPPORTED_HINTS` 把"错误文本"映射回"被拒绝的参数名"，再按参数归属决定重试方式：

| 参数归属 | 参数 | 重试方式 |
|---|---|---|
| 客户端级 `CLIENT_LEVEL_FEATURES` | `temperature`、`provider_reasoning` | 重建 `ChatOpenAI` 客户端 |
| 请求级 `OPTIONAL_TOOL_CALL_FEATURES` | `parallel_tool_calls` | 重新 `bind_tools` 即可 |

关键约束：**只有"系统自己加上去的"参数才允许降级**，用户显式配置的参数不降级。

- `_degradable_client_features()` 对 `provider == "custom"` 不返回 `provider_reasoning`，意思是：用户在自定义端点里显式关掉了 thinking，这个意图不能被静默丢弃；
- `tools`、`tool_choice`、`response_format` 三个核心协议字段**故意不在** `UNSUPPORTED_HINTS` 里。端点拒绝它们说明它根本不能承担研究 Agent 的职责，此时直接 `ModelProtocolError`，而不是"把 schema 去掉再试一次"。

降级循环有次数上限（`len(UNSUPPORTED_HINTS) + 1`），全部组合都被拒绝时报"大模型端点拒绝了全部可选请求参数组合"。

#### 1.2.2 上游代理不转发原生能力

这是踩得最疼的一组，有两种表现：

1. **代理丢弃 `response_format` / schema**：请求里带了结构化输出契约，模型侧看不到，返回自由文本。
2. **代理丢弃 `tools`**：模型根本不知道有研究函数可选，于是不返回调用、或返回空。

处置：显式 `prompt_json` 兼容模式，把契约从"API 字段"搬到"系统消息"：

- 结构化输出：`_prompt_json_messages()` 把 `model_json_schema()` 原样写进系统消息，并声明"只返回符合该 Schema 的 JSON，不要 Markdown/代码围栏/额外字段"；
- 函数选择：`_prompt_tool_call_messages()` 把**函数白名单**（name/description/parameters）写进系统消息；
- 之后仍然走本地严格校验：`_PromptToolCalls` / `_PromptToolTurn` 用 `extra="forbid"` 校验信封，函数名必须逐字来自白名单，`_validate_tool_calls()` 拒绝任何白名单外的函数名。

这条路径必须在诊断里**明确标注 `schema_enforcement=local`**，因为它不宣称服务端执行了 schema；把它当成"原生结构化输出"会掩盖真实风险。

历史教训：最初修好了普通结构化对象（`invoke_structured`），却漏了研究函数选择（`invoke_tool_calls`）——结构化输出选项只控制前者。在代理丢弃 `tools` 的端点上，P1 方案因此无法生成。

#### 1.2.3 非标准 HTTP 响应信封

现象：网关把 JSON 响应体**双重编码**（外层是字符串化的 JSON），OpenAI SDK 解析直接失败。这类问题在请求本身完全正确时也会发生，靠改请求参数永远修不好。

处置：`build_compatible_http_client()` 返回一个 `httpx.Client` 子类，在 `send()` 里尝试 `unwrap_double_encoded_body()`；命中则重建响应，并剥掉 `content-length` / `content-encoding`。

实现细节值得保留：**用子类而不是自定义 transport**，是为了让 httpx 继续使用它自己的环境代理发现逻辑。流式响应直接透传。

#### 1.2.4 连接身份与路由不匹配

| 问题 | 处置 |
|---|---|
| 用 Cherry 风格的 `vertexai:` 模型前缀配 Gemini 原生适配器 | 直接报错并给出正确的填写方式（填 `gemini-2.5-flash`，Vertex 另开关） |
| API Key 被写成 `none` / `null` 字符串 | 显示层过滤为空，不冒充凭据 |
| Base URL 写法不统一（大小写、默认端口、尾部斜杠、query 里的密钥） | `normalize_base_url()` 统一规范化并**丢弃 query**，避免密钥进入路由键或日志 |
| Vertex 缺 project / location | 配置校验阶段就拦住 |
| 缺 Base URL 或模型名 | `ModelConfigurationError`，提示"大模型尚未配置" |

路由身份由 `provider | 规范化 base_url | model` 组成，用于匹配 `model_profiles.json` 中的能力画像。匹配不到时标记为 `unverified`，**不编造上下文窗口和最大输出**。

#### 1.2.5 响应内容不合格

HTTP 200 不代表结果可用。以下都在本地被拦截，且**不允许"恰好能过 schema 就算成功"**：

| 现象 | 判定 | 处置 |
|---|---|---|
| 空回复 | `invoke_text` 检查 `visible_text` | `ModelResponseError("大模型返回了空回复。")` |
| 输出截断 | `finish_reason` ∈ {`length`, `max_tokens`, `max_token`, `max_output_tokens`, `token_limit`}；或 SDK 提前抛 `LengthFinishReasonError` / `MaxTokensError` | `ModelOutputTruncatedError`；即使解析成功也判失败 |
| 思考内容泄漏 | `reasoning_content` / `additional_kwargs["reasoning"]` / reasoning 内容块 / 正文内联 `<think>...</think>` | 默认 `strip` 丢弃；严格策略 `reject` 时抛 `ModelThinkingError` |
| 请求放不进上下文 | 调用前用 `structured_request_budget()` 连系统提示、正文、Schema、输出预留和安全余量一起算 | `ModelContextLimitError`，**不裁剪正文、不静默降级** |
| Schema 解析失败 | `parsing_error` 非空或 `parsed` 为空 | 打印原始输出 + 校验错误供诊断，然后抛协议错误 |
| 函数参数不合 schema | 有 `tool_calls` 但解析失败 | `ModelResponseError`，不猜测、不补救 |
| 参数是 JSON 字符串 | 部分端点原样返回编码后的参数字符串 | `ModelToolCall._decode_arguments()` 预解码，解不开就保留原值交由校验失败 |
| 调用了白名单外的函数 | `_validate_tool_calls()` | 明确列出未授权的函数名 |
| 缺 provider `call_id` | `invoke_tool_turn` 校验 | `ModelResponseError("大模型函数调用缺少 provider call_id。")`，因为回传 tool 结果时无法对应 |

`prompt_json` 模式的 `call_id` 需要自己造，且必须是**确定性的**：`invoke_tool_turn` 用 `sha256(对话历史 + 序号 + 调用 JSON)` 生成，这样同一轮对话重放不会产生不同的调用身份。

消息契约本身也是校验点（`app/llm/gateway.py`）：tool 消息必须有 `tool_call_id`；assistant 带 `tool_calls` 时每个调用必须有 `call_id`；assistant 必须有正文或调用之一，不能两者皆空。

#### 1.2.6 异常分档与失败语义

最初的问题是把所有端点异常当成一类"调用失败"，结果一次 HTTP 500 就让整批 benchmark 中止。现在按可修复性分三档：

| 档位 | 判定（`compat.py`） | 语义 | 处置 |
|---|---|---|---|
| 拒绝请求 | 400 / 404 / 422，或 `BadRequestError` / `UnprocessableEntityError` / `NotFoundError` | 端点不接受这个请求形状，换参数重试也没用 | 转 `ModelProtocolError`，fail closed |
| 瞬时故障 | ≥500 或 408 / 429，或超时/连接错误/限流 | 只影响本次调用 | 转 `ModelTransientError`，**按篇/步骤隔离**，不中止全局 |
| 传输失败（非瞬时） | ≥500 或 401 / 403 / 408 / 429 | 认证、权限、配置问题 | 全局停止，不重试 |
| 内容层失败 | 见 1.2.5 | 响应拿到了但不可用 | 结构化失败，不进证据 |

配套的产品口径（`docs/data-panel-ui-spec.md`）：面向业务人员的面板文案对"连接超时、认证失败、隧道未建、库不存在、权限不足"是**同一套说法**，差异只进 `TraceEvent.details` 和日志——业务人员对这些区别做不出不同动作。

但内部审计口径必须相反：**模型或网关故障不得冒充业务结论**。`docs/phase2-status.md` 明确写了，端点协议错误导致的运行"不能计为准确率"，模型故障不得伪装成"正确隔离"。

#### 1.2.7 能力探测：把"看起来能用"变成"验证过能用"

两个脚本专门用来在跑基准之前拆穿伪兼容：

- `scripts/probe_structured_output.py`：每次用 `uuid4` 生成**随机字段名**的 schema。原因是早期探针用固定字段名，模型可以从提示词里猜出格式，于是"通过"是假的；随机字段名能证明 schema 真的被端点传递和执行了。`prompt_json` 模式下它验证的是"提示遵循 + 本地校验"，并明确输出"这不表示服务商提供了服务端原生 schema 约束"。
- `scripts/probe_dynamic_tool_turn.py`：发一轮工具请求，用**同一个 provider call ID** 回传一个假的工具结果，再验证第二轮能否解码。它同时检查 call ID 是否缺失、重复、跨轮复用。

结论是否通过直接决定能不能跑 P2 模型抽取基准。

### 1.3 MCP 工具服务

MCP 是第三方提供的**工具进程**，问题不在协议本身，而在"它是个外部可执行程序"：

| 问题 | 处置 |
|---|---|
| 版本漂移：`npx` 每次都可能拉到新实现 | 配置里固定 `@playwright/mcp@0.0.80` |
| 环境缺失：Node < 18、`npx` 不在 PATH | `resolved_command()` 用 `shutil.which` 检查，找不到就抛 `MCPConfigError` |
| 工具面过大 | 只注册 6 个只读工具；`browser_run_code_unsafe`、上传、填表、点击**不注册给研究 Agent** |
| 远端返回 `isError` | 转成本地 `MCPClientError` 并带出远端文本，**不当成功** |
| 调用/读取超时 | `read_timeout_seconds`（默认 60s，上限 600s），连接和调用两侧都传 |
| 凭据注入 | `environment_from` 声明"子进程变量 ← 父进程变量"，缺变量直接报错；配置本身不存凭据值 |
| 登录态 | 默认无头隔离浏览器，不复用登录态；需要登录的站点当前不支持，代码里写明 |
| 审计隐私 | 审计只存参数哈希、结果哈希、工具名、耗时、错误状态，不存凭据和网页正文 |
| 时间语义 | 网页和目标文件都没有发布时间时采集失败，**不拿抓取时间冒充发布时间** |

MCP 到研究工具的桥接也是收窄的：`tool_adapter.py` 只暴露一个高层只读调用 `browser_collect_news_page`，而不是把任意浏览器操作工具直接给模型。

### 1.4 外部数据源

| 问题 | 处置 |
|---|---|
| 凭据缺失 | 报错时**列出缺失的具体环境变量名**，不是笼统的"数据库连接失败" |
| 连接卡死 | `connect_timeout=10`、`read_timeout=120`、`write_timeout=10` |
| 误写生产库 | 会话级 `SET SESSION TRANSACTION READ ONLY` + `START TRANSACTION READ ONLY`；`_query_frame()` 只放行 `^SELECT` |
| 表名是猜的 | 先查 `information_schema` 确认表和列存在，再取数 |
| 接口契约未取得 | 外部新闻 API 是 **fail-closed 占位适配器**：调用即抛"提供样例响应和字段映射后才能启用"，不用假数据糊过去 |
| 数据可用时点 | 与端点无关但同样会让评估失败：历史价格 96 个点里复现当前提前量时只有 1 个可见，结论是 `not_evaluable`，**不报成模型表现差** |

---

## 2. 三个可追溯的真实事故

| 事故 | 表现 | 根因 | 修复 |
|---|---|---|---|
| 端点 404 | `custom + chat + vertexai:gemini-3.8-flash + prompt_json` 全链路 404 | 模型名前缀走错了协议路径 | 修正配置；此后确认代理会丢弃 `tools`，兼容模式改为**同时**传函数白名单 |
| 函数选择没走兼容模式 | 结构化输出探针 1/1 通过，但 P1 方案生成不了 | 兼容模式只接在 `invoke_structured` 上，`invoke_tool_calls` 没接 | 把 `prompt_json` 扩展到函数选择，保留本地 Schema、白名单、编译器和审批 |
| HTTP 500 中止全批 | 一次 500 让整个长文本 benchmark 停止 | 瞬时传输失败和全局配置失败被归为同一异常 | 新增 invocation-scoped transient failure；408/429/5xx/超时按篇隔离，认证/配置/协议仍全局停止 |

---

## 3. 可复用的原则

1. **核心协议 fail closed，可选参数才降级。** 判断标准是"这个字段缺失后，系统还能不能负责地回答用户"。
2. **只降级系统自己加的参数。** 用户显式配置的意图不能被静默丢弃。
3. **拒绝请求 ≠ 瞬时故障 ≠ 认证问题。** 三者的重试策略和影响范围不同；混在一起会导致"小故障中止全批"或"配置错误反复重试"。
4. **兼容层必须做本地严格校验，并如实标注校验发生在哪一侧**（`schema_enforcement=local`）。
5. **失败不得伪装成成功或业务结论。** 截断、空回复、缺 call ID、未授权函数名，全部当失败处理。
6. **能力探测要用不可预测的输入。** 随机字段名，否则测的是模型的猜题能力。
7. **外部依赖的版本、超时、工具面、凭据来源都要显式写进配置**，不要依赖默认值或"最新版"。
8. **错误文案分两层**：给人看的一致简短，给审计看的完整详细。

---

## 4. 仍未解决 / 待确认

- 端点真实 usage / 货币成本不可用，网关不返回可靠 usage，性能与成本只能用品代指标；
- 自定义端点的线程安全、限流和并发计费口径未验证，并发始终默认关闭；
- MCP 需要登录态的站点尚无方案；
- 外部新闻 API 的样例响应、分页、修订语义、稳定 ID、许可与留存限制尚未取得，占位适配器仍不可用；
- 供应商兼容性未单独量化，三套长文本方案共用同一 `ModelGateway`，差异未拆分测量。
