# 结构化模型新闻抽取实现说明

## 当前状态

已完成“结构化模型抽取 + 确定性校验 + 安全隔离”的代码边界。现有规则抽取器继续作为合成夹具回归基线；
模型抽取器必须由调用方显式注入，不会因为本次修改而自动访问外部模型。

2026-09-03，`vertexai:gemini-3.8-flash` 经 Cherry `prompt_json` 路径在 10 条真实来源开发集上完成过运行，逐案完整准确率为 80%。2026-09-06 复核确认旧准入规则过宽：完整字段准确率和证据完整性没有阻止 `qualified`。当前 schema/prompt 已更新到 1.2.0，旧运行只证明调用链可用，不再作为合格结论；必须重新运行开发集与盲测集。

## 处理链路

```text
NewsDocument
→ 原生结构化输出，或 Cherry 提示约束 JSON
→ 原文证据定位
→ 字段值与原文、时间、单位、语义和置信度校验
→ 有效 EventRecord + 文档级或候选级 ExtractionQuarantine
→ 带更正状态时间线的 as_of 合并
→ 现有价格分析和证据包
```

主要实现位于：

- `app/research/news/model_extraction.py`：Prompt、模型输出 schema、转换和校验；
- `app/research/news/contracts.py`：时间精度、事件状态、物理影响、量值语义及抽取追踪；
- `app/research/news/merging.py`：多事件合并、模型结果复用和追踪传播；
- `app/research/news/evidence.py`：同文多事件证据隔离及结论级抽取哈希；
- `tests/research/test_structured_news_extraction.py`：真实开发集和失败注入测试。

## 新增契约

### 时间（schema 1.1.0 起重构）

模型不再直接返回 `datetime`，而是返回原文字符串，`datetime` 一律由本地代码构造——与“单位换算不交给模型”同一原则。这样“把只有日期的时间塞进时刻字段”从结构上不可能发生：

- `time_precision` 必填：`instant/hour/day/month/range/vague/unknown`；
- `time_text` 保留原文时间短语（如 `5月24日`、`0856 hrs`）；
- `event_instant` 只允许在 `instant/hour` 精度下出现，携带带时区偏移的完整 ISO-8601 字符串和 `basis`（原文绝对时间 / 原文日期与时刻组合 / 基于发布时间解析的相对时间），由本地严格解析：只有日期、缺时区偏移或组合时间未用市场当日偏移，都会进入隔离区而不是让 schema 崩溃；
- 只有日期、只有月份或模糊时段时 `event_instant` 必须为空；日期级短期事件因此进入隔离区，不会被伪造成午夜时刻。
- schema 1.2.0 起，本地还会把模型生成的标准时间反向核对原文日期、钟点和市场时区；格式正确但与原文不同的时间同样会被拒绝。

默认执行 **3 趟独立抽取**（`extraction_passes`）：端点在 `temperature=0` 下不可复现，一趟分不清
“有把握”和“碰巧”，三趟能。趟间结论不一致即隔离待复核，不做并集合并——并集会提高召回，却让某一趟的
臆造事件不经复核直接进表。代价是每篇 3 倍模型调用，基准报告里记录实际调用次数与耗时。

模型响应结构化校验失败时，抽取器会**把校验错误回传给模型让它自改一次**（`max_repair_attempts`，默认 1）：这类错误多为机械错误（枚举写错、漏必填字段、把日期塞进时刻），错误信息本身就点名了问题，一次纠正通常即可救回；重试只是再问一遍，绝不放宽 schema。重试用尽仍不合规时，该篇被隔离为 `model_response_invalid`，不再中断整批——端点级故障（配置/协议/思考策略）仍然快速失败，不进入重试。

### 量值

量值保留原始数值、单位、原文、MW 换算值和语义。当前支持 `kW`、`MW`、`GW`、`万千瓦`、
`亿千瓦`。容量水平、需求水平和出力水平不会写入兼容字段 `capacity_mw`；只有变化量、损失量和恢复量
可以进入下游容量特征。**机组编号或序号（如“1号机组”“Units 1 and 2”）是资产名称，不是数量**：
数量原文必须带功率单位（含 `瓦` 或拉丁 `W`），且解析出的数值和单位换算结果必须与模型字段一致；否则按 `ambiguous_quantity` 隔离，避免把编号当成 1 台/1 MW，也避免原文 `870 MW` 被模型写成 `900 GW`。

### 实体规范化（受控词表）

模型对同一个电网可能写成 `辽宁`、`辽宁电网` 或 `辽宁省`，也可能这一趟把电网名填进 `affected_regions`、
下一趟同时填进 `affected_assets`。事件身份由「区域 + 资产 + 时刻 + 类型」算出，因此这种措辞漂移会让同一条
新闻在两次运行里变成两个事件，**去重与合并静默失效**。

`app/research/news/entities.py` 把「记录原文」和「决定身份」分开，与 EDC 式流水线同一思路：

- 抽取层照原文保存 `affected_regions` / `affected_assets`，逐字证据不受影响；
- `region_keys` / `asset_keys` 是规范化后的键：NFKC、去标点空白、折叠大小写，再剥掉 `电网`、`省`、`公司`
  等运营主体或行政后缀，最后查受控词表映射为区域码（`辽宁电网 → CN-LIAONING`、`Britain → GB`）；
- **词表不需要完整**：查不到时仍按剥后缀的确定性结果折叠，因此新区域也不会引入抖动；
- **区域与资产的归属由词表决定，而不是由模型决定**：凡能解析为已知区域的名称，无论模型填在哪一侧都算区域；
  词表不认识的名字（电厂、机组、线路）留在资产侧；
- 身份、合并谱系键和基准比分全部使用规范化键；金标准也走同一次规范化，避免答案文件变成字符串猜谜。
- 每个资产必须能在对应引文中找到原词或足够明确的组成词；区域也可由经过适配器确认的市场标签支持。只有一段真实引文、但引文并不支持模型填写的资产，仍按无效证据处理。

### 多趟抽取与一致性闸门

端点在 `temperature=0` 下依然不可复现（盲测 10 篇里 9 篇每次原始输出哈希都不同），单趟结果无法区分
「稳的判断」和「碰巧的判断」。`extraction_passes` 大于 1 时，同一篇独立跑多趟并要求结论一致：

- 处置类别（事件 / 无关 / 隔离）不一致 → 隔离为 `inconsistent_extraction`；
- 全部隔离 → 保留出现最多的真实原因，不用一致性标签掩盖；
- 全部产出事件但**规范化后**内容不一致 → 隔离为 `inconsistent_extraction`。

比较只看会影响下游结论的内容：事件类型、相关性、规范化实体键、时刻、容量、方向、状态与物理影响；资产集合与它所属的具体事件一起比较，不能把两个事件的资产交换后仍视为一致；
证据偏移、置信度和实体原文措辞在趟与趟之间本就会变，纳入比较等于把噪声当成分歧。

刻意**不采用并集合并**（LangExtract 式多趟提召回）：并集能提高召回，但会让某一趟的臆造事件无人复核直接进表，
与本系统其余闸门一致的取向相反。不一致就送复核，是把静默漏报变成显式待办。

### 事件语义

抽取层记录事件状态和物理影响。价格方向由物理影响确定性映射得到，不要求模型直接预测价格涨跌。

### 可复现性

每个模型事件记录：

- 模型名；
- Prompt 版本；
- 输出 schema 版本；
- 模型输入哈希；
- 结构化输出哈希。

这些字段会传播到合并事件与结论证据链，并汇入**来源指纹**（`provenance_hash`）。

它们**不进入结论指纹**。`output_hash` 是模型原始结构化输出的哈希，而所配端点在 `temperature=0` 下
也不可复现——盲测集 10 篇里有 9 篇每次运行的 `output_hash` 都不同。若把它折进结论指纹，
那么每次重新抽取都会得到新指纹，哪怕结论一字未变，指纹也就回答不了它唯一要回答的问题：
**结论变了吗？** 因此指纹一分为二：

| 指纹 | 覆盖 | 重新抽取时 |
|---|---|---|
| 结论指纹（`analysis_hash`、`event_hash`、`feature_hash`、`package_hash`、`price_hash`） | 规范化实体键、时间、量值、方向、事件 ID、来源正文哈希、引文、隔离清单 | 结论未变则**不变** |
| 来源指纹（`provenance_hash`） | 模型名、Prompt 版本、schema 版本、输入与输出哈希 | 模型输出变化时**会变**，属预期 |

实体以规范化键（`region_keys`/`asset_keys`）进入结论指纹，理由相同：同一个电网这次写作
`辽宁`、下次写作 `辽宁电网`，原文照记用于证据，身份则必须稳定。

## 确定性闸门（不依赖模型配合）

多趟一致性闸门只能抓住**不稳定**的错误：三趟结论不同就送复核。模型**稳定地**判错时，闸门看到的是
“一致”，照样放行。以下检查直接读取原文或候选结果，因此不依赖模型自行纠错：

- **拒绝存疑的“无关”**：`irrelevant` 是终局判定，既不产生事件也不进复核区。标题或正文若含功率量值
  （`100MW`、`4775万千瓦`，含源站把“千瓦”误写为“干瓦”的情况）或运行词汇（停运、跳闸、负荷、并网……），
  则拒绝该判定，改为隔离 `disputed_irrelevance`。词表刻意收窄：`电力`、`发电` 单独出现会误伤
  电力业务许可通报这类真无关新闻，因此不在其中。
- **重复事件提示**：日期数量本身不代表事件数量，成立日期、历史对比日期等背景信息不会再使整篇新闻被隔离。只有正文同时出现多个时间表达式和“再次、再创、先后、twice”等明确重复事件措辞，且模型没有全部认领时，系统才增加一个候选级 `ambiguous_multi_event` 复核项；已经验证的事件仍然保留。
- **部分成功**：一篇新闻有多个候选时逐个校验。某个候选时间不足或证据无效，只隔离该候选；其他通过校验的事件继续进入合并层。所有候选都失败时才形成文档级隔离。

这些规则已经进入确定性回归；真实模型仍需用 schema 1.2.0 重新跑开发集和盲测集验证整体效果。

## 安全门禁

以下情况不会生成可分析事件：

- 模型协议或 schema 无效（隔离为 `model_response_invalid`，整批继续）；
- 多趟结论不一致（`inconsistent_extraction`）；
- 判为无关但正文含功率量值或运行词汇（`disputed_irrelevance`）；
- 正文明确描述重复事件、但部分时间没有对应候选（候选级 `ambiguous_multi_event`）；
- 引文不是标题或正文的真实子串，或引文虽存在但不支持对应资产；
- 必填字段缺少原文证据（含 relevance 必须单独给证据）；
- 短期事件缺少精确开始时刻，或时间精度低于 `hour`；
- 数量语义为未知、数量原文没有功率单位，或模型数值/单位与原文不一致；
- 置信度低于门槛；
- 同文事件无法形成不同的稳定身份；
- 多趟独立抽取的结论不一致（`inconsistent_extraction`）；
- 抽取器时区与价格市场时钟不一致。

新闻正文按不可信输入处理，正文中的提示指令不得改变抽取规则。

## 端点兼容（结构化输出协议）

系统优先接受服务商**原生**结构化输出；只有显式选择 `prompt_json` 时才启用提示约束兼容路径。
`ModelGateway` 保持统一调用接口，但不同服务商由不同适配器负责协议转换：

- Gemini 使用 `ChatGoogleGenerativeAI` 直连 Google API，并固定使用原生 `json_schema`；
- DeepSeek、Qwen 和自定义 OpenAI 兼容端点使用 `ChatOpenAI`，按配置选择 `function_calling`、
  `json_schema` 或 `json_mode`；
- Gemini 原生路径不通过 Cherry Studio 的 OpenAI 兼容接口转发。该接口会接受 `response_format`，但不代表
  它会把 schema 转换成 Gemini 的 `responseSchema`；这种情况下模型会自行设计字段，随后在本地校验失败。

必须通过 Cherry Studio 调用 Gemini 时，显式选择 `prompt_json` 兼容模式。该模式把 Pydantic 生成的
完整 JSON Schema 附加到系统消息，通过 Cherry 的 JSON Object 模式获取响应，然后执行与原生路径
相同的 Pydantic 严格校验。模型输出不符合契约时只允许一次带错误反馈的修复重试，仍失败则隔离。
报告将其标记为 `schema_enforcement=local`，不得表述为服务商原生约束。

先运行协议探针：

```powershell
.\.venv\Scripts\python.exe scripts\probe_structured_output.py
```

探针每次生成一套随机字段名。原生模式使用它验证服务端是否收到 schema；`prompt_json` 模式使用它
验证模型能否读取提示中的 schema 并连续通过本地校验。默认连续验证三次，需要查看原始响应时增加
`--show-raw`。探针通过后再运行真实新闻基准，避免把格式遵循问题误判为新闻抽取准确率问题。

Gemini Developer API 在桌面“服务商”中选择“Gemini（原生 API）”，填写 Google API key 和
`gemini-*` 模型名，接口地址保持为空。Vertex AI 通过 `.env` 配置 `VPP_LLM_GOOGLE_VERTEXAI=true`、
项目和区域，并使用 Google 应用默认凭据（ADC）。

## 运行方式

规则基线仍是默认值：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_real_news_extraction.py
```

只有显式指定时才使用已配置的结构化模型：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_real_news_extraction.py `
  --extractor structured-model `
  --show-model-output `
  --require-qualified
```

三趟一致性是默认值；`--extraction-passes 1` 可退回单趟（更便宜，但会恢复原来的判定摆动）。

盲测集，并重复 3 次测量稳定性：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_real_news_extraction.py `
  --news tests\fixtures\news_holdout\source_news.jsonl `
  --gold tests\fixtures\news_holdout\gold_manifest.yaml `
  --output artifacts\acceptance\holdout `
  --extractor structured-model `
  --repeat 3
```

`--repeat` 会在报告里写入 `stability` 段：逐案通过次数、发生变化的字段，以及
`metric_stability_is_incidental`——当仍有案例在摆动、只是恰好互相抵消时该标志为真，
提醒不要把“三次指标相同”读成“已经稳定”。`cost` 段记录实际模型调用次数与每篇耗时。

专项测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/research/test_structured_news_extraction.py `
  tests/research/test_real_news_extraction_benchmark.py
```

## 尚未完成

1. 固定生产候选模型、Cherry 和 Prompt/schema 的精确版本；
2. 建立不参与 Prompt 调整的真实新闻盲测集；
3. 检查至少五次重复运行的一致性、成本和延迟；
4. 通过盲测后再将模型抽取器设为生产默认值；
5. 接入外部新闻 API 的分页、修订和许可语义。
