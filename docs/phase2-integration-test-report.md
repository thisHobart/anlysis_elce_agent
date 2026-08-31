# P2 新闻模块集成测试与问题报告

| 字段 | 内容 |
|---|---|
| 测试日期 | 2026-08-31 |
| 测试对象 | `app/research/news/` 完整链路 |
| 输入 | 10 篇合成新闻、10 篇真实来源短摘录、8 周合成价格、故障注入数据 |
| 既有基线 | P2 59 项测试通过 |
| 新增诊断 | 8 类问题、10 个故障注入回归、6 个结果质量验收 |
| 修复状态 | 8 类问题均已修复；原 `strict xfail` 已转为普通回归 |
| 结论 | 合成数据内部有效性通过；真实来源抽取基准不合格，当前规则抽取器不可用于真实新闻 |

## 1. 测试范围

本次按实际集成顺序运行：

```text
JSONL Adapter
→ NewsNormalizer
→ NewsVersionStore
→ Event Extractor
→ AsOfEventAssembler
→ MarketClock / PriceObservations
→ EventPriceAnalyzer
→ Event Features
→ Evidence Package / P1 CSV hand-off
```

先运行现有 P2 测试建立基线，再注入以下异常：互相矛盾的时钟、跨市场新闻、无结束时间的恢复事件、价格缺口/错格/NaN、修订事件时间、无证据抽取、非整点时区以及与真实事件重叠的安慰剂窗口。

## 2. 基线结果

- Ruff：通过。
- 项目虚拟环境当前全量回归：394 passed（131.28s；包含 Agent 目标验证、真实来源基准与结构化模型抽取安全测试）。
- 现有 P2 范围：59 passed。
- 正常夹具能够形成事件视图、价格分析、事件特征、证据包和 P1 可读取的 CSV。
- 同一输入重复运行的包指纹一致。
- 10 个故障注入回归全部通过，分别覆盖下面 8 类问题。
- 新增 9 项运行后质量门禁：单一时钟、市场覆盖、价格网格、事件身份、统计合法性、证据链、特征可获得性、事件去向和指纹。
- 新增 6 个结果质量测试，验证正向注入、负对照、统计范围、点事件生命周期、报告证据和质量门禁的反例识别。
- 真实来源开发集：10 条中仅无关负样本通过；可抽取事件召回率 0%、应隔离新闻召回率 0%、`unknown` 比率 100%，总体 `not qualified`。详见[真实来源新闻抽取基准](real-news-extraction-benchmark.md)。

## 3. 发现的问题

### P2-INT-001：同一次研究可以同时使用两套市场时钟（高，已修复）

**位置**：`app/research/news/pipeline.py:77`、`app/research/news/pipeline.py:94`、`app/research/news/pipeline.py:96`

修复前，`clock` 参数控制抽取器和事件特征，但价格分析始终使用 `prices.clock`。现在入口要求研究时钟与价格时钟完全一致，否则抛出 `NewsPipelineError`。

**影响**：报告和特征表看似都成功，但时间格、市场和统计结果不是同一研究口径。

**建议**：入口只允许一套权威时钟；如允许显式 `clock`，必须与 `prices.clock` 的市场、时区、间隔和版本完全一致，否则 fail closed。

### P2-INT-002：新闻市场标签没有与价格市场核对（高，已修复）

**位置**：`app/research/news/contracts.py:54`、`app/research/news/normalization.py:91`、`app/research/news/pipeline.py`

修复前，`market_tags` 只保留不校验。现在每个新闻版本必须包含权威价格市场；缺标签或跨市场输入在进入版本库前失败。

**影响**：真实接入后可能把一个区域的新闻解释为另一个市场的价格驱动，且不会产生错误提示。

**建议**：在版本库或事件抽取前建立市场准入门禁；明确多市场新闻、无标签新闻和市场映射表的处理规则。

### P2-INT-003：无结束时间的恢复事件会永久保持活跃（高，已修复）

**位置**：`app/research/news/features.py:34-43`

修复前，`generation_restore` 持续 2316 个结算区间。现在它是单区间点事件，脉冲锚定在 `max(事件生效时间, 新闻可用时间)` 后首个可决策区间，既不前视也不丢失晚到公告。

**影响**：P1 接收到的事件特征长期带有恢复事件、350 MW 和下行方向，污染后续关系或预测分析。

**建议**：按事件类型定义生命周期。恢复、公告等点事件应采用单区间脉冲或明确衰减；持续状态事件缺少结束时间时应拒绝导出或要求上限，而不是默认无限期。

### P2-INT-004：价格契约接受缺口、错格时间和非有限数值（高，已修复）

**位置**：`app/research/news/prices.py:36-48`

修复前只检查首尾时间。现在所有时间统一为 UTC，并逐点校验落格、唯一、严格连续；所有价格必须通过 `math.isfinite`。以下三类输入均被拒绝：

- `00:00, 00:30, 01:30`，中间缺一个结算区间；
- `00:00, 00:31, 01:00`，中间时间戳不在 30 分钟格上；
- 价格中包含 `NaN`。

**影响**：`index_of` 假设等间隔位置，缺口会导致索引错位；NaN 会进入基线、p 值和哈希，使结论失真或不可移植。

**建议**：`PriceObservations.__post_init__` 校验所有时间戳带时区、全部落格、相邻差严格等于结算间隔，并用 `math.isfinite` 拒绝 NaN/Inf。

### P2-INT-005：更正事件时间会把同一事件拆成两个事件（高，已修复）

**位置**：`app/research/news/extraction.py:340-343`

事件抽取身份仍可包含开始时间，但合并层现在同时使用“抽取 event ID”和“来源文档谱系 + 事件类型”建立连通分组。时间更正后保持最早 event ID，最终视图只有一个事件、两个版本并采用最新时间。

**影响**：时间恰恰是最可能被更正的字段之一；一旦更正，去重、事件数量、特征和统计检验都会重复。

**建议**：把“事件谱系身份”和“可更正字段”分开。优先使用来源事件 ID；没有来源 ID 时，使用稳定文档谱系/资产/事件类别建立候选身份，再由合并层处理时间修订和冲突。

### P2-INT-006：证据覆盖率可以显示 100%，但所有结论都不可追溯（中高，已修复）

**位置**：`app/research/news/evidence.py:247-252`

质量报告现在按每个事件已填充的关键字段逐项检查 `EvidenceSpan`。移除全部文本证据后，覆盖率为 0%，运行后质量评估明确失败 `complete_evidence_chain`。

**影响**：质量摘要与实际证据链互相矛盾，可能让报告在缺少原文依据时被误判为完整。

**建议**：按事件类型定义关键字段集合，逐字段核对最终定值版本的 evidence span；存在不可追溯结论时，证据包生成应失败或显式降级，不能只提供辅助查询方法。

### P2-INT-007：市场网格按 UTC 午夜锚定，不是市场本地午夜（中，已修复）

**位置**：`app/research/news/clock.py:55-61`

`floor` 现在先转换到市场本地时钟，按本地当日分钟取整，再转换回 UTC。`Asia/Kathmandu` 60 分钟边界及原 UTC 市场均通过回归。

**影响**：当时区偏移不能被结算间隔整除时，所有窗口和特征格都会错位。对当前 UTC 夹具没有影响，但 `MarketClock` 的通用契约不成立。

**建议**：在市场本地时间上取整后再转换为 UTC；同时增加 DST、半小时/45 分钟时区回归。若产品只支持少数市场，也应在配置层显式限制，而不是静默错位。

### P2-INT-008：安慰剂窗口可能包含另一个真实事件（高，已修复）

**位置**：`app/research/news/analysis.py:540-556`

安慰剂窗口现在复用控制池的 occupied mask；任何与已知事件重叠的偏移窗口均跳过。故障注入中的 `+100` 真实事件不再进入安慰剂分布。

**影响**：真实事件被当成“无事件噪声”，会把稳定关联降级为不稳定或相反结论。

**建议**：安慰剂窗口复用与 control pool 相同的 occupied mask；任何区间与已知事件重叠就跳过，并在样本不足时返回明确限制。

## 4. 已知限制，但不是本次新发现的回归

- 外部 API 仍是 fail-closed 占位符。
- 规则抽取器只适合显式合成文本，真实中文新闻、中文资产名、万千瓦和模糊时段尚不能可靠处理。
- `capacity_mw` 同时承载容量、出力和需求变化，字段语义需要拆分。
- P2 尚未接入 Graph、桌面流程和报告落盘。
- `fuel_supply_change` 规则没有夹具覆盖。

这些限制决定 P2 还不是生产能力，但与上面 8 个“当前实现会静默给出错误或矛盾结果”的问题应分开管理。

## 5. 已执行的修复顺序

1. **先封住错误输入**：P2-INT-001、002、004。
2. **再修事件语义**：P2-INT-003、005。
3. **再修统计与证据门禁**：P2-INT-008、006。
4. **最后完善通用市场时钟**：P2-INT-007；若生产市场先确定，可按目标市场风险调整优先级。

全部 `xfail` 标记已删除，10 个探针现为普通回归。合成数据结果质量仍限定为 `synthetic_p2_internal_validity`；新增真实来源开发集已经证明当前规则抽取器缺少外部有效性，但尚未验证真实事件—价格因果效应。

## 6. 复现命令

现有 P2 基线：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/research/test_news_pipeline_integration.py `
  tests/research/test_news_research_questions.py `
  tests/research/test_news_normalization_and_extraction.py `
  tests/research/test_import_boundaries.py
```

问题探针：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/research/test_news_integration_regressions.py `
  tests/research/test_news_result_quality.py
```

回归文件：[P2 integration regressions](../tests/research/test_news_integration_regressions.py)。结果质量文件：[P2 result quality](../tests/research/test_news_result_quality.py)。
