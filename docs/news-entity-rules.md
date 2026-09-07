# P2 实体来源、类别和组合表达

| 字段 | 内容 |
|---|---|
| 文档类型 | P2 当前实体契约与验收说明 |
| 版本 | schema/Prompt `1.3.0`；实体解析快照 `2.0.0` |
| 状态入口 | [P2 进展状态](phase2-status.md) |
| 配套实现 | [结构化模型新闻抽取](structured-news-extraction.md)、[真实来源抽取基准](real-news-extraction-benchmark.md) |

新事件保存 `entity_resolution.version=2.0.0`。
实体解析由本地代码执行，不要求模型输出规范化键或来源标签。

## 保存的含义

每个新 `EventRecord` 的 `entity_resolution` 保存以下内容，并传播到 `MergedEvent` 与
`EventStateRevision`。使用一个完整快照，避免合并时让实体键与来源记录脱离。

| 字段 | 含义与证据 |
|---|---|
| `market_scope` | 输入标签的候选市场范围；允许多个值。每项记录规范化值、原标签、`source_type=market_tag`、文档版本和 `market_tags[index]`。 |
| `mentioned_regions` | 标题或正文确实出现的区域；保存原词、规范化键、规则和原始提及引用。 |
| `affected_assets` | 具体资产的规范化名称与键；组合拆分结果标记为 `rule_derived / explicit_unit_list`。 |
| `asset_groups` | AGRs、coal fleet、wind generation 等集合；受控别名映射为类别键，其他明确放入集合字段的原文表达保留原词键。 |
| `raw_entity_mentions` | 真实原文、标题/正文字段、字符偏移、文档版本与稳定提及 ID；原文来源为 `source_text`。 |

规范化结果的 `source_type=rule_derived` 表示键或名称经过规则处理；它引用的原始提及仍是
`source_text`。标签来源不使用标题或正文引文冒充。来源完整性检查同时验证原文切片和标签位置。
无法安全解析的候选进入隔离，已验证的实体引文保存在隔离记录中。

兼容字段 `affected_regions` 只保留原文明示的区域措辞；`affected_assets` 保存具体资产名称。
运行时代码可通过事件的 `market_scope`、`mentioned_regions`、`asset_groups`、`raw_entity_mentions`
属性访问解析快照；JSON 中这些内容保存在 `entity_resolution` 内。

## 三项规则

区域先验证原词，再用受控词表确定规范化键。模型把输入标签写入区域列表而原文没有提及时，
本地只保存标签来源，不生成区域引文。标签和明确区域冲突、标签存在多个市场却没有明确事件范围，
或者跨市场总容量不能分配时，事件保留在事件表中，并说明未进入目标市场电价分析的原因；
容量特征使用相同市场规则。

已知集合即使被模型放进资产列表，也确定性移入 `asset_groups`。未知而明显属于集合的表达
不能作为具体资产通过。没有名称的“7 AGRs”不会展开为七个资产；数字保留在原始提及中。

组合拆分只接受带电厂/电站名称的明确编号枚举，例如：

- `Alpha Power Station Generating Units 1 and 2`；
- `Delta Power Plant Units 1, 2 and 3`；
- `松林电厂1、2号机组`。

两个子资产引用同一个完整组合提及。模型已分别输出子资产时，同样从原始组合表达校验每个编号；
只输出一个子资产不会自动补齐其他子资产。范围、二选一、缺少父电厂、重复编号均不猜测拆分。
`Unit 1` 不能由 `Unit 10` 或引文中其他位置的数字支持。

拆分只改变资产集合。R06 仍为一个事件、两个机组、总损失 870 MW；不复制容量，也不推算各机容量。

## 身份、一致性和回放

三趟抽取先做上述规范化，再检查区域、市场范围、类别键和具体资产与事件的对应关系。
所有趟必须一致，不再以 2:1 的多数票选择不同资产。

类别键参与新事件身份。没有具体资产且没有明确开始时刻的事件增加来源谱系约束，避免
“同区域、同类型、无时刻”的不同长期事件自动合并；不同来源的这类事件保守保持独立。
有具体对象和时刻的明确转载仍使用规范化身份去重。恢复事件不得跨市场结束另一市场的停运状态。

旧事件缺少 `entity_resolution` 时保持原来的实体键和事件 ID。新记录保存解析版本与规范化键，
历史快照只使用当时可见的记录；合并后的每一版状态都保留当时的实体来源。重抽取属于新版本处理，
不能把新规则结果静默覆盖为旧结果。验收要求是：未受影响样本不变，预期修正可解释，固定版本
历史回放可复现；不要求已证实错误的旧分类或旧合并数量永久不变。

## 验证

`tests/research/test_entity_resolution_rules.py` 包含五类各 10 条独立、人工写定预期键的合成规则样本，
并覆盖误拆、虚构编号、三趟身份、容量、市场冲突、来源伪造、旧记录读取和修订回放。
它们验证规则实现，不等于真实模型盲测准确率。

开发集 R05/R06 的金标准增加独立的区域键、资产键、市场键和类别键；R06 的原文明示区域改为空。
新闻原文与保留集金标准保持不变。基准报告新增规范化键、市场/类别匹配结果和事件 ID，供多次运行
核对。完整准确率与证据完整性门槛仍为 100%。

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m ruff check app tests scripts
.\venv\Scripts\python.exe scripts/benchmark_real_news_extraction.py --extractor structured-model --extraction-passes 3 --require-qualified
```

真实模型结果受实际端点配置约束。端点调用失败只能记录为验证未完成，不能算作准确率通过。
2026-09-06 的开发集、保留集和端点结果统一记录在[真实来源新闻抽取基准](real-news-extraction-benchmark.md)，
避免在实体契约中重复维护易过期的运行数字。
