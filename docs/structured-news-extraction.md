# 结构化模型新闻抽取实现说明

## 当前状态

已完成“结构化模型抽取 + 确定性校验 + 安全隔离”的代码边界。现有规则抽取器继续作为合成夹具回归基线；
模型抽取器必须由调用方显式注入，不会因为本次修改而自动访问外部模型。

这表示模型接入能力已经具备，但**不表示任何真实模型已经通过准确率门禁**。测试中的脚本化模型响应只验证
结构化输出能否被正确转换、拒绝和追踪，不能代替独立盲测。

## 处理链路

```text
NewsDocument
→ 原生结构化模型输出
→ 原文证据定位
→ 时间、单位、语义和置信度校验
→ EventRecord / ExtractionQuarantine
→ as_of 合并
→ 现有价格分析和证据包
```

主要实现位于：

- `app/research/news/model_extraction.py`：Prompt、模型输出 schema、转换和校验；
- `app/research/news/contracts.py`：时间精度、事件状态、物理影响、量值语义及抽取追踪；
- `app/research/news/merging.py`：多事件合并、模型结果复用和追踪传播；
- `app/research/news/evidence.py`：同文多事件证据隔离及结论级抽取哈希；
- `tests/research/test_structured_news_extraction.py`：真实开发集和失败注入测试。

## 新增契约

### 时间

- 区分 `instant/hour/day/month/range/vague/unknown`；
- 区分原文绝对时间、原文日期与时刻组合、基于发布时间解析的相对时间；
- 日期级或模糊的短期事件不能伪造成午夜时刻，必须进入隔离区。

### 量值

量值保留原始数值、单位、原文、MW 换算值和语义。当前支持 `kW`、`MW`、`GW`、`万千瓦`、
`亿千瓦`。容量水平、需求水平和出力水平不会写入兼容字段 `capacity_mw`；只有变化量、损失量和恢复量
可以进入下游容量特征。

### 事件语义

抽取层记录事件状态和物理影响。价格方向由物理影响确定性映射得到，不要求模型直接预测价格涨跌。

### 可复现性

每个模型事件记录：

- 模型名；
- Prompt 版本；
- 输出 schema 版本；
- 模型输入哈希；
- 结构化输出哈希。

这些字段会传播到合并事件、结论证据链和事件指纹。

## 安全门禁

以下情况不会生成可分析事件：

- 模型协议或 schema 无效；
- 引文不是标题或正文的真实子串；
- 必填字段缺少原文证据；
- 短期事件缺少精确开始时刻；
- 数量语义为未知；
- 置信度低于门槛；
- 同文事件无法形成不同的稳定身份；
- 抽取器时区与价格市场时钟不一致。

新闻正文按不可信输入处理，正文中的提示指令不得改变抽取规则。

## 运行方式

规则基线仍是默认值：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_real_news_extraction.py
```

只有显式指定时才使用已配置的结构化模型：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_real_news_extraction.py `
  --extractor structured-model `
  --require-qualified
```

专项测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/research/test_structured_news_extraction.py `
  tests/research/test_real_news_extraction_benchmark.py
```

## 尚未完成

1. 选择并固定用于新闻抽取的真实模型；
2. 对真实模型运行当前 10 条开发集，记录逐案结果；
3. 建立不参与 Prompt 调整的真实新闻盲测集；
4. 检查至少五次重复运行的一致性、成本和延迟；
5. 达到门槛后再将模型抽取器设为生产默认值；
6. 接入外部新闻 API 的分页、修订和许可语义。

