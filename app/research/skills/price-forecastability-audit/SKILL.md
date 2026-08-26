---
name: price-forecastability-audit
description: Audit whether an electricity-price series is forecastable at all, using only the target series. Use when the question is about price shape, spikes and negative prices, volatility regimes, stationarity and differencing, trend and seasonal structure, autoregressive memory, how hard the series is to predict, what naive benchmark a model has to beat, or when no exogenous driver data has been loaded yet.
metadata:
  version: "1.0.0"
  domain: eda
  protocol-file: references/research-protocol.yaml
allowed-tools: data_quality price_descriptive_distribution price_duration_curve price_tukey_outer_fence price_spike_regime_profile price_calendar_group_profile price_rolling_mean_std price_stationarity_tests price_seasonal_decomposition price_lag_autocorrelation price_partial_autocorrelation price_segment_distribution_comparison price_variance_stabilization_check price_naive_baseline_benchmark
---

# 电价可预测性审计 Skill

只使用目标电价序列，回答一个问题：**这条价格序列到底有多可预测，后续建模的起点和底线在哪里。**

适用场景：还没有外生变量数据、外生变量质量不过关，或者用户先想弄清价格自身的规律与难度。本 Skill 不涉及任何外生变量函数。

## 研究顺序

唯一顺序由本地 `research_protocol` 固定，共五个阶段：

1. **市场时钟**：明确市场产品、粒度、时区、预测起点与预测范围；缺失即标记为限制。
2. **数据体检**：时间轴、覆盖率、缺失与重复，确认样本足以支撑时序结论。
3. **价格形态**：水平与分布、持续曲线、极端值阈值、尖峰与负价状态，以及用户明确指定的分段对比。
4. **价格动态**：滚动波动、平稳性、趋势与日周季节分解、自相关与偏自相关记忆。
5. **建模就绪**：方差稳定变换建议与朴素基线误差底线。

## 选择原则

- 只选回答当前问题所需的最小函数集合。
- 每个函数每个计划最多调用一次；多滞后合并为一个最大 `max_lag`，多个子样本合并为一次 `segments`。
- `data_quality` 由编译器强制加入。
- `selected_variables` 始终为空列表——本 Skill 不使用外生变量。
- `segments` 的小时、月份和时间边界必须来自用户或研究配置的明确值。

## 判定口径

- **可预测性高**：季节强度高、偏自相关有清晰有限阶结构、朴素基线误差相对价格量级较小。
- **可预测性受限**：残差方差占比高、尖峰频繁且聚集、平稳性结论冲突。
- **需先做预处理**：存在单位根、厚尾明显或方差稳定检查建议 asinh 变换。

以上判定只描述序列自身难度，不构成任何模型选择结论。

## 方法边界

- 负价与尖峰保留并标注，不截尾、不删除、不插补。
- 平稳性、季节分解与 PACF 需要连续序列，函数内部只为这些方法做时间插值并报告插值比例。
- 朴素基线在全样本上计算，只作为误差底线的量级参考；正式实验必须改为按时间顺序滚动重估。
- 本 Skill 不产生任何预测值，也不比较真实模型。

外部依据见 [../price-exogenous-eda/references/methodology.md](../price-exogenous-eda/references/methodology.md) 中记录的公开工作。
