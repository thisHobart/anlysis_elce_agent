---
name: price-forecastability-audit
description: Audit whether an electricity-price series is forecastable at all, using only the target series. Use when the question is about price shape, spikes and negative prices, volatility regimes, stationarity and differencing, trend and seasonal structure, autoregressive memory, how hard the series is to predict, what naive benchmark a model has to beat, or when no exogenous driver data has been loaded yet.
metadata:
  version: "1.2.1"
  domain: eda
  protocol-file: references/research-protocol.yaml
allowed-tools: data_quality price_descriptive_distribution price_duration_curve price_tukey_outer_fence price_spike_regime_profile price_calendar_group_profile price_rolling_mean_std price_stationarity_tests price_seasonal_decomposition price_lag_autocorrelation price_partial_autocorrelation price_segment_distribution_comparison price_variance_stabilization_check price_naive_baseline_benchmark
---

# 电价可预测性审计

只用目标电价序列，给出两个结论：这条序列的可预测性属于哪一档，后续模型必须超越的误差底线是多少。结论只描述序列自身难度，不推荐模型，也不产生预测值。

## 假设的写法

每条假设都必须能被**本轮选择的函数**判定，否则本轮无法收敛，会被退回要求改写。写不出对应函数的猜想不要列入；一条假设只对应一个判据。

## 裁决优先级

- **预处理 > 定级**：平稳性 verdict 为 `unit_root`，或 `recommended_transform` 为 `asinh_median_mad` 时，先给预处理结论，再定难度档。
- **多指标 > 单指标**：不得只凭 `seasonal_strength` 或只凭偏自相关定级，三项证据（季节强度、记忆结构、朴素基线）必须同时引用。
- **尖峰聚集 > 平均误差**：`clustering.persistence_ratio` 明显大于 1 时，朴素基线的平均误差低估真实难度，定级要下调并写明原因。
- **样本量 > 结论强度**：有效样本不足两个完整日周期时不做季节分解，改记限制，不用短样本硬给周期结论。

## 平稳性判读

| verdict | ADF / KPSS | 结论与下一步 |
|---|---|---|
| `stationary` | 拒绝 / 不拒绝 | 可用水平值建模 |
| `unit_root` | 不拒绝 / 拒绝 | 存在单位根，先差分再谈记忆结构 |
| `trend_or_break_suspected` | 拒绝 / 拒绝 | 趋势项或结构突变，用 `trend_strength` 或 `price_segment_distribution_comparison` 定位 |
| `inconclusive` | 不拒绝 / 不拒绝 | 样本偏短或噪声偏大，记为限制，不在两者中硬选一个 |

`first_difference` 仍非平稳时不继续加阶，直接报告限制。

## 形态与极端状态

- `price_descriptive_distribution` 先做，为后续所有阈值提供量纲与分位基准。`skewness` 绝对值 ≥ 0.5 或 `kurtosis`（超额峰度）≥ 1.0 即认定明显偏斜或厚尾。
- 集中度看 `price_duration_curve`：少数时段贡献大部分价格水平时，平均误差指标不足以刻画难度。
- 极端阈值取 `price_tukey_outer_fence` 的 outer fence，除非用户明确给出领域阈值；只标记，不删除。
- `price_spike_regime_profile` 判断尖峰是孤立还是成片：`persistence_ratio` 大于 1 表示尖峰有聚集性；`concentration.top_hours` 与 `top_months` 说明风险集中在哪些时段；`negative_share` 非零时单独说明负价。

## 动态与记忆

- `price_calendar_group_profile` 确认日内、周内与月度结构是否稳定。没有分解强度时，用分组自身的组间方差占比判断：某个分组解释了 ≥ 6% 的电价方差即认定存在该层结构。
- `price_rolling_mean_std` 判断波动是否分阶段切换：`rolling_statistics.one_day` 的 `maximum_std` 与 `minimum_std` 比值 ≥ 2 即认定分阶段。波动分段切换会让全样本统计量失去代表性。
- `price_seasonal_decomposition`：`seasonal_strength` ≥ 0.6 为强周期结构，0.3–0.6 为中等，< 0.3 视为无稳定周期结构；同时读 `variance_share.remainder`，残差占比过半时季节结论一律降级表述。
- `price_lag_autocorrelation` 看整体持续性与日／周滞后衰减形状；`price_partial_autocorrelation` 看直接记忆，取 `suggested_autoregressive_order` 与 `significance_band`。偏自相关一阶后即落入置信带说明接近随机游走。
- Ljung-Box 各阶均不拒绝时序列接近白噪声，此时不要再用衰减形状讲"记忆"。

## 可预测性定级

按季节强度、记忆结构和朴素基线三项联合定级：

- **结构清晰**：日或周 `seasonal_strength` ≥ 0.6，偏自相关在有限阶后落入置信带，且 `best_baseline` 为 `daily_naive` 或 `weekly_naive`——序列有可利用的周期结构。
- **以持续性为主**：`best_baseline` 为 `persistence` 且日／周 naive 的 `mae_ratio_to_persistence` 均大于 1，偏自相关一阶后迅速衰减——接近随机游走，模型增益空间有限，必须在结论中写明。
- **难度高**：`seasonal_strength` < 0.3 且残差方差占比过半，或尖峰成片聚集，或 verdict 为 `trend_or_break_suspected`。
- **需先预处理**：verdict 为 `unit_root`，或 `recommended_transform` 为 `asinh_median_mad`。

误差底线一律引用 `best_baseline_label` 与 `best_mae` 原值，并附全样本计算的限制。`near_zero_share` ≥ 1% 时不得使用 MAPE 一类百分比误差。

## 提案约束

- 只选回答当前问题所需的最小函数集合。
- 每个函数每个方案最多调用一次；多滞后合并为一个最大 `max_lag`，多子样本合并为一次 `segments`。
- `selected_variables` 始终为空列表，本 Skill 不涉及外生变量函数。
- `segments` 的小时、月份和时间边界必须由用户明确给出。
- 负价与尖峰保留并标注，不截尾、不删除、不插补。
- 平稳性、季节分解与偏自相关只在方法内部为连续性做时间插值，并报告 `interpolated_share_*`。
- 朴素基线为全样本计算，只作误差量级参考；正式实验须改为按时间顺序滚动重估。
