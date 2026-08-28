---
name: price-exogenous-eda
description: Build forecast-readiness evidence for electricity prices together with their exogenous drivers. Use when the question involves market-time alignment, price level and shape, spikes and negative prices, stationarity, seasonality, price memory, driver data quality, multicollinearity, contemporaneous or lagged price-driver relationships, nonlinear dependence, relationship stability, leakage risk, or which candidate features deserve a forecasting experiment.
metadata:
  version: "3.2.1"
  domain: eda
  protocol-file: references/research-protocol.yaml
allowed-tools: data_quality price_descriptive_distribution price_rolling_mean_std price_tukey_outer_fence price_lag_autocorrelation price_calendar_group_profile price_stationarity_tests price_seasonal_decomposition price_partial_autocorrelation price_spike_regime_profile price_duration_curve price_segment_distribution_comparison exogenous_descriptive_distribution exogenous_iqr_outliers exogenous_linear_index_trend exogenous_pearson_collinearity exogenous_variance_inflation exogenous_stationarity_tests relationship_scipy_pearson_pairwise relationship_scipy_spearman_pairwise relationship_pearson_positive_lead_scan relationship_pearson_by_hour relationship_pearson_by_month relationship_feature_quartile_response relationship_mutual_information_scan relationship_granger_causality_scan relationship_rolling_correlation_stability relationship_pearson_segment_comparison price_variance_stabilization_check price_naive_baseline_benchmark
---

# 电价与外生变量联合研究

判定候选驱动变量能否进入电价预测特征集。每个变量必须同时通过四道判据才能提升为预测候选：预测起点可获得、成对有效样本达标、关系在时段与时间上稳定、排除共同趋势造成的伪相关。任一项不满足即降级为描述性证据，并写明是哪一项没过。

## 假设的写法

每条假设都必须能被**本轮选择的函数**判定，否则本轮无法收敛，会被退回要求改写。

- 写得出对应函数的才写。「碳配额成本可能传导到批发市场」这类猜想没有对应的确定性检验，不要列入。
- 一条假设对应一个判据。不要把「电价有季节结构且负荷领先两小时」写成一条。
- 只想确认数据能不能用时，可以不提任何假设，只做数据可用性核验。

## 跨阶段裁决

结论冲突时按以下优先级取舍，不按效应量大小取舍。

- **可获得性 > 效应量**：`availability_type` 为 `observed_only` 或 `unknown` 的变量，即使 |r| 最高也只能进入泄漏风险，不得列为预测候选；`forecast` 但 `availability_timestamp_present` 为 false 时同样降级。
- **平稳性 > 相关性**：`price_stationarity_tests` 与 `exogenous_stationarity_tests` 的 verdict 同时为 `unit_root` 时，同期 Pearson 按伪相关处理，必须与平稳性结论一起给出，并说明需在一阶差分上重估。
- **稳定性 > 显著性**：`relationship_rolling_correlation_stability` 的 `stability` 为 `sign_unstable` 时，全样本相关不成立；为 `magnitude_unstable` 时只能写"方向一致、强度随时间变化"。
- **依赖 > 线性缺失**：Pearson 接近 0 而 `nonlinearity_flagged` 为 true 时，结论写"存在非线性依赖"，不写"无关系"。
- **样本量 > 扫描范围**：成对有效样本不足时缩小结论，不通过扩大 `max_lag` 或变量集来制造关系证据。

## 数据可用性核验

`data_quality` 每个方案强制执行一次，从中读三件事：

- 目标有效样本是否支撑时序结论。不足两个完整日周期时不做季节分解与偏自相关，改记限制。
- 每个变量的 `availability_type` 与 `median_availability_lag_hours`，据此确定该变量的候选资格，并写进方案假设。
- 覆盖缺口与重复时间戳的位置。缺口成片出现时，滚动统计和季节分解结论都要标注受影响区间。

## 电价结构证据

平稳性按 `price_stationarity_tests` 返回的 verdict 直接判读：

| verdict | ADF / KPSS | 结论与下一步 |
|---|---|---|
| `stationary` | 拒绝 / 不拒绝 | 可用水平值做关系分析与建模 |
| `unit_root` | 不拒绝 / 拒绝 | 存在单位根，关系证据须在一阶差分上重估 |
| `trend_or_break_suspected` | 拒绝 / 拒绝 | 趋势项或结构突变，用 `trend_strength` 或 `price_segment_distribution_comparison` 定位 |
| `inconclusive` | 不拒绝 / 不拒绝 | 样本偏短或噪声偏大，记为限制，不在两者中硬选一个 |

`first_difference` 仍非平稳时不继续加阶，直接报告限制。

分布形态：

- `distribution.skewness` 绝对值 ≥ 0.5，或 `distribution.kurtosis`（超额峰度）≥ 1.0，即认定明显偏斜或厚尾；两项都不到则视为接近对称。

波动状态：

- 取 `rolling_statistics.one_day` 的 `maximum_std` 与 `minimum_std`，比值 ≥ 2 认定波动分阶段变化；此时全样本相关和全样本误差指标都失去代表性，结论要标明。

季节与记忆：

- `seasonal_strength` ≥ 0.6 为强周期结构，0.3–0.6 为中等，< 0.3 视为无稳定周期结构。同时读 `dominant_seasonality` 与 `variance_share.remainder`；残差占比过半时任何季节结论都要降级表述。
- 自回归阶数取 `suggested_autoregressive_order` 与 `strongest_lags`。偏自相关在一阶后即落入 `significance_band`，说明序列接近随机游走、模型增益空间有限，必须写明。
- Ljung-Box 各阶均不拒绝时序列接近白噪声，此时不要再用自相关衰减形状讲"记忆"。

极端状态：

- 没有趋势季节分解时，用日历分组自身的组间方差占比判断：某个分组（小时／星期／月份）解释了 ≥ 6% 的电价方差即认定存在该层结构。有分解强度时以强度为准。
- 尖峰阈值取 `price_spike_regime_profile` 的 Tukey outer fence，除非用户明确给出领域阈值。
- `clustering.persistence_ratio` 明显大于 1 表示尖峰成片出现，误差集中在少数时段，此时朴素基线的平均误差会低估真实难度，结论中要一并说明。
- `negative_share` 非零时单独说明负价出现的时段与量级，不并入"极端值"一句带过。

## 驱动质量

- 任何变量首次进入研究范围先做 `exogenous_descriptive_distribution`，确认量纲与覆盖后再谈关系。`coverage_rate` < 90% 的变量不足以支撑关系分析，先说明缺口再谈相关。
- 漂移量级用 `trend_slope_per_interval` × `observations` 与该变量自身 `std` 比较：全期漂移超过一个标准差才算长期漂移，否则只是噪声。
- `exogenous_iqr_outliers` 的异常点先归因（传感异常、真实极端天气、单位错误），未归因的异常不得当作信号。
- `exogenous_linear_index_trend` 显示显著长期漂移且目标同样非平稳时，同期相关一律先按共同趋势处置。
- 两个及以上变量时用 `exogenous_pearson_collinearity` 看两两冗余；仍怀疑多变量共同解释时才加 `exogenous_variance_inflation`。`severity` 为 severe 的一组变量在特征建议中只保留一个代表，并说明保留理由。

## 关系证据的推进顺序

1. `relationship_scipy_pearson_pairwise` 先做。|r| 低不等于结束，进第 2 步。
2. 怀疑单调非线性或极端值主导时做 `relationship_scipy_spearman_pairwise`。秩相关明显高于 Pearson，说明关系单调但非线性，或 Pearson 被少数极端点拉动。
3. 两者都低时做 `relationship_mutual_information_scan`。以 `nonlinearity_flagged` 为准：归一化互信息要在**最优滞后**上高于同滞后的 `absolute_pearson_at_best_lag`，只有 lag 0 偏高不算。
4. 出现候选关系后用 `relationship_pearson_positive_lead_scan` 定位滞后。正滞后固定表示 `feature[t-lag]` 与 `price[t]`。最优滞后为 0 说明该变量没有领先性，只能作同期特征，且要求预测起点已有它的预测值。
5. 稳定性至少验一项，否则结论只能写"全样本平均关系"：日内用 `relationship_pearson_by_hour`，季节用 `relationship_pearson_by_month`，长期用 `relationship_rolling_correlation_stability`，用户给定分段用 `relationship_pearson_segment_comparison`。
6. 需要区分"变量历史是否优于纯自回归"时才用 `relationship_granger_causality_scan`，结论只能表述为样本内前置性。
7. 关注响应形状（是否存在阈值或饱和）时用 `relationship_feature_quartile_response`。分位桶的 `target_mean` 不再单调即认定响应非线性；只描述形状，不作因果解释。

## 可预测性定级

- 误差底线引用 `best_baseline_label` 与 `best_mae` 原值。后续模型必须在同一时间切分下低于该值才构成预测增益。
- `near_zero_share` ≥ 1% 时不得使用 MAPE 一类百分比误差。
- `recommended_transform` 为 `asinh_median_mad` 时，先给预处理结论，再谈难度。
- 每个候选变量最终归入一类：**可进入算法实验**（四道判据全过）、**仅可描述**（稳定性或伪相关判据未过）、**存在泄漏风险**（`observed_only` 或可获得时点未知）、**证据不足**（样本量或检验不支持）。

## 提案约束

- 只选回答当前问题所需的最小函数集合。
- 每个函数每个方案最多调用一次；多变量用 `variables`，多滞后用一个最大 `max_lag`，多子样本用一次 `segments`。
- 变量名取自当前已识别的数据字段；目标电价不进 `selected_variables`。
- `segments` 的小时、月份和时间边界必须由用户明确给出，不得自拟峰谷时段、季节定义或政策事件日期。
- 负价、零价、上限附近值和尖峰保留并标注，本阶段不截尾、不删除、不插补。
- 平稳性、STL/MSTL 与偏自相关只在方法内部为连续性做时间插值，并报告 `interpolated_share_*`；其余统计量使用成对有效样本。
- 朴素基线为全样本计算，只作误差量级参考；正式实验须改为按时间顺序滚动重估。
