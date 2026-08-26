---
name: price-exogenous-eda
description: Build forecast-readiness evidence for electricity prices together with their exogenous drivers. Use when the question involves market-time alignment, price level and shape, spikes and negative prices, stationarity, seasonality, price memory, driver data quality, multicollinearity, contemporaneous or lagged price-driver relationships, nonlinear dependence, relationship stability, leakage risk, or which candidate features deserve a forecasting experiment.
metadata:
  version: "3.0.0"
  domain: eda
  protocol-file: references/research-protocol.yaml
allowed-tools: data_quality price_descriptive_distribution price_rolling_mean_std price_tukey_outer_fence price_lag_autocorrelation price_calendar_group_profile price_stationarity_tests price_seasonal_decomposition price_partial_autocorrelation price_spike_regime_profile price_duration_curve price_segment_distribution_comparison exogenous_descriptive_distribution exogenous_iqr_outliers exogenous_linear_index_trend exogenous_pearson_collinearity exogenous_variance_inflation exogenous_stationarity_tests relationship_scipy_pearson_pairwise relationship_scipy_spearman_pairwise relationship_pearson_positive_lead_scan relationship_pearson_by_hour relationship_pearson_by_month relationship_feature_quartile_response relationship_mutual_information_scan relationship_granger_causality_scan relationship_rolling_correlation_stability relationship_pearson_segment_comparison price_variance_stabilization_check price_naive_baseline_benchmark
---

# 电价与外生变量联合研究 Skill

为电价预测建立可复现、可追溯的目标序列与驱动变量证据。本阶段只做离线只读 EDA：不训练模型，也不产生任何预测值。

## 研究顺序

唯一顺序由本地 `research_protocol` 固定，共六个阶段：

1. **市场时钟**：明确市场产品、时间粒度、时区、预测起点、预测范围与决策截止时点；缺失信息显式标记，不用常识补齐。
2. **数据体检**：时间轴、覆盖率、重复、缺失、数值有效性，以及每个变量在预测起点是否真实可获得。
3. **电价自身规律**：水平与分布、极端与负价状态、持续曲线、日历结构、波动状态、平稳性、趋势季节分解、自相关与偏自相关记忆。
4. **驱动质量**：候选驱动的覆盖、异常、漂移、两两相关、多重共线性与自身平稳性，并保留预测时点可用性标签。
5. **关系证据**：只对通过可获得性检查的变量研究同期线性与秩关系、领先滞后、非线性依赖、样本内前置性、分时段差异与随时间稳定性。
6. **可预测性判定**：给出朴素基线误差底线与方差稳定建议，把结论归类为可进入算法实验、仅可描述、存在泄漏风险或证据不足。

## 选择原则

- 选择回答当前问题所需的**最小函数集合**，不因为函数可用就全部调用。
- 每个原子函数每个计划最多调用一次；多变量用 `variables`，多滞后用一个最大 `max_lag`，多个子样本用一次 `segments`。
- `data_quality` 由编译器强制加入，模型不得重复调用。
- 变量名必须来自当前研究配置；目标电价不能放进外生变量列表。
- `exogenous_variance_inflation` 至少需要两个变量。
- 正滞后统一表示 `feature[t-lag]` 与 `price[t]`；不能把事后信息包装成领先信号。
- `segments` 中的小时、月份与时间边界必须来自用户或研究配置的明确值，不得自行猜测峰谷时段、季节定义或政策事件日期。

## 方法边界

- 负价、零价与尖峰是电力市场结构证据，默认保留并标注，EDA 阶段不静默截尾、删除或插补。
- 平稳性检验、STL/MSTL 分解和 PACF 需要连续序列，函数内部只为这些方法做时间插值，并在结果中报告插值比例；其余统计量一律使用成对有效样本。
- 同期相关、秩相关、互信息与 Granger 前置性都不是因果关系，也不等于按时间顺序切分后的样本外预测增益。
- 目标与驱动同时非平稳时，同期相关可能来自共同趋势；此时必须同时引用平稳性证据。
- 朴素基线在全样本上计算，只作为**误差底线的量级参考**，正式实验必须改为按时间顺序滚动重估。

## API 模型边界

- 模型只解析用户研究意图，并在授权集合内提出具体 Function Call。
- 阶段顺序、函数权限、参数门禁、执行、证据校验与停止条件由本地代码和版本化协议控制。
- 可观察的研究依据只包括协议阶段、具体函数调用、确定性结果、门禁结论和用户反馈。

## 输出契约

返回供用户确认的结构化方案，记录 Skill、领域协议、函数与版本、变量、参数、假设与限制。执行后只用确定性证据给出结论，并明确指出下一阶段仍需按时间顺序完成基线与增量验证。

外部依据与采用/舍弃说明见 [references/methodology.md](references/methodology.md)。
