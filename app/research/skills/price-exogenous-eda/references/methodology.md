# 领域协议依据

本文件记录两个内置 Skill 的外部调研依据与取舍，不作为可执行代码，也不复制外部实现。

## 采用的公开工作

- [Lago 等（2021）Forecasting day-ahead electricity prices](https://arxiv.org/abs/2008.08004)：采用时间顺序切分、训练/验证/测试隔离、**必须与朴素基线比较**、完整记录输入与日期、避免把 MAPE 作为负价与近零电价主要指标等原则。`price_naive_baseline_benchmark` 直接实现其中的“误差底线”要求；时间顺序滚动重估属于后续算法实验阶段的门禁。
- [EPFToolbox](https://github.com/jeslago/epftoolbox)：采用可复现研究、公开基线、滚动/日度重估、明确输入与统计检验的原则。
- [Weron（2014）Electricity price forecasting: A review](https://doi.org/10.1016/j.ijforecast.2014.08.008)：采用把电价特征归纳为季节性多重叠加、均值回复、波动聚集与尖峰状态四类的分析框架，对应 `price_calendar_group_profile`、`price_stationarity_tests`、`price_rolling_mean_std` 与 `price_spike_regime_profile`。
- [Uniejewski、Weron 等 Variance Stabilizing Transformations for Electricity Spot Price Forecasting](https://ieeexplore.ieee.org/document/7997921/)：采用先用 median 与 MAD 稳健标准化、再评估 asinh 变换的预处理顺序，作为 `price_variance_stabilization_check` 的口径；本阶段只输出建议，不改写报告统计量。
- [Hyndman & Athanasopoulos, Forecasting: Principles and Practice](https://otexts.com/fpp3/)：采用 STL/MSTL 分解与“季节强度 / 趋势强度 = 1 − Var(残差)/Var(残差+成分)”的度量定义，作为 `price_seasonal_decomposition` 的口径。
- [statsmodels time-series diagnostics](https://www.statsmodels.org/stable/tsa.html)：ADF、KPSS、PACF、Ljung-Box 与 MSTL 全部直接调用其成熟实现，不自行重写统计量。
- [OpenSTEF](https://github.com/OpenSTEF/openstef)：采用数据处理、特征工程、训练、回测、评估相互分层的工程边界。
- [GB day-ahead electricity price forecast](https://github.com/zsheikhnajdi/gb-day-ahead-electricity-price-forecast)：采用预测截止时点、发布时点可获得性、用过去信息构造滞后与滚动特征的防泄漏思路；不采用其中针对 GB 市场的具体截止时刻作为其他市场默认值。
- [Energy Analytics Lecture Series](https://github.com/lipiecki/energy-analytics)：采用朴素模型、点预测到概率预测、预测组合和预测性能检验分层推进的研究顺序。

## 搜到但未采用的做法

- [MemCast time-series SKILL.md](https://github.com/ndpvt-web/arxiv-claude-skills/blob/master/skills/memcast-memory-driven-time-series/SKILL.md) 依赖 LLM 生成、保存和反思推理轨迹，并让 LLM 直接参与数值预测。它与本系统“本地确定性计算、API 仅作受限路由与 Function Calling、不保存模型私有思考链”的边界冲突，因此不复制、不安装。
- 通用能源预测 Skills 多面向宏观事件概率、负荷预测或外部 MCP 服务，无法直接约束当前电价—外生变量 EDA 函数。
- 自动特征筛选（如基于相关性阈值的自动入选）未采用：Lago 等明确指出特征价值必须由样本外实验判定，EDA 阶段只输出候选与风险。
- 尖峰截尾/Winsorize 预处理未采用：尖峰是电力市场的结构信息，第一阶段只标注不修改数据。

## 本地化取舍

1. 将领域推理显式化为版本化 `research_protocol`，而不是自然语言思考轨迹。
2. 内置只保留两个核心 Skill：`price-exogenous-eda`（电价 + 外生变量联合研究）与 `price-forecastability-audit`（仅目标序列的可预测性审计）。后者用于没有外生变量数据、或需要先判断序列难度的场景。
3. API 模型只输出路由或具体 Function Call；本地编译器负责排序、权限、参数和版本校验。
4. 第一阶段不训练模型，因此不宣称变量具有样本外预测增益；该结论必须由后续时间顺序实验产生。
5. 同一函数每个计划最多调用一次；多个变量、滞后和子样本分别通过 `variables`、`max_lag` 与 `segments` 在一次调用中处理。
6. 需要连续序列的方法（ADF/KPSS、MSTL/STL、PACF、Ljung-Box）在函数内部做时间插值，并把插值比例写进结构化证据；其余统计量一律使用成对有效样本。
7. 长序列在平稳性检验与季节分解前按固定阶梯降采样（1h → 2h → … → 1D），并在结果中记录实际分析粒度，避免检验耗时不可控。
8. Granger 检验只表述为“样本内前置性”，互信息只表述为“统计依赖强度”，两者都不写成因果或预测增益。
