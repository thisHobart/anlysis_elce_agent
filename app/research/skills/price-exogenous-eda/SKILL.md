---
name: price-exogenous-eda
description: Design reproducible electricity-price and exogenous-variable EDA plans when the user asks about price structure, external drivers, correlation, lag, seasonality, volatility, extremes, or data quality.
metadata:
  version: "1.0.0"
  domain: eda
allowed-tools: data_quality price_profile exogenous_profile relationship_analysis
---

# 电价与外生变量 EDA

为离线、只读的电价与外生变量数据集设计可复现的探索性分析方案。规划结果供用户确认，统计计算由本地确定性工具完成。

## 规划目标

1. 先查看时间轴、覆盖率、重复、缺失和变量可获得时点。
2. 识别用户关注的是电价结构、外生变量画像，还是变量与电价的描述性关系。
3. 选择回答当前问题所需的最少工具、method key、变量和滞后范围。
4. 把方案作为只读建议交给用户确认；运行结果来自确定性工具。
5. 用户提出修改时生成新版本方案；用户确认后才进入执行阶段。
6. 解释只使用工具产生的证据，并明确相关关系与因果结论的区别。

## 可用工具

- `data_quality`：时间轴、覆盖率、缺失、重复和可获得性检查；编译器会把它作为必需步骤加入方案。
- `price_profile`：电价分布、波动、极端、自相关和季节性。
- `exogenous_profile`：外生变量分布、异常、趋势和共线性。
- `relationship_analysis`：同期、秩相关、领先滞后、分组和分位数组关系。

方案中的工具、method key 和变量均使用当前上下文中的目录条目与精确名称。

## 输出

返回结构化 EDA 方案，包含研究目标、假设、工具调用、具体方法、变量、参数、限制和实现版本。
