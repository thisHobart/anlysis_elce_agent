# P3 山东实时电价最小预测

## 当前能力

P3 首版只支持山东省级实时电价。用户在桌面端选择山东并完成取数后，可以在对话中提出“预测山东明天实时电价”或“预测山东未来24小时电价”。程序固定生成次日自然日 00:00—23:45 的 96 个15分钟点，不支持四川、日前价、节点价或任意本地文件预测。

预测必须经过独立确认卡显式批准，不使用 EDA 的自动审批倒计时。批准后依次执行3个历史锚点回测和1次未来预测；回测锚点位于最近90天内、各自具有完整96点真实价格，并至少相隔7天。

## 数据边界

`app/research/forecasting/data.py` 通过只读 `RegionalSnapshot` 接口复用地区取数层。每个历史锚点只查询当时已经可见的数据，`available_at/create_time` 不晚于锚点；实际价格、负荷和出力在锚点之后不可见。预测运行只读取计划目录中冻结的4份 Parquet 输入，不修改数据面板缓存指针，不连接任何写库发布器。

算法字段适配如下：

| 项目字段 | 模型字段 |
|---|---|
| `forecast_load` | `fcst_load_type_11` |
| `forecast_generation` | `fcst_total_gen` |
| 风电预测 + 光伏预测 | `fcst_re_total` |
| `forecast_solar` | `fcst_new_energy_pv_unified` |
| 天气字段 | `wx_*` |

实际电价超过2小时未更新会写入警告，超过24小时则拒绝预测。负荷、发电、风电或光伏预测至少有一项覆盖72/96点；天气不足仅警告并使用缺失掩码，不以未来实况填补。

## 算法与评估

`app/research/forecasting/model.py` 是从内部山东算法参考实现中隔离出的 CPU 点预测版本：CTM-Base 连续状态、多 tick 确定性、市场状态监督、多因素相似日以及有界融合。运行不依赖参考目录，也不包含 Q-QRA、LightGBM、Legacy LSTM、Stage3专家或后置 rescue 规则。

生产默认参数为：96点历史窗口、96点预测范围、hidden 80、6 ticks、6步记忆、48个同步对、最多55轮、早停8轮、种子42和 `[-120, 1500]` 价格边界。PyTorch 强制使用 CPU 与确定性算法。

每折同时计算模型、持续法、前一日同刻朴素法和一周前同刻朴素法的 MAE、RMSE、Bias，四种方法严格使用同一组共同有效点。每折共同有效点少于72个时不得宣称优于基线。回测不阻断未来预测；若样本不足或模型聚合 MAE 没有优于前一日/一周前基线，结果卡和报告醒目标记“未验证出预测增益”。

## 研究包与恢复

每份计划目录包含：

- `prediction.csv`
- `backtest_predictions.parquet`
- `metrics.json`
- `report.md`
- `figures/forecast_curve.svg`
- `figures/backtest_comparison.svg`
- `data/` 下4份固定模型输入及3份回测真实值
- `provenance/forecast_plan.json`
- `provenance/snapshot_manifest.json`
- `provenance/environment.json`
- `provenance/checkpoint.json`
- 根目录 `manifest.json` 及各产物 SHA-256

`app/research/forecasting/workflow.py` 将 `backtest_1`、`backtest_2`、`backtest_3` 和 `future` 编排为4个 LangGraph 节点。每个回测折完成后写入检查点；相同计划和数据指纹重启执行时会复用已完成折，随后继续下一个节点。未来预测完成后保存版本化 `ForecastRunResult`。失败和取消状态写入 provenance，不写业务数据库。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/research/test_forecasting.py tests/desktop/test_forecast_ui.py -q
.\.venv\Scripts\python.exe -m ruff check app tests scripts/build_desktop_exe.py
```

PyInstaller 构建不再排除 PyTorch、scikit-learn 和 PyTorch 所需的 SymPy；EXE 启动探针会实际导入 P3 模型并验证其参数位于 CPU。源代码测试覆盖确定性、防未来真实价格泄漏、锚点规则、研究包生成、检查点恢复和桌面确认卡。

真实数据库验收仍受数据新鲜度门禁约束。若源库最后实际价格超过24小时，验收应记录为数据门禁拒绝，不得通过放宽规则或填补未来真实值伪造成功。
