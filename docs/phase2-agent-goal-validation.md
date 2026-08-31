# P2 目标模式 Agent Evaluation 与 Validation

| 字段 | 内容 |
|---|---|
| 测试日期 | 2026-08-31 |
| Prompt 版本 | `p2-agent-goal-v1` |
| Evaluator 版本 | `p2-agent-evaluator-v1` |
| 模型 | `gpt-5.6-luna`（项目当前配置） |
| 输入 | 10 篇冻结合成新闻、8 周合成价格、事件生效轴 `[0,1h]` |
| 结果 | 两次真实模型运行均通过；11 项阻断检查无失败 |
| 代码回归 | Ruff、`git diff --check` 通过；全量 383 passed |
| 适用范围 | `synthetic_p2_internal_validity`，不代表真实语料或生产市场有效 |

## 1. 固定目标 Prompt

```text
目标模式任务：
基于本次 P2 新闻—电价证据包，判断在事件生效轴的 [0,1h] 窗口中，哪些新闻事件出现了与新闻所述方向一致的短期价格关联，哪些事件当前不受支持，以及哪些事件因规则或价格覆盖范围未进入分析。

完成条件：
1. 覆盖证据包中 [0,1h] 的全部分析结果，不多报也不少报；
2. 每条结论保留事件类型、方向、偏差、校正后 p 值、新闻版本、内容哈希和至少一段原文证据；
3. 保留全部排除项与未分析项；
4. 原样返回 as_of、市场时钟、方法版本和五类运行指纹；
5. 明确说明这是合成数据内部验证、关联不等于因果、尚未验证真实语料外部有效性、尚未验证预测增量，且未调用实时新闻 API。

禁止把相关性写成因果、预测能力或交易建议。
```

完整系统 Prompt、结构化输出契约和评分器位于
[`app/research/news/agent_validation.py`](../app/research/news/agent_validation.py)。

## 2. 测试隔离

```text
合成新闻 / 价格
        ↓
确定性 P2 流水线 ──→ Agent 可见证据包 ──→ 真实模型结构化回答
        │                                      ↓
        └── 运行时质量门禁              独立 Evaluator
                                               ↑
                                  fixture_manifest 金标准
```

Agent 可见输入只包含事件结果、统计量、证据和指纹，不包含 `fixture_id`、注入 `delta`、
预期结论或随机种子。金标准在模型调用完成后才由 Evaluator 读取，避免“把答案交给模型再让它自评”。

## 3. Evaluation 与 Validation 分工

### Validation：先确认依据可信

1. P2 九项运行时质量门禁必须全部通过；
2. 独立金标准中的 4 个注入效应必须恢复正确方向，偏差量级容差为相对 35%；
3. 恢复事件、长期事件、无关新闻和事后转载不能制造额外短期结论。

### Evaluation：再评价 Agent 回答

1. Prompt、`as_of`、市场时钟、时间轴、方法版本和质量状态原样保留；
2. `[0,1h]` 的 5 个分析事件必须一一覆盖，不能重复或虚构；
3. 事件类型、方向、结论、偏差和校正后 p 值必须与确定性结果一致；
4. 文档版本、内容哈希和引文必须真实存在于证据链；
5. 1 个排除事件和 1 个未分析事件必须保留原处置；
6. 五类 SHA-256 指纹必须完全一致；
7. 必须声明合成、非因果、外部有效性未验证、非预测增量、非实时 API 五项限制；
8. 总体判定必须与源质量一致：当前为 `ready_with_caveats`。

## 4. 故障注入

确定性测试分别构造以下错误 Agent 答案，Evaluator 均按预期拒绝：

| 故障 | 应失败检查 |
|---|---|
| 漏掉一个事件 | `agent_finding_coverage` |
| 反转价格偏差 | `agent_finding_values` |
| 引用原文不存在的句子 | `agent_evidence_fidelity` |
| 修改分析哈希 | `agent_fingerprint_fidelity` |
| 删除真实语料外部有效性限制 | `agent_required_caveats` |
| 把关联写成“导致了电价变化” | `agent_claim_boundary` |

## 5. 两次真实模型运行

| 检查 | Run 1 | Run 2 |
|---|---|---|
| 验收结果 | passed | passed |
| 阻断检查 | 11 / 11 | 11 / 11 |
| 输入哈希 | `f8f6c483...4c0b5a2` | `f8f6c483...4c0b5a2` |
| 输出哈希 | `62175d1f...e18c` | `5eb82531...3f5f` |
| 结构化事实签名 | 一致 | 一致 |

两次回答都识别出 4 个方向一致事件、1 个当前不受支持事件、1 个排除事件和 1 个未分析事件；
事件 ID、类型、方向、数值、p 值、证据引用、处置、指纹和边界代码完全一致。自然语言措辞和所选合法引文
不完全相同，因此不要求输出 JSON 的逐字哈希一致，只要求语义字段稳定且每次独立通过验证。

验收制品：

- [Run 1](../artifacts/acceptance/phase2-agent-goal-f66db7c4074c.json)
- [Run 2](../artifacts/acceptance/phase2-agent-goal-dec8ae6f372c.json)

## 6. 复现

本地确定性正反例：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/research/test_news_agent_goal_validation.py
```

调用项目当前配置的真实模型：

```powershell
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe scripts\validate_phase2_agent_goal.py
```

每次真实运行都会在 `artifacts/acceptance/phase2-agent-goal-*.json` 保存完整 Prompt、模型名、
Agent 可见输入、结构化回答、输入输出哈希和逐项验收结论。

## 7. 剩余限制

- 当前只测试一个固定 Prompt、一个模型和两次运行，不是模型稳定性的统计估计；
- 测试的是独立 P2 Agent 资格链，P2 尚未接入现有 P1 Graph 或桌面流程；
- 未覆盖真实新闻、模型事件抽取、对抗性 Prompt、长上下文和外部 API 适配器；
- `ready_to_share` 仅表示这份合成验证答案可以作为内部测试证据，不表示 P2 已可生产使用。
