# 工具调用独立调试

P1 工具调用可以绕过桌面、ResearchCoordinator 和 LangGraph，按“请求、选择、编译、执行、回传”逐段调试。独立入口与生产研究图共用 `build_tool_request`、`compile_tool_turn`、`ToolExecutor`、`validate_tool_result` 和工具回传构造函数。

## 输入文件

`scripts/debug_tool_turn.py` 接受一个 JSON 对象。基础字段：

```json
{
  "scope": {"这里放完整的 EDAResearchScope": "..."},
  "study_config_path": "G:/absolute/path/to/study.yaml",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

`request` 和 `select` 使用 `messages`；`compile` 另外读取 `turn`；`execute` 另外读取 `call`；`round` 使用 messages 并完成一轮真实选择、编译、执行和回传。工具执行需要 scope 的数据指纹对应已冻结数据，或当前配置仍能产生相同指纹。

## 三个最短调试过程

只看模型实际收到的消息和工具 Schema：

```powershell
.\venv\Scripts\python.exe scripts\debug_tool_turn.py request --input .\debug-case.json
```

只调用一次真实模型，观察原始工具选择：

```powershell
.\venv\Scripts\python.exe scripts\debug_tool_turn.py select --input .\debug-case.json --output .\artifacts\proposal.json
```

离线复现提议编译。把 `proposal.json` 中的 `turn` 放回 debug case 后执行：

```powershell
.\venv\Scripts\python.exe scripts\debug_tool_turn.py compile --input .\debug-case.json --output .\artifacts\compiled.json
```

直接执行单个编译调用时，把 `compiled.calls` 中的一项放到输入的 `call` 字段：

```powershell
.\venv\Scripts\python.exe scripts\debug_tool_turn.py execute --input .\debug-call.json --output .\artifacts\result.json
```

命令默认不重试。输出保留原始提议参数和编译参数，因此本地注入不会掩盖模型参数错误。

## 工具选择评测

首批 12 个审阅案例位于 `tests/fixtures/tool_calling/cases.json`。评分器只检查原始 `ModelToolTurn` 的工具名和模型负责的参数，不调用研究结果评估器，也不生成报告。

真实模型评测需要一个 context JSON，其中包含 `base_scope`、`quality_report` 和可选的 `data_profile`：

```powershell
.\venv\Scripts\python.exe scripts\benchmark_tool_calling.py --context .\tool-benchmark-context.json --output .\artifacts\tool-selection-report.json
```

传输或端点协议失败单独记录为 `infrastructure_error`，不会计作工具选择准确率。普通 pytest 使用固定模型回复验证评分器和调用流程，不冒充真实模型结果。
