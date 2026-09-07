# 第一阶段发布验收说明

> 文档定位：当前发布验收规程。实际是否通过以 [P1 验收记录](phase1-acceptance.md)为准；历史运行结果不能替代重新验收。

本文说明如何验证当前源码、真实研究循环、桌面流程和 Windows EXE 属于同一个可发布版本。

## 验收范围

发布验收分为四层，前一层失败时不应继续发布：

1. 静态检查和自动化测试；
2. 真实模型、真实数据的持久化研究循环；
3. 真实桌面交互以及关闭、重启恢复；
4. 基于当前源码重新构建的 Windows EXE。

自动化测试主要验证确定性逻辑和边界条件，不能替代真实模型、桌面和 EXE 验收。

## 方案字段约定

当前 `EDAPlanStep` 使用以下字段：

| 含义 | 当前字段 | 已废弃字段 |
|---|---|---|
| 研究函数名称 | `function` | `tool` |
| 研究函数版本 | `function_version` | `tool_version` |

`scripts/validate_phase1.py` 生成的 `enabled_functions` 必须读取 `step["function"]`。旧字段只允许在历史状态迁移边界出现，不能用于当前运行或验收产物。

对应回归测试为：

```powershell
.\venv\Scripts\python.exe -m pytest -q tests/research/test_release_validation.py
```

## 验收前提

- 已在当前虚拟环境安装项目运行、开发和构建依赖；
- `.env` 已配置可用的模型 Base URL、API Key 和模型名称；
- `data/` 下的默认目标、实际外生变量和预测外生变量文件可读；
- 工作区中不存在不准备纳入发布的代码修改；
- `dist/` 中的旧 EXE 不作为当前源码的验收证据。

## 执行顺序

### 1. 静态检查和全套测试

```powershell
.\venv\Scripts\python.exe -m ruff check app tests scripts
$env:QT_QPA_PLATFORM="offscreen"
.\venv\Scripts\python.exe -m pytest -q
```

两条命令都必须以退出码 `0` 结束。

### 2. 真实持久化循环

```powershell
.\venv\Scripts\python.exe scripts\validate_phase1.py
```

脚本会使用真实端点顺序完成两个场景：

1. 只有目标电价时选择 `price-forecastability-audit`，执行电价可预测性函数并追问结果；
2. 有目标电价和外生变量时选择 `price-exogenous-eda`，完成自然语言修订、审批、函数执行、确定性评估和结果追问。

评估接受和 `result_limitations` 都会按各自公开的 interrupt 契约处理；受限结果先完成证据追问，再用 `stop` 保留当前结果，不调用不存在的“接受限制”动作。成功后写入：

```text
artifacts/acceptance/phase1-loop-*.json
```

重点检查：

- `status` 为 `passed`；
- `scenarios` 同时包含两个内置 Skill；
- 每个场景的 `enabled_functions` 使用当前研究函数名称；
- 每个场景的 `plan_id`、`run_id`、`episode_id` 均存在；
- `report_path` 和同目录 `methods.md` 实际存在；
- `evaluation` 是当前评估器的有效决策，结果追问回到原结果门禁。

验证进程使用 120 秒模型请求超时下限，不修改 `.env`。模型追问瞬时失败时只允许通过持久化 `response_error → retry` 恢复一次，已有函数结果不得重跑或丢失。

### 3. 真实桌面流程和重启恢复

```powershell
.\venv\Scripts\python.exe scripts\validate_desktop_phase1.py
```

应确认桌面能够完成方案修订、审批、执行、结果解释，以及关闭后重新打开会话并继续追问。脚本输出桌面阶段状态；超时清理会先协作取消 Worker，再关闭原窗口和恢复窗口，offscreen 模式不得停在模态关闭提示框。

### 4. 重建并检查 EXE

```powershell
.\build_exe.bat --onefile
Get-FileHash .\dist\PriceResearchAgent.exe -Algorithm SHA256
```

必须使用当前提交重新构建。构建脚本会清理可能污染 Qt 依赖分析的外部 ICU PATH，随后自动启动当前 EXE 并检查：

- 两个内置 Skill 均可加载；
- 桌面窗口可构造；
- 30 个研究函数均已注册；
- Skill 加载没有错误；
- 研究报告和内部状态写入用户目录，而不是 EXE 或临时解压目录。

自动 smoke 通过后，发布前仍应人工打开可见窗口，完成一次真实数据研究并观察报告排版与重启恢复。

## 发布完成条件

- Ruff 和全套测试通过；
- 真实循环验收文件生成成功；
- 真实桌面及重启恢复通过；
- EXE 由当前源码重新构建并通过启动检查；
- EXE 哈希和验收时间已更新到正式验收记录；
- README、验收文档、schema 版本和实际代码保持一致。
