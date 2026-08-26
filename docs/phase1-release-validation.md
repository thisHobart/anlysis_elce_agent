# 第一阶段发布验收说明

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

脚本应完成：生成方案、用户修订、审批、函数执行、确定性评估和结果追问。成功后在本轮研究包内写入：

```text
phase1_loop_acceptance.json
```

重点检查：

- `status` 为 `passed`；
- `enabled_functions` 使用当前研究函数名称；
- `plan_id`、`run_id`、`episode_id` 均存在；
- `report_path` 指向实际存在的报告；
- `evaluation` 是当前评估器的有效决策。

### 3. 真实桌面流程和重启恢复

```powershell
.\venv\Scripts\python.exe scripts\validate_desktop_phase1.py
```

应确认桌面能够完成方案修订、审批、执行、结果解释，以及关闭后重新打开会话并继续追问。

### 4. 重建并检查 EXE

```powershell
.\build_exe.bat --onefile
Get-FileHash .\dist\PriceResearchAgent.exe -Algorithm SHA256
```

必须使用当前提交重新构建，记录构建时间、文件大小和 SHA-256。启动 EXE 后至少检查：

- 两个内置 Skill 均可加载；
- 可打开桌面并创建研究会话；
- 能完成一次真实数据研究；
- 重启后能够恢复已完成会话；
- 研究报告和内部状态写入用户目录，而不是 EXE 或临时解压目录。

## 发布完成条件

- Ruff 和全套测试通过；
- 真实循环验收文件生成成功；
- 真实桌面及重启恢复通过；
- EXE 由当前源码重新构建并通过启动检查；
- EXE 哈希和验收时间已更新到正式验收记录；
- README、验收文档、schema 版本和实际代码保持一致。
