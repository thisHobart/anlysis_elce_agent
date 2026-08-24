# 电价研究 Agent

面向非开发用户的 PySide6 Windows 桌面应用。用户在连续会话中提出电价研究问题，Agent 根据可用数据推荐可调整的 EDA 计划，确定性算法负责计算，评估器检查证据并生成可复现研究包。

## 桌面界面

```text
历史会话 | 用户与 Agent 的连续对话 | 文件输入、当前方案、Agent 运行轨迹
```

- 每次启动直接进入一个新的“新会话”，历史有效会话仍保留在左侧；空白旧会话会自动清理。
- 历史会话支持新建、搜索、重命名、删除和重启恢复。
- 对话时间线包含用户消息、Agent 回复、数据检查、计划卡、工具进度和结果卡。
- Agent 统一路由研究讨论、新方案、方案修订、结果解释和执行确认；可以直接用自然语言调整当前方案。
- 文件输入按角色管理，但文件名不受限制：研究配置、目标电价、实际外生变量、预测外生变量。
- YAML、实际变量和预测变量均为可选；没有 YAML 时自动识别 CSV/Parquet 的时间列、数值列和频率。
- 真正执行 EDA 时至少需要目标电价数据；没有文件时仍可正常讨论研究方向。
- 当前方案由大模型生成并在中间计划卡只读展示；用户在对话中提出修改意见，由大模型生成修订版。
- 每版方案等待用户反馈 30 秒；没有反馈时自动锁定并执行，输入修改意见时倒计时暂停。
- Agent 运行轨迹独立滚动，记录可审计的计划、工具、评估、耗时和产物，不展示隐藏推理过程。
- 会话保存最新结构化证据和多轮运行血缘，重启后仍可继续追问已有结果。

## Agent 架构

桌面端通过 `ResearchCoordinator` 驱动唯一的持久化 `ResearchLoopGraph`。主研究 Agent、Skill、EDA Subagent、计划校验、用户审批、逐工具执行、结果校验、评估回流和结果追问都处于同一个有界循环。

```text
PySide6 Desktop
→ ResearchCoordinator
→ app/research/graph
→ MainResearchAgent
→ SkillRegistry / resolve_skill
→ EDASubagent
→ 受约束 Function Calls
→ ToolRegistry / ToolExecutor
→ 工具结果校验
→ 确定性评估器
→ accept / 自动 revise / need_user / reject
→ SQLite checkpoint / interrupt / resume
```

主 Agent 和 Subagent 共用 `app/llm` 中唯一的模型网关；只有该基础设施层可以创建 `ChatOpenAI`。不同 Agent 的区别是职责、提示词和结构化契约，而不是各自维护模型连接。

方案校验、工具错误和评估结果通过结构化 `FeedbackPacket` 回流。自动修订最多执行两轮并受原审批权限约束；无法继续时暂停并等待用户，不会形成无限循环。应用关闭期间不后台执行，过期审批在重启后必须明确确认。

第一阶段内置 `price-exogenous-eda` Skill。外部专业 Skill 可以放在项目或 EXE 同级的 `skills/<skill-name>/SKILL.md`，也可以通过 `VPP_SKILL_PATHS` 添加搜索目录。外部 Skill 只提供专业流程和工具许可，不会自动执行其 `scripts/`。

EDA 工具位于 `app/research/tools/eda`。模型只能看到注册工具的 Function Schema；最终调用必须经过计划编译、用户反馈窗口、Skill 权限策略和确定性 `ToolExecutor`。

## 运行

需要 Python 3.11+。

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -e ".[dev]"
.\venv\Scripts\python.exe -m app.desktop
```

也可以双击：

```text
start_desktop.bat
```

## 文件输入

右侧有四个可选角色槽位：

| 角色 | 支持格式 | 必需性 |
|---|---|---|
| 研究配置 | YAML | 可选；用于提供字段、时区、频率和方法参数 |
| 目标电价 | CSV / Parquet | 执行 EDA 时必需；配置文件可自动提供路径 |
| 实际外生变量 | CSV / Parquet | 可选 |
| 预测外生变量 | CSV / Parquet | 可选 |

可以点击选择，也可以把文件拖到对应角色槽位。更换文件后，旧数据画像和分析计划会自动失效。

## LLM 配置

复制 `.env.example` 为 `.env`，填写 OpenAI 兼容模型：

```dotenv
VPP_LLM_BASE_URL=http://127.0.0.1:xxxx/v1
VPP_LLM_API_KEY=local-placeholder
VPP_LLM_MODEL=your-model-name
VPP_LLM_STRUCTURED_MODE=native
VPP_SKILL_PATHS=
```

研究对话、方案生成和方案修订必须调用大模型。模型未配置、调用失败或返回无效方案时，任务会停止并提示重试，不存在本地规划回退。界面不再提供“启用大模型规划”开关，只保留模型连接参数。

`native` 使用模型 Function Calling；不支持 Function Calling 的兼容接口可以显式改为 `json_prompt`。不存在失败后自动切换到本地规则规划的路径。

大模型只能从版本化 Skill、工具和方法目录中选择具体实现。执行计划保存 Skill、工具、方法实现 ID、版本、参数、输入数据指纹和代码环境；版本不匹配时拒绝执行。

源码运行时将读取仓库根目录的 `.env`；打包后的程序读取 `PriceResearchAgent.exe` 同目录的 `.env`。

## 生成 Windows EXE

PyInstaller 必须在 Windows 上构建 Windows 程序。第一次建议使用默认的单目录模式，便于排查依赖；确认无误后可加 `--onefile` 生成单文件版本。

```powershell
.\build_exe.bat
```

单文件版本：

```powershell
.\build_exe.bat --onefile
```

输出位于：

```text
dist/PriceResearchAgent/
```

或单文件模式下的 `dist/PriceResearchAgent.exe`。

## 确定性 EDA 命令

桌面端是主要用户入口。算法回归仍可直接运行：

```powershell
.\venv\Scripts\python.exe -m app.research.cli --config configs/research/price_exogenous_eda.yaml
```

## 测试

```powershell
.\venv\Scripts\python.exe -m ruff check app tests
$env:QT_QPA_PLATFORM="offscreen"
.\venv\Scripts\python.exe -m pytest -q
```

第一阶段说明见 [docs/phase1-price-exogenous-eda.md](docs/phase1-price-exogenous-eda.md)，长期路线见 [docs/research-agent-development.md](docs/research-agent-development.md)。

循环设计见 [docs/research-loop.md](docs/research-loop.md)；真实模型、真实数据、桌面恢复和 EXE 验收记录见 [docs/phase1-acceptance.md](docs/phase1-acceptance.md)。
