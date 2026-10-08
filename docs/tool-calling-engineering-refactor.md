# 工具调用代码整改：缩短调用链，独立调试与评测

2026-09-21，基于当前源码的工程整改方案。请求构建、调用编译、统一校验、回传构造、独立调试和首批选择评测已接入生产代码；checkpoint 字段与恢复节点保持不变。

本方案取代此前以工具说明、追踪和指标建设为主的整改方向。首先解决调用代码不能独立调试的问题；工具手册和评测指标只是后续使用这些接口的调用方。

## 1. 代码问题与证据

| 位置 | 当前事实 | 工程影响 |
|---|---|---|
| `graph/workflow.py` | 4,431 行；`build_research_workflow` 内定义 56 个函数 | 大量逻辑被闭包依赖锁在图构建函数里，难以直接 import、单步执行和局部测试 |
| `workflow.py:analysis_selector` | 410 行，同时处理授权、预算、消息协议、模型调用、配方展开、编译、记录及 UI 事件 | “测试模型下一步选什么”与“推动整个研究流程”耦合 |
| `workflow.py:execute_tool` | 137 行，混合执行、错误分类、重试路由、状态记录和 UI 事件 | 一个异常要跨多个函数和状态字段才能追清，业务执行难以独立复现 |
| `workflow.py:complete_tool_batch` / `_dynamic_tool_error_update` | 分别 154 / 167 行；成功和失败路径均构造消息、处理调用组与证据 | 回传闭合语义分散，修一个路径容易漏另一个 |
| `application/execution.py` | 同时负责数据准备/缓存、调用编译、执行、结果校验、评估和报告 | 调用工具需要依赖研究应用服务，测试容易被报告与研究评估牵连 |
| `execute_scope_call` → 图的 `_validate_tool_result` | 前者已校验动态结果，后者再次调用同一校验函数 | 新鲜结果在同一链路重复校验；固定计划路径又用 `validate_result=False` 改变行为 |
| `prepare_scope` | 每次 `execute_scope_call` 都调用，当前未使用固定计划路径的准备缓存 | 各子调用重复进入配置/Skill/快照准备，不能把已准备的数据明确交给执行器 |
| `graph/state.py` | 70 个顶层字段；队列、当前批次、记录、结果、证据、预算均由不同节点写入 | 调试一个调用需要理解大量全局状态；其中合理索引也缺少清晰的写入责任 |
| `application/coordinator.py` | 根据 planner 内部 gateway 是否有 `invoke_tool_turn` 自动组装动态 agent | 执行方式取决于对象内部形状；替换测试对象可能改变生产路径 |
| `agent/dynamic.py` | 已有可直接调用的 `select`，但 Schema 构建私有且与传输放在一起 | 可以复用的入口存在；缺的是可保存请求、直接测试编译和回传的对应入口 |
| `tools/eda/functions.py` | 部分工具经闭包 handler → method 字符串 → 聚合分析函数再分派 | 断点常停在通用 handler，需要运行时检查名字才能找到统计实现 |
| `test_dynamic_function_calling.py` | 配方测试构建 coordinator、提交消息、批准范围，并 monkeypatch 研究评估器 | 适合作为集成测试，但不适合日常调试工具选择与配方编译 |

这些不是“多写日志”能消除的问题。核心是职责与依赖交叉，而非所有抽象都应该删除：ModelGateway 有真实多供应商实现，ToolRegistry 有真实工具发现用途，两者保留。

## 2. 目标：普通函数可直接执行，图负责持久化调度

当前动态路径的主要阶段（不是一个连续的 Python 调用栈）：

```text
Coordinator → Graph/analysis_selector → DynamicAnalysisAgent → ModelGateway
                      ↓ 配方展开 / EDAExecutionService.compile_dynamic_call
            Graph/mark_tool_running → Graph/execute_tool
                      ↓ EDAExecutionService.prepare_scope / execute_scope_call
                    ToolExecutor → handler → 统计实现
            Graph/validate_tool_result → Graph/complete_tool_batch → 下一模型轮
```

目标采用三个可分别运行的操作，使用现有类型，少量普通函数组织：

```python
# 示意接口，尚未实现。
request = build_tool_request(scope, skill, messages, registry)
turn = gateway.invoke_tool_turn(**request.as_gateway_kwargs())

batch = compile_tool_turn(turn, scope=scope, config=config, registry=registry,
                          recipes=recipes, prior_groups=groups, next_sequence=sequence)

for call in batch.calls:
    result = executor.execute(call, context=prepared.context,
                              policy=prepared.policy,
                              data_fingerprint=prepared.data_fingerprint)
    validate_tool_result(call, result, registry=registry,
                         data_fingerprint=prepared.data_fingerprint)
```

示例省略逐调用持久化、错误分支与回传。生产保留这些阶段边界；不能把整批执行塞进一个不可恢复的图节点。调试脚本在没有 checkpoint 的情况下顺序调用相同函数，不重新实现编译、校验和回传规则。

核心操作不得导入 Coordinator、ResearchLoopState、LangGraph、桌面 UI、报告生成器；参数中不得携带整份图状态或万能 `services` 对象。

## 3. 具体代码去向

### 3.1 模型请求：改现有 `agent/dynamic.py`

- 把 `_tool_schemas` 与消息组装整理为公开、无 I/O 的 `build_tool_request`；返回实际 messages、tools、purpose，不调用模型。
- `select` 仅调用该函数及现有 gateway；不新增 SelectorService/ModelInvoker 等转发类。
- ModelGateway 的签名固定包含 purpose，更新项目内测试替身以符合 Protocol；逐步移除为旧测试替身服务的运行时签名探测。若确有外部旧实现，在装配处提供一个兼容适配器。
- 提供保存请求及重新提交请求的最小入口。可以只测模型输出，不必准备统计数据、审批界面或研究报告。

### 3.2 调用编译：迁移成 `tools/calls.py` 中的普通函数

- 从 execution 移出 `compile_dynamic_call`，从 analysis_selector 移出配方展开及批次编译，合并为可直接调用的入口。
- 把现有 `compile_function_parameters` 下沉到工具层，计划编译器和动态调用共同引用；工具核心不再反向依赖“研究计划编译”模块。
- 分清边界：模型参数检查 → 本地配置注入 → 执行参数校验；原始提议与编译结果都返回，不静默混合。
- 返回 calls、provider→child groups、逐提议错误和必要的协议回复数据；函数不写图状态、不做 UI 展示、不读文件、不调用模型。
- 保留现有合法提议与非法提议隔离、配方整体预检、预算不足不执行、provider ID 不重复、确定性 work ID/call ID 语义。
- 固定计划与动态范围保留不同授权规则，适配为同一个已编译 ToolCall 执行入口；不要为统一外观丢失审批内容绑定。

### 3.3 工具执行：保留 `tools/executor.py`，去掉中间转发依赖

- 执行器只接收编译调用、已准备 context、policy、数据指纹。数据准备在一次范围/计划进入执行阶段时完成；恢复时按相同指纹重建或复用缓存。
- `EDAExecutionService` 留给应用层准备数据和最终组装研究结果。图的单工具执行不再经过一个同时懂报告和评估的服务。
- 固定与动态结果校验共享一个核心函数；范围权限检查与调用内容完整校验保留各自职责。
- 新鲜结果在指定的验证节点校验一次；缓存加载、checkpoint 恢复、最终证据合并仍在信任边界复核。不能把这些必要复核误删为“重复”。
- 底层异常保留 `raise ... from ...` 的原始异常链。批次逻辑统一映射为调用错误，图再决定重试/暂停；统计函数不构建 FeedbackPacket 或 UI 文案。

### 3.4 图节点：移走业务逻辑，而非仅拆文件

- analysis_selector 只负责读取本轮输入、调用模型入口与编译入口、写回状态和路由。
- mark / execute / validate / complete 保留逐工具 checkpoint 边界。缓存与重试状态推进由明确的函数负责，不让各节点自行修改同一计数。
- 从成功和失败路径提取统一的调用组结算与消息构建逻辑；一个 provider call 只关闭一次，局部失败保留成功子结果。
- UI 事件由调用状态变化生成，放在图适配层或已有 narration/process_events 模块；核心逻辑不发桌面进度信号。
- 首轮不改 checkpoint JSON 字段结构。先限制每组字段的写入者并移出纯逻辑，后续确有需要再做 schema 版本迁移；不复制一份新旧状态长期双写。

### 3.5 组装与 handler：移除隐式行为和二次分派

- 把默认 gateway/agents/registry 的组装放在明确的工厂中，coordinator 接收明确依赖。动态/固定模式显式选择，不根据是否存在某个方法推断。
- 为常用工具提供具名 handler，可直接跳转和设置断点。随后将聚合分析函数中的具体统计操作提成可直接引用的函数，旧聚合入口也调用同一实现，避免复制统计代码。
- handler 可以保留一层必要的 context/输出适配；不要把每个统计函数继续包装成 ToolService → Runner → Adapter → Handler。
- 工具说明继续使用现有 catalog，不为工程整改引入新的工具 DSL、插件框架或元数据平台。

## 4. 调试与模型准确性测试的工程入口

新增一个轻量脚本，例如 `scripts/debug_tool_turn.py`。不创建 coordinator，不启动桌面，不自动生成报告，默认执行一轮，不自动重试。

| 模式 | 依赖 | 用途 |
|---|---|---|
| request | 案例上下文 + 工具目录 | 导出模型实际会收到的 messages/tools |
| select | 上述请求 + 真实 gateway | 只获取 ModelToolTurn，观察选工具准确性 |
| compile | 已保存 ModelToolTurn + 明确范围/配置 | 不调用模型，直接复现配方展开、参数或权限问题 |
| execute | 编译调用 + 冻结数据 context | 不调用模型，直接调试 handler 与输出校验 |
| round | 同一组输入 | 顺序运行选择、编译、执行、回传，保存各阶段结果 |

这些模式调用生产使用的同一函数。脱离图只代表省去 UI 和持久化调度，不代表绕过权限、数据指纹、预算或结果校验。fixture 中的已批准范围是明确的测试输入。

选择评测采用普通参数化案例：输入为准确的请求上下文，输出为原始 ModelToolTurn，对照预期工具集合与参数断言即可。运行器替换 gateway 时无需改流程；评分器只接收案例期望与实际调用，不导入研究 evaluator，不需要猴子补丁。

保存 request.json、proposal.json、compiled.json、outputs.json、tool_messages.json，使同一次错误可以从任意阶段复现。编译、执行和回传的问题可以不消耗模型调用反复调试。研究评估仍负责“证据是否足以回答研究问题”，独立评分器负责“模型的工具提议是否正确”。

多轮首次保留现有图集成测试，不另造一套生产循环。待单轮接口稳定，才能提取相同的状态推进函数供无图驱动器使用；涉及恢复、审批、取消仍由图集成测试验收。

## 5. 按依赖顺序迁移

1. **第一批：打开独立调试入口。** 暴露模型请求构建；提取批次编译及配方展开；让图调用新函数；增加 select/compile 脚本入口。将选工具、参数、配方的大部分断言迁移为直接函数测试，保留代表性图集成用例。此批必须交付可运行代码，不能只新建目录和转发类。
2. **第二批：收短执行链。** 将参数编译下沉，明确准备数据与单工具执行的边界，合并结果校验实现；增加 execute 模式；证明一个工具可从冻结 context 直接执行。
3. **第三批：收敛图内状态操作。** 统一批次成功/失败结算、错误回传与 UI 事件投影；显式装配依赖；保持旧 checkpoint 可恢复。
4. **第四批：消除 handler 二次分派，批量测真实模型。** 逐工具改成具名统计入口；在已稳定的 select 接口上批量运行测试案例，避免再次绑定完整研究工作流。

每批验证行为等价：原始提议、最终参数、工具输出、provider 回传及错误分类。除时间戳等运行字段外，同一冻结输入应得到一致结果；统计值按既定精度比较。优先覆盖配方局部失败、混合有效/无效提议、重复 ID、预算不足、缓存复用、范围越界和执行中恢复。

## 6. 工程验收标准

- 能在 PyCharm 中直接调试一轮模型选择、一个提议编译、一个工具执行，不创建 ResearchCoordinator。
- 核心测试无需 monkeypatch 研究评估器、报告写出或 UI 进度函数；只替换真正的外部依赖。
- 图节点不再实现配方展开、参数默认注入、工具输出裁剪等业务细节；普通函数不接收完整 ResearchLoopState。
- 编译与执行入口唯一，调试脚本、评测脚本、生产图共享；没有“测试通过的是另一条简化路径”。
- 从 provider call ID 能找到所有原子调用；工具完成后崩溃恢复不重复执行已提交子结果。保留目前的持久化保证，不宣称对所有外部副作用实现 exactly-once。
- 第一批完成后即可测真实模型工具选择；不以新增界面、补齐所有工具文档或跑完整研究报告作为前置条件。
