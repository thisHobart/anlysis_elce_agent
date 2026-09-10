# 数据面板实现规格

> 职责：桌面右侧数据面板的 UI 层实现规格（设计方案类文档）
> 上游：数据接入架构改造 —— 数据库取数、物化快照、审批契约
> 影响文件：`app/desktop/panes.py` · `app/desktop/input_config.py` · `app/desktop/session.py` · `app/desktop/workspace.py` · `app/desktop/theme.py` · `app/desktop/message_widgets.py`
> 状态：设计方案，未实现

---

## 0 这份文档要解决什么

右侧「数据文件」面板被「数据」面板取代：用户不再选文件，数据由 Agent 探查数据库后确定。

**面向的使用者是电力业务人员，不是程序员。** 这一条不是措辞偏好，是本规格的第一约束——它决定了下面每一处文案、每一个字段和一条硬边界。

---

## 1 术语红线

界面上**可以**出现的词，是电力行业日常用语：

> 实时电价、日前电价、节点电价、统调负荷、风电出力、光伏出力、外来电、非市场化机组出力、峰谷、检修、15 分钟一个点

界面上**不得**出现的词，是程序与数据库概念：

> 表名（`t_market_rt_price`）、字段名（`rt_node_price_2b`、`ts_day`、`p001`）、SQL 片段、`WHERE`/`JOIN`、数据指纹、取数契约、快照 ID、Skill 版本号、`max_lag`、宽表 / 长表 / unpivot、连接串、SSH 隧道、端口

**边界例外**：「查看取数细节」弹窗（§7.3）是唯一允许出现红线词的地方。它默认不可见，需两次点击才能到达。

**这条红线适用于**：控件文本、占位符、`QMessageBox` 文案、`SessionMessage.content`、`TraceEvent.summary`、异常给用户看的部分。**不适用于**：日志、研究包内部 JSON、`TraceEvent.details`。

红线可作为 review checklist；如果后续加 `CLAUDE.md`，这一节应原样搬过去。

---

## 2 硬边界：UI 只认 `DataSummary`

**UI 层不得 import 任何 `app.research.data.sources.*` 的契约类型**（`DatasetSpec`、`ConnectionProfile`、`MaterializedSnapshot`），也不得自行拼接展示字符串。

理由：人话摘要必须是**确定性**的，且只生成一次。如果散在控件里现拼，每个控件都要懂数据库结构；如果交给模型生成，同一份数据每次显示的措辞都不一样，研究包也不再可复现。

### 2.1 契约

新增 `app/research/data/sources/summary.py`，纯展示数据结构，无行为：

```python
@dataclass(frozen=True)
class VariableLabel:
    display: str            # "风电出力"
    resolved: bool          # False 表示没查到中文名，display 回落为原列名

@dataclass(frozen=True)
class DataSummary:
    price_label: str            # "山东电网 实时电价"
    variables: list[VariableLabel]
    start_date: str             # "2026年1月1日"
    end_date: str               # "2026年9月9日"
    granularity_text: str       # "每 15 分钟一个点"
    gap_text: str | None        # "缺 47 个点" / None 表示无缺口
    fetched_at_text: str        # "今天 15:32"
```

日期与时间已在此层格式化成中文，UI 直接贴。`start_date` 等不是 `datetime`，正是为了防止 UI 层自行决定格式。

### 2.2 谁生成

`materialize.py` 在物化完成后产出 `DataSummary`，随 `MaterializedSnapshot` 一起返回。勘探中途的部分摘要由 `explore` 子 Agent 节点产出同一结构（未确定的字段留空）。

### 2.3 变量中文名从哪来

数据库列名是 `wind_power` / `p001` / `type=2`，界面要显示「风电出力」。这个映射：

- **必须确定性**：一张词表，不是模型翻译。模型翻译会让同一列在两次会话里叫两个名字，可复现性直接失效。
- **位置**：`app/research/data/sources/naming.py` + `configs/variable_names.yaml`
- **匹配顺序**：`(表名, 列名)` 精确匹配 → `列名` 精确匹配 → `列名` 规范化后匹配（小写、去下划线）→ 未命中
- **未命中的兜底**：`display` 回落为原列名，`resolved=False`，**不阻断流程**；同时写一条 `TraceEvent`（category=`input`，status=`warning`，summary 用人话：「有 1 项数据没有中文名，先按原名显示」）。UI 对 `resolved=False` 的项加 `#B45309` 色，鼠标悬停提示「这项数据还没配中文名」。

宁可显示 `p001` 也不要显示「变量 1」——前者至少能让业务人员找运维问清楚，后者什么信息都没有。

---

## 3 组件结构

| 现有 | 变更 | 说明 |
|---|---|---|
| `FileSlotRow` | **删除** | 拖拽、`QFileDialog`、`FILE_FILTERS` 一并移除 |
| `InputFilesPanel` | **替换为 `DataPanel`** | `objectName` 由 `inputFilesPanel` 改为 `dataPanel` |
| `ContextPane.inputs` | 改名 `ContextPane.data_panel` | `setSizes([290, 510])` → `setSizes([262, 538])` |

`DataPanel` 对外接口：

```python
class DataPanel(QFrame):
    refetch_requested = Signal()        # 「取最新的」
    reselect_requested = Signal()       # 「换一批数据」
    details_requested = Signal()        # 「查看取数细节」
    retry_requested = Signal()          # 取不到态的「再试一次」
    local_file_requested = Signal()     # 取不到态的「用本地文件」

    def set_state(self, state: DataPanelState, summary: DataSummary | None) -> None: ...
    def set_busy(self, busy: bool) -> None: ...
```

```python
DataPanelState = Literal["empty", "exploring", "ready", "unavailable"]
```

`empty`（新会话、还没提问）显示一行灰字：**还没开始。说说你想研究什么，我来找数据。**

---

## 4 三态 × 文案表

**下列字符串是规格的一部分，实现时逐字照抄，不要改写。**

### 4.1 通用

| 位置 | 文案 |
|---|---|
| 面板标题 | `数据` |
| 状态 · ready | `已就绪`（✓ 图标，`#0F766E`） |
| 状态 · exploring | `正在找数据`（旋转图标，`#3F51B5`） |
| 状态 · unavailable | `取不到`（⚠ 图标，`#BE123C`） |
| 字段名 | `电价` / `影响因素` / `时间范围` |

### 4.2 ready

| 位置 | 文案 |
|---|---|
| 电价 | `{price_label}` |
| 影响因素 | 变量中文名以 `、` 连接，不截断、不写「等 N 项」 |
| 时间范围 | `{start_date} — {end_date}`（全角破折号前后各一个空格） |
| 时间范围副行 | `{granularity_text}，{gap_text}`；无缺口时只显示 `{granularity_text}` |
| 底部 | `数据取自{fetched_at_text}`（**不加空格**，如「数据取自今天 15:32」） |
| 按钮 | `取最新的` |
| 灰字链接 | `换一批数据 · 查看取数细节` |

### 4.3 exploring

| 位置 | 文案 |
|---|---|
| 进度条上方 | `正在看有哪些数据能用…` |
| 已确定项 | 正常显示（✓ `#0F766E`） |
| 进行中项 | `#98A2B3`，内容形如 `已找到风电、光伏、统调负荷，还在看气温` |
| 未开始项 | `待定`（`#98A2B3`，空心圆图标 `#C9CDD4`） |
| 底部说明 | `现在只是在看有什么数据，还没开始取。找完会先给你确认。` |

最后这句消解「它是不是已经在动我的库了」的疑虑，不能省。

### 4.4 unavailable

| 位置 | 文案 |
|---|---|
| 提示标题 | `现在取不到数据` |
| 提示正文 | `和数据服务器连不上。稍等一下再试；一直不行就找运维看看，或者先用本地文件继续。` |
| 按钮 | `再试一次`（primary） / `用本地文件`（quiet） |
| 历史卡标题 | `上次用的数据` |
| 底部说明 | `上次取的数据还在，可以直接接着分析，只是不是最新的。` |

**无论底层异常是什么**（连接超时、认证失败、隧道未建、库不存在、权限不足），面板文案都是上面这一套。差异只进 `TraceEvent.details` 和日志。业务人员对这些区别做不出任何不同的动作，区分只会制造焦虑。

上次数据不存在时（首次使用即失败），隐藏历史卡与底部说明。

---

## 5 布局与样式

沿用 `theme.py` 现有令牌，不新增颜色。

```
padding 8px · 元素间距 6px
标题行     : contextTitle（13.5px / 600）+ 右侧状态（12px）
主卡片     : 白底 · 1px #E7EAEF · 圆角 6px
  条目     : padding 9px 10px；条目间 1px #EEF0F3 分隔（新增分隔色，仅此处）
  字段名   : 12px #667085
  字段值   : 13px / line-height 1.55
  副行     : 12px #98A2B3
底部行     : 12px #667085 + quietButton（28px）
灰字链接   : 12px #98A2B3
```

`theme.py` 新增 `objectName`：`dataPanel`（替换 `inputFilesPanel`）、`dataCard`、`dataKey`、`dataValue`、`dataSub`、`dataQuietLink`。`fileSlot` / `fileName` / `fileRole` 三条规则删除。

面板高度约 262px，比现有 290px 略矮，「研究过程」表格相应多出约 2 行。分栏比例不需要额外调整。

---

## 6 交互与失效规则

| 操作 | 行为 |
|---|---|
| `取最新的` | 用同一份取数口径重新物化。数据变了 → 走 §6.1 失效 |
| `换一批数据` | 丢弃当前口径，重新进入勘探；等价于换了研究输入 → 走 §6.1 |
| `再试一次` | 仅重连，不改口径；成功后回到 `ready`，不失效 |
| `用本地文件` | 回落到文件模式（`source_kind="file"`），弹出 `QFileDialog` |
| 全部按钮 | `is_busy` 为真时禁用，与现有 `set_busy` 一致 |

### 6.1 方案失效

现有 `_invalidate_plan_for_input_change()` 保留，触发条件由「文件路径变了」改为「**新快照的数据指纹与当前方案所锚定的不同**」。

指纹相同则不失效——用户点了「取最新的」但数据库其实没新数据，这是常见操作，不该白白作废一个已批准的方案。

失效通知文案改为：

> `数据换了，之前的分析方案已经作废。重新问一次，我按新数据给方案。`

---

## 7 确认卡片

### 7.1 呈现位置

会话流里的一条 `SessionMessage`，`kind` 新增 `"data_plan"`，在 `message_widgets.py` 中渲染。沿用 `planMessage` 外观（白底 · 1px `#E7EAEF` · 圆角 10px）。

它取代现有的纯方案审批卡：**取数口径与分析步骤合并为一次确认**。用户批准的从来不是「这些分析步骤」，而是「用这些数据做这些分析」，拆成两次会产生「批了数据没批方案」的中间态。

### 7.2 内容与文案

| 位置 | 文案 |
|---|---|
| 标题 | `开始之前，跟你确认一下` |
| 倒计时 chip | `{n} 分 {m} 秒后自动开始` |
| 分节一 | `要用的数据` |
| 分节一内容 | `电价` / `影响因素` / `时间范围` 三行，与面板同源同措辞 |
| 分节二 | `要做的事` |
| 分节二内容 | 每条一行，无编号前缀，写成动作而不是方法名 |
| 底部提示 | `点「可以开始」之后，这批数据会先固定下来。后面数据库再更新，也不会影响这一轮的结论。` |
| 按钮 | `可以开始` / `我想改改` / `先不做` |

分节二的每一条由 Skill 的研究协议提供人话描述，**不是**函数名或参数。`price-exogenous-eda` 的 `references/research-protocol.yaml` 需为每个步骤补一个 `display_text` 字段。这是本规格对后端提的唯一新增要求。

### 7.3 查看取数细节

`QDialog`，从面板灰字链接或卡片进入。这是唯一允许出现红线词的界面：库名、表名、字段名、过滤条件、时间窗、快照指纹、SQL 原文（只读、可复制）。

出了怪结果时排查用，普通用户永远不会点开。默认不显示，无快捷键。

---

## 8 会话持久化与迁移

`SessionInputFile` / `InputRole` / `default_input_files()` 保留，用于 `source_kind="file"` 兜底路径。

`ResearchSession` 新增：

```python
source_kind: Literal["database", "file"] = "database"
data_state: DataPanelState = "empty"
data_summary: DataSummary | None = None
dataset_fingerprint: str | None = None
```

`ResearchSession` 是 `extra="ignore"`，旧会话缺这些字段时按默认值加载，落到 `empty` 态——旧会话打开后需要重新提问才能继续，这是可接受的。`inputs` 字段在旧会话里仍存在且被忽略，无需清理。

**不要**把 `DatasetSpec` 或 `MaterializedSnapshot` 写进 `ResearchSession`：它们是 `extra="forbid"` 的契约模型，进了长期存活的会话文件就会在下次加字段时炸掉旧会话。会话里只存 `dataset_fingerprint` 这个字符串。

---

## 9 验收清单

1. 全新会话打开 → 面板 `empty` 态，一行灰字，无任何按钮。
2. 提问后 → `exploring` 态逐项点亮，底部说明常驻。
3. 勘探完成 → 会话流出现确认卡片，面板进 `ready` 态，两处措辞逐字一致。
4. 点「可以开始」→ 执行；期间手工向数据库插入新数据，结论不受影响，`fetched_at_text` 不变。
5. 点「取最新的」且数据无变化 → 已批准方案**不**失效。
6. 点「取最新的」且数据有变化 → 方案失效，通知文案为 §6.1 原文。
7. 拔掉网络 → `unavailable` 态，文案为 §4.4 原文，历史卡显示上次数据。
8. 词表删掉一项 → 该项显示原列名 + `#B45309`，流程不中断，研究过程多一条 warning。
9. **术语扫描**：对全部界面字符串（含 `QMessageBox` 与 `TraceEvent.summary`）grep §1 红线词，除「查看取数细节」弹窗外零命中。
10. 打开一个改造前保存的旧会话，不抛异常。

---

## 10 待确认

1. **电价数据的真实库表位置**。`shandong_db` 中只有负荷与新能源出力表，未见电价表。这不影响本规格，但决定 §2.3 词表的初始内容。
2. **宽表时间轴形态**。若为「一行一天 + 96 点为列」，展开逻辑放在物化层且必须确定性可复现；界面上只体现为一句 `每 15 分钟一个点`，用户不感知这一步。
3. **`换一批数据` 是否保留**。它给了用户推翻 Agent 判断的入口，但也可能被误点导致重跑一轮勘探。若实测中很少用到，可只保留在细节弹窗里。
