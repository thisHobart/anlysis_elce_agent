# 山东 P2 新闻数据准备

`scripts.prepare_shandong_news` 从两个只读 MySQL 角色生成一次不可改写的山东新闻研究快照。数据库密码只从 `VPP_SHANDONG_DB_FEATURE_*` 和 `VPP_SHANDONG_DB_ANALYTICS_*` 环境变量读取，既不进入日志，也不进入 manifest。

## 两阶段运行

先生成数据库画像和搜索计划：

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_shandong_news `
  --as-of 2026-09-10T18:00:00+08:00 `
  --run-id 20260910T100000Z-plan
```

按 `search_plan.json` 执行检索，并把结果冻结为 JSONL。每行必须包含 `source_name`、`source_url`、`source_type`、`title`、`content`、带时区的 `published_at` 与 `fetched_at`，以及对应的 `search_query_id`。`content_scope` 必须明确为 `full_text`、`excerpt` 或 `summary`；只有 `full_text` 可以进入真实事件抽取。标题和正文始终是不可信数据，不会被执行为指令。

搜索服务另行输出执行账本 JSONL，每条包含 `query_id`、带时区的 `executed_at`、`search_service` 和 `returned_result_count`。没有执行账本时，系统只标记为“已导入结果”，不会声称查询已经执行；没有结果、也没有执行记录的查询标记为 `not_run`。

然后建立完整快照并调用现有三趟结构化抽取：

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_shandong_news `
  --as-of 2026-09-10T18:00:00+08:00 `
  --run-id 20260910T100000Z `
  --search-results .\collected-search-results.jsonl `
  --search-executions .\search-executions.jsonl `
  --extractor structured-model --extraction-passes 3
```

如果真实模型不可用，不要改用规则抽取器冒充资格。使用 `--extractor none --extraction-blocker "可审计的阻断原因"` 固化数据库、搜索和规范化结果；该运行会返回退出码 2，并在 manifest 中标记 `needs_review`。

## 关键边界

- 实际价格主表是 `t_data_province_days_cleared_price` 与 `t_data_province_real_time_cleared_price`；`t_data_province_days_forecase_production_price` 始终标记为预测结果。
- `ts_day + hour + min` 构成观察时间；`hour=24,min=0` 转换为次日 00:00，其他 `hour=24` 组合拒绝。
- 数据库快照同时按观察时间和 `create_time` 截止到统一 `as_of`，避免后来回填的数据进入历史运行。
- `database_profile.json` 保存账号实际可见的数据库名和当前库表名目录，但不保存无限量 schema 或样本行。
- 异常日必须具有至少 80% 的预期日内点数，避免将残缺分区误判为异常。
- `published_at` 不等于系统取得时间。本次历史搜索使用真实 `fetched_at` 作为 `first_seen_at`；不能回填历史时间来制造可回测特征。
- `raw_news` 中的 `snapshot_kind` 明确区分全文和人工摘录；摘录文件不能被描述为网页原始正文。
- 每个完成快照以 `manifest.json` 为完成标记，记录所有文件 SHA-256、统一 `as_of` 和版本，但不记录凭据。
