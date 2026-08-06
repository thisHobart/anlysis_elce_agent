"""VPP 数字人 FAQ 知识库（从 digital-human/skills/digital-human-faq 迁移）。

结构化的高频问答条目，供 ``app.services.faq_service`` 加载与匹配。
每条 ``FaqEntry`` 承载：触发词、回答模板、数据需求（路由到工具）、大屏联动指令。

迁移说明
--------
- **共 29 条**：源编号 FAQ-01~FAQ-30，但源文件缺失 FAQ-11（``02-实时数据类`` 从
  FAQ-10 直接跳到 FAQ-12），故实际 29 条。编号保持与源一致，不重新编号。
- **主/备数据源**：部分 FAQ 同一变量既有主源（api）又有备源（db）。备源条目标记
  ``fallback: True``，供 P5 取数失败时回退。
- **省略项**：源 md 中以 ``#`` 注释的"备选大屏 API 端点"提示未纳入结构化数据
  （它们是开发备注，非激活路由），如需可回源文档查阅。
- ``source`` 取值：``api`` | ``db`` | ``knowledge`` | ``static``，是 ``query_data``
  节点的路由依据。``static`` 表示模板已含文案、无需取数。

数据 schema
-----------
FaqEntry = {
    "faq_id": str,
    "category": str,                 # basic_info|realtime|statistics|trend|alert
    "triggers": list[str],
    "template": str,                 # {{var}} / {{if}} / {{#each}}
    "data_requirements": list[DataRequirement],
    "screen_action": str,
}
DataRequirement = {
    "variable": str | None,
    "source": "api"|"db"|"knowledge"|"static",
    "vpp_command": str,              # source == "api"
    "vpp_args": dict,                # source == "api"
    "db_command": str,               # source == "db"
    "db_query": str,                 # source == "db"（自定义 SELECT）
    "response_hint": str,            # 取字段提示（供 LLM/开发参考）
    "call_group": str,               # 同组变量合并为一次调用
    "note": str,
    "fallback": bool,                # True = 备用数据源
}
"""

from __future__ import annotations

from typing import Any

# 类别码 -> 中文名（供展示/统计）
CATEGORIES: dict[str, str] = {
    "basic_info": "基本信息类",
    "realtime": "实时数据类",
    "statistics": "统计数据类",
    "trend": "趋势对比类",
    "alert": "预警事件类",
}

# 无数据时的引导问题（源 guaranteed_questions.md），兜底随机挑 2-3 个
GUARANTEED_QUESTIONS: list[str] = [
    "什么是虚拟电厂？",
    "电厂接入了多少家企业？总容量多大？",
    "碳减排量是多少？",
    "哪种资源可调能力最强？",
]


FAQS: list[dict[str, Any]] = [
    # ============================ 01 基本信息类 (FAQ-01~05) ============================
    {
        "faq_id": "FAQ-01",
        "category": "basic_info",
        "triggers": ["什么是虚拟电厂", "虚拟电厂是什么", "虚拟电厂概念", "VPP是什么", "VPP概念"],
        "template": (
            "虚拟电厂就像一个电力管家，它不发电厂，但通过智能控制把分散的太阳能、储能、"
            "充电桩、工厂空调等资源聚合起来，统一参与电网调度。这样既能帮电网缓解压力，也能帮用户赚钱。"
        ),
        "data_requirements": [
            {"variable": "none", "source": "static", "description": "纯静态文本，无需数据注入"},
        ],
        "screen_action": "播放30秒概念动画",
    },
    {
        "faq_id": "FAQ-02",
        "category": "basic_info",
        "triggers": ["介绍虚拟电厂", "我们的虚拟电厂", "聊城虚拟电厂", "虚拟电厂介绍"],
        "template": (
            "聊城虚拟电厂由界无际能源科技公司运营，投运于2024年6月。目前已接入{{total_enterprises}}家企业，"
            "总聚合容量{{total_capacity_mw}}兆瓦，最大可调能力{{max_adjustable_capacity_mw}}兆瓦，"
            "覆盖聊城经济技术开发区、高新区等区域。"
        ),
        "data_requirements": [
            {"variable": "total_enterprises", "source": "db", "db_command": "vpp-overview", "description": "已接入企业数"},
            {"variable": "total_capacity_mw", "source": "db", "db_command": "vpp-overview", "description": "总聚合容量（兆瓦）"},
            {"variable": "max_adjustable_capacity_mw", "source": "db", "db_command": "vpp-overview", "description": "最大可调能力（兆瓦）"},
        ],
        "screen_action": "切换到区域用能总览，高亮企业总数和容量卡片",
    },
    {
        "faq_id": "FAQ-03",
        "category": "basic_info",
        "triggers": ["有哪些资源类型", "资源类型有哪些", "资源种类", "有哪些资源", "光伏储能充电桩"],
        "template": (
            "目前虚拟电厂已接入的资源涵盖{{type_count}}种类型、共{{total_count}}个站点，"
            "总接入容量{{total_capacity_kw}}千瓦。包括分布式光伏{{pv_count}}个站点、储能系统{{storage_count}}个站点、"
            "充换电设备{{charging_count}}个站点，以及柴油发电机组{{generator_count}}台作为可调节负荷资源。"
        ),
        "data_requirements": [
            {
                "variable": "resource_stats",
                "source": "db",
                "db_query": (
                    "SELECT COUNT(DISTINCT type) AS type_count, COUNT(*) AS total_count, "
                    "COALESCE(SUM(capacity),0) AS total_capacity_kw, "
                    "SUM(CASE WHEN type='fbsgf' THEN 1 ELSE 0 END) AS pv_count, "
                    "SUM(CASE WHEN type IN ('gsycn','qtcn') THEN 1 ELSE 0 END) AS storage_count, "
                    "SUM(CASE WHEN type='chdsb' THEN 1 ELSE 0 END) AS charging_count, "
                    "SUM(CASE WHEN type='cyfdj' THEN 1 ELSE 0 END) AS generator_count "
                    "FROM t_res_resource WHERE del_flag=0 OR del_flag IS NULL"
                ),
                "description": "资源类型统计（一次查询返回全部计数字段）",
            },
        ],
        "screen_action": "切换到能源结构总览，展示资源类型分布饼图",
    },
    {
        "faq_id": "FAQ-04",
        "category": "basic_info",
        "triggers": ["怎么赚钱", "盈利模式", "收益来源", "如何盈利", "怎么盈利"],
        "template": (
            "我们主要参与电力需求响应和辅助服务市场。当电网出现缺口时，我们组织用户压降负荷，"
            "获得电网补偿。补偿收益按比例分给用户，平台留取一部分作为运营费用。"
        ),
        "data_requirements": [
            {"variable": "none", "source": "static", "description": "纯静态文本，商业模式介绍；如需详细政策可扩展为 knowledge 来源"},
        ],
        "screen_action": "展示收益构成饼图",
    },
    {
        "faq_id": "FAQ-05",
        "category": "basic_info",
        "triggers": ["接入多少企业", "多少家企业", "总容量多大", "企业数和容量", "企业总数"],
        "template": (
            "目前已接入{{total_enterprises}}家企业，总接入容量{{total_capacity_mw}}兆瓦，"
            "其中工业占60%，商业占25%，充换电占15%。"
        ),
        "data_requirements": [
            {"variable": "total_enterprises", "source": "db", "db_command": "vpp-overview", "description": "已接入企业数"},
            {"variable": "total_capacity_mw", "source": "db", "db_command": "vpp-overview", "description": "总接入容量（兆瓦）"},
        ],
        "screen_action": "高亮企业总数和容量卡片，显示行业占比饼图",
    },

    # ============================ 02 实时数据类 (FAQ-06~15, 缺 FAQ-11) ============================
    {
        "faq_id": "FAQ-06",
        "category": "realtime",
        "triggers": ["当前发电功率", "总发电功率", "发电功率多少", "现在发电多少", "实时发电"],
        "template": "当前总发电功率为{{total_power}}兆瓦，其中光伏{{pv_power}}兆瓦，储能放电{{storage_discharge}}兆瓦。",
        "data_requirements": [
            {"variable": "total_power", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["total_generation_power"]}, "response_hint": "data.total", "call_group": "power",
             "description": "总发电功率（兆瓦）"},
            {"variable": "pv_power", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["total_generation_power"]}, "response_hint": "data.solar", "call_group": "power",
             "description": "光伏发电功率（兆瓦）"},
            {"variable": "storage_discharge", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["total_generation_power"]}, "response_hint": "data.storage_discharge",
             "call_group": "power", "description": "储能放电功率（兆瓦）"},
        ],
        "screen_action": "切换到\"能源运行管理\"视图，高亮总发电功率仪表盘",
    },
    {
        "faq_id": "FAQ-07",
        "category": "realtime",
        "triggers": ["储能SOC", "储能状态", "SOC多少", "电池电量", "储能电量"],
        "template": "储能系统平均SOC为{{avg_soc}}%。站点A {{soc_a}}%（偏高），站点B {{soc_b}}%（正常），站点C {{soc_c}}%（偏低）。",
        "data_requirements": [
            {"variable": "avg_soc", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["storage_soc"]}, "response_hint": "data.average", "call_group": "soc",
             "description": "平均SOC（百分比）"},
            {"variable": "soc_a", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["storage_soc"]}, "response_hint": "data.stations[0].soc", "call_group": "soc",
             "description": "站点A的SOC"},
            {"variable": "soc_b", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["storage_soc"]}, "response_hint": "data.stations[1].soc", "call_group": "soc",
             "description": "站点B的SOC"},
            {"variable": "soc_c", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["storage_soc"]}, "response_hint": "data.stations[2].soc", "call_group": "soc",
             "description": "站点C的SOC"},
        ],
        "screen_action": "GIS地图上储能站点以红/黄/绿圆点标注，右侧SOC柱状图",
    },
    {
        "faq_id": "FAQ-08",
        "category": "realtime",
        "triggers": ["总可调能力", "平台可调能力", "总可调节容量", "当前可调容量", "削峰能力", "填谷能力", "可调节容量", "当前可调节容量"],
        "template": "当前可调节容量为{{total_adjustable}}兆瓦，其中削峰能力{{peak_shaving}}兆瓦，填谷能力{{valley_filling}}兆瓦。",
        "data_requirements": [
            {"variable": "total_adjustable", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["adjustable_capacity"]}, "response_hint": "data.peak_shaving + data.valley_filling",
             "call_group": "adjustable", "description": "总可调节容量（兆瓦）"},
            {"variable": "peak_shaving", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["adjustable_capacity"]}, "response_hint": "data.peak_shaving",
             "call_group": "adjustable", "description": "削峰能力（兆瓦）"},
            {"variable": "valley_filling", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["adjustable_capacity"]}, "response_hint": "data.valley_filling",
             "call_group": "adjustable", "description": "填谷能力（兆瓦）"},
        ],
        "screen_action": "切换到\"可调资源总览\"，高亮可调能力卡片",
    },
    {
        "faq_id": "FAQ-09",
        "category": "realtime",
        "triggers": ["企业实时负荷", "XX企业负荷", "企业负荷多少", "负载是多少", "实时负荷", "负荷是多少", "的实时负荷", "负荷多少"],
        "template": "{{enterprise_name}}当前实时负荷为{{load}}千瓦，占其可调能力的{{ratio}}%。",
        "data_requirements": [
            {"variable": "enterprise_name", "source": "static", "description": "从用户问题中提取的企业名称"},
            {"variable": "load", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["enterprise_realtime_load"], "enterprise_ids": ["{{企业ID}}"]},
             "response_hint": "data.enterprises[0].load", "call_group": "ent_load",
             "note": "需先调 vpp_db companies 将企业名称解析为企业ID", "description": "实时负荷（千瓦）"},
            {"variable": "ratio", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["enterprise_realtime_load"], "enterprise_ids": ["{{企业ID}}"]},
             "response_hint": "data.enterprises[0].ratio", "call_group": "ent_load",
             "description": "负荷占可调能力百分比"},
        ],
        "screen_action": "切换到企业详情页，高亮实时负荷曲线",
    },
    {
        "faq_id": "FAQ-10",
        "category": "realtime",
        "triggers": ["今天响应情况", "今日响应", "响应任务", "响应率", "负荷响应"],
        "template": (
            "今日截至目前，共执行{{task_count}}个响应任务。总中标量{{bid_mw}}兆瓦，"
            "实际响应量{{actual_mw}}兆瓦，总体响应率{{response_rate}}%。"
        ),
        "data_requirements": [
            {"variable": "task_count", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "response_count", "period": "day", "start": "{{today}}", "end": "{{today}}"},
             "response_hint": "data.value", "call_group": "today_response", "description": "今日响应任务数量"},
            {"variable": "bid_mw", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "response_count", "period": "day", "start": "{{today}}", "end": "{{today}}"},
             "response_hint": "data.bid_mw", "call_group": "today_response", "description": "总中标量（兆瓦）"},
            {"variable": "actual_mw", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "response_count", "period": "day", "start": "{{today}}", "end": "{{today}}"},
             "response_hint": "data.actual_mw", "call_group": "today_response", "description": "实际响应量（兆瓦）"},
            {"variable": "response_rate", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "response_success_rate", "period": "day", "start": "{{today}}", "end": "{{today}}"},
             "response_hint": "data.value 或 data.rate", "call_group": "today_response", "description": "总体响应率（百分比）"},
        ],
        "screen_action": "切换到\"执行监测\"视图，显示响应率环形图及实时曲线",
    },
    # NOTE: 源文件缺失 FAQ-11（02-实时数据类 从 FAQ-10 直接到 FAQ-12），此处不虚构。
    {
        "faq_id": "FAQ-12",
        "category": "realtime",
        "triggers": ["多少设备在线", "设备在线率", "在线离线", "设备状态", "设备在线"],
        "template": "当前在线设备{{online_count}}台，离线{{offline_count}}台，总体在线率{{online_rate}}%。",
        "data_requirements": [
            {"variable": "online_count", "source": "db", "db_command": "device-online", "description": "在线设备数"},
            {"variable": "offline_count", "source": "db", "db_command": "device-online", "description": "离线设备数"},
            {"variable": "online_rate", "source": "db", "db_command": "device-online", "description": "在线率（百分比）"},
        ],
        "screen_action": "切换到\"资源运行监测\"视图，显示在线/离线仪表盘",
    },
    {
        "faq_id": "FAQ-13",
        "category": "realtime",
        "triggers": ["储能充放", "储能充电还是放电", "储能模式", "储能现在", "储能状态"],
        "template": (
            "储能系统当前处于{{mode}}状态，总功率{{power}}兆瓦。"
            "其中站点A充电{{charge_a}}兆瓦，站点B放电{{discharge_b}}兆瓦。"
        ),
        "data_requirements": [
            {"variable": "mode", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["storage_soc"]}, "response_hint": "data.stations[0].mode（需后端扩展充放状态字段）",
             "call_group": "storage_mode", "description": "充电/放电/待机"},
            {"variable": "power", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["storage_soc"]}, "response_hint": "data.stations[0].power（需后端扩展）",
             "call_group": "storage_mode", "description": "总功率（兆瓦）"},
            {"variable": "charge_a", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["storage_soc"]}, "response_hint": "data.stations[0].charge_power（需后端扩展）",
             "call_group": "storage_mode", "description": "站点A充电功率"},
            {"variable": "discharge_b", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["storage_soc"]}, "response_hint": "data.stations[1].discharge_power（需后端扩展）",
             "call_group": "storage_mode", "description": "站点B放电功率"},
        ],
        "screen_action": "高亮储能充放电功率曲线，标注充放状态",
    },
    {
        "faq_id": "FAQ-14",
        "category": "realtime",
        "triggers": ["缺口任务", "当前任务", "有没有任务", "进行中的任务", "响应任务"],
        "template": (
            "{{if has_task}}当前有一个进行中的缺口任务：'{{task_name}}'，响应时段{{time_slot}}，"
            "缺口量{{gap_mw}}兆瓦。{{else}}当前没有进行中的缺口任务。{{endif}}"
        ),
        "data_requirements": [
            {"variable": "has_task", "source": "api", "vpp_command": "gap-list",
             "vpp_args": {"page_num": 1, "page_size": 10, "status": [1]},
             "response_hint": "records 中存在 status=1(已发布) 或 step=1 的任务", "call_group": "active_gaps",
             "description": "是否有进行中任务"},
            {"variable": "task_name", "source": "api", "vpp_command": "gap-list",
             "vpp_args": {"page_num": 1, "page_size": 10, "status": [1]},
             "response_hint": "records[0].gapName", "call_group": "active_gaps", "description": "任务名称"},
            {"variable": "time_slot", "source": "api", "vpp_command": "gap-list",
             "vpp_args": {"page_num": 1, "page_size": 10, "status": [1]},
             "response_hint": "records[0].interval", "call_group": "active_gaps", "description": "响应时段描述"},
            {"variable": "gap_mw", "source": "api", "vpp_command": "gap-list",
             "vpp_args": {"page_num": 1, "page_size": 10, "status": [1]},
             "response_hint": "records[0].totalLoad(kW) 转兆瓦", "call_group": "active_gaps", "description": "缺口量（兆瓦）"},
            {"variable": "has_task", "source": "db", "db_command": "active-gaps", "fallback": True,
             "description": "是否有进行中任务（备用数据源）"},
        ],
        "screen_action": "若有任务，切换到\"邀约跟踪\"视图并高亮该任务",
    },
    {
        "faq_id": "FAQ-15",
        "category": "realtime",
        "triggers": ["企业可调能力", "企业的可调能力", "XX可调能力", "可调能力怎么样", "的可调能力"],
        "template": (
            "{{enterprise_name}}当前可调能力为{{adjustable}}兆瓦，其中削峰能力{{peak}}兆瓦，"
            "填谷能力{{valley}}兆瓦。该企业保供等级为{{level}}。"
        ),
        "data_requirements": [
            {"variable": "enterprise_name", "source": "static", "description": "从用户问题中提取的企业名称"},
            {"variable": "adjustable", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["enterprise_realtime_load"], "enterprise_ids": ["{{企业ID}}"]},
             "response_hint": "data.enterprises[0].adjustable（需后端扩展）", "call_group": "ent_adjustable",
             "note": "先调 vpp_db companies 获取企业名→ID 映射", "description": "可调能力（兆瓦）"},
            {"variable": "peak", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["adjustable_capacity"], "enterprise_ids": ["{{企业ID}}"]},
             "response_hint": "data.peak_shaving（需支持企业粒度）", "call_group": "ent_adjustable",
             "description": "削峰能力（兆瓦）"},
            {"variable": "valley", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["adjustable_capacity"], "enterprise_ids": ["{{企业ID}}"]},
             "response_hint": "data.valley_filling（需支持企业粒度）", "call_group": "ent_adjustable",
             "description": "填谷能力（兆瓦）"},
            {"variable": "level", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["enterprise_realtime_load"], "enterprise_ids": ["{{企业ID}}"]},
             "response_hint": "data.enterprises[0].level（需后端扩展）", "call_group": "ent_adjustable",
             "description": "保供等级（A/B/C）"},
        ],
        "screen_action": "高亮该企业的可调能力卡片，显示等级标签",
    },

    # ============================ 03 统计数据类 (FAQ-16~23) ============================
    {
        "faq_id": "FAQ-16",
        "category": "statistics",
        "triggers": ["本月发电量", "月累计发电", "本月发了多少", "月度发电"],
        "template": "本月（{{month}}）截至今日，累计发电量{{generation}}万度，其中光伏{{pv_gen}}万度，储能放电{{storage_gen}}万度。",
        "data_requirements": [
            {"variable": "month", "source": "static", "description": "当前月份（从系统时间获取）"},
            {"variable": "generation", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "month", "start": "{{本月1日}}", "end": "{{今天}}", "group_by": "none"},
             "response_hint": "data.value 或 data.total", "call_group": "monthly_gen", "description": "本月累计发电量（万度）"},
            {"variable": "pv_gen", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "month", "start": "{{本月1日}}", "end": "{{今天}}", "group_by": "resource_type"},
             "response_hint": "data.breakdown.solar", "call_group": "monthly_gen_by_resource", "description": "光伏发电量（万度）"},
            {"variable": "storage_gen", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "month", "start": "{{本月1日}}", "end": "{{今天}}", "group_by": "resource_type"},
             "response_hint": "data.breakdown.storage", "call_group": "monthly_gen_by_resource", "description": "储能放电量（万度）"},
        ],
        "screen_action": "切换到\"能源结构总览\"，展示本月发电量柱状图",
    },
    {
        "faq_id": "FAQ-17",
        "category": "statistics",
        "triggers": ["上周响应成功率", "响应成功率", "上周响应", "成功率怎么样", "上周任务"],
        "template": (
            "上周（{{start_date}}至{{end_date}}）共执行响应任务{{task_count}}次，总成功率{{success_rate}}%。"
            "最高为{{max_date}}任务（{{max_rate}}%），最低为{{min_date}}任务（{{min_rate}}%）。"
        ),
        "data_requirements": [
            {"variable": "start_date", "source": "static", "description": "上周起始日期（据当前日期推算）"},
            {"variable": "end_date", "source": "static", "description": "上周结束日期（据当前日期推算）"},
            {"variable": "task_count", "source": "api", "vpp_command": "gap-list",
             "vpp_args": {"page_num": 1, "page_size": 50, "respond_day": "{{上周年月}}"},
             "response_hint": "上周缺口列表条数", "call_group": "weekly_gaps", "description": "响应任务数"},
            {"variable": "success_rate", "source": "api", "vpp_command": "monitor-detail",
             "vpp_args": {"gap_code": "{{逐个gapCode}}"},
             "response_hint": "逐缺口调 monitor-detail，汇总 loadPercentage 求均值", "call_group": "weekly_monitor",
             "description": "总成功率（百分比）"},
            {"variable": "max_date", "source": "api", "vpp_command": "monitor-detail",
             "vpp_args": {"gap_code": "{{逐个}}"}, "response_hint": "平均响应率最高缺口的 respondDay",
             "call_group": "weekly_monitor", "description": "最高成功率日期"},
            {"variable": "max_rate", "source": "api", "vpp_command": "monitor-detail",
             "vpp_args": {"gap_code": "{{逐个}}"}, "response_hint": "该缺口平均响应率", "call_group": "weekly_monitor",
             "description": "最高成功率"},
            {"variable": "min_date", "source": "api", "vpp_command": "monitor-detail",
             "vpp_args": {"gap_code": "{{逐个}}"}, "response_hint": "平均响应率最低缺口的 respondDay",
             "call_group": "weekly_monitor", "description": "最低成功率日期"},
            {"variable": "min_rate", "source": "api", "vpp_command": "monitor-detail",
             "vpp_args": {"gap_code": "{{逐个}}"}, "response_hint": "该缺口平均响应率", "call_group": "weekly_monitor",
             "description": "最低成功率"},
            {"variable": "task_count", "source": "db", "db_command": "weekly-response", "fallback": True, "description": "响应任务数（备用）"},
            {"variable": "success_rate", "source": "db", "db_command": "weekly-response", "fallback": True, "description": "总成功率（备用）"},
            {"variable": "max_date", "source": "db", "db_command": "weekly-response", "fallback": True, "description": "最高成功率日期（备用）"},
            {"variable": "max_rate", "source": "db", "db_command": "weekly-response", "fallback": True, "description": "最高成功率（备用）"},
        ],
        "screen_action": "切换到\"保供活动分析\"视图，折线图标注最高最低点",
    },
    {
        "faq_id": "FAQ-18",
        "category": "statistics",
        "triggers": ["这个月收益", "本月收益", "收益比上个月", "环比收益", "收益多少"],
        "template": "本月截至今日收益为{{current_revenue}}万元。上月全月收益为{{last_revenue}}万元，环比{{change_direction}}{{change_percent}}%。",
        "data_requirements": [
            {"variable": "current_revenue", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "revenue", "period": "month", "start": "{{本月1日}}", "end": "{{今天}}", "compare": "mom"},
             "response_hint": "data.value 或 data.current", "call_group": "revenue", "description": "本月收益（万元）"},
            {"variable": "last_revenue", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "revenue", "period": "month", "start": "{{上个月1日}}", "end": "{{上个月最后一天}}", "compare": "mom"},
             "response_hint": "data.previous", "call_group": "revenue_prev", "description": "上月全月收益（万元）"},
            {"variable": "change_direction", "source": "static", "description": "增长/下降（由回答时比较 current vs last 得出）"},
            {"variable": "change_percent", "source": "static",
             "response_hint": "data.change_percent，或 (current-last)/last*100", "description": "环比变化百分比"},
        ],
        "screen_action": "展示收益对比柱状图，环比箭头动画",
    },
    {
        "faq_id": "FAQ-19",
        "category": "statistics",
        "triggers": ["累计响应", "响应了多少次", "压降电量", "累计响应次数", "最大响应负荷"],
        "template": "自运营以来，累计响应{{response_count}}次，最大响应负荷{{max_response_mw}}兆瓦，累计压降电量{{curtailment}}万度。",
        "data_requirements": [
            {"variable": "response_count", "source": "db", "db_command": "cumulative", "description": "累计响应次数（整数）"},
            {"variable": "max_response_mw", "source": "db", "db_command": "cumulative", "description": "最大响应负荷（兆瓦）"},
            {"variable": "curtailment", "source": "db", "db_command": "cumulative", "description": "累计压降电量（万度）"},
        ],
        "screen_action": "切换到\"可调资源总览\"，高亮累计响应卡片",
    },
    {
        "faq_id": "FAQ-20",
        "category": "statistics",
        "triggers": ["碳减排", "减排多少", "碳排放", "碳中和", "绿色减排", "环保"],
        "template": "截至目前，虚拟电厂累计碳减排约{{carbon}}吨，相当于种植{{tree_count}}万棵树。",
        "data_requirements": [
            {"variable": "carbon", "source": "db", "db_command": "carbon", "description": "碳减排量（吨）"},
            {"variable": "tree_count", "source": "db", "db_command": "carbon", "description": "等效植树（万棵）"},
        ],
        "screen_action": "展示碳减排仪表盘，等效植树图标",
    },
    {
        "faq_id": "FAQ-21",
        "category": "statistics",
        "triggers": ["哪个企业响应量大", "企业响应排名", "响应量最大", "响应企业排名", "最多响应"],
        "template": (
            "{{time_range}}内，响应量最大的企业是{{top_enterprise}}，总响应量{{top_response}}兆瓦时；"
            "第二名{{second_enterprise}}，{{second_response}}兆瓦时；第三名{{third_enterprise}}，{{third_response}}兆瓦时。"
        ),
        "data_requirements": [
            {"variable": "time_range", "source": "static", "description": "时间范围，从用户问题中提取"},
            {"variable": "top_enterprise", "source": "api", "vpp_command": "score-users",
             "vpp_args": {"page_num": 1, "page_size": 10, "sort": "desc"}, "response_hint": "records[0].companyName",
             "call_group": "ranking", "description": "第一名企业"},
            {"variable": "top_response", "source": "api", "vpp_command": "score-users",
             "vpp_args": {"page_num": 1, "page_size": 10, "sort": "desc"}, "response_hint": "records[0].totalScore",
             "call_group": "ranking", "description": "第一名响应积分"},
            {"variable": "second_enterprise", "source": "api", "vpp_command": "score-users",
             "vpp_args": {"page_num": 1, "page_size": 10, "sort": "desc"}, "response_hint": "records[1].companyName",
             "call_group": "ranking", "description": "第二名企业"},
            {"variable": "second_response", "source": "api", "vpp_command": "score-users",
             "vpp_args": {"page_num": 1, "page_size": 10, "sort": "desc"}, "response_hint": "records[1].totalScore",
             "call_group": "ranking", "description": "第二名响应积分"},
            {"variable": "third_enterprise", "source": "api", "vpp_command": "score-users",
             "vpp_args": {"page_num": 1, "page_size": 10, "sort": "desc"}, "response_hint": "records[2].companyName",
             "call_group": "ranking", "description": "第三名企业"},
            {"variable": "third_response", "source": "api", "vpp_command": "score-users",
             "vpp_args": {"page_num": 1, "page_size": 10, "sort": "desc"}, "response_hint": "records[2].totalScore",
             "call_group": "ranking", "description": "第三名响应积分"},
            {"variable": "ranking_top3", "source": "db", "fallback": True,
             "db_query": (
                 "SELECT company_name, ROUND(SUM(response_load)/1000, 2) AS total_mw "
                 "FROM t_load_gap_monitor WHERE response_load > 0 GROUP BY company_name "
                 "ORDER BY total_mw DESC LIMIT 3"
             ),
             "response_hint": "按 total_mw 降序，records[0..2] 为前三名（备用数据源）",
             "description": "按聚合响应量排序的企业前三名"},
        ],
        "screen_action": "切换到\"响应认定\"视图，企业列表按响应量排序并高亮前三名",
    },
    {
        "faq_id": "FAQ-22",
        "category": "statistics",
        "triggers": ["今天最高负荷", "今日最高负荷", "最高负荷什么时候", "峰值负荷"],
        "template": "今日最高负荷出现在{{peak_time}}，为{{peak_load}}兆瓦。当前负荷为{{current_load}}兆瓦。",
        "data_requirements": [
            {"variable": "peak_time", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["total_consumption"]}, "response_hint": "data.peak_time（需后端扩展）",
             "call_group": "peak_load", "note": "vpp-api 暂无最高负荷指标，建议后端扩展 total_consumption",
             "description": "最高负荷出现时间"},
            {"variable": "peak_load", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["total_consumption"]}, "response_hint": "data.peak_load（需后端扩展）",
             "call_group": "peak_load", "description": "最高负荷值（兆瓦）"},
            {"variable": "current_load", "source": "api", "vpp_command": "realtime",
             "vpp_args": {"metrics": ["total_consumption"]}, "response_hint": "data.total", "call_group": "peak_load",
             "description": "当前负荷（兆瓦）"},
            {"variable": "peak_time", "source": "db", "fallback": True,
             "db_query": (
                 "SELECT LPAD(h,2,'0') AS peak_h, LPAD(mi,2,'0') AS peak_mi FROM t_iot_electrical_meter_revised "
                 "WHERE ts_day = CAST(DATE_FORMAT(UTC_DATE(),'%Y%m%d') AS UNSIGNED) AND curppower > 0 "
                 "ORDER BY curppower DESC LIMIT 1"
             ),
             "response_hint": "peak_h:peak_mi 拼 'HH:MM'；UTC 需 +8h 转北京时间", "description": "最高负荷时间（备用/高压表）"},
            {"variable": "peak_load", "source": "db", "fallback": True,
             "db_query": (
                 "SELECT ROUND(MAX(curppower)/1000, 2) AS peak_mw FROM t_iot_electrical_meter_revised "
                 "WHERE ts_day = CAST(DATE_FORMAT(UTC_DATE(),'%Y%m%d') AS UNSIGNED) AND curppower > 0"
             ),
             "response_hint": "peak_mw 即今日最高负荷（MW）", "description": "今日最高负荷（备用/高压表）"},
            {"variable": "current_load", "source": "db", "fallback": True,
             "db_query": (
                 "SELECT ROUND(curppower/1000, 2) AS current_mw FROM t_iot_electrical_meter_revised "
                 "WHERE ts_day = CAST(DATE_FORMAT(UTC_DATE(),'%Y%m%d') AS UNSIGNED) AND curppower > 0 "
                 "ORDER BY h DESC, mi DESC LIMIT 1"
             ),
             "response_hint": "current_mw 即当前负荷（MW）", "description": "当前负荷（备用/高压表）"},
        ],
        "screen_action": "大屏负荷曲线图上标注最高点",
    },
    {
        "faq_id": "FAQ-23",
        "category": "statistics",
        "triggers": ["数据完整率", "设备完整率", "漏采", "数据采集", "完整率怎么样"],
        "template": "今日设备数据完整率为{{completeness_rate}}%，漏采最严重的设备是{{worst_device}}，漏采率{{loss_rate}}%。近一周完整率趋势平稳。",
        "data_requirements": [
            {"variable": "completeness_rate", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "day", "start": "{{today}}", "end": "{{today}}"},
             "response_hint": "vpp-api 暂不支持完整率指标，需后端扩展 data_completeness", "call_group": "completeness",
             "note": "⚠️ 后端需新增 data_completeness 指标：返回 {rate, devices:[{name, loss_rate}]}",
             "description": "综合完整率（百分比）"},
            {"variable": "worst_device", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "day"}, "response_hint": "需后端扩展", "call_group": "completeness",
             "description": "漏采最严重设备名"},
            {"variable": "loss_rate", "source": "api", "response_hint": "需后端扩展", "call_group": "completeness",
             "description": "该设备漏采率"},
        ],
        "screen_action": "切换到\"设备漏采分析\"视图，展示完整率趋势图",
    },

    # ============================ 04 趋势对比类 (FAQ-24~27) ============================
    {
        "faq_id": "FAQ-24",
        "category": "trend",
        "triggers": ["一周发电趋势", "最近一周发电", "发电趋势", "近一周趋势", "发电量趋势"],
        "template": (
            "最近一周总发电量呈{{trend}}趋势。{{max_date}}达到峰值{{max_value}}万度，"
            "{{min_date}}为低谷{{min_value}}万度。日均发电{{avg_value}}万度。"
        ),
        "data_requirements": [
            {"variable": "trend", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "day", "start": "{{7天前}}", "end": "{{今天}}", "group_by": "none"},
             "response_hint": "data.daily[] 逐日发电量，自行分析趋势方向", "call_group": "weekly_gen",
             "description": "上升/下降/平稳"},
            {"variable": "max_date", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "day", "start": "{{7天前}}", "end": "{{今天}}"},
             "response_hint": "data.daily[] 最大值对应日期", "call_group": "weekly_gen", "description": "峰值日期"},
            {"variable": "max_value", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "day", "start": "{{7天前}}", "end": "{{今天}}"},
             "response_hint": "data.daily[].value 最大值", "call_group": "weekly_gen", "description": "峰值发电量（万度）"},
            {"variable": "min_date", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "day", "start": "{{7天前}}", "end": "{{今天}}"},
             "response_hint": "data.daily[] 最小值对应日期", "call_group": "weekly_gen", "description": "低谷日期"},
            {"variable": "min_value", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "day", "start": "{{7天前}}", "end": "{{今天}}"},
             "response_hint": "data.daily[].value 最小值", "call_group": "weekly_gen", "description": "低谷发电量（万度）"},
            {"variable": "avg_value", "source": "static", "description": "日均发电量（由逐日数据计算均值）"},
        ],
        "screen_action": "展示近7日发电量折线图，标注峰谷点",
    },
    {
        "faq_id": "FAQ-25",
        "category": "trend",
        "triggers": ["对比企业", "企业对比", "XX和YY对比", "企业响应率对比", "对比响应率"],
        "template": (
            "{{enterprise_A}}本月响应率{{rate_A}}%，{{enterprise_B}}本月响应率{{rate_B}}%。"
            "{{better_one}}表现更好，高出{{diff}}个百分点。"
        ),
        "data_requirements": [
            {"variable": "enterprise_A", "source": "static", "description": "从用户问题提取的企业A名称"},
            {"variable": "enterprise_B", "source": "static", "description": "从用户问题提取的企业B名称"},
            {"variable": "rate_A", "source": "db",
             "db_query": "SELECT response_rate FROM t_enterprise_stats WHERE enterprise_name = %s",
             "note": "⚠️ 源用 t_enterprise_stats，需确认该表是否存在；参数化传入 enterprise_A",
             "description": "企业A响应率"},
            {"variable": "rate_B", "source": "db",
             "db_query": "SELECT response_rate FROM t_enterprise_stats WHERE enterprise_name = %s",
             "note": "参数化传入 enterprise_B", "description": "企业B响应率"},
            {"variable": "better_one", "source": "static", "description": "rate_A/rate_B 较大者企业名（回答时计算）"},
            {"variable": "diff", "source": "static", "description": "|rate_A - rate_B|（回答时计算）"},
        ],
        "screen_action": "双柱对比图，高亮差异",
    },
    {
        "faq_id": "FAQ-26",
        "category": "trend",
        "triggers": ["哪种资源可调能力最强", "资源可调能力", "最强可调资源", "资源能力对比", "可调能力构成"],
        "template": (
            "可调节负荷类型最强，占总可调能力的55%；其次是储能，占30%；光伏占15%。"
            "具体来说，工业空调和电炉贡献了大部分可中断负荷。"
        ),
        "data_requirements": [
            {"variable": "none", "source": "static", "description": "静态比例（55%/30%/15%）；后续可由 api 驱动饼图数据"},
        ],
        "screen_action": "展示可调能力构成饼图，点击图例钻取",
    },
    {
        "faq_id": "FAQ-27",
        "category": "trend",
        "triggers": ["储能放电量变化", "近三月储能", "储能月度趋势", "储能放电趋势", "三个月储能"],
        "template": (
            "2月放电量{{feb}}万度，3月{{mar}}万度，4月截至今日{{apr}}万度。"
            "预计4月全月{{apr_forecast}}万度，呈持续增长态势。"
        ),
        "data_requirements": [
            {"variable": "feb", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "month", "start": "{{2月1日}}", "end": "{{2月28日}}", "group_by": "resource_type"},
             "response_hint": "data.breakdown.storage", "call_group": "storage_monthly", "description": "2月放电量（万度）"},
            {"variable": "mar", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "month", "start": "{{3月1日}}", "end": "{{3月31日}}", "group_by": "resource_type"},
             "response_hint": "data.breakdown.storage", "call_group": "storage_monthly", "description": "3月放电量（万度）"},
            {"variable": "apr", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "month", "start": "{{4月1日}}", "end": "{{今天}}", "group_by": "resource_type"},
             "response_hint": "data.breakdown.storage", "call_group": "storage_monthly", "description": "4月截至今日放电量（万度）"},
            {"variable": "apr_forecast", "source": "api", "vpp_command": "statistics",
             "vpp_args": {"metric": "generation", "period": "month", "start": "{{4月1日}}", "end": "{{4月30日}}", "group_by": "resource_type"},
             "response_hint": "支持则 data.forecast.storage；否则据日均推算", "call_group": "storage_monthly_forecast",
             "note": "需 statistics 支持预测；否则 AI 按 (apr/已过天数*全月天数) 估算", "description": "4月预测全月值（万度）"},
        ],
        "screen_action": "折线图+预测虚线",
    },

    # ============================ 05 预警事件类 (FAQ-28~30) ============================
    {
        "faq_id": "FAQ-28",
        "category": "alert",
        "triggers": ["设备异常", "哪些设备异常", "设备故障", "异常设备", "告警"],
        "template": "当前有{{count}}个设备异常：{{#each events}}{{device_name}}（{{message}}）；{{/each}}需要我列出详情吗？",
        "data_requirements": [
            {"variable": "count", "source": "db", "db_command": "anomaly-devices", "description": "异常设备数量"},
            {"variable": "events", "source": "db", "db_command": "anomaly-devices",
             "description": "异常事件列表，每项含 device_name 和 message"},
        ],
        "screen_action": "切换到\"异常数据检测\"视图，列表展示，地图图标闪烁",
    },
    {
        "faq_id": "FAQ-29",
        "category": "alert",
        "triggers": ["响应率偏低", "哪些企业响应率低", "低响应率", "响应不合格", "响应偏低"],
        "template": (
            "当前任务中，响应率低于90%的企业有{{#each enterprises}}{{name}}（{{rate}}%）；{{/each}}。"
            "其中{{lowest}}最低，仅{{lowest_rate}}%。"
        ),
        "data_requirements": [
            {"variable": "enterprises", "source": "api", "vpp_command": "monitor-detail",
             "vpp_args": {"gap_code": "{{最新缺口}}", "page_size": 100, "warn_status": 2},
             "response_hint": "records where status=2，取 companyName/loadPercentage，按 loadPercentage 升序",
             "call_group": "low_response", "note": "先调 gap-list 获取最新缺口 code", "description": "低响应率企业列表"},
            {"variable": "lowest", "source": "api", "vpp_command": "monitor-detail",
             "vpp_args": {"gap_code": "{{最新缺口}}", "page_size": 100, "warn_status": 2},
             "response_hint": "升序[0].companyName", "call_group": "low_response", "description": "最低响应率企业名"},
            {"variable": "lowest_rate", "source": "api", "vpp_command": "monitor-detail",
             "vpp_args": {"gap_code": "{{最新缺口}}", "page_size": 100, "warn_status": 2},
             "response_hint": "升序[0].loadPercentage", "call_group": "low_response", "description": "最低响应率"},
            {"variable": "enterprises", "source": "db", "db_command": "low-response", "fallback": True,
             "description": "低响应率企业列表，每项含 name 和 rate（备用）"},
        ],
        "screen_action": "执行监测用户列表按响应率升序排列，低响应率企业行标红",
    },
    {
        "faq_id": "FAQ-30",
        "category": "alert",
        "triggers": ["边缘终端", "终端A正常", "终端状态", "边缘终端状态", "XX终端"],
        "template": (
            "边缘终端A最后心跳时间为{{last_heartbeat}}，状态{{status}}。"
            "今日上报完整率{{completeness}}%，平均执行偏差{{deviation}}%。"
        ),
        "data_requirements": [
            {"variable": "last_heartbeat", "source": "db",
             "db_query": "SELECT heartbeat_time FROM t_edge_terminal WHERE terminal_name = %s",
             "note": "⚠️ 源用 t_edge_terminal，需确认该表是否存在；参数化传入 terminal_name",
             "description": "最后心跳时间"},
            {"variable": "status", "source": "db",
             "db_query": "SELECT status FROM t_edge_terminal WHERE terminal_name = %s", "description": "正常/离线/异常"},
            {"variable": "completeness", "source": "db",
             "db_query": "SELECT report_completeness FROM t_edge_terminal WHERE terminal_name = %s", "description": "上报完整率（百分比）"},
            {"variable": "deviation", "source": "db",
             "db_query": "SELECT avg_deviation FROM t_edge_terminal WHERE terminal_name = %s", "description": "平均执行偏差百分比"},
        ],
        "screen_action": "展示终端详情卡片，高亮关键指标",
    },
]
