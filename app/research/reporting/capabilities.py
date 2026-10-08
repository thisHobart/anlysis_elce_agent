"""Single source of truth for figures the deterministic report can render."""

from __future__ import annotations

from typing import Any

FIGURE_CATALOG: dict[str, dict[str, str]] = {
    "price_timeseries": {"title": "电价时间序列", "form": "折线图"},
    "price_distribution": {"title": "电价分布", "form": "直方图"},
    "price_duration_curve": {"title": "电价持续曲线", "form": "曲线图"},
    "seasonal_patterns": {"title": "分小时平均电价", "form": "柱状图"},
    "price_weekday_profile": {"title": "分星期平均电价", "form": "柱状图"},
    "price_month_profile": {"title": "分月份平均电价", "form": "柱状图"},
    "price_autocorrelation": {"title": "电价自相关", "form": "茎叶图"},
    "price_partial_autocorrelation": {"title": "电价偏自相关", "form": "茎叶图"},
    "price_decomposition_variance": {"title": "趋势与季节成分方差占比", "form": "堆叠占比图"},
    "price_daily_shape": {"title": "分解得到的日内季节形态", "form": "发散柱状图"},
    "price_naive_baselines": {"title": "朴素基线误差底线", "form": "排序条形图"},
    "exogenous_multicollinearity": {"title": "影响因素方差膨胀因子", "form": "排序条形图"},
    "correlation_ranking": {"title": "电价—因素同期相关排序", "form": "排序条形图"},
    "correlation_matrix": {"title": "变量相关矩阵", "form": "热力图"},
    "lag_relationships": {"title": "因素领先电价的相关曲线", "form": "多折线图"},
    "mutual_information": {"title": "非线性依赖强度（归一化互信息）", "form": "排序条形图"},
    "granger_precedence": {"title": "加入变量历史后的残差方差下降", "form": "排序条形图"},
    "rolling_stability": {"title": "关系随时间的稳定性", "form": "区间图"},
    "segment_price_comparison": {"title": "分段电价均值对比", "form": "柱状图"},
    "segment_relationship_comparison": {"title": "分段关系强度对比", "form": "排序条形图"},
}

FIGURE_TITLES: dict[str, str] = {
    key: specification["title"] for key, specification in FIGURE_CATALOG.items()
}


def report_capabilities_context() -> dict[str, Any]:
    """Describe product output capabilities without implying that a run already exists."""

    return {
        "input_data_access": "read_only",
        "analysis_function_output": "validated_structured_evidence",
        "generation_trigger": "after_successful_plan_execution_and_finalization",
        "automatic_report_generation": True,
        "report_format": "Markdown",
        "report_file": "report.md",
        "report_file_contains": "结论与适用边界、数据口径，以及每个分析步骤的图表、读图说明和关键数字",
        "methods_file": "methods.md",
        "methods_file_contains": "研究设计、逐函数参数与判读口径、判读阈值、有效性核验与假设验收",
        "figure_format": "SVG",
        "figures_are_derived_from_executed_evidence": True,
        "desktop_report_reader": True,
        "supported_workflows": {
            "eda": "电价与数值型外生变量的确定性分析",
            "news": "读取冻结且可追溯的新闻JSONL，抽取事件、复核时点并生成数值特征",
            "forecast": "P1初步分析 → P2新闻分析 → P1综合分析 → P3三折回测与山东次日96点预测",
            "forecast_requires_separate_approval": True,
        },
        "figure_catalog": [
            {"key": key, **specification}
            for key, specification in FIGURE_CATALOG.items()
        ],
        "model_boundary": {
            "direct_statistical_computation": False,
            "direct_arbitrary_chart_generation": False,
            "direct_image_generation": False,
        },
        "limitations": [
            "P2只读取已冻结且带来源与取得时间的新闻快照；没有合格快照时不能声称已经分析新闻。",
            "只为本轮已执行且具有可绘制证据的分析生成图表，不是每个研究函数都有对应图表。",
            "没有完成执行与报告最终化时，只能说明项目支持哪些图，不能声称当前已有图表。",
            "未列入 figure_catalog 的图形不承诺生成；当前月度画像是均值柱状图，不是箱线图。",
            "当前季节分解图展示成分方差占比和日内季节形态，不是完整趋势、季节与残差序列面板。",
            "方法、参数、判读阈值、有效性核验与假设验收写在 methods.md；report.md 不重复这些内容。",
        ],
    }


def current_artifacts_context(latest_run: dict[str, Any] | None) -> dict[str, Any]:
    """Project authoritative current-run artifact metadata into model context."""

    run = latest_run or {}
    figure_paths = run.get("figure_paths") if isinstance(run.get("figure_paths"), dict) else {}
    figure_keys = [str(key) for key in figure_paths]
    report_path = str(run.get("report_path") or "") or None
    if not report_path:
        return {
            "status": "not_generated_for_current_episode",
            "run_id": None,
            "report_path": None,
            "figure_count": 0,
            "figures": [],
        }
    return {
        "status": "available",
        "run_id": run.get("run_id"),
        "report_path": report_path,
        "figure_count": len(figure_keys),
        "figures": [
            {
                "key": key,
                "title": FIGURE_TITLES.get(key, key),
            }
            for key in figure_keys
        ],
    }
