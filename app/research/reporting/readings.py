"""Say what each figure and evidence block actually shows, from the numbers alone.

Every sentence is assembled from executed statistics and the thresholds in
``evaluation.criteria``; no wording is left for a model to invent. A reading
returns an empty string whenever the evidence it needs is absent, so a section
degrades to its table instead of asserting something the run cannot support.

Register: a term an undergraduate statistics course covers is used directly; one
it does not is explained in the same sentence. No internal vocabulary (判据,
阈值, 水平序列, 口径) and no chat register. Every number is followed by what it
means for the price series.
"""

from __future__ import annotations

from typing import Any

from app.research.evaluation.criteria import (
    CALENDAR_EFFECT_THRESHOLD,
    DISTRIBUTION_KURTOSIS_THRESHOLD,
    DISTRIBUTION_SKEW_THRESHOLD,
    MINIMUM_DRIVER_COVERAGE,
    SEASONAL_STRENGTH_THRESHOLD,
    SEASONAL_STRONG_THRESHOLD,
    VOLATILITY_REGIME_RATIO,
    group_variance_share,
)
from app.research.evaluation.wording import (
    GROUP_LABELS,
    SEASONALITY_TEXT,
    stationarity_text,
    transform_text,
)


def format_statistic(value: Any, unit: str = "") -> str:
    """Print one statistic at the precision the evaluator uses, in readings and tables alike.

    Chart axes use the compact ``svg_charts.format_number``; prose and tables must
    not round 269.583 to 270 while the sentence beside them quotes three decimals.
    """

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    magnitude = abs(float(value))
    if magnitude and magnitude < 0.001:
        text = f"{float(value):.2e}"
    else:
        text = f"{float(value):,.3f}".rstrip("0").rstrip(".")
    return f"{text} {unit}".strip() if unit else text


def _number(value: Any, unit: str = "") -> str:
    return format_statistic(value, unit)


def _share(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{float(value) * 100:.2f}%"


def _float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _names(values: Any, limit: int = 5) -> str:
    items = [str(name) for name in list(values or [])[:limit]]
    return "、".join(items)


def _paragraph(sentences: list[str]) -> str:
    return "".join(sentence for sentence in sentences if sentence)


def distribution_reading(distribution: dict[str, Any], unit: str) -> str:
    mean = _float(distribution.get("mean"))
    median = _float(distribution.get("median"))
    skewness = _float(distribution.get("skewness"))
    kurtosis = _float(distribution.get("kurtosis"))
    if mean is None or median is None:
        return ""
    pull = "把平均值往上拉了" if mean > median else "把平均值往下拉了"
    side = "高价" if mean > median else "低价"
    sentences = [
        (
            f"多数时候电价在 {_number(median, unit)} 附近，平均值 {_number(mean, unit)}，"
            f"两者相差 {_number(abs(mean - median), unit)}，说明少数{side}时段{pull}。"
        )
    ]
    if skewness is not None and kurtosis is not None:
        skewed = abs(skewness) >= DISTRIBUTION_SKEW_THRESHOLD
        heavy = kurtosis >= DISTRIBUTION_KURTOSIS_THRESHOLD
        if not skewed and not heavy:
            sentences.append(
                f"分布的形状接近对称，两端也没有出现特别多的极端值（偏度 {_number(skewness)}、"
                f"超额峰度 {_number(kurtosis)}）。用平均值和标准差概括这条序列是合适的。"
            )
        else:
            lean = "高价" if skewness > 0 else "低价"
            sentences.append(
                f"这条序列向{lean}一侧倾斜（偏度 {_number(skewness)}，衡量分布是否对称），"
                f"出现极端价格的机会也高于正态分布（超额峰度 {_number(kurtosis)}，衡量两端有多重）。"
                "因此只看平均值和标准差会低估极端情况，读误差时要同时看中位数和分位数。"
            )
    return _paragraph(sentences)


def timeseries_reading(price: dict[str, Any], unit: str) -> str:
    """Read the time path itself, so the chart is never left standing without a sentence."""

    sentences: list[str] = []
    quantiles = (price.get("distribution") or {}).get("quantiles") or {}
    middle = (_float(quantiles.get("0.25")), _float(quantiles.get("0.75")))
    tails = (_float(quantiles.get("0.01")), _float(quantiles.get("0.99")))
    if all(value is not None for value in (*middle, *tails)):
        sentences.append(
            f"图中一半的时间价格在 {_number(middle[0], unit)} 到 {_number(middle[1], unit)} 之间，"
            f"只有很少的时段会低于 {_number(tails[0], unit)} 或高于 {_number(tails[1], unit)}。"
        )
    elif price.get("observations"):
        sentences.append(
            f"图上共 {int(price['observations']):,} 个有效时点"
            f"（{str(price.get('start_time') or '')[:16]} 到 {str(price.get('end_time') or '')[:16]}），"
            f"数据完整度 {_share(price.get('coverage_rate'))}；空白处是数据缺口，不是价格为零。"
        )
    volatility = volatility_reading(price)
    if volatility:
        sentences.append(volatility)
    return _paragraph(sentences)


def volatility_reading(price: dict[str, Any]) -> str:
    rolling = (price.get("rolling_statistics") or {}).get("one_day") or {}
    minimum = _float(rolling.get("minimum_std"))
    maximum = _float(rolling.get("maximum_std"))
    if minimum is None or maximum is None:
        return ""
    if minimum == 0:
        # A day whose price never moves (held at a cap or floor) is itself strong
        # evidence that the swings are not uniform; the ratio simply cannot be formed.
        return (
            f"价格的波动幅度很不均匀：有整整一天几乎没有变化，而波动最大的一天标准差达到 {_number(maximum)}。"
            "用一个整体的波动水平代表全时段并不合适，分析和建模都要分时段来看。"
        )
    ratio = maximum / minimum
    if ratio >= VOLATILITY_REGIME_RATIO:
        return (
            f"价格的波动幅度并不均匀：以一天为窗口计算标准差，最平静的一天是 {_number(minimum)}，"
            f"最剧烈的一天是 {_number(maximum)}，相差 {_number(ratio)} 倍。"
            "用一个整体的波动水平代表全时段并不合适，分析和建模都要分时段来看。"
        )
    return (
        f"以一天为窗口计算标准差，最平静和最剧烈的一天分别是 {_number(minimum)} 和 {_number(maximum)}，"
        f"相差 {_number(ratio)} 倍，波动在整段时间里比较均匀。"
    )


def calendar_reading(seasonality: dict[str, Any]) -> str:
    hours = [row.get("mean") for row in seasonality.get("hour_of_day") or [] if row.get("mean") is not None]
    shares = {
        label: share
        for label in ("hour_of_day", "day_of_week", "month")
        if (share := group_variance_share(seasonality.get(label))) is not None
    }
    if not hours and not shares:
        return ""
    sentences: list[str] = []
    if hours:
        sentences.append(f"一天之内，不同小时的平均电价最多相差 {_number(max(hours) - min(hours))}。")
    if shares:
        ranked = sorted(shares.items(), key=lambda item: item[1], reverse=True)
        detail = "；".join(
            f"按{GROUP_LABELS.get(label, label)}分组能解释 {value:.1%}" for label, value in ranked
        )
        strongest_label, strongest_value = ranked[0]
        if strongest_value >= CALENDAR_EFFECT_THRESHOLD:
            verdict = (
                f"{GROUP_LABELS.get(strongest_label, strongest_label)}是这条序列里最明显的重复规律，"
                "做特征时应当把它包含进来。"
            )
        else:
            verdict = "没有哪一层达到这个水平，本轮不认为存在稳定的日历规律。"
        sentences.append(
            f"把价格分组后比较，{detail} 的价格波动"
            f"（一般以 {CALENDAR_EFFECT_THRESHOLD:.0%} 作为这一层是否存在规律的门槛）。{verdict}"
        )
    return _paragraph(sentences)


def decomposition_reading(decomposition: dict[str, Any]) -> str:
    strengths = {
        label: value
        for label, value in (decomposition.get("seasonal_strength") or {}).items()
        if _float(value) is not None
    }
    remainder = _float((decomposition.get("variance_share") or {}).get("remainder"))
    if not strengths and remainder is None:
        return ""
    sentences: list[str] = []
    if strengths:
        label, value = max(strengths.items(), key=lambda item: float(item[1]))
        strength = float(value)
        if strength >= SEASONAL_STRONG_THRESHOLD:
            verdict = "属于很稳定的周期。"
        elif strength >= SEASONAL_STRENGTH_THRESHOLD:
            verdict = "周期存在但不算稳定。"
        else:
            verdict = "本轮不认为存在稳定的周期。"
        sentences.append(
            f"把价格拆成长期趋势、重复出现的周期和剩下的部分之后，最强的周期是"
            f"{SEASONALITY_TEXT.get(label, label)}，强度 {_number(strength)}"
            f"（取值 0 到 1，越接近 1 说明这个周期每次都重复得越像；"
            f"{SEASONAL_STRONG_THRESHOLD:.1f} 以上算强，{SEASONAL_STRENGTH_THRESHOLD:.1f} 以下算弱）。{verdict}"
        )
    if remainder is not None:
        sentences.append(
            f"拆完之后仍有 {_share(remainder)} 的波动落在无法用趋势和周期解释的部分，"
            "这部分决定了只靠时间规律能做到的上限。"
        )
    return _paragraph(sentences)


def spike_regime_reading(regime: dict[str, Any]) -> str:
    high_share = _float(regime.get("high_spike_share"))
    negative_share = _float(regime.get("negative_share"))
    clustering = regime.get("clustering") or {}
    conditional = _float(clustering.get("probability_spike_follows_spike"))
    longest = regime.get("high_episodes", {}).get("max_duration_intervals")
    if high_share is None and negative_share is None:
        return ""
    sentences: list[str] = []
    if high_share is not None and conditional is not None:
        sentences.append(
            f"价格尖峰很少见，只占 {_share(high_share)} 的时段，但它们是扎堆出现的："
            f"出现一次尖峰后，下一个时段仍是尖峰的概率有 {_share(conditional)}；"
            f"如果尖峰是随机分布的，这个概率只有 {_share(high_share)}。"
            + (f"最长的一次连续了 {longest} 个时段。" if longest else "")
            + "所以尖峰不适合当成孤立的异常点删掉，更应该看作一段会持续一阵的状态。"
        )
    elif high_share is not None:
        sentences.append(f"价格尖峰占 {_share(high_share)} 的时段。")
    if negative_share is not None and negative_share > 0:
        sentences.append(
            f"负价占 {_share(negative_share)}"
            + (
                "，属于常态而不是偶发情况；这条序列上不能使用以价格为分母的百分比误差指标。"
                if negative_share >= 0.01
                else "，属于偶发。"
            )
        )
    return _paragraph(sentences)


def duration_curve_reading(duration_curve: dict[str, Any]) -> str:
    top_share = _float(duration_curve.get("top_5_percent_share_of_positive_value"))
    above_zero = _float(duration_curve.get("share_above_zero"))
    ratio = _float(duration_curve.get("ratio_p95_to_median"))
    if top_share is None and above_zero is None:
        return ""
    sentences: list[str] = ["这条曲线把所有时段按价格从高到低排序。"]
    if top_share is not None:
        sentences.append(f"最贵的 5% 时段贡献了全部正价格的 {_share(top_share)}。")
    if above_zero is not None:
        sentences.append(f"价格高于零的时段占 {_share(above_zero)}。")
    if ratio is not None:
        sentences.append(
            f"95% 分位是中位数的 {_number(ratio)} 倍——曲线在高价一端越陡，收益就越集中在少数时段。"
        )
    return _paragraph(sentences)


def stationarity_reading(stationarity: dict[str, Any]) -> str:
    verdict = str(stationarity.get("verdict") or "")
    if not verdict:
        return ""
    difference = str((stationarity.get("first_difference") or {}).get("verdict") or "")
    sentences = [
        (
            "平稳性检验判断的是价格在围绕一个固定水平上下波动，还是会持续漂移。"
            f"这里的结论是：{stationarity_text(verdict)}。"
        )
    ]
    if verdict == "trend_or_break_suspected":
        sentences.append("两项检验（ADF 与 KPSS）给出了相反的结果，这种情况通常意味着序列里有趋势或结构变化。")
    if difference:
        sentences.append(f"改用相邻时段的差值之后，结论变成：{stationarity_text(difference)}。")
    transform = stationarity.get("recommended_transform")
    if transform:
        sentences.append(f"因此{transform_text(transform)}。")
    return _paragraph(sentences)


def autocorrelation_reading(rows: list[dict[str, Any]]) -> str:
    usable = [row for row in rows or [] if _float(row.get("correlation")) is not None]
    if not usable:
        return ""
    strongest = max(usable, key=lambda row: abs(float(row["correlation"])))
    first = usable[0]
    return (
        "自相关衡量的是当前价格和它自己过去取值的相似程度。"
        f"相邻两个时段的自相关是 {_number(first.get('correlation'))}，"
        f"扫描范围内最强的是相隔 {strongest.get('lag')} 个时段的 {_number(strongest.get('correlation'))}。"
        "自相关越强，说明光靠价格自己的历史就能解释相当一部分变化；"
        "评估其他变量有没有价值时，要先把这部分扣掉。"
    )


def partial_autocorrelation_reading(partial: dict[str, Any]) -> str:
    order = partial.get("suggested_autoregressive_order")
    count = partial.get("significant_lag_count")
    if order is None and count is None:
        return ""
    sentences = ["偏自相关衡量的是扣掉中间时段的影响之后，某个滞后还剩多少直接关系。"]
    if count is not None:
        sentences.append(f"共有 {count} 个滞后超出了随机波动的范围。")
    if order:
        sentences.append(f"用价格自身历史做基线时，大约需要往前看 {order} 个时段，再往前的直接影响基本消失。")
    ljung = [row for row in partial.get("ljung_box") or [] if row.get("rejects_independence_at_5_percent")]
    if ljung:
        sentences.append("Ljung-Box 检验也否定了「序列没有规律」这一假设，说明价格里仍有可利用的结构。")
    return _paragraph(sentences)


def extremes_reading(extremes: dict[str, Any]) -> str:
    high = extremes.get("high_spike_count")
    low = extremes.get("extreme_low_count")
    if high is None and low is None:
        return ""
    return (
        f"用四分位距规则（把上下四分位之间的距离乘 {extremes.get('iqr_multiplier')} 作为界线）划定极端值："
        f"高于 {_number(extremes.get('upper_threshold'))} 的极端高价有 {high} 个，"
        f"低于 {_number(extremes.get('lower_threshold'))} 的极端低价有 {low} 个。"
        "这一步只做标记，不删除数据；是否剔除留到建模阶段再决定。"
    )


def driver_coverage_reading(series: dict[str, Any]) -> str:
    rates = {
        name: value
        for name, item in (series or {}).items()
        if (value := _float(item.get("coverage_rate"))) is not None
    }
    if not rates:
        return ""
    below = sorted(name for name, value in rates.items() if value < MINIMUM_DRIVER_COVERAGE)
    lowest_name, lowest_value = min(rates.items(), key=lambda item: item[1])
    if below:
        verdict = (
            f"其中 {len(below)} 个低于这个水平：{_names(below)}，它们单独支撑不了关系结论，"
            "要先说明缺口再谈相关。"
        )
    else:
        verdict = "全部达标，数据量上足以进入关系分析。"
    return (
        f"{len(rates)} 个候选变量中，数据最不完整的是 {lowest_name}（{_share(lowest_value)}）。"
        f"一般认为完整度低于 {MINIMUM_DRIVER_COVERAGE:.0%} 的变量不足以单独支撑关系结论。{verdict}"
        "需要注意，完整度只回答「能不能算」，不回答「预测的时候能不能拿到」。"
    )


def collinearity_reading(exogenous: dict[str, Any]) -> str:
    pairs = exogenous.get("strong_collinearity_pairs")
    multicollinearity = exogenous.get("multicollinearity") or {}
    severe = multicollinearity.get("severe_variables") or []
    if pairs is None and not multicollinearity:
        return ""
    sentences: list[str] = []
    if pairs is not None:
        threshold = exogenous.get("strong_collinearity_threshold")
        sentences.append(
            f"有 {len(pairs)} 对变量彼此高度相关（相关系数超过 {threshold}），它们提供的信息大量重叠。"
            if pairs
            else "没有哪两个变量之间达到高度相关。"
        )
    if severe:
        sentences.append(
            f"方差膨胀因子进一步显示，{len(severe)} 个变量可以被其余变量共同解释：{_names(severe)}。"
            "建模时这一组里保留一个代表就够了，同时放进去会让系数不稳定。"
        )
    elif multicollinearity:
        sentences.append("方差膨胀因子也没有发现被其余变量共同解释的变量。")
    return _paragraph(sentences)


def driver_stationarity_reading(driver_stationarity: dict[str, Any]) -> str:
    series = driver_stationarity.get("series") or {}
    if not series:
        return ""
    drifting = sorted(
        name
        for name, item in series.items()
        if str(item.get("verdict")) in {"unit_root", "trend_or_break_suspected"}
    )
    if not drifting:
        return "这些变量自身都围绕稳定水平波动，和电价形成虚假共同趋势的风险不大。"
    return (
        f"{len(drifting)}/{len(series)} 个变量自身存在趋势或长期漂移：{_names(drifting)}。"
        "带趋势的变量和同样带趋势的电价直接算相关，容易得出只是「两者都在往同一个方向走」的假关系。"
    )


def relationship_reading(series: dict[str, Any]) -> str:
    coefficients = {
        name: abs(value)
        for name, result in (series or {}).items()
        if (value := _float(((result.get("contemporaneous") or {}).get("pearson") or {}).get("correlation")))
        is not None
    }
    if not coefficients:
        return ""
    strongest_name, strongest_value = max(coefficients.items(), key=lambda item: item[1])
    meaningful = sum(1 for value in coefficients.values() if value >= 0.1)
    leading = [
        name
        for name, result in series.items()
        if int((result.get("best_absolute_lag") or {}).get("lag") or 0) > 0
    ]
    sentences = [
        (
            f"在同一时刻比较，与电价关系最紧的是 {strongest_name}，相关系数 {_number(strongest_value)}；"
            f"{len(coefficients)} 个变量里有 {meaningful} 个达到 0.1 以上。"
        )
    ]
    if leading:
        sentences.append(
            f"把变量往前挪几个时段再比，{len(leading)}/{len(series)} 个变量在正的提前量上相关最强，"
            "说明它们的变化出现在电价变化之前。"
        )
    sentences.append(
        "需要提醒的是，这些相关还没有排除共同趋势、日内规律和价格自身的延续性，"
        "只能作为下一步的候选线索，不能当作因果关系。"
    )
    return _paragraph(sentences)


def mutual_information_reading(mutual_information: dict[str, Any]) -> str:
    candidates = mutual_information.get("nonlinear_candidates")
    if candidates is None:
        return ""
    lead = "互信息衡量两个变量之间的依赖强度，和相关系数不同的是，它也能捕捉不成直线的关系。"
    if not candidates:
        return lead + "本轮没有变量的互信息明显高于它的相关系数，因此不提示额外的非线性关系。"
    return (
        lead + f"{len(candidates)} 个变量的互信息明显高于它们的相关系数：{_names(candidates)}。"
        "对这些变量，只看相关系数会低估它们和电价的关系。"
    )


def granger_reading(granger: dict[str, Any]) -> str:
    preceding = granger.get("variables_with_precedence")
    if preceding is None:
        return ""
    lead = (
        "这项检验比较两件事：只用电价自己的历史来拟合，和额外加入某个变量的历史来拟合，"
        "后者的误差是不是明显更小。"
    )
    if not preceding:
        return lead + "本轮没有变量通过检验，也就没有可用的时间先后证据。"
    return (
        lead + f"{len(preceding)} 个变量（占 {_share(granger.get('share_with_precedence'))}）通过了检验："
        f"{_names(preceding)}。这只说明它们的变化在时间上先于电价，"
        "既不等于因果，也不等于放进模型就能提高预测精度。"
    )


def rolling_stability_reading(stability: dict[str, Any]) -> str:
    unstable = stability.get("unstable_variables")
    if unstable is None:
        return ""
    window = stability.get("window_days")
    lead = (
        f"用 {_number(window)} 天的滑动窗口反复计算相关系数，可以看出这个关系是不是一直成立。"
        if window
        else "用滑动窗口反复计算相关系数，可以看出这个关系是不是一直成立。"
    )
    if not unstable:
        return lead + "本轮没有变量出现明显漂移或正负反转，整段时间用同一个相关系数是合适的。"
    return (
        lead + f"{len(unstable)} 个变量的相关系数出现了明显漂移，甚至正负反转：{_names(unstable)}。"
        "对它们来说，一个覆盖全时段的相关系数会掩盖阶段之间的差别。"
    )


def segment_price_reading(comparison: dict[str, Any], unit: str) -> str:
    contrasts = [
        row
        for row in comparison.get("contrasts") or []
        if _float(row.get("mean_difference_left_minus_right")) is not None
    ]
    if not contrasts:
        return ""
    segments = comparison.get("segments") or {}

    def label(segment_id: Any) -> str:
        return str((segments.get(str(segment_id)) or {}).get("label") or segment_id)

    largest = max(contrasts, key=lambda row: abs(float(row["mean_difference_left_minus_right"])))
    difference = abs(float(largest["mean_difference_left_minus_right"]))
    return (
        f"差距最大的一对是{label(largest.get('left_segment_id'))}与{label(largest.get('right_segment_id'))}，"
        f"平均价格相差 {_number(difference, unit)}。"
        "这只是两组数据的直接比较，还没有排除趋势、季节和同时发生的其他因素。"
    )


def baseline_reading(baselines: dict[str, Any], unit: str) -> str:
    best_mae = _float(baselines.get("best_mae"))
    if best_mae is None:
        return ""
    rows = baselines.get("baselines") or []
    persistence = next((row for row in rows if row.get("baseline") == "persistence"), None)
    sentences = [
        (
            "最简单的预测办法是直接照抄历史价格。"
            f"这里表现最好的是「{baselines.get('best_baseline_label')}」，"
            f"平均绝对误差 {_number(best_mae, unit)}。"
            "后续模型在同样的时间切分下必须低于这个数字，才算真的带来了预测价值。"
        )
    ]
    if (
        baselines.get("best_baseline") != "persistence"
        and persistence is not None
        and _float(persistence.get("mae")) is not None
    ):
        sentences.append(f"作为对照，直接沿用上一时刻价格的平均绝对误差是 {_number(persistence.get('mae'), unit)}。")
    if baselines.get("percentage_errors_reliable") is False:
        sentences.append(
            f"序列里有 {_share(baselines.get('near_zero_share'))} 的取值接近零，"
            "用百分比表示的误差指标在这里会失真，评估要用平均绝对误差或均方根误差。"
        )
    return _paragraph(sentences)


def variance_stabilization_reading(stabilization: dict[str, Any]) -> str:
    rationale = str(stabilization.get("rationale") or "").strip()
    raw = _float((stabilization.get("raw_standardized") or {}).get("excess_kurtosis"))
    transformed = _float((stabilization.get("asinh_transformed") or {}).get("excess_kurtosis"))
    if raw is None or transformed is None:
        return rationale
    return (
        f"超额峰度衡量极端值出现得有多频繁。标准化之后是 {_number(raw)}，"
        f"再做一次 asinh 变换（一种压缩极端值的常用变换）后降到 {_number(transformed)}。{rationale}"
    )
