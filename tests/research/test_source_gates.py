"""The deterministic gates that catch a model being confidently, repeatably wrong.

The consistency gate only sees disagreement. These read the article, so they still work
when every pass makes the same mistake — which is how both cases here actually failed.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.research.news.source_gates import (
    operational_signals,
    operational_terms,
    power_quantities,
    time_anchors,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _bodies(corpus: str) -> list[str]:
    path = FIXTURES / corpus / "source_news.jsonl"
    return [json.loads(line)["body"] for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_a_full_datetime_counts_as_one_anchor_not_a_date_plus_a_clock_time() -> None:
    """Splitting these would make almost every article look like it had two events."""

    assert time_anchors("2026年8月3日11时06分，辽宁电网最大用电负荷冲至4775万千瓦。") == (
        "2026年8月3日11时06分",
    )


def test_the_sentence_that_reports_a_record_broken_twice_yields_two_anchors() -> None:
    """The holdout case: a date-only first record and a precise second one, in one sentence."""

    body = "安徽全省最大用电负荷继8月3日（6922万干瓦）首创新高后，在4日午间12时45分再创历史新高，达6999万千瓦。"
    assert time_anchors(body) == ("8月3日", "4日午间12时45分")


def test_a_month_reference_needs_the_month_marker_so_that_early_july_is_not_counted() -> None:
    assert time_anchors("截至2026年7月份，全省100MW及以上统调机组异常停运48次。") == ("2026年7月份",)
    assert time_anchors("7月10日，全国用电负荷较7月初上涨。") == ("7月10日",)


def test_two_quantities_under_one_time_are_not_two_events() -> None:
    """R04 states a threshold and a value; that is one record, not two."""

    body = "江苏电网7月31日最高用电负荷首次突破1.6亿千瓦，达1.609亿千瓦。"
    assert len(time_anchors(body)) == 1
    assert len(power_quantities(body)) == 2


def test_power_quantities_read_both_scripts_and_the_source_typo() -> None:
    assert power_quantities("870 MW") == ("870 MW",)
    assert power_quantities("85,464 MW") == ("85,464 MW",)
    assert power_quantities("1.31亿千瓦") == ("1.31亿千瓦",)
    # The published source really does write 干瓦 for 千瓦; the gate must not miss it.
    assert power_quantities("6922万干瓦") == ("6922万干瓦",)


def test_counts_and_dates_are_not_mistaken_for_power_figures() -> None:
    assert power_quantities("共办理电力业务许可21件") == ()
    assert power_quantities("异常停运48次") == ()
    assert power_quantities("2026年7月份") == ()


def test_operational_vocabulary_excludes_words_that_appear_in_licence_notices() -> None:
    """电力 and 发电 alone would flag the licence bulletin, which really is irrelevant."""

    assert operational_terms("8月份，全省办理电力业务许可共21件，其中发电类19件。") == ()
    assert "停运" in operational_terms("统调机组异常停运48次")


def test_exactly_one_holdout_document_states_two_times() -> None:
    """The multi-event gate must be selective, or it converts good extractions into review."""

    multi = [body for body in _bodies("news_holdout") if len(time_anchors(body)) > 1]
    assert len(multi) == 1
    assert "再创历史新高" in multi[0]


def test_no_development_document_states_two_times() -> None:
    assert [body for body in _bodies("news_realistic") if len(time_anchors(body)) > 1] == []


def test_the_genuinely_irrelevant_articles_carry_no_operational_signal() -> None:
    """Both corpora's irrelevant cases must keep their terminal verdict."""

    licence = "8月份，全省办理电力业务许可共21件，其中发电类电力业务许可19件、供电类电力业务许可1件、配售电类电力业务许可1件。"
    community = "服务队在37个社区开展爱心电力活动，长期提供管家式服务。"
    assert operational_signals(licence) == ()
    assert operational_signals(community) == ()


def test_the_outage_tally_does_carry_one() -> None:
    """This is the article the model dismissed as irrelevant on two runs out of three."""

    assert operational_signals("截至2026年7月份，全省100MW及以上统调机组异常停运48次。")
