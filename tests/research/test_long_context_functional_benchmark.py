from scripts.benchmark_long_context_strategy import (
    _document,
    load_cases,
    run_benchmark,
    run_context_boundary_checks,
)


def test_shared_long_context_corpus_covers_required_failure_shapes() -> None:
    cases = load_cases()
    descriptions = " ".join(case["description"] for case in cases)

    assert len(cases) == 10
    assert "尾部" in descriptions
    assert "更正" in descriptions
    assert "撤回" in descriptions
    assert "不同段落" in descriptions
    assert "切分边界" in descriptions
    assert "最大 Token" in descriptions
    assert "上下文限制" in descriptions
    assert any(case["expectation"] == "safe_failure" for case in cases)


def test_strategy_functional_benchmark_never_admits_incomplete_evidence_to_price_analysis() -> None:
    report = run_benchmark()

    assert report["case_count"] == 10
    assert report["evidence_integrity"] == 1.0
    assert report["unsafe_price_admissions"] == 0
    assert report["context_boundary_checks"]["passed"]


def test_complete_request_budget_accepts_below_and_equal_but_rejects_above_limit() -> None:
    report = run_context_boundary_checks()

    assert report["passed"]
    assert [item["accepted"] for item in report["profiles"]] == [True, True, False]


def test_gold_descriptions_do_not_leak_into_model_visible_titles() -> None:
    case = next(item for item in load_cases() if item["case_id"] == "LC07")

    assert case["description"] not in _document(case).title
