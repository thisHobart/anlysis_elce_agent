from scripts.benchmark_long_context_strategy import load_cases, run_benchmark


def test_shared_long_context_corpus_covers_required_failure_shapes() -> None:
    cases = load_cases()
    descriptions = " ".join(case["description"] for case in cases)

    assert len(cases) == 6
    assert "尾部" in descriptions
    assert "更正" in descriptions
    assert "不同段落" in descriptions
    assert any(case["expectation"] == "safe_failure" for case in cases)


def test_strategy_functional_benchmark_never_admits_incomplete_evidence_to_price_analysis() -> None:
    report = run_benchmark()

    assert report["case_count"] == 6
    assert report["evidence_integrity"] == 1.0
    assert report["unsafe_price_admissions"] == 0
