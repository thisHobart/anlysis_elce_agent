from app.research.news.contracts import CharacterRange, ExtractionCoverage


def test_coverage_accepts_overlaps_but_detects_gaps_and_failures() -> None:
    complete = ExtractionCoverage(
        strategy="chunk_merge",
        total_characters=100,
        processed_ranges=(CharacterRange(start_char=0, end_char=60), CharacterRange(start_char=50, end_char=100)),
    )
    gap = complete.model_copy(
        update={"processed_ranges": (CharacterRange(start_char=0, end_char=49), CharacterRange(start_char=50, end_char=100))}
    )
    failed = complete.model_copy(
        update={"failed_ranges": (CharacterRange(start_char=20, end_char=30),)}
    )

    assert complete.complete
    assert not gap.complete
    assert not failed.complete
