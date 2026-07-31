from app.services.faq_service import FAQService


def test_faq_matches_known_question():
    result = FAQService().match("请问什么是虚拟电厂？")
    assert result is not None
    assert "聚合" in result.answer


def test_faq_returns_none_for_unknown_question():
    assert FAQService().match("查询今天的所有可用功率") is None
