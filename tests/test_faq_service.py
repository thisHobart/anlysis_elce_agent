from app.services.faq_service import FAQService


def test_faq_matches_known_question():
    result = FAQService().match("请问什么是虚拟电厂？")
    assert result is not None
    assert result.faq_id == "FAQ-01"
    assert result.category == "basic_info"


def test_faq_returns_none_for_unrelated_question():
    assert FAQService().match("今天天气怎么样") is None
    assert FAQService().match("帮我写一段 Python 代码") is None


def test_faq_get_by_id():
    svc = FAQService()
    entry = svc.get("FAQ-20")
    assert entry is not None
    assert entry["category"] == "statistics"
    assert svc.get("FAQ-11") is None  # 源缺失编号
