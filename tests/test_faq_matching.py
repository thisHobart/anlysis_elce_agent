"""FAQ 知识库结构 + 匹配器 + 模板渲染测试（P2）。"""

import pytest

from app.knowledge.faqs import CATEGORIES, FAQS
from app.services.faq_service import FAQService
from app.services.template_service import render_faq_template

VALID_SOURCES = {"api", "db", "knowledge", "static"}


# ----------------------------- 结构完整性 -----------------------------

def test_faq_count_and_unique_ids():
    ids = [f["faq_id"] for f in FAQS]
    assert len(FAQS) == 29  # 源缺 FAQ-11
    assert len(ids) == len(set(ids)), "存在重复 faq_id"
    assert "FAQ-11" not in ids


def test_every_faq_has_required_fields():
    for f in FAQS:
        assert f["faq_id"] and f["triggers"] and f["template"], f["faq_id"]
        assert f["category"] in CATEGORIES, f["faq_id"]
        assert isinstance(f["screen_action"], str)
        assert f["data_requirements"], f["faq_id"]


def test_data_requirement_sources_valid():
    for f in FAQS:
        # call_group 内允许合并调用：组内至少一条定义 vpp_command，其余共享
        group_has_cmd = {
            req["call_group"]
            for req in f["data_requirements"]
            if req["source"] == "api" and req.get("call_group") and "vpp_command" in req
        }
        for req in f["data_requirements"]:
            assert req["source"] in VALID_SOURCES, (f["faq_id"], req)
            if req["source"] == "api":
                # 有自己的命令，或与同组某条共享命令（call_group 合并调用）
                assert "vpp_command" in req or req.get("call_group") in group_has_cmd, (f["faq_id"], req.get("variable"))
            if req["source"] == "db":
                assert "db_command" in req or "db_query" in req, (f["faq_id"], req.get("variable"))


# ----------------------------- 匹配效果 -----------------------------

# 覆盖 5 大类的代表性问题 → 期望命中的 FAQ
MATCH_CASES = [
    ("什么是虚拟电厂", "FAQ-01"),
    ("介绍一下聊城虚拟电厂", "FAQ-02"),
    ("虚拟电厂有哪些资源类型", "FAQ-03"),
    ("虚拟电厂怎么赚钱", "FAQ-04"),
    ("接入了多少家企业", "FAQ-05"),
    ("当前总发电功率是多少", "FAQ-06"),
    ("储能SOC怎么样", "FAQ-07"),
    ("当前可调节容量是多少", "FAQ-08"),
    ("今天响应情况怎么样", "FAQ-10"),
    ("有多少设备在线", "FAQ-12"),
    ("当前有缺口任务吗", "FAQ-14"),
    ("本月发电量多少", "FAQ-16"),
    ("碳减排量是多少", "FAQ-20"),
    ("最近一周发电趋势怎么样", "FAQ-24"),
    ("当前有哪些设备异常", "FAQ-28"),
    ("哪些企业响应率偏低", "FAQ-29"),
]


@pytest.mark.parametrize("question, expected_id", MATCH_CASES)
def test_representative_questions_match(question, expected_id):
    result = FAQService().match(question)
    assert result is not None, question
    assert result.faq_id == expected_id, f"{question} -> {result.faq_id}, 期望 {expected_id}"


def test_short_keyword_query_matches():
    # 用户只打关键词（短于触发词）也应命中
    assert FAQService().match("发电功率").faq_id == "FAQ-06"


@pytest.mark.parametrize("question", ["今天天气如何", "北京到上海怎么走", "讲个笑话"])
def test_non_vpp_questions_do_not_match(question):
    assert FAQService().match(question) is None


# ----------------------------- 模板渲染 -----------------------------

def test_static_faq_renders_fully():
    entry = FAQService().get("FAQ-01")
    rendered = render_faq_template(entry["template"], {})
    assert "〔" not in rendered  # 静态模板无占位符
    assert "虚拟电厂" in rendered


def test_var_faq_shows_placeholder_when_no_data():
    entry = FAQService().get("FAQ-06")
    rendered = render_faq_template(entry["template"], {})
    assert "〔total_power〕" in rendered


def test_var_faq_fills_with_data():
    entry = FAQService().get("FAQ-06")
    rendered = render_faq_template(
        entry["template"], {"total_power": 125.6, "pv_power": 45.2, "storage_discharge": 30.1}
    )
    assert "125.6" in rendered and "45.2" in rendered
    assert "〔" not in rendered


def test_if_template_both_branches():
    entry = FAQService().get("FAQ-14")
    yes = render_faq_template(entry["template"], {"has_task": True, "task_name": "迎峰度夏", "time_slot": "14:00-15:00", "gap_mw": 12.0})
    no = render_faq_template(entry["template"], {"has_task": False})
    assert "迎峰度夏" in yes
    assert "没有进行中的缺口任务" in no


def test_each_template_loops_items():
    entry = FAQService().get("FAQ-28")
    rendered = render_faq_template(
        entry["template"],
        {"count": 2, "events": [{"device_name": "储能PCS-01", "message": "离线"}, {"device_name": "光伏逆变器-03", "message": "数据采集异常"}]},
    )
    assert "储能PCS-01（离线）" in rendered
    assert "光伏逆变器-03（数据采集异常）" in rendered
