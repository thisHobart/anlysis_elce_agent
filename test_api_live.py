#!/usr/bin/env python3
"""测试 VPP Agent API"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查"""
    print("=" * 60)
    print("1. 测试健康检查")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    print(f"状态: {data.get('status')}")
    print(f"环境: {data.get('environment')}")
    print()

def test_query_faq():
    """测试 FAQ 查询"""
    print("=" * 60)
    print("2. 测试 FAQ 查询")
    print("=" * 60)

    payload = {
        "session_id": "test-001",
        "question": "什么是虚拟电厂？",
        "role": "viewer"
    }

    print(f"问题: {payload['question']}")
    response = requests.post(f"{BASE_URL}/v1/query", json=payload)
    data = response.json()

    print(f"回答: {data['answer']}")
    print(f"路由: {data['route']}")
    print(f"FAQ ID: {data.get('faq_id')}")
    print(f"分类来源: {data['classification_source']}")
    print()

def test_query_with_action():
    """测试带大屏动作的查询"""
    print("=" * 60)
    print("3. 测试带大屏动作的查询")
    print("=" * 60)

    payload = {
        "session_id": "test-002",
        "question": "当前总发电功率是多少？",
        "role": "viewer",
        "request_action": True
    }

    print(f"问题: {payload['question']}")
    response = requests.post(f"{BASE_URL}/v1/query", json=payload)
    data = response.json()

    print(f"回答: {data['answer']}")
    print(f"路由: {data['route']}")
    print(f"FAQ ID: {data.get('faq_id')}")
    print(f"大屏动作: {data.get('screen_action')}")
    print()

def test_multi_turn():
    """测试多轮对话"""
    print("=" * 60)
    print("4. 测试多轮对话")
    print("=" * 60)

    session_id = "test-multi-turn"

    # 第一轮
    payload1 = {
        "session_id": session_id,
        "question": "山东鲁能新能源的实时负荷是多少？",
        "role": "viewer"
    }

    print(f"Q1: {payload1['question']}")
    response1 = requests.post(f"{BASE_URL}/v1/query", json=payload1)
    data1 = response1.json()
    print(f"A1: {data1['answer']}")
    print()

    # 第二轮（使用上下文）
    payload2 = {
        "session_id": session_id,
        "question": "它的可调能力怎么样？",
        "role": "viewer"
    }

    print(f"Q2: {payload2['question']}")
    response2 = requests.post(f"{BASE_URL}/v1/query", json=payload2)
    data2 = response2.json()
    print(f"A2: {data2['answer']}")
    print()

def test_knowledge_query():
    """测试知识检索"""
    print("=" * 60)
    print("5. 测试知识检索")
    print("=" * 60)

    payload = {
        "session_id": "test-003",
        "question": "什么是需求响应？",
        "role": "viewer"
    }

    print(f"问题: {payload['question']}")
    response = requests.post(f"{BASE_URL}/v1/query", json=payload)
    data = response.json()

    print(f"回答: {data['answer'][:200]}...")
    print(f"路由: {data['route']}")
    print()

if __name__ == "__main__":
    try:
        test_health()
        test_query_faq()
        test_query_with_action()
        test_multi_turn()
        test_knowledge_query()

        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务，请确保服务已启动")
        print("   运行: python -m uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
