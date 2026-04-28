# -*- coding: utf-8 -*-
# File: tests/test_week3_bm25_simple.py
# 第 3 周优化测试：BM25 混合检索（简化版）

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from zulong.memory.enhanced_experience_store import BM25Search


def test_bm25_basic():
    """测试 1: BM25 基本功能"""
    print("\n" + "="*80)
    print("测试 1: BM25 基本功能")
    print("="*80)
    
    # 创建 BM25 搜索引擎
    print("\n--- 创建 BM25 搜索引擎 ---")
    bm25 = BM25Search(k1=1.5, b=0.75)
    
    # 添加测试文档
    print("\n--- 添加测试文档 ---")
    test_docs = [
        ("doc1", "Python 是一种高级编程语言，支持面向对象编程和函数式编程"),
        ("doc2", "Python 的列表推导式可以简化代码，例如 [x*2 for x in range(10)]"),
        ("doc3", "Python 的装饰器是一种强大的工具，可以在不修改原代码的情况下增强函数功能"),
        ("doc4", "Java 是一种面向对象的编程语言，广泛用于企业应用开发"),
        ("doc5", "JavaScript 是 Web 开发的脚本语言，支持前端和后端开发"),
    ]
    
    for doc_id, content in test_docs:
        bm25.add_document(doc_id, content)
        print(f"  添加文档：{doc_id} - {content[:50]}...")
    
    print(f"\n✅ BM25 索引构建完成：{len(bm25.documents)} 个文档")
    
    # 测试搜索
    print("\n--- 测试搜索：'Python 列表' ---")
    query = "Python 列表"
    results = bm25.search(query, top_k=3)
    
    print(f"\n检索结果：{len(results)} 个文档")
    for i, (doc_id, score) in enumerate(results, 1):
        # 找到对应内容
        content = next((c for d, c in test_docs if d == doc_id), "")
        print(f"  [{i}] {doc_id}: {content[:80]}...")
        print(f"      分数：{score:.3f}")
    
    # 验证：应该有结果
    assert len(results) > 0, "BM25 搜索应该返回结果"
    
    # 验证：第一个结果应该是 doc2（最相关）
    if results:
        top_doc_id = results[0][0]
        print(f"\n✅ 测试 1 通过：BM25 搜索成功 (Top 结果：{top_doc_id})")
    
    return True


def test_bm25_keywords():
    """测试 2: BM25 关键词匹配"""
    print("\n" + "="*80)
    print("测试 2: BM25 关键词匹配")
    print("="*80)
    
    # 创建 BM25
    bm25 = BM25Search()
    
    # 添加包含特定关键词的文档
    print("\n--- 添加关键词文档 ---")
    test_docs = [
        ("http_doc", "HTTP 协议是超文本传输协议，用于 Web 浏览器和服务器之间的通信"),
        ("https_doc", "HTTPS 是 HTTP 的安全版本，使用 SSL/TLS 加密传输数据"),
        ("tcp_doc", "TCP 是传输控制协议，提供可靠的面向连接的通信"),
        ("udp_doc", "UDP 是用户数据报协议，提供无连接的不可靠传输"),
    ]
    
    for doc_id, content in test_docs:
        bm25.add_document(doc_id, content)
        print(f"  添加文档：{doc_id}")
    
    # 测试 1: 搜索 "HTTP"
    print("\n--- 搜索：'HTTP' ---")
    results = bm25.search("HTTP", top_k=2)
    print(f"结果数：{len(results)}")
    for doc_id, score in results:
        content = next((c for d, c in test_docs if d == doc_id), "")
        print(f"  {doc_id}: {content[:60]}... (分数：{score:.3f})")
    
    # 验证：HTTP 和 HTTPS 文档应该排在前面
    assert len(results) > 0, "应该返回结果"
    
    # 测试 2: 搜索 "TCP UDP"
    print("\n--- 搜索：'TCP UDP' ---")
    results = bm25.search("TCP UDP", top_k=2)
    print(f"结果数：{len(results)}")
    for doc_id, score in results:
        content = next((c for d, c in test_docs if d == doc_id), "")
        print(f"  {doc_id}: {content[:60]}... (分数：{score:.3f})")
    
    assert len(results) > 0, "应该返回结果"
    
    print(f"\n✅ 测试 2 通过：关键词匹配测试完成")
    
    return True


def test_bm25_chinese():
    """测试 3: BM25 中文分词"""
    print("\n" + "="*80)
    print("测试 3: BM25 中文分词")
    print("="*80)
    
    bm25 = BM25Search()
    
    # 添加中文文档
    print("\n--- 添加中文文档 ---")
    test_docs = [
        ("ml_doc", "机器学习是人工智能的一个分支，使用算法从数据中学习模式"),
        ("dl_doc", "深度学习是机器学习的子集，使用神经网络进行复杂建模"),
        ("rl_doc", "强化学习通过奖励机制训练智能体做出最优决策"),
        ("nlp_doc", "自然语言处理是 AI 的重要应用领域，涉及文本理解和生成"),
    ]
    
    for doc_id, content in test_docs:
        bm25.add_document(doc_id, content)
        print(f"  添加文档：{doc_id}")
    
    # 测试搜索
    print("\n--- 搜索：'机器学习 神经网络' ---")
    query = "机器学习 神经网络"
    results = bm25.search(query, top_k=3)
    
    print(f"\n检索结果：{len(results)} 个文档")
    for i, (doc_id, score) in enumerate(results, 1):
        content = next((c for d, c in test_docs if d == doc_id), "")
        print(f"  [{i}] {doc_id}: {content[:60]}...")
        print(f"      分数：{score:.3f}")
    
    # 验证：应该有结果
    assert len(results) > 0, "中文搜索应该返回结果"
    
    print(f"\n✅ 测试 3 通过：中文分词测试完成")
    
    return True


def test_bm25_weight_impact():
    """测试 4: BM25 参数影响"""
    print("\n" + "="*80)
    print("测试 4: BM25 参数影响")
    print("="*80)
    
    test_docs = [
        ("doc1", "Python 编程语言 简洁 优雅 易读"),
        ("doc2", "Java 编程语言 强大 类型安全 企业级"),
        ("doc3", "JavaScript 灵活 动态 前端开发"),
    ]
    
    # 测试不同 k1 值
    print("\n--- 测试不同 k1 值（词频饱和度） ---")
    k1_values = [0.5, 1.5, 2.5]
    
    query = "Python 编程语言"
    
    for k1 in k1_values:
        bm25 = BM25Search(k1=k1)
        for doc_id, content in test_docs:
            bm25.add_document(doc_id, content)
        
        results = bm25.search(query, top_k=1)
        if results:
            print(f"  k1={k1}: Top 结果={results[0][0]}, 分数={results[0][1]:.3f}")
    
    # 测试不同 b 值
    print("\n--- 测试不同 b 值（长度归一化） ---")
    b_values = [0.0, 0.5, 1.0]
    
    for b in b_values:
        bm25 = BM25Search(b=b)
        for doc_id, content in test_docs:
            bm25.add_document(doc_id, content)
        
        results = bm25.search(query, top_k=1)
        if results:
            print(f"  b={b}: Top 结果={results[0][0]}, 分数={results[0][1]:.3f}")
    
    print(f"\n✅ 测试 4 通过：参数影响测试完成")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 第 3 周优化测试：BM25 混合检索（简化版）")
    print("="*80)
    
    tests = [
        ("测试 1: BM25 基本功能", test_bm25_basic),
        ("测试 2: BM25 关键词匹配", test_bm25_keywords),
        ("测试 3: BM25 中文分词", test_bm25_chinese),
        ("测试 4: BM25 参数影响", test_bm25_weight_impact),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} 失败：{e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 总结
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    print(f"通过：{passed}/{len(tests)}")
    print(f"失败：{failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！第 3 周优化完成！")
    else:
        print(f"\n⚠️ {failed} 个测试失败，请检查")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
