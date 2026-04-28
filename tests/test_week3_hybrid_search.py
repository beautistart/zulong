# -*- coding: utf-8 -*-
# File: tests/test_week3_hybrid_search.py
# 第 3 周优化测试：BM25 混合检索

import sys
from pathlib import Path
import time

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from zulong.memory.rag_manager import RAGManager, RAGConfig
from zulong.l2.rag_node import RAGIntegrationNode


async def test_hybrid_search():
    """测试 1: 混合检索（BM25 + 向量）"""
    print("\n" + "="*80)
    print("测试 1: 混合检索（BM25 + 向量）")
    print("="*80)
    
    # 创建 RAG 管理器
    print("\n--- 初始化 RAG 管理器 ---")
    config = RAGConfig(
        vector_dimension=768,
        base_path="./data/rag_test"
    )
    rag_manager = RAGManager(config)
    
    # 添加测试经验
    print("\n--- 添加测试经验 ---")
    test_experiences = [
        {
            "content": "Python 是一种高级编程语言，支持面向对象编程和函数式编程",
            "category": "programming",
            "importance": "important"
        },
        {
            "content": "Python 的列表推导式可以简化代码，例如 [x*2 for x in range(10)]",
            "category": "programming",
            "importance": "useful"
        },
        {
            "content": "Python 的装饰器是一种强大的工具，可以在不修改原代码的情况下增强函数功能",
            "category": "programming",
            "importance": "advanced"
        },
        {
            "content": "Python 的 GIL（全局解释器锁）限制了多线程程序的性能",
            "category": "programming",
            "importance": "critical"
        }
    ]
    
    for exp in test_experiences:
        doc_id = rag_manager.add_experience(
            content=exp["content"],
            category=exp["category"],
            importance=exp["importance"]
        )
        print(f"  添加经验：{doc_id[:8]}... - {exp['content'][:50]}...")
    
    # 等待 Embedding 模型加载
    await asyncio.sleep(2)
    
    # 创建 RAG 节点（启用混合检索）
    print("\n--- 创建 RAG 节点（混合检索） ---")
    rag_node = RAGIntegrationNode(
        rag_manager=rag_manager,
        max_results=3,
        enable_hybrid_search=True,  # 启用混合检索
        bm25_weight=0.4,  # BM25 权重 0.4
        top_k_bm25=10  # BM25 初检 10 个
    )
    
    # 同步 BM25 索引
    print("\n--- 同步 BM25 索引 ---")
    rag_node.set_rag_manager(rag_manager)
    await asyncio.sleep(1)
    
    # 测试查询
    print("\n--- 测试查询：'Python 列表推导式' ---")
    query = "Python 列表推导式"
    
    # 模拟状态
    state = {
        "query": query,
        "context": {},
        "rag_results": [],
        "retrieved_docs": [],
        "target_rag": "experience_rag",
        "search_metadata": {},
        "messages": []
    }
    
    # 执行检索
    result_state = rag_node.retrieve(state)
    
    # 检查结果
    documents = result_state.get("retrieved_docs", [])
    print(f"\n检索结果：{len(documents)} 个文档")
    
    for i, doc in enumerate(documents, 1):
        print(f"\n  [{i}] {doc['rag']}")
        print(f"      内容：{doc['content'][:100]}...")
        print(f"      融合分数：{doc['similarity']:.3f}")
        if 'bm25_score' in doc and 'vector_score' in doc:
            print(f"      BM25: {doc['bm25_score']:.3f}, Vector: {doc['vector_score']:.3f}")
    
    # 验证：应该有结果
    assert len(documents) > 0, "混合检索应该返回结果"
    
    # 验证：第一个结果应该相关
    if documents:
        top_doc = documents[0]
        assert top_doc['similarity'] > 0.3, "最高分应该大于阈值"
        print(f"\n✅ 测试 1 通过：混合检索成功 (Top 结果分数：{top_doc['similarity']:.3f})")
    
    return True


async def test_bm25_vs_vector():
    """测试 2: BM25 vs 向量检索对比"""
    print("\n" + "="*80)
    print("测试 2: BM25 vs 向量检索对比")
    print("="*80)
    
    # 创建 RAG 管理器
    print("\n--- 初始化 RAG 管理器 ---")
    config = RAGConfig(vector_dimension=768, base_path="./data/rag_test2")
    rag_manager = RAGManager(config)
    
    # 添加包含专业术语的经验
    print("\n--- 添加专业术语经验 ---")
    test_experiences = [
        {
            "content": "HTTP 协议是超文本传输协议，用于 Web 浏览器和服务器之间的通信",
            "category": "network",
            "importance": "important"
        },
        {
            "content": "HTTPS 是 HTTP 的安全版本，使用 SSL/TLS 加密传输数据",
            "category": "network",
            "importance": "important"
        },
        {
            "content": "TCP 是传输控制协议，提供可靠的面向连接的通信",
            "category": "network",
            "importance": "basic"
        },
        {
            "content": "UDP 是用户数据报协议，提供无连接的不可靠传输",
            "category": "network",
            "importance": "basic"
        }
    ]
    
    for exp in test_experiences:
        doc_id = rag_manager.add_experience(
            content=exp["content"],
            category=exp["category"],
            importance=exp["importance"]
        )
        print(f"  添加经验：{doc_id[:8]}...")
    
    await asyncio.sleep(2)
    
    # 测试 1: 纯向量检索
    print("\n--- 测试 1: 纯向量检索 ---")
    rag_node_vector = RAGIntegrationNode(
        rag_manager=rag_manager,
        max_results=3,
        enable_hybrid_search=False  # 不启用混合检索
    )
    
    query = "HTTP 和 HTTPS 的区别"
    state = {"query": query, "context": {}, "rag_results": [], "retrieved_docs": [], 
             "target_rag": "experience_rag", "search_metadata": {}, "messages": []}
    
    result_state_vector = rag_node_vector.retrieve(state)
    vector_docs = result_state_vector.get("retrieved_docs", [])
    
    print(f"向量检索结果：{len(vector_docs)} 个文档")
    for i, doc in enumerate(vector_docs, 1):
        print(f"  [{i}] {doc['content'][:80]}... (分数：{doc['similarity']:.3f})")
    
    # 测试 2: 混合检索
    print("\n--- 测试 2: 混合检索 (BM25 + 向量) ---")
    rag_node_hybrid = RAGIntegrationNode(
        rag_manager=rag_manager,
        max_results=3,
        enable_hybrid_search=True,
        bm25_weight=0.4
    )
    rag_node_hybrid.set_rag_manager(rag_manager)
    await asyncio.sleep(1)
    
    result_state_hybrid = rag_node_hybrid.retrieve(state)
    hybrid_docs = result_state_hybrid.get("retrieved_docs", [])
    
    print(f"混合检索结果：{len(hybrid_docs)} 个文档")
    for i, doc in enumerate(hybrid_docs, 1):
        print(f"  [{i}] {doc['content'][:80]}... (融合：{doc['similarity']:.3f}, BM25: {doc.get('bm25_score', 0):.3f}, Vector: {doc.get('vector_score', 0):.3f})")
    
    # 验证：混合检索应该返回结果
    assert len(hybrid_docs) > 0, "混合检索应该返回结果"
    
    print(f"\n✅ 测试 2 通过：BM25 vs 向量检索对比完成")
    
    return True


async def test_bm25_weight_impact():
    """测试 3: BM25 权重影响"""
    print("\n" + "="*80)
    print("测试 3: BM25 权重影响")
    print("="*80)
    
    # 创建 RAG 管理器
    config = RAGConfig(vector_dimension=768, base_path="./data/rag_test3")
    rag_manager = RAGManager(config)
    
    # 添加测试数据
    test_experiences = [
        {"content": "机器学习是人工智能的一个分支，使用算法从数据中学习", "category": "ai", "importance": "important"},
        {"content": "深度学习是机器学习的子集，使用神经网络进行建模", "category": "ai", "importance": "important"},
        {"content": "强化学习通过奖励机制训练智能体做出决策", "category": "ai", "importance": "useful"},
    ]
    
    for exp in test_experiences:
        rag_manager.add_experience(content=exp["content"], category=exp["category"], importance=exp["importance"])
    
    await asyncio.sleep(2)
    
    # 测试不同权重
    query = "机器学习 深度学习"
    state = {"query": query, "context": {}, "rag_results": [], "retrieved_docs": [], 
             "target_rag": "experience_rag", "search_metadata": {}, "messages": []}
    
    weights = [0.2, 0.4, 0.6, 0.8]
    
    print(f"\n查询：'{query}'")
    print(f"\n不同 BM25 权重的结果对比:")
    
    for weight in weights:
        rag_node = RAGIntegrationNode(
            rag_manager=rag_manager,
            max_results=3,
            enable_hybrid_search=True,
            bm25_weight=weight
        )
        rag_node.set_rag_manager(rag_manager)
        await asyncio.sleep(0.5)
        
        result_state = rag_node.retrieve(state)
        docs = result_state.get("retrieved_docs", [])
        
        print(f"\n  BM25 权重={weight}:")
        if docs:
            print(f"    Top 结果分数：{docs[0]['similarity']:.3f}")
            if 'bm25_score' in docs[0] and 'vector_score' in docs[0]:
                print(f"    BM25: {docs[0]['bm25_score']:.3f}, Vector: {docs[0]['vector_score']:.3f}")
    
    print(f"\n✅ 测试 3 通过：BM25 权重影响测试完成")
    
    return True


async def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 第 3 周优化测试：BM25 混合检索")
    print("="*80)
    
    tests = [
        ("测试 1: 混合检索", test_hybrid_search),
        ("测试 2: BM25 vs 向量检索", test_bm25_vs_vector),
        ("测试 3: BM25 权重影响", test_bm25_weight_impact),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
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
    success = asyncio.run(main())
    exit(0 if success else 1)
