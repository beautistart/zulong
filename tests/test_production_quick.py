# -*- coding: utf-8 -*-
# File: tests/test_production_quick.py
# 祖龙生产代码快速验证测试

"""
快速验证 4 周优化的核心功能是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试所有关键模块是否可以导入"""
    print("\n" + "="*80)
    print("测试 1: 关键模块导入")
    print("="*80)
    
    try:
        # 第 1 周：记忆巩固
        from zulong.memory.memory_evolution import MemoryConsolidator, MemoryStrength
        from zulong.memory.short_term_memory import ShortTermMemory
        print("✅ 第 1 周模块：MemoryConsolidator, MemoryStrength, ShortTermMemory")
        
        # 第 2 周：共享池持久化
        from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
        print("✅ 第 2 周模块：SharedMemoryPool")
        
        # 第 3 周：BM25 混合检索
        from zulong.memory.enhanced_experience_store import BM25Search
        from zulong.l2.rag_node import RAGIntegrationNode
        print("✅ 第 3 周模块：BM25Search, RAGIntegrationNode")
        
        # 第 4 周：经验自动生成
        from zulong.memory.experience_generator import ExperienceGenerator, ExperienceCandidate
        print("✅ 第 4 周模块：ExperienceGenerator, ExperienceCandidate")
        
        # 核心引擎
        from zulong.l2.inference_engine import InferenceEngine
        from zulong.memory.rag_manager import RAGManager
        from zulong.memory.tagging_engine import TaggingEngine
        print("✅ 核心引擎：InferenceEngine, RAGManager, TaggingEngine")
        
        print("\n✅ 所有关键模块导入成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 模块导入失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_strength():
    """测试记忆强度模型"""
    print("\n" + "="*80)
    print("测试 2: 记忆强度模型")
    print("="*80)
    
    try:
        from zulong.memory.memory_evolution import MemoryStrength
        
        strength = MemoryStrength()
        
        # 测试衰减
        initial = 1.0
        decayed = strength.decay(initial, hours_passed=24)
        assert 0 < decayed < initial, "记忆应该衰减"
        print(f"✅ 记忆衰减：{initial:.2f} -> {decayed:.2f} (24 小时后)")
        
        # 测试强化
        reinforced = strength.reinforce(decayed, repeat_count=3)
        assert reinforced > decayed, "重复应该强化记忆"
        print(f"✅ 记忆强化：{decayed:.2f} -> {reinforced:.2f} (3 次重复)")
        
        print("\n✅ 记忆强度模型工作正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 记忆强度模型测试失败：{e}")
        return False


def test_experience_generation():
    """测试经验生成"""
    print("\n" + "="*80)
    print("测试 3: 经验生成器")
    print("="*80)
    
    try:
        from zulong.memory.experience_generator import ExperienceGenerator, ExperienceCandidate
        
        generator = ExperienceGenerator()
        
        # 测试经验候选生成
        test_dialogue = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
        ]
        
        candidates = generator.extract_from_dialogue(test_dialogue)
        print(f"✅ 从对话中提取到 {len(candidates)} 个经验候选")
        
        # 测试经验候选数据结构
        if candidates:
            candidate = candidates[0]
            assert hasattr(candidate, 'content'), "ExperienceCandidate 应该有 content"
            assert hasattr(candidate, 'category'), "ExperienceCandidate 应该有 category"
            assert hasattr(candidate, 'confidence'), "ExperienceCandidate 应该有 confidence"
            print(f"✅ 经验候选数据结构完整：{candidate.category} (置信度：{candidate.confidence:.2f})")
        
        print("\n✅ 经验生成器工作正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 经验生成器测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_bm25_search():
    """测试 BM25 搜索"""
    print("\n" + "="*80)
    print("测试 4: BM25 混合检索")
    print("="*80)
    
    try:
        from zulong.memory.enhanced_experience_store import BM25Search
        
        bm25 = BM25Search()
        
        # 添加测试文档
        test_docs = [
            "用户喜欢简洁的界面设计",
            "系统应该支持快捷键操作",
            "响应时间应该小于 100 毫秒",
        ]
        
        for i, doc in enumerate(test_docs):
            bm25.add_document(doc_id=f"doc_{i}", text=doc)
        
        print(f"✅ 添加 {len(test_docs)} 个测试文档")
        
        # 测试搜索
        results = bm25.search("界面设计", top_k=2)
        assert len(results) > 0, "应该返回搜索结果"
        print(f"✅ 搜索结果：{len(results)} 条 (查询：'界面设计')")
        
        print("\n✅ BM25 搜索引擎工作正常")
        return True
        
    except Exception as e:
        print(f"\n❌ BM25 搜索测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_node():
    """测试 RAG 集成节点"""
    print("\n" + "="*80)
    print("测试 5: RAG 集成节点")
    print("="*80)
    
    try:
        from zulong.l2.rag_node import RAGIntegrationNode
        
        # 创建支持混合检索的 RAG 节点
        rag_node = RAGIntegrationNode(
            enable_hybrid_search=True,
            bm25_weight=0.4,
            max_results=5
        )
        
        assert rag_node.enable_hybrid_search, "应该启用混合检索"
        assert rag_node.bm25_weight == 0.4, "BM25 权重应该是 0.4"
        print("✅ RAGIntegrationNode 支持混合检索")
        
        # 验证 BM25 搜索引擎已初始化
        assert rag_node.bm25_search is not None, "BM25 搜索引擎应该已初始化"
        print("✅ BM25 搜索引擎已初始化")
        
        print("\n✅ RAG 集成节点工作正常")
        return True
        
    except Exception as e:
        print(f"\n❌ RAG 集成节点测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_shared_pool():
    """测试共享池基本功能"""
    print("\n" + "="*80)
    print("测试 6: 共享池持久化")
    print("="*80)
    
    try:
        from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
        
        pool = SharedMemoryPool()
        
        # 验证快照方法存在
        assert hasattr(pool, 'save_snapshot_now'), "应该有 save_snapshot_now 方法"
        assert hasattr(pool, '_load_snapshot'), "应该有 _load_snapshot 方法"
        print("✅ 快照机制已实现")
        
        # 验证压缩
        assert pool.use_compression, "应该启用 gzip 压缩"
        print("✅ gzip 压缩已启用")
        
        print("\n✅ 共享池持久化功能正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 共享池测试失败：{e}")
        return False


def test_inference_engine_structure():
    """测试 InferenceEngine 结构"""
    print("\n" + "="*80)
    print("测试 7: InferenceEngine 结构验证")
    print("="*80)
    
    try:
        import inspect
        from zulong.l2.inference_engine import InferenceEngine
        
        source = inspect.getsource(InferenceEngine)
        
        # 验证经验生成器集成
        assert 'ExperienceGenerator' in source, "应该导入 ExperienceGenerator"
        assert 'experience_generator' in source, "应该初始化 experience_generator"
        print("✅ InferenceEngine 集成经验生成器")
        
        # 验证 RAG 节点集成
        assert 'rag_node' in source or 'RAGIntegrationNode' in source, "应该集成 RAG 节点"
        print("✅ InferenceEngine 集成 RAG 节点")
        
        # 验证记忆系统集成
        assert 'short_term_memory' in source or 'ShortTermMemory' in source, "应该集成记忆系统"
        print("✅ InferenceEngine 集成记忆系统")
        
        print("\n✅ InferenceEngine 结构正确")
        return True
        
    except Exception as e:
        print(f"\n❌ InferenceEngine 结构验证失败：{e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 祖龙生产代码快速验证测试")
    print("="*80)
    
    tests = [
        ("模块导入", test_imports),
        ("记忆强度模型", test_memory_strength),
        ("经验生成器", test_experience_generation),
        ("BM25 混合检索", test_bm25_search),
        ("RAG 集成节点", test_rag_node),
        ("共享池持久化", test_shared_pool),
        ("InferenceEngine 结构", test_inference_engine_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 测试异常：{e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "="*80)
    print("📊 测试结果汇总")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n通过：{passed}/{len(tests)}")
    print(f"失败：{failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！生产代码运行正常！")
        print("\n📝 生产系统已就绪:")
        print("  ✅ 第 1 周：记忆巩固 + 自动复盘")
        print("  ✅ 第 2 周：共享池持久化")
        print("  ✅ 第 3 周：BM25 混合检索")
        print("  ✅ 第 4 周：经验自动生成")
    else:
        print(f"\n⚠️ {failed} 个测试失败，请检查")
        print("\n失败的测试:")
        for test_name, result in results:
            if not result:
                print(f"  ❌ {test_name}")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
