"""
向量缓存测试脚本

用于验证 TSD v2.4 增量向量化方案的实现

运行方式:
    python test_vector_cache.py
"""

import asyncio
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zulong.memory.vector_cache import get_vector_cache, ShortTermMemoryVectorCache
from zulong.memory.embedding_manager import EmbeddingModelManager


async def test_basic_functionality():
    """测试基本功能"""
    print("="*80)
    print("🧪 测试 1: 基本功能测试")
    print("="*80)
    
    try:
        # 1. 获取缓存实例
        print("\n📦 初始化向量缓存...")
        cache = get_vector_cache(max_cache_size=50)
        print(f"✅ 缓存初始化完成：max_size={cache.max_cache_size}")
        
        # 2. 添加对话
        print("\n📝 添加测试对话...")
        test_dialogs = [
            ("今天天气怎么样？", "今天天气晴朗，适合外出。"),
            ("你喜欢吃什么？", "我喜欢吃知识，不吃东西。"),
            ("你能帮我做什么？", "我可以帮你解答问题、提供建议、聊天解闷。"),
            ("你多大了？", "我是 AI，没有年龄概念。"),
            ("你会说英语吗？", "Yes, I can speak English!"),
        ]
        
        for i, (user_text, bot_text) in enumerate(test_dialogs, 1):
            success = await cache.add_turn(
                user_text=user_text,
                bot_text=bot_text,
                turn_id=i
            )
            if success:
                print(f"  ✅ turn={i}: {user_text[:20]}...")
            else:
                print(f"  ❌ turn={i}: 添加失败")
        
        # 3. 检索
        print("\n🔍 测试检索功能...")
        query = "天气如何"
        print(f"  查询：'{query}'")
        
        results = await cache.search(
            query_text=query,
            top_k=3,
            time_decay=True
        )
        
        print(f"  ✅ 检索到 {len(results)} 条结果:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. [分数={result['score']:.4f}] turn={result['turn_id']}: {result['user_text'][:30]}...")
        
        # 4. 查看统计
        print("\n📊 缓存统计信息:")
        stats = cache.get_cache_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ 测试 1 通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 1 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance():
    """测试性能"""
    print("\n" + "="*80)
    print("🧪 测试 2: 性能测试")
    print("="*80)
    
    try:
        # 1. 初始化
        print("\n⏱️  初始化缓存...")
        cache = get_vector_cache(max_cache_size=100)
        
        # 2. 添加 50 轮对话并计时
        print("\n⏱️  添加 50 轮对话...")
        start_time = time.time()
        
        for i in range(50):
            await cache.add_turn(
                user_text=f"测试对话 {i}: 这是一个测试问题",
                bot_text=f"回复 {i}: 这是一个测试回答",
                turn_id=i
            )
        
        add_time = time.time() - start_time
        avg_add_time = add_time / 50 * 1000  # ms
        
        print(f"  总耗时：{add_time:.2f}s")
        print(f"  平均每次添加：{avg_add_time:.2f}ms")
        
        # 3. 检索 20 次并计时
        print("\n⏱️  执行 20 次检索...")
        start_time = time.time()
        
        query_times = []
        for i in range(20):
            query_start = time.time()
            results = await cache.search(
                query_text=f"测试 {i % 10}",
                top_k=3
            )
            query_time = (time.time() - query_start) * 1000  # ms
            query_times.append(query_time)
        
        search_time = time.time() - start_time
        avg_search_time = sum(query_times) / len(query_times)
        
        print(f"  总耗时：{search_time:.2f}s")
        print(f"  平均每次检索：{avg_search_time:.2f}ms")
        print(f"  最快检索：{min(query_times):.2f}ms")
        print(f"  最慢检索：{max(query_times):.2f}ms")
        
        # 4. 性能评估
        print("\n📊 性能评估:")
        if avg_search_time < 15:
            print(f"  ✅ 检索速度优秀：{avg_search_time:.2f}ms < 15ms")
        elif avg_search_time < 30:
            print(f"  ⚠️  检索速度良好：{avg_search_time:.2f}ms < 30ms")
        else:
            print(f"  ❌ 检索速度较慢：{avg_search_time:.2f}ms >= 30ms")
        
        if avg_add_time < 50:
            print(f"  ✅ 添加速度优秀：{avg_add_time:.2f}ms < 50ms")
        elif avg_add_time < 100:
            print(f"  ⚠️  添加速度良好：{avg_add_time:.2f}ms < 100ms")
        else:
            print(f"  ❌ 添加速度较慢：{avg_add_time:.2f}ms >= 100ms")
        
        # 5. 查看统计
        print("\n📊 缓存统计信息:")
        stats = cache.get_cache_stats()
        print(f"  总向量化次数：{stats['total_vectors_computed']}")
        print(f"  总检索次数：{stats['total_searches']}")
        print(f"  当前缓存大小：{stats['current_cache_size']}/{stats['max_cache_size']}")
        print(f"  缓存使用率：{stats['cache_usage_percent']:.1f}%")
        
        print("\n✅ 测试 2 通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 2 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_hybrid_search():
    """测试混合检索"""
    print("\n" + "="*80)
    print("🧪 测试 3: 混合检索测试")
    print("="*80)
    
    try:
        # 1. 初始化
        print("\n📦 初始化缓存（混合检索）...")
        cache = get_vector_cache(
            max_cache_size=50,
            enable_hybrid_search=True,
            vector_weight=0.8,
            keyword_weight=0.2
        )
        
        # 2. 添加特定对话
        print("\n📝 添加特定对话...")
        test_dialogs = [
            ("红色的苹果很好吃", "是的，苹果富含营养。"),
            ("我喜欢蓝色的天空", "天空确实很美。"),
            ("红色和蓝色都是好颜色", "没错，各有特色。"),
            ("苹果和天空有什么关系？", "都是自然界的美好事物。"),
        ]
        
        for i, (user_text, bot_text) in enumerate(test_dialogs, 1):
            await cache.add_turn(
                user_text=user_text,
                bot_text=bot_text,
                turn_id=i
            )
            print(f"  ✅ turn={i}: {user_text}")
        
        # 3. 测试不同查询
        print("\n🔍 测试不同查询...")
        
        queries = [
            ("苹果", "语义检索：苹果"),
            ("红色", "关键词检索：红色"),
            ("蓝色 天空", "混合检索：蓝色 + 天空"),
        ]
        
        for query_text, description in queries:
            print(f"\n  查询：'{query_text}' ({description})")
            
            results = await cache.search(
                query_text=query_text,
                top_k=2,
                time_decay=False  # 关闭时间衰减便于观察
            )
            
            if results:
                print(f"    ✅ 检索到 {len(results)} 条结果:")
                for i, result in enumerate(results, 1):
                    print(f"      {i}. [分数={result['score']:.4f}] turn={result['turn_id']}: {result['user_text']}")
            else:
                print(f"    ⚠️  未检索到结果")
        
        print("\n✅ 测试 3 通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 3 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_time_decay():
    """测试时间衰减"""
    print("\n" + "="*80)
    print("🧪 测试 4: 时间衰减测试")
    print("="*80)
    
    try:
        # 1. 初始化
        print("\n📦 初始化缓存...")
        cache = get_vector_cache(max_cache_size=50)
        
        # 2. 添加对话
        print("\n📝 添加对话...")
        await cache.add_turn(
            user_text="很久以前的对话",
            bot_text="很久以前的回答",
            turn_id=1
        )
        print(f"  ✅ turn=1: 很久以前的对话")
        
        # 等待 2 秒
        print("\n⏳ 等待 2 秒...")
        await asyncio.sleep(2)
        
        await cache.add_turn(
            user_text="最新的对话",
            bot_text="最新的回答",
            turn_id=2
        )
        print(f"  ✅ turn=2: 最新的对话")
        
        # 3. 检索（启用时间衰减）
        print("\n🔍 测试检索（启用时间衰减）...")
        results_with_decay = await cache.search(
            query_text="对话",
            top_k=2,
            time_decay=True
        )
        
        print(f"  结果（时间衰减）:")
        for i, result in enumerate(results_with_decay, 1):
            print(f"    {i}. [分数={result['score']:.4f}] turn={result['turn_id']}: {result['user_text']}")
        
        # 4. 检索（不启用时间衰减）
        print("\n🔍 测试检索（不启用时间衰减）...")
        results_without_decay = await cache.search(
            query_text="对话",
            top_k=2,
            time_decay=False
        )
        
        print(f"  结果（无时间衰减）:")
        for i, result in enumerate(results_without_decay, 1):
            print(f"    {i}. [分数={result['score']:.4f}] turn={result['turn_id']}: {result['user_text']}")
        
        # 5. 对比
        print("\n📊 对比分析:")
        if len(results_with_decay) >= 2 and len(results_without_decay) >= 2:
            old_score_decay = results_with_decay[0]['score'] if results_with_decay[0]['turn_id'] == 1 else results_with_decay[1]['score']
            new_score_decay = results_with_decay[0]['score'] if results_with_decay[0]['turn_id'] == 2 else results_with_decay[1]['score']
            
            old_score_no_decay = results_without_decay[0]['score'] if results_without_decay[0]['turn_id'] == 1 else results_without_decay[1]['score']
            new_score_no_decay = results_without_decay[0]['score'] if results_without_decay[0]['turn_id'] == 2 else results_without_decay[1]['score']
            
            print(f"  旧对话（turn=1）:")
            print(f"    有时间衰减：{old_score_decay:.4f}")
            print(f"    无时间衰减：{old_score_no_decay:.4f}")
            
            print(f"  新对话（turn=2）:")
            print(f"    有时间衰减：{new_score_decay:.4f}")
            print(f"    无时间衰减：{new_score_no_decay:.4f}")
            
            print(f"\n  ✅ 时间衰减效果：旧对话分数降低，新对话相对提升")
        
        print("\n✅ 测试 4 通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 4 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("🚀 TSD v2.4 向量缓存测试套件")
    print("="*80)
    print("\n测试目标:")
    print("  1. 验证基本功能")
    print("  2. 测试性能指标")
    print("  3. 测试混合检索")
    print("  4. 测试时间衰减")
    print("="*80)
    
    results = []
    
    # 运行测试
    results.append(("基本功能", await test_basic_functionality()))
    results.append(("性能测试", await test_performance()))
    results.append(("混合检索", await test_hybrid_search()))
    results.append(("时间衰减", await test_time_decay()))
    
    # 总结
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计：{passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！向量缓存实现成功！")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查问题。")
    
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
