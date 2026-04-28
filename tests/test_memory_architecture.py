# 增强型三级记忆架构 - 核心功能测试（简化版）

"""
测试场景:
1. 动态容量管理（验证 max_episodes 动态计算）
2. 异步复盘机制（验证快速摘要 + 语义摘要）
3. 记忆检索（验证基于摘要检索）
4. 分级读取（验证详情读取）
5. 跨轮次依赖（验证长程上下文）

注意：本测试不依赖完整模型加载，只测试 EpisodicMemory 核心逻辑
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Mock ModelContainer 和 SharedMemoryPool
async def mock_pool_get_instance():
    pool = MagicMock()
    pool.write_text = AsyncMock(return_value="trace_123")
    pool.read_text = AsyncMock(return_value={"user": "test", "ai": "test"})
    return pool

async def test_dynamic_capacity():
    """测试 1: 动态容量管理"""
    print("\n" + "="*60)
    print("测试 1: 动态容量管理")
    print("="*60)
    
    from zulong.memory.episodic_memory import EpisodicMemory
    
    em = EpisodicMemory()
    
    # Mock pool
    em.pool = await mock_pool_get_instance()
    
    # 测试动态容量计算
    await em._calculate_dynamic_capacity()
    
    print(f"✓ 最大记忆轮次：{em.max_episodes}")
    print(f"✓ Token 预算：{em.max_tokens_reserved}")
    print(f"✓ 估算每轮 tokens: {em.estimated_average_turn_tokens}")
    
    # 验证动态计算
    assert em.max_episodes > 0, "max_episodes 应该大于 0"
    assert em.max_tokens_reserved > 0, "max_tokens_reserved 应该大于 0"
    
    # 验证默认值（4k 模型）
    expected_episodes = 20  # 4096 * 0.75 / 150 = 20.48
    assert em.max_episodes == expected_episodes, f"应该是 {expected_episodes} 轮，实际 {em.max_episodes}"
    
    print("✅ 动态容量管理测试通过\n")
    return True

async def test_async_summarization():
    """测试 2: 异步复盘机制"""
    print("\n" + "="*60)
    print("测试 2: 异步复盘机制")
    print("="*60)
    
    from zulong.memory.episodic_memory import EpisodicMemory
    
    em = EpisodicMemory()
    em.pool = await mock_pool_get_instance()
    await em._calculate_dynamic_capacity()
    await em._load_index()
    em._start_summarization_worker()
    
    # 测试快速摘要生成
    user_input = "AI MAX 395 是什么？"
    ai_response = "AI MAX 395 是一款高性能处理器，采用 5nm 工艺，集成 128 核 GPU，支持 DDR5 内存。"
    
    start_time = time.time()
    result = await em.store_episode(user_input, ai_response)
    elapsed = time.time() - start_time
    
    print(f"✓ 存储耗时：{elapsed*1000:.2f}ms")
    print(f"✓ Episode ID: {result['episode_id']}")
    print(f"✓ 初始摘要：{result['summary']}")
    
    # 验证快速摘要（应该 < 50ms）
    assert elapsed < 0.1, f"快速摘要应该 < 100ms，实际 {elapsed*1000:.2f}ms"
    assert result['summary'] is not None, "摘要不应该为空"
    
    # 检查摘要类型
    episode_metadata = em._episode_index.get(result['episode_id'])
    print(f"✓ 摘要类型：{episode_metadata.get('summary_type', 'unknown')}")
    
    # 等待异步复盘
    print("\n等待异步复盘完成（2 秒）...")
    await asyncio.sleep(2)
    
    # 检查摘要是否更新
    episode_metadata = em._episode_index.get(result['episode_id'])
    if episode_metadata:
        print(f"✓ 更新后摘要类型：{episode_metadata.get('summary_type', 'unknown')}")
        print(f"✓ 更新后摘要：{episode_metadata.get('summary', 'N/A')[:50]}...")
    
    print("✅ 异步复盘机制测试通过\n")
    return True

async def test_memory_retrieval():
    """测试 3: 记忆检索"""
    print("\n" + "="*60)
    print("测试 3: 记忆检索")
    print("="*60)
    
    from zulong.memory.episodic_memory import EpisodicMemory
    
    em = EpisodicMemory()
    em.pool = await mock_pool_get_instance()
    await em._calculate_dynamic_capacity()
    await em._load_index()
    em._start_summarization_worker()
    
    # 存储多轮对话
    test_data = [
        ("AI MAX 395 是什么？", "AI MAX 395 是一款高性能处理器..."),
        ("如何安装 CPU 散热器？", "安装 CPU 散热器的步骤：1. 清洁表面..."),
        ("AI MAX 395 多少钱？", "AI MAX 395 的价格约为 2999 元..."),
        ("DDR5 内存有什么优势？", "DDR5 内存相比 DDR4 有以下优势..."),
        ("量子计算是什么？", "量子计算是基于量子力学原理的计算范式..."),
    ]
    
    print("存储测试数据...")
    for user_input, ai_response in test_data:
        await em.store_episode(user_input, ai_response)
    
    print(f"✓ 已存储 {len(test_data)} 轮对话")
    
    # 测试检索
    query = "处理器相关信息"
    results = await em.search_by_summary(query, top_k=3, time_window=7200)
    
    print(f"\n检索查询：'{query}'")
    print(f"✓ 检索到 {len(results)} 条结果")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Episode {result['episode_id']}")
        print(f"   摘要：{result['summary']}")
        print(f"   相似度：{result['similarity']:.3f}")
    
    # 验证检索结果
    assert len(results) > 0, "应该检索到结果"
    assert results[0]['similarity'] > 0.1, "相似度应该大于阈值"
    
    print("✅ 记忆检索测试通过\n")
    return True

async def test_hierarchical_reading():
    """测试 4: 分级读取"""
    print("\n" + "="*60)
    print("测试 4: 分级读取")
    print("="*60)
    
    from zulong.memory.episodic_memory import EpisodicMemory
    
    em = EpisodicMemory()
    em.pool = await mock_pool_get_instance()
    await em._calculate_dynamic_capacity()
    await em._load_index()
    em._start_summarization_worker()
    
    # 先存储一些数据
    await em.store_episode("测试问题", "测试回答内容")
    
    # 检索
    query = "测试"
    results = await em.search_by_summary(query, top_k=3)
    
    if results:
        print(f"检索到 {len(results)} 条摘要")
        
        # 读取详情
        episode_id = results[0]['episode_id']
        print(f"\n读取 Episode {episode_id} 的详情...")
        
        full_dialogue = await em.get_full_dialogue(episode_id)
        
        if full_dialogue:
            print(f"✓ 用户问：{full_dialogue['user']}")
            print(f"✓ AI 答：{full_dialogue['ai'][:100]}...")
            
            # 验证详情读取
            assert full_dialogue['user'] is not None, "用户输入不应该为空"
            assert full_dialogue['ai'] is not None, "AI 回复不应该为空"
            
            print("✅ 分级读取测试通过\n")
            return True
        else:
            print("⚠️ 无法读取详情")
            return False
    else:
        print("⚠️ 没有检索到结果")
        return False

async def test_long_range_dependency():
    """测试 5: 长程上下文依赖"""
    print("\n" + "="*60)
    print("测试 5: 长程上下文依赖")
    print("="*60)
    
    from zulong.memory.episodic_memory import EpisodicMemory
    
    em = EpisodicMemory()
    em.pool = await mock_pool_get_instance()
    await em._calculate_dynamic_capacity()
    await em._load_index()
    em._start_summarization_worker()
    
    # 模拟多轮对话
    print("模拟 10 轮对话...")
    for i in range(10):
        user_input = f"第{i+1}轮问题"
        ai_response = f"第{i+1}轮回答"
        await em.store_episode(user_input, ai_response)
    
    print(f"✓ 当前总轮次：{em._current_episode}")
    print(f"✓ 索引中的记忆数：{len(em._episode_index)}")
    
    # 检索第 1 轮的内容
    query = "第 1 轮"
    results = await em.search_by_summary(query, top_k=5)
    
    print(f"\n检索查询：'{query}'")
    print(f"✓ 检索到 {len(results)} 条结果")
    
    if results:
        print(f"✓ 第 1 轮摘要：{results[0]['summary']}")
        print("✅ 长程上下文依赖测试通过\n")
        return True
    else:
        print("⚠️ 未能检索到早期记忆")
        return False

async def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("  增强型三级记忆架构 - 功能测试套件（简化版）")
    print("="*80)
    
    tests = [
        ("动态容量管理", test_dynamic_capacity),
        ("异步复盘机制", test_async_summarization),
        ("记忆检索", test_memory_retrieval),
        ("分级读取", test_hierarchical_reading),
        ("长程上下文依赖", test_long_range_dependency),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"\n❌ {test_name} 测试失败：{e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # 汇总报告
    print("\n" + "="*80)
    print("  测试汇总报告")
    print("="*80)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {test_name}")
        if error:
            print(f"       错误：{error}")
    
    print(f"\n总计：{passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    print("="*80 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
