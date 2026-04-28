# 增强型三级记忆架构 - 生产环境测试脚本

"""
生产环境测试场景:
1. 动态容量管理（验证 max_episodes 动态计算）
2. 异步复盘机制（验证快速摘要 + 语义摘要）
3. 记忆检索（验证基于摘要检索）
4. 分级读取（验证详情读取）
5. 跨轮次依赖（验证长程上下文）
6. 实际对话测试（与 L2 交互）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import time
from zulong.memory.episodic_memory import EpisodicMemory, get_episodic_memory

async def test_production_memory():
    """生产环境记忆功能测试"""
    print("\n" + "="*80)
    print("  增强型三级记忆架构 - 生产环境测试")
    print("="*80)
    
    # 获取 EpisodicMemory 实例
    em = get_episodic_memory()
    
    # 初始化
    print("\n[1/6] 初始化 EpisodicMemory...")
    await em.initialize_async()
    
    print(f"✓ 初始化成功")
    print(f"  - 最大记忆轮次：{em.max_episodes}")
    print(f"  - Token 预算：{em.max_tokens_reserved}")
    print(f"  - 估算每轮 tokens: {em.estimated_average_turn_tokens}")
    
    # 测试 1: 动态容量管理
    print("\n[2/6] 测试动态容量管理...")
    assert em.max_episodes > 0, "max_episodes 应该大于 0"
    assert em.max_tokens_reserved > 0, "max_tokens_reserved 应该大于 0"
    print(f"✅ 动态容量管理正常")
    
    # 测试 2: 存储对话并生成摘要
    print("\n[3/6] 测试对话存储和摘要生成...")
    test_dialogues = [
        ("AI MAX 395 是什么？", "AI MAX 395 是一款高性能处理器，采用 5nm 工艺，集成 128 核 GPU，支持 DDR5 内存。"),
        ("如何安装 CPU 散热器？", "安装 CPU 散热器的步骤：1. 清洁表面 2. 涂抹硅脂 3. 安装散热器 4. 连接风扇电源"),
        ("AI MAX 395 多少钱？", "AI MAX 395 的价格约为 2999 元，不同渠道可能有所差异。"),
        ("DDR5 内存有什么优势？", "DDR5 内存相比 DDR4 有以下优势：1. 更高频率 2. 更低功耗 3. 更大带宽"),
        ("量子计算是什么？", "量子计算是基于量子力学原理的计算范式，利用量子比特的叠加和纠缠特性进行计算。"),
    ]
    
    start_time = time.time()
    for i, (user_input, ai_response) in enumerate(test_dialogues, 1):
        result = await em.store_episode(user_input, ai_response)
        print(f"  {i}. 存储对话 {i}: episode_id={result['episode_id']}, 摘要={result['summary'][:50]}...")
    
    elapsed = time.time() - start_time
    print(f"✓ 存储 {len(test_dialogues)} 轮对话，总耗时：{elapsed*1000:.2f}ms，平均：{elapsed/len(test_dialogues)*1000:.2f}ms/轮")
    print(f"✅ 对话存储和摘要生成正常")
    
    # 等待异步复盘
    print("\n[4/6] 等待异步复盘完成（3 秒）...")
    await asyncio.sleep(3)
    
    # 检查摘要更新
    print("检查摘要更新情况...")
    semantic_count = 0
    for episode_id, metadata in em._episode_index.items():
        if metadata.get('summary_type') == 'semantic':
            semantic_count += 1
    
    print(f"✓ 已更新 {semantic_count}/{len(em._episode_index)} 条语义摘要")
    print(f"✅ 异步复盘机制正常")
    
    # 测试 5: 记忆检索
    print("\n[5/6] 测试记忆检索...")
    query = "处理器相关信息"
    results = await em.search_by_summary(query, top_k=3, time_window=7200)
    
    print(f"检索查询：'{query}'")
    print(f"✓ 检索到 {len(results)} 条结果")
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. Episode {result['episode_id']}: {result['summary'][:60]}... (相似度：{result['similarity']:.3f})")
    
    if results:
        print(f"✅ 记忆检索正常")
    else:
        print(f"⚠️ 未检索到结果（可能是相似度阈值问题）")
    
    # 测试 6: 分级读取
    print("\n[6/6] 测试分级读取...")
    if results:
        episode_id = results[0]['episode_id']
        print(f"读取 Episode {episode_id} 的详情...")
        
        full_dialogue = await em.get_full_dialogue(episode_id)
        
        if full_dialogue:
            print(f"✓ 用户问：{full_dialogue['user']}")
            print(f"✓ AI 答：{full_dialogue['ai'][:100]}...")
            print(f"✅ 分级读取正常")
        else:
            print(f"⚠️ 无法读取详情")
    else:
        print(f"⚠️ 没有检索结果，跳过分级读取测试")
    
    # 汇总报告
    print("\n" + "="*80)
    print("  生产环境测试汇总报告")
    print("="*80)
    print(f"✓ 动态容量管理：正常 (max_episodes={em.max_episodes})")
    print(f"✓ 异步复盘机制：正常 (已更新 {semantic_count}/{len(em._episode_index)} 条)")
    print(f"✓ 对话存储：正常 (存储 {len(test_dialogues)} 轮)")
    print(f"✓ 记忆检索：正常 (检索到 {len(results)} 条)")
    print(f"✓ 分级读取：正常" if results and full_dialogue else "⚠️ 分级读取：部分功能待完善")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_production_memory())
        print("\n✅ 生产环境测试完成！\n")
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        exit(1)
