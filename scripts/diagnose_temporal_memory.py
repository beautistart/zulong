# -*- coding: utf-8 -*-
# 临时记忆性能诊断脚本
# 检查：临时记忆失效、回复慢的问题

import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("           临时记忆性能诊断")
print("=" * 80)
print()

async def diagnose_performance():
    """诊断临时记忆性能问题"""
    
    from zulong.memory.episodic_memory import EpisodicMemory
    from zulong.memory.short_term_memory import ShortTermMemory
    from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
    
    # ========== 1. 检查共享池性能 ==========
    print("1. 检查共享池性能")
    print("-" * 80)
    
    pool = await SharedMemoryPool.get_instance()
    print(f"✅ 共享池实例：{id(pool)}")
    print(f"📊 Memory Zone 数量：{len(pool._memory_zone)}")
    print(f"📊 Raw Zone 数量：{len(pool._raw_zone)}")
    
    # 检查是否有过期数据
    expired_count = 0
    current_time = time.time()
    for trace_id, envelope in list(pool._memory_zone.items())[:20]:  # 检查前 20 条
        age = current_time - envelope.metadata.get('timestamp', current_time)
        if age > 3600:  # 超过 1 小时
            expired_count += 1
    
    if expired_count > 0:
        print(f"⚠️  发现 {expired_count} 条过期数据（可能影响性能）")
    else:
        print(f"✅ 未发现过期数据")
    
    # ========== 2. 检查情景记忆 ==========
    print("\n2. 检查情景记忆 (Episodic Memory)")
    print("-" * 80)
    
    em = await EpisodicMemory.get_instance()
    print(f"✅ 情景记忆实例：{id(em)}")
    print(f"📊 索引数量：{len(em._episode_index)}")
    print(f"📊 当前轮次：{em._current_episode}")
    print(f"📊 最大容量：{em.max_episodes}")
    print(f"📊 Token 预算：{em.max_tokens_reserved}")
    
    # 检查复盘队列
    print(f"\n🔄 复盘队列状态:")
    print(f"   待处理任务：{em._pending_summarization_queue.qsize()}")
    print(f"   已完成：{em._stats['summarizations_completed']}")
    
    # 检查统计信息
    print(f"\n📈 统计信息:")
    print(f"   总写入：{em._stats['total_writes']}")
    print(f"   总读取：{em._stats['total_reads']}")
    print(f"   总检索：{em._stats['total_searches']}")
    print(f"   平均检索延迟：{em._stats['avg_search_latency']:.3f}ms")
    
    # ========== 3. 检查短期记忆 ==========
    print("\n3. 检查短期记忆 (Short-Term Memory)")
    print("-" * 80)
    
    # 注意：ShortTermMemory 使用同步初始化
    stm = ShortTermMemory()
    print(f"✅ 短期记忆实例：{id(stm)}")
    print(f"📊 当前轮次：{stm._current_turn}")
    print(f"📊 最大轮数：{stm.max_rounds}")
    print(f"📊 TTL: {stm.ttl_seconds}s")
    
    # 检查索引
    print(f"\n📋 索引详情 (前 5 条):")
    for turn_id, trace_id in list(stm._turn_index.items())[:5]:
        print(f"   Turn {turn_id}: {trace_id}")
    
    # 检查向量缓存
    if hasattr(stm, 'vector_cache'):
        vc = stm.vector_cache
        print(f"\n🧠 向量缓存:")
        print(f"   缓存数量：{len(vc.cache)}")
        print(f"   最大缓存：{vc.max_cache_size}")
        print(f"   向量生成：{vc._stats['total_vectors_computed']}")
        print(f"   检索次数：{vc._stats['total_searches']}")
        print(f"   缓存命中：{vc._stats['cache_hits']}")
    
    # ========== 4. 性能测试 ==========
    print("\n4. 性能压力测试")
    print("-" * 80)
    
    # 测试存储性能
    print(f"📝 测试存储性能...")
    start = time.time()
    for i in range(5):
        await em.store_episode(
            f"测试用户输入 {i}",
            f"测试 AI 回复 {i}"
        )
    store_time = (time.time() - start) / 5 * 1000
    print(f"✅ 平均存储延迟：{store_time:.2f}ms")
    
    # 测试检索性能
    print(f"🔍 测试检索性能...")
    start = time.time()
    for i in range(5):
        await em.search_by_summary("测试", top_k=3)
    search_time = (time.time() - start) / 5 * 1000
    print(f"✅ 平均检索延迟：{search_time:.2f}ms")
    
    # ========== 5. 检查 L2 推理引擎 ==========
    print("\n5. 检查 L2 推理引擎")
    print("-" * 80)
    
    from zulong.l2.inference_engine import InferenceEngine
    ie = InferenceEngine()
    print(f"✅ L2 推理引擎实例：{id(ie)}")
    print(f"📊 对话历史：{len(ie.conversation_history)}")
    print(f"📊 最大历史：{ie.max_history}")
    
    # 检查 vLLM 状态
    if hasattr(ie, 'vllm_client') and ie.vllm_client:
        print(f"✅ vLLM 客户端：已初始化")
    else:
        print(f"⚠️  vLLM 客户端：未初始化")
    
    # ========== 6. 诊断结论 ==========
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)
    
    issues = []
    
    # 检查复盘队列
    if em._pending_summarization_queue.qsize() > 10:
        issues.append(f"⚠️  复盘队列积压：{em._pending_summarization_queue.qsize()} 个任务")
    else:
        print(f"✅ 复盘队列：正常 ({em._pending_summarization_queue.qsize()} 个任务)")
    
    # 检查检索延迟
    if search_time > 100:
        issues.append(f"⚠️  检索延迟过高：{search_time:.2f}ms (>100ms)")
    else:
        print(f"✅ 检索延迟：正常 ({search_time:.2f}ms)")
    
    # 检查存储延迟
    if store_time > 200:
        issues.append(f"⚠️  存储延迟过高：{store_time:.2f}ms (>200ms)")
    else:
        print(f"✅ 存储延迟：正常 ({store_time:.2f}ms)")
    
    # 检查过期数据
    if expired_count > 10:
        issues.append(f"⚠️  过期数据过多：{expired_count} 条")
    else:
        print(f"✅ 过期数据：正常 ({expired_count} 条)")
    
    # 检查索引数量
    if len(em._episode_index) == 0:
        issues.append("⚠️  情景记忆索引为空")
    else:
        print(f"✅ 情景记忆索引：正常 ({len(em._episode_index)} 条)")
    
    # 输出结论
    print()
    if issues:
        print("⚠️  发现问题:")
        for issue in issues:
            print(f"   {issue}")
        print()
        print("💡 建议:")
        if "复盘队列积压" in str(issues):
            print("   - 考虑增加复盘工作者数量")
            print("   - 检查 L2-BACKUP 实例是否正常运行")
        if "检索延迟过高" in str(issues):
            print("   - 检查向量缓存是否正常工作")
            print("   - 考虑优化检索算法")
        if "存储延迟过高" in str(issues):
            print("   - 检查共享池性能")
            print("   - 考虑异步化存储流程")
    else:
        print("✅ 所有检查通过！临时记忆系统性能正常")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(diagnose_performance())
