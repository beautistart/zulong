# -*- coding: utf-8 -*-
# 记忆系统完整诊断脚本
# 检查：索引生成 → 注入模型 → 读取记忆

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("           记忆系统完整诊断")
print("=" * 80)
print()

async def diagnose():
    from zulong.memory.episodic_memory import EpisodicMemory
    from zulong.memory.short_term_memory import ShortTermMemory
    
    # ========== 1. 检查情景记忆索引 ==========
    print("1. 检查情景记忆索引生成")
    print("-" * 80)
    
    em = EpisodicMemory()
    print(f"✅ 情景记忆实例：已创建")
    print(f"📊 索引数量：{len(em._episode_index)}")
    
    if len(em._episode_index) > 0:
        print(f"\n📋 索引详情:")
        for eid, meta in list(em._episode_index.items())[:5]:  # 显示前 5 条
            summary = meta.get('summary', 'N/A')
            trace_id = meta.get('trace_id', 'N/A')
            timestamp = meta.get('timestamp', 'N/A')
            print(f"   Episode {eid}:")
            print(f"      摘要：{summary}")
            print(f"      Trace ID: {trace_id}")
            print(f"      时间戳：{timestamp}")
            print()
    else:
        print(f"⚠️  索引为空，可能原因:")
        print(f"   - 尚未进行对话")
        print(f"   - 存储失败")
        print(f"   - 初始化未完成")
    
    # ========== 2. 检查共享池数据 ==========
    print("2. 检查共享池数据存储")
    print("-" * 80)
    
    if em.pool:
        print(f"✅ 共享池：已初始化")
        
        # 尝试读取第一个 episode
        if len(em._episode_index) > 0:
            first_episode_id = list(em._episode_index.keys())[0]
            metadata = em._episode_index[first_episode_id]
            trace_id = metadata.get('trace_id')
            
            if trace_id:
                print(f"\n📖 尝试读取 Episode {first_episode_id}:")
                print(f"   Trace ID: {trace_id}")
                
                envelope = await em.pool.read_memory(trace_id)
                if envelope:
                    payload = envelope.payload
                    print(f"   ✅ 读取成功")
                    print(f"   用户：{payload.get('user', 'N/A')}")
                    print(f"   AI:   {payload.get('ai', 'N/A')}")
                else:
                    print(f"   ❌ 读取失败：envelope 为 None")
            else:
                print(f"⚠️  Episode {first_episode_id} 没有 trace_id")
    else:
        print(f"❌ 共享池：未初始化")
    
    # ========== 3. 测试摘要检索 ==========
    print("\n3. 测试摘要检索功能")
    print("-" * 80)
    
    if len(em._episode_index) > 0:
        # 测试不同关键词
        test_queries = ["名字", "北京", "参观", "吃的"]
        
        for query in test_queries:
            episodes = await em.search_by_summary(query, top_k=3)
            print(f"🔍 检索关键词：'{query}'")
            print(f"   结果数量：{len(episodes)}")
            
            if episodes:
                for ep in episodes:
                    print(f"   - Episode {ep['episode_id']}: {ep['summary']}")
            print()
    else:
        print(f"⚠️  没有索引，无法测试检索")
    
    # ========== 4. 检查短期记忆 ==========
    print("4. 检查短期记忆存储")
    print("-" * 80)
    
    stm = ShortTermMemory()
    print(f"✅ 短期记忆实例：已创建")
    
    # 检查共享池
    if stm.pool:
        print(f"✅ 短期记忆共享池：已初始化")
    else:
        print(f"⚠️  短期记忆共享池：未初始化")
    
    # ========== 5. 完整流程测试 ==========
    print("\n5. 完整流程测试")
    print("-" * 80)
    
    print(f"📝 存储测试对话...")
    result = await em.store_episode("我叫小明", "你好，小明！很高兴认识你。")
    print(f"✅ 存储结果：{result}")
    
    episode_id = result.get('episode_id')
    if episode_id:
        print(f"\n📖 读取完整对话...")
        full_data = await em.get_full_dialogue(episode_id)
        
        if full_data:
            print(f"✅ 读取成功")
            print(f"   用户：{full_data.get('user')}")
            print(f"   AI:   {full_data.get('ai')}")
        else:
            print(f"❌ 读取失败")
    
    print(f"\n🔍 检索测试...")
    episodes = await em.search_by_summary("名字", top_k=5)
    print(f"检索到 {len(episodes)} 条相关记忆")
    
    if episodes:
        for ep in episodes:
            print(f"   - Episode {ep['episode_id']}: {ep['summary']}")
    
    # ========== 6. 诊断结论 ==========
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)
    
    issues = []
    
    # 检查索引
    if len(em._episode_index) == 0:
        issues.append("❌ 情景记忆索引为空")
    else:
        print(f"✅ 情景记忆索引：正常 ({len(em._episode_index)} 条)")
    
    # 检查共享池
    if not em.pool:
        issues.append("❌ 情景记忆共享池未初始化")
    else:
        print(f"✅ 情景记忆共享池：正常")
    
    # 检查读取
    if episode_id:
        full_data = await em.get_full_dialogue(episode_id)
        if not full_data:
            issues.append("❌ 无法读取完整对话")
        else:
            print(f"✅ 完整对话读取：正常")
    
    # 检查检索
    episodes = await em.search_by_summary("测试", top_k=1)
    if len(episodes) == 0:
        issues.append("⚠️  检索功能可能有问题")
    else:
        print(f"✅ 摘要检索：正常")
    
    # 输出结论
    print()
    if issues:
        print("⚠️  发现问题:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ 所有检查通过！记忆系统工作正常")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(diagnose())
