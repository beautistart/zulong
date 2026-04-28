# -*- coding: utf-8 -*-
# 测试情景记忆存储和检索

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("           情景记忆测试")
print("=" * 80)

async def test():
    from zulong.memory.episodic_memory import EpisodicMemory
    
    em = EpisodicMemory()
    
    # 1. 测试存储
    print("\n1. 测试存储")
    print("-" * 80)
    result = await em.store_episode("我叫小明", "你好，小明！")
    print(f"存储结果：{result}")
    
    result = await em.store_episode("我住在北京", "北京是个好地方")
    print(f"存储结果：{result}")
    
    # 2. 检查索引
    print("\n2. 检查索引")
    print("-" * 80)
    print(f"索引数量：{len(em._episode_index)}")
    for eid, meta in em._episode_index.items():
        print(f"Episode {eid}: {meta.get('summary')}")
    
    # 3. 测试检索
    print("\n3. 测试检索")
    print("-" * 80)
    episodes = await em.search_by_summary("名字", top_k=5)
    print(f"检索到 {len(episodes)} 条记忆")
    for ep in episodes:
        print(f"  - {ep['summary']}")
    
    # 4. 测试读取
    print("\n4. 测试读取")
    print("-" * 80)
    full_data = await em.get_full_dialogue(1)
    if full_data:
        print(f"用户：{full_data.get('user')}")
        print(f"AI:   {full_data.get('ai')}")
    else:
        print("读取失败")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())
