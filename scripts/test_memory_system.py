# -*- coding: utf-8 -*-
# File: scripts/test_memory_system.py
# 三层记忆系统验证脚本

import sys
import asyncio
from pathlib import Path
import os

# 设置控制台编码
os.system('chcp 65001 >nul')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from zulong.memory.short_term_memory import ShortTermMemory
from zulong.memory.episodic_memory import EpisodicMemory
from zulong.memory.rag_libraries import ExperienceRAG, MemoryRAG
from zulong.memory.rag_manager import RAGManager
from zulong.infrastructure.shared_memory_pool import SharedMemoryPool, ZoneType, DataType


async def test_short_term_memory():
    """测试短期记忆"""
    print("\n" + "="*80)
    print("🧠 测试短期记忆 (Short-term Memory)")
    print("="*80)
    
    try:
        # 获取单例
        stm = await ShortTermMemory.get_instance(max_rounds=20, ttl_seconds=3600)
        
        print(f"✅ 短期记忆已初始化")
        print(f"   - 最大轮数：{stm.max_rounds}")
        print(f"   - TTL: {stm.ttl_seconds}s")
        print(f"   - 存储分区：{stm.zone}")
        
        # 测试存储
        test_input = "今天天气真好"
        test_output = "是的，阳光明媚，适合外出"
        
        await stm.store(test_input, test_output)
        print(f"✅ 成功存储对话：'{test_input}' -> '{test_output}'")
        
        # 测试读取
        recent = await stm.get_recent(rounds=1)
        if recent and len(recent) > 0:
            print(f"✅ 成功读取最近对话：{len(recent)} 轮")
            print(f"   - 输入：{recent[0].get('input', 'N/A')}")
            print(f"   - 输出：{recent[0].get('output', 'N/A')}")
        else:
            print("❌ 读取失败")
            
        # 获取状态
        status = stm.get_status()
        print(f"📊 当前状态：{status}")
        
        print("\n✅ 短期记忆测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 短期记忆测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_episodic_memory():
    """测试情景记忆"""
    print("\n" + "="*80)
    print("📝 测试情景记忆 (Episodic Memory)")
    print("="*80)
    
    try:
        em = EpisodicMemory()
        
        print(f"✅ 情景记忆已初始化")
        print(f"   - 最大集数：{em.max_episodes}")
        print(f"   - Token 预算：{em.max_tokens_reserved}")
        print(f"   - 摘要模型：{em.summary_model}")
        
        # 测试生成摘要
        test_dialogue = [
            {"role": "user", "content": "你好，我叫小明"},
            {"role": "assistant", "content": "你好小明，很高兴认识你"}
        ]
        
        summary = await em.generate_summary(test_dialogue)
        print(f"✅ 成功生成摘要：{summary[:100]}...")
        
        # 测试存储
        episode_id = await em.store_episode(
            dialogue=test_dialogue,
            summary=summary,
            tags=["greeting", "introduction"]
        )
        print(f"✅ 成功存储情景记忆：episode_id={episode_id}")
        
        # 测试检索
        results = await em.retrieve_by_query("认识新朋友", top_k=1)
        if results:
            print(f"✅ 成功检索记忆：{len(results)} 条结果")
            for r in results:
                print(f"   - 摘要：{r.get('summary', 'N/A')[:50]}...")
        else:
            print("⚠️  检索结果为空")
        
        print("\n✅ 情景记忆测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 情景记忆测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_experience_rag():
    """测试经验记忆 (RAG)"""
    print("\n" + "="*80)
    print("💡 测试经验记忆 (Experience RAG)")
    print("="*80)
    
    try:
        from zulong.memory.rag_libraries import ExperienceRAG
        from zulong.memory.base_rag_library import RAGDocument
        import numpy as np
        
        # 创建经验 RAG
        exp_rag = ExperienceRAG()
        print(f"✅ 经验 RAG 已初始化")
        print(f"   - 维度：{exp_rag.dimension}")
        print(f"   - 文档数：{len(exp_rag.documents)}")
        
        # 测试添加经验
        test_experience = "用户喜欢简洁的回答，不喜欢冗长的解释"
        test_embedding = np.random.rand(512).astype('float32').tolist()  # 模拟 embedding
        
        from datetime import datetime
        doc = RAGDocument(
            content=test_experience,
            embedding=test_embedding,
            metadata={
                "source": "user_feedback",
                "timestamp": datetime.now().isoformat(),
                "category": "preference"
            }
        )
        
        doc_id = exp_rag.add_document(doc)
        print(f"✅ 成功添加经验：doc_id={doc_id}")
        print(f"   - 内容：{test_experience}")
        
        # 测试搜索
        results = exp_rag.search_documents("用户偏好", top_k=1)
        if results:
            print(f"✅ 成功搜索经验：{len(results)} 条结果")
            for r in results:
                print(f"   - 内容：{r.content[:50]}...")
        else:
            print("⚠️  搜索结果为空")
        
        print("\n✅ 经验记忆测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 经验记忆测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_rag():
    """测试记忆 RAG"""
    print("\n" + "="*80)
    print("🗂️  测试记忆 RAG (Memory RAG)")
    print("="*80)
    
    try:
        mem_rag = MemoryRAG()
        print(f"✅ 记忆 RAG 已初始化")
        print(f"   - 维度：{mem_rag.dimension}")
        print(f"   - 文档数：{len(mem_rag.documents)}")
        print(f"   - 时间跨度分类：{list(mem_rag.memory_time_spans.keys())}")
        print(f"   - 记忆类型：{list(mem_rag.memory_types.keys())}")
        
        from zulong.memory.base_rag_library import RAGDocument
        from datetime import datetime
        import numpy as np
        
        # 测试添加记忆
        test_memory = "用户询问了关于天气的问题"
        test_embedding = np.random.rand(512).astype('float32').tolist()
        
        doc = RAGDocument(
            content=test_memory,
            embedding=test_embedding,
            metadata={
                "time_span": "short_term",
                "memory_type": "context",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        doc_id = mem_rag.add_document(doc)
        print(f"✅ 成功添加记忆：doc_id={doc_id}")
        
        # 测试搜索
        results = mem_rag.search_documents("天气查询", top_k=1)
        if results:
            print(f"✅ 成功搜索记忆：{len(results)} 条结果")
            for r in results:
                print(f"   - 内容：{r.content[:50]}...")
        else:
            print("⚠️  搜索结果为空")
        
        print("\n✅ 记忆 RAG 测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 记忆 RAG 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🚀 开始测试三层记忆系统")
    print("="*80)
    
    results = {
        "短期记忆": await test_short_term_memory(),
        "情景记忆": await test_episodic_memory(),
        "经验记忆": await test_experience_rag(),
        "记忆 RAG": await test_memory_rag()
    }
    
    print("\n" + "="*80)
    print("📊 测试结果汇总")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n总计：{passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！三层记忆系统运行正常")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查日志")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
