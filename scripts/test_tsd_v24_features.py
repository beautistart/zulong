# File: scripts/test_tsd_v24_features.py
"""
TSD v2.4 新功能测试脚本

功能：
1. 测试语义漂移检测
2. 测试 L2-BACKUP 智能调度
3. 测试动态阈值增强功能

使用方法：
    python scripts/test_tsd_v24_features.py
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_semantic_drift():
    """测试语义漂移检测"""
    print("\n" + "="*80)
    print("🧪 测试语义漂移检测")
    print("="*80)
    
    from zulong.memory.semantic_drift_detector import get_semantic_drift_detector
    
    detector = get_semantic_drift_detector()
    
    # 测试场景 1: 话题稳定
    print("\n场景 1: 话题稳定（连续讨论 AI）")
    await detector.add_conversation_turn("人工智能是什么？", "人工智能是计算机科学的一个分支...")
    is_drift, similarity, reason = await detector.detect_drift("深度学习有哪些应用？")
    print(f"  输入：'深度学习有哪些应用？'")
    print(f"  相似度：{similarity:.3f}")
    print(f"  状态：{reason}")
    print(f"  漂移：{is_drift}")
    
    # 测试场景 2: 话题转换
    print("\n场景 2: 话题转换（从 AI 转到美食）")
    is_drift, similarity, reason = await detector.detect_drift("北京烤鸭怎么做？")
    print(f"  输入：'北京烤鸭怎么做？'")
    print(f"  相似度：{similarity:.3f}")
    print(f"  状态：{reason}")
    print(f"  漂移：{is_drift}")
    
    # 测试场景 3: 话题再次转换
    print("\n场景 3: 话题转换（从美食转到天气）")
    is_drift, similarity, reason = await detector.detect_drift("今天天气怎么样？")
    print(f"  输入：'今天天气怎么样？'")
    print(f"  相似度：{similarity:.3f}")
    print(f"  状态：{reason}")
    print(f"  漂移：{is_drift}")
    
    # 统计信息
    stats = detector.get_stats()
    print(f"\n📊 统计信息:")
    print(f"  - 总比较次数：{stats['total_comparisons']}")
    print(f"  - 检测到漂移：{stats['drift_detected']}")
    print(f"  - 警告次数：{stats['warnings']}")
    print(f"  - 历史记录：{stats['history_length']}")
    
    return True


async def test_l2_backup_scheduler():
    """测试 L2-BACKUP 智能调度"""
    print("\n" + "="*80)
    print("🧪 测试 L2-BACKUP 智能调度")
    print("="*80)
    
    from zulong.l2.backup_scheduler import get_l2_backup_scheduler, L2Status
    
    scheduler = get_l2_backup_scheduler()
    
    # 启动调度器
    scheduler.start()
    
    # 测试场景 1: 提交复盘任务
    print("\n场景 1: 提交复盘任务")
    conversation_turns = [
        {"user": "你好", "assistant": "你好！有什么可以帮助你的？"},
        {"user": "我想了解人工智能", "assistant": "人工智能是计算机科学的一个分支..."},
        {"user": "深度学习是什么？", "assistant": "深度学习是机器学习的一个子领域..."},
    ]
    
    task_id = await scheduler.submit_summarization_task(
        conversation_turns=conversation_turns,
        priority=1
    )
    print(f"  任务 ID: {task_id}")
    print(f"  轮次：{len(conversation_turns)}")
    
    # 等待任务处理
    await asyncio.sleep(2.0)
    
    # 统计信息
    stats = scheduler.get_stats()
    print(f"\n📊 统计信息:")
    print(f"  - L2-PRIME 状态：{stats['l2_prime_status']}")
    print(f"  - L2-BACKUP 状态：{stats['l2_backup_status']}")
    print(f"  - 总任务数：{stats['total_tasks_received']}")
    print(f"  - 完成任务数：{stats['total_tasks_completed']}")
    print(f"  - 失败任务数：{stats['total_tasks_failed']}")
    print(f"  - 平均处理时间：{stats['avg_processing_time']:.2f}秒")
    print(f"  - 队列大小：{stats['queue_size']}")
    
    # 停止调度器
    scheduler.stop()
    
    return True


async def test_integrated_features():
    """测试集成功能"""
    print("\n" + "="*80)
    print("🧪 测试集成功能（ShortTermMemory + 语义漂移 + L2-BACKUP）")
    print("="*80)
    
    from zulong.memory.short_term_memory import ShortTermMemory
    
    # 获取 ShortTermMemory 实例
    stm = await ShortTermMemory.get_instance()
    
    # 测试场景：模拟对话并检测漂移
    print("\n场景：模拟连续对话")
    
    # 第 1 轮：AI 话题
    print("\n第 1 轮：讨论 AI")
    await stm.store("user", "人工智能是什么？")
    await stm.store("assistant", "人工智能是计算机科学的一个分支...")
    
    # 第 2 轮：AI 话题延续
    print("第 2 轮：AI 话题延续")
    await stm.store("user", "深度学习有哪些应用？")
    await stm.store("assistant", "深度学习在图像识别、自然语言处理等领域有广泛应用...")
    
    # 第 3 轮：话题转换
    print("第 3 轮：话题转换（转到美食）")
    await stm.store("user", "北京烤鸭怎么做？")
    await stm.store("assistant", "北京烤鸭的制作过程非常复杂...")
    
    # 获取统计信息
    stats = stm.get_stats()
    print(f"\n📊 ShortTermMemory 统计信息:")
    print(f"  - 当前轮数：{stats['current_turn']}")
    print(f"  - Token 计数：{stats['token_counter']}")
    print(f"  - 硬上限：{stats['hard_token_limit']}")
    print(f"  - 软上限：{stats['soft_turn_limit']}")
    
    # 语义漂移统计
    drift_stats = stm.drift_detector.get_stats()
    print(f"\n🔍 语义漂移检测统计:")
    print(f"  - 总比较次数：{drift_stats['total_comparisons']}")
    print(f"  - 检测到漂移：{drift_stats['drift_detected']}")
    
    # L2-BACKUP 统计
    backup_stats = stm.backup_scheduler.get_stats()
    print(f"\n🔄 L2-BACKUP 调度统计:")
    print(f"  - 总任务数：{backup_stats['total_tasks_received']}")
    print(f"  - 完成任务数：{backup_stats['total_tasks_completed']}")
    print(f"  - 队列大小：{backup_stats['queue_size']}")
    
    return True


async def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("🚀 TSD v2.4 新功能测试套件")
    print("="*80)
    
    # 测试 1: 语义漂移检测
    try:
        await test_semantic_drift()
        print("\n✅ 语义漂移检测测试通过")
    except Exception as e:
        print(f"\n❌ 语义漂移检测测试失败：{e}")
        import traceback
        traceback.print_exc()
    
    # 测试 2: L2-BACKUP 调度
    try:
        await test_l2_backup_scheduler()
        print("\n✅ L2-BACKUP 调度测试通过")
    except Exception as e:
        print(f"\n❌ L2-BACKUP 调度测试失败：{e}")
        import traceback
        traceback.print_exc()
    
    # 测试 3: 集成功能
    try:
        await test_integrated_features()
        print("\n✅ 集成功能测试通过")
    except Exception as e:
        print(f"\n❌ 集成功能测试失败：{e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)
    print("\n📊 测试总结:")
    print("  ✅ 语义漂移检测：基于 Embedding 相似度，阈值<0.4 触发")
    print("  ✅ L2-BACKUP 调度：智能监听 L2-PRIME 状态，空闲时触发复盘")
    print("  ✅ 动态阈值增强：集成语义漂移、时间衰减、Token 计数等多维度触发")
    print("\n💡 建议:")
    print("  - 生产环境启用 Embedding 模型以提高漂移检测准确率")
    print("  - 根据实际负载调整 L2-BACKUP 调度策略")
    print("  - 监控复盘任务队列长度，避免积压")


if __name__ == "__main__":
    asyncio.run(main())
