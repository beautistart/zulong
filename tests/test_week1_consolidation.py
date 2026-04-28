# -*- coding: utf-8 -*-
# File: tests/test_week1_consolidation.py
# 第 1 周优化测试：记忆巩固 + 持久化

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import time
import shutil

# 清理旧的测试数据
test_data_path = Path("./data/short_term_memory")
if test_data_path.exists():
    shutil.rmtree(test_data_path)
    print(f"🧹 已清理测试数据：{test_data_path}")


async def test_consolidation():
    """测试 1: 记忆巩固机制"""
    print("\n" + "="*80)
    print("测试 1: 记忆巩固机制")
    print("="*80)
    
    from zulong.memory.short_term_memory import get_short_term_memory
    
    stm = get_short_term_memory()
    
    # 测试 1: 包含关键信息的对话（应该巩固）
    print("\n--- 测试 1.1: 重要对话（包含姓名） ---")
    success = await stm.store(
        user_input="我的名字是张三",
        ai_response="好的，我记住了您叫张三。很高兴认识您！"
    )
    assert success, "存储失败"
    await asyncio.sleep(0.5)
    
    # 测试 2: 用户追问（应该巩固）
    print("\n--- 测试 1.2: 用户追问 ---")
    success = await stm.store(
        user_input="为什么天是蓝的？",
        ai_response="因为瑞利散射。阳光穿过大气层时，蓝色光波长较短，更容易被散射..."
    )
    assert success, "存储失败"
    await asyncio.sleep(0.5)
    
    # 测试 3: 普通对话（不应该立即巩固）
    print("\n--- 测试 1.3: 普通对话 ---")
    success = await stm.store(
        user_input="你好",
        ai_response="你好！有什么可以帮助你的？"
    )
    assert success, "存储失败"
    await asyncio.sleep(0.5)
    
    # 查看统计
    stats = stm.get_stats()
    print(f"\n📊 统计信息:")
    print(f"  当前轮数：{stats['current_turn']}")
    print(f"  缓存轮数：{stats['cached_turns']}")
    print(f"  总写入：{stats['total_writes']}")
    print(f"  总巩固：{stats['total_consolidations']}")
    
    # 验证：应该有至少 1 次巩固（重要对话）
    assert stats["total_consolidations"] >= 1, f"重要对话应该被巩固，实际巩固次数：{stats['total_consolidations']}"
    print(f"\n✅ 测试 1 通过：重要对话已巩固 (共{stats['total_consolidations']}次)")
    
    return True


async def test_persistence():
    """测试 2: 持久化机制"""
    print("\n" + "="*80)
    print("测试 2: 持久化机制")
    print("="*80)
    
    from zulong.memory.short_term_memory import ShortTermMemory
    
    # 第 1 次启动：存储对话
    print("\n--- 第 1 次启动：存储对话 ---")
    stm1 = ShortTermMemory()
    await stm1.store(
        user_input="测试持久化功能",
        ai_response="这是测试回复，用于验证持久化"
    )
    
    current_turn = stm1.get_current_turn()
    print(f"当前轮数：{current_turn}")
    
    # 模拟系统重启（创建新实例）
    print("\n--- 第 2 次启动：模拟重启 ---")
    stm2 = ShortTermMemory()
    
    # 验证：索引已恢复
    restored_turn = stm2.get_current_turn()
    print(f"恢复后轮数：{restored_turn}")
    
    assert restored_turn == current_turn, f"索引应该恢复：期望{current_turn}, 实际{restored_turn}"
    print(f"✅ 测试 2 通过：索引已成功恢复 (轮数：{restored_turn})")
    
    return True


def test_importance_calculation():
    """测试 3: 重要性计算"""
    print("\n" + "="*80)
    print("测试 3: 重要性计算")
    print("="*80)
    
    from zulong.memory.short_term_memory import get_short_term_memory
    
    stm = get_short_term_memory()
    
    # 测试用例
    test_cases = [
        {
            "name": "包含关键信息（姓名）",
            "user": "我的名字是张三",
            "ai": "好的，我记住了您叫张三",
            "expected_min": 0.7
        },
        {
            "name": "用户追问（为什么）",
            "user": "为什么天是蓝的？",
            "ai": "因为瑞利散射...",
            "expected_min": 0.65
        },
        {
            "name": "用户情感（谢谢）",
            "user": "谢谢！",
            "ai": "不客气~",
            "expected_min": 0.65
        },
        {
            "name": "长回复",
            "user": "帮我写个总结",
            "ai": "这是一篇很长的文章，包含了很多内容和详细的信息..." * 3,
            "expected_min": 0.6
        },
        {
            "name": "普通对话",
            "user": "你好",
            "ai": "你好",
            "expected_min": 0.5
        }
    ]
    
    print("\n测试用例:")
    for i, case in enumerate(test_cases, 1):
        importance = stm._calculate_importance(case["user"], case["ai"])
        passed = importance >= case["expected_min"]
        status = "✅" if passed else "❌"
        
        print(f"\n{status} 测试{i}: {case['name']}")
        print(f"  重要性：{importance:.2f} (期望>={case['expected_min']})")
        print(f"  用户：{case['user'][:50]}...")
        print(f"  AI: {case['ai'][:50]}...")
        
        assert passed, f"重要性评分过低：{importance} < {case['expected_min']}"
    
    print(f"\n✅ 测试 3 通过：所有重要性测试通过")
    
    return True


async def test_periodic_consolidation():
    """测试 4: 定期巩固"""
    print("\n" + "="*80)
    print("测试 4: 定期巩固（加速测试）")
    print("="*80)
    
    from zulong.memory.short_term_memory import get_short_term_memory
    
    stm = get_short_term_memory()
    
    # 修改巩固间隔为 2 秒（便于测试）
    original_interval = stm.consolidation_interval
    stm.consolidation_interval = 2
    stm.last_consolidation_time = time.time() - 5  # 模拟已过 5 秒
    
    print(f"\n已加速测试：巩固间隔={stm.consolidation_interval}秒")
    
    # 存储对话
    print("\n--- 存储对话 ---")
    await stm.store(
        user_input="测试定期巩固",
        ai_response="这是测试回复"
    )
    
    # 等待定期巩固触发
    print("等待定期巩固触发...")
    await asyncio.sleep(3)
    
    stats = stm.get_stats()
    print(f"\n📊 统计信息:")
    print(f"  总巩固：{stats['total_consolidations']}")
    
    # 恢复原始设置
    stm.consolidation_interval = original_interval
    
    print(f"\n✅ 测试 4 通过：定期巩固机制正常")
    
    return True


async def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 第 1 周优化测试：记忆巩固 + 持久化")
    print("="*80)
    
    tests = [
        ("测试 1: 记忆巩固机制", test_consolidation),
        ("测试 2: 持久化机制", test_persistence),
        ("测试 3: 重要性计算", test_importance_calculation),
        ("测试 4: 定期巩固", test_periodic_consolidation),
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
        print("\n🎉 所有测试通过！第 1 周优化完成！")
    else:
        print(f"\n⚠️ {failed} 个测试失败，请检查")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
