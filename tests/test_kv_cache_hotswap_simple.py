# File: tests/test_kv_cache_hotswap_simple.py
"""
KV Cache 热切换功能测试 (无 pytest 依赖)
TSD v1.9: 验证热交换机制的正确性
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_hardware_aware_kv_pool():
    """测试硬件抽象层"""
    print("\n" + "="*60)
    print("测试 1: HardwareAwareKVPool 初始化")
    print("="*60)
    
    from zulong.adapters.memory_backend import HardwareAwareKVPool, KVPoolConfig
    
    config = KVPoolConfig(
        max_blocks=128,
        block_size=16,
        hidden_size=512,
        num_layers=8,
    )
    
    pool = HardwareAwareKVPool(config)
    assert pool.max_blocks == 128
    assert pool.block_size == 16
    assert not pool._initialized
    print("✓ KV Pool 配置正确")
    
    success = pool.initialize()
    assert success
    assert pool._initialized
    print("✓ KV Pool 初始化成功")
    
    stats = pool.get_statistics()
    print(f"  统计信息: {stats['block_manager']}")
    
    pool.shutdown()
    print("✓ KV Pool 关闭成功")


def test_block_allocation():
    """测试块分配"""
    print("\n" + "="*60)
    print("测试 2: Block 分配与释放")
    print("="*60)
    
    from zulong.adapters.memory_backend import HardwareAwareKVPool, KVPoolConfig
    
    config = KVPoolConfig(max_blocks=64)
    pool = HardwareAwareKVPool(config)
    pool.initialize()
    
    blocks = pool.allocate_for_task("task_1", 4)
    assert len(blocks) == 4
    print(f"✓ 分配 4 个块: {blocks}")
    
    stats = pool.get_statistics()
    assert stats["block_manager"]["allocated_blocks"] == 1
    assert stats["block_manager"]["free_blocks"] == 60
    print(f"✓ 统计验证通过: 已分配={stats['block_manager']['allocated_blocks']}, 空闲={stats['block_manager']['free_blocks']}")
    
    freed = pool.free_task("task_1")
    assert len(freed) == 4
    print(f"✓ 释放 4 个块: {freed}")
    
    stats = pool.get_statistics()
    assert stats["block_manager"]["allocated_blocks"] == 0
    print("✓ 所有块已释放")
    
    pool.shutdown()


def test_oom_handling():
    """测试 OOM 处理"""
    print("\n" + "="*60)
    print("测试 3: OOM 处理")
    print("="*60)
    
    from zulong.adapters.memory_backend import HardwareAwareKVPool, KVPoolConfig
    
    config = KVPoolConfig(max_blocks=8)
    pool = HardwareAwareKVPool(config)
    pool.initialize()
    
    pool.allocate_for_task("task_1", 4)
    pool.allocate_for_task("task_2", 4)
    print("✓ 分配了 8 个块 (满)")
    
    try:
        pool.allocate_for_task("task_3", 4)
        assert False, "应该抛出 OOM 异常"
    except RuntimeError as e:
        assert "OOM" in str(e)
        print(f"✓ 正确抛出 OOM 异常: {e}")
    
    pool.shutdown()


def test_scheduler_initialization():
    """测试调度器初始化"""
    print("\n" + "="*60)
    print("测试 4: L1B_HotSwapScheduler 初始化")
    print("="*60)
    
    from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
    from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig
    
    config = KVPoolConfig(max_blocks=64)
    pool = MockKVPool(config)
    pool.initialize()
    
    scheduler_config = HotSwapConfig(switch_threshold=2)
    scheduler = L1B_HotSwapScheduler(pool, scheduler_config)
    
    assert not scheduler._initialized
    print("✓ 调度器创建成功 (未初始化)")
    
    success = scheduler.initialize()
    assert success
    assert scheduler._initialized
    assert scheduler.L2_PRIME is not None
    assert scheduler.L2_BACKUP is not None
    print("✓ 调度器初始化成功")
    print(f"  L2_PRIME: {scheduler.L2_PRIME.engine_id}")
    print(f"  L2_BACKUP: {scheduler.L2_BACKUP.engine_id}")
    
    scheduler.shutdown()
    pool.shutdown()
    print("✓ 调度器关闭成功")


def test_single_task_processing():
    """测试单任务处理"""
    print("\n" + "="*60)
    print("测试 5: 单任务处理")
    print("="*60)
    
    from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
    from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig
    
    config = KVPoolConfig(max_blocks=64)
    pool = MockKVPool(config)
    pool.initialize()
    
    scheduler = L1B_HotSwapScheduler(pool, HotSwapConfig(switch_threshold=2))
    scheduler.initialize()
    
    response = scheduler.on_user_input("你好")
    assert response is not None
    print(f"✓ 收到响应: {response[:50]}...")
    
    assert len(scheduler.active_tasks) == 1
    assert scheduler.task_counter == 1
    print(f"✓ 活跃任务数: {len(scheduler.active_tasks)}")
    
    scheduler.shutdown()
    pool.shutdown()


def test_hot_swap_trigger():
    """测试热交换触发"""
    print("\n" + "="*60)
    print("测试 6: 热交换触发 (核心功能)")
    print("="*60)
    
    from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
    from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig, TaskStatus
    
    config = KVPoolConfig(max_blocks=64)
    pool = MockKVPool(config)
    pool.initialize()
    
    scheduler = L1B_HotSwapScheduler(pool, HotSwapConfig(switch_threshold=2, block_per_task=4))
    scheduler.initialize()
    
    print("  发送任务 1...")
    scheduler.on_user_input("任务1")
    assert len(scheduler.active_tasks) == 1
    assert scheduler._hot_swap_count == 0
    print(f"  活跃任务: {len(scheduler.active_tasks)}, 热交换次数: {scheduler._hot_swap_count}")
    
    print("  发送任务 2...")
    scheduler.on_user_input("任务2")
    assert len(scheduler.active_tasks) == 2
    assert scheduler._hot_swap_count == 0
    print(f"  活跃任务: {len(scheduler.active_tasks)}, 热交换次数: {scheduler._hot_swap_count}")
    
    print("  发送任务 3 (触发阈值)...")
    scheduler.on_user_input("任务3")
    
    assert scheduler._hot_swap_count == 1, f"热交换次数应为 1，实际为 {scheduler._hot_swap_count}"
    assert len(scheduler.active_tasks) == 1, f"活跃任务应为 1，实际为 {len(scheduler.active_tasks)}"
    assert len(scheduler.freeze_stack) == 2, f"冻结栈应为 2，实际为 {len(scheduler.freeze_stack)}"
    print(f"✓ 热交换触发成功!")
    print(f"  活跃任务: {len(scheduler.active_tasks)}")
    print(f"  冻结任务: {len(scheduler.freeze_stack)}")
    print(f"  热交换次数: {scheduler._hot_swap_count}")
    
    for ctx in scheduler.freeze_stack:
        assert ctx.status == TaskStatus.MIGRATED
    print("✓ 冻结任务状态正确 (MIGRATED)")
    
    scheduler.shutdown()
    pool.shutdown()


def test_task_resume():
    """测试任务恢复"""
    print("\n" + "="*60)
    print("测试 7: 任务恢复")
    print("="*60)
    
    from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
    from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig
    
    config = KVPoolConfig(max_blocks=64)
    pool = MockKVPool(config)
    pool.initialize()
    
    scheduler = L1B_HotSwapScheduler(pool, HotSwapConfig(switch_threshold=2, block_per_task=4))
    scheduler.initialize()
    
    scheduler.on_user_input("任务1")
    scheduler.on_user_input("任务2")
    scheduler.on_user_input("任务3")
    
    assert len(scheduler.freeze_stack) == 2
    print(f"✓ 冻结栈有 {len(scheduler.freeze_stack)} 个任务")
    
    task_id = scheduler.freeze_stack[0].task_id
    print(f"  尝试恢复任务: {task_id}")
    
    response = scheduler.resume_task(task_id)
    assert response is not None
    print(f"✓ 任务恢复成功: {response[:50]}...")
    
    assert len(scheduler.freeze_stack) == 1
    assert len(scheduler.active_tasks) == 2
    print(f"  冻结栈: {len(scheduler.freeze_stack)}, 活跃任务: {len(scheduler.active_tasks)}")
    
    scheduler.shutdown()
    pool.shutdown()


def test_model_loader():
    """测试模型加载器"""
    print("\n" + "="*60)
    print("测试 8: 模型加载器")
    print("="*60)
    
    from zulong.adapters.model_loader import (
        detect_hardware, 
        auto_select_model, 
        get_model_info,
        init_l2_engines,
    )
    
    hw_type = detect_hardware()
    print(f"✓ 检测到硬件类型: {hw_type.value}")
    
    config = auto_select_model()
    print(f"✓ 自动选择模型: {config.model_name}")
    print(f"  Model ID: {config.model_id}")
    print(f"  GPU 利用率: {config.gpu_memory_utilization}")
    
    info = get_model_info()
    print(f"✓ 模型信息: {info['model_name']}")
    
    prime, backup = init_l2_engines(use_vllm=False)
    assert prime is not None
    assert backup is not None
    print(f"✓ Mock 引擎初始化成功")
    print(f"  L2_PRIME: {prime.engine_id}")
    print(f"  L2_BACKUP: {backup.engine_id}")


def test_full_integration():
    """完整集成测试"""
    print("\n" + "="*60)
    print("测试 9: 完整热交换流程 (集成测试)")
    print("="*60)
    
    from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
    from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig
    
    config = KVPoolConfig(max_blocks=128)
    pool = MockKVPool(config)
    assert pool.initialize()
    print("✓ KV Pool 初始化成功")
    
    scheduler = L1B_HotSwapScheduler(
        pool, 
        HotSwapConfig(
            switch_threshold=2,
            block_per_task=8,
            max_freeze_stack_size=5,
        )
    )
    assert scheduler.initialize()
    print("✓ 调度器初始化成功")
    
    print("\n  步骤 1: 发送第一个任务")
    scheduler.on_user_input("第一个任务：今天天气怎么样？")
    print(f"    活跃任务: {len(scheduler.active_tasks)}")
    
    print("\n  步骤 2: 发送第二个任务")
    scheduler.on_user_input("第二个任务：讲个笑话")
    print(f"    活跃任务: {len(scheduler.active_tasks)}")
    
    print("\n  步骤 3: 发送第三个任务 (触发阈值)")
    response = scheduler.on_user_input("第三个任务：救命！")
    assert response is not None
    print(f"    收到响应: {response[:50]}...")
    print(f"    热交换次数: {scheduler._hot_swap_count}")
    print(f"    冻结任务: {len(scheduler.freeze_stack)}")
    print(f"    活跃任务: {len(scheduler.active_tasks)}")
    
    assert scheduler._hot_swap_count == 1
    assert len(scheduler.freeze_stack) == 2
    assert len(scheduler.active_tasks) == 1
    print("✓ 热交换验证通过")
    
    print("\n  步骤 4: 恢复冻结任务")
    frozen_task_id = scheduler.freeze_stack[0].task_id
    resume_response = scheduler.resume_task(frozen_task_id)
    assert resume_response is not None
    print(f"    恢复响应: {resume_response[:50]}...")
    print(f"    冻结任务: {len(scheduler.freeze_stack)}")
    print(f"    活跃任务: {len(scheduler.active_tasks)}")
    
    print("\n  步骤 5: 查看统计信息")
    stats = scheduler.get_statistics()
    print(f"    任务计数: {stats['task_counter']}")
    print(f"    热交换次数: {stats['hot_swap_count']}")
    print(f"    L2_PRIME 状态: {stats['l2_prime_status']}")
    print(f"    L2_BACKUP 状态: {stats['l2_backup_status']}")
    
    scheduler.shutdown()
    pool.shutdown()
    print("\n✓ 完整集成测试通过!")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("KV Cache 热切换功能测试")
    print("TSD v1.9 验证")
    print("="*60)
    
    tests = [
        test_hardware_aware_kv_pool,
        test_block_allocation,
        test_oom_handling,
        test_scheduler_initialization,
        test_single_task_processing,
        test_hot_swap_trigger,
        test_task_resume,
        test_model_loader,
        test_full_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ 测试失败: {test.__name__}")
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
