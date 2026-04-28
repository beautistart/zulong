# File: tests/test_kv_cache_hotswap.py
"""
KV Cache 热切换功能测试
TSD v1.9: 验证热交换机制的正确性
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHardwareAwareKVPool:
    """测试硬件抽象层"""
    
    def test_kv_pool_initialization(self):
        """测试 KV Pool 初始化"""
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
        
        success = pool.initialize()
        assert success
        assert pool._initialized
        
        pool.shutdown()
    
    def test_block_allocation(self):
        """测试块分配"""
        from zulong.adapters.memory_backend import HardwareAwareKVPool, KVPoolConfig
        
        config = KVPoolConfig(max_blocks=64)
        pool = HardwareAwareKVPool(config)
        pool.initialize()
        
        blocks = pool.allocate_for_task("task_1", 4)
        assert len(blocks) == 4
        assert all(0 <= b < 64 for b in blocks)
        
        stats = pool.get_statistics()
        assert stats["block_manager"]["allocated_blocks"] == 1
        assert stats["block_manager"]["free_blocks"] == 60
        
        pool.shutdown()
    
    def test_block_free(self):
        """测试块释放"""
        from zulong.adapters.memory_backend import HardwareAwareKVPool, KVPoolConfig
        
        config = KVPoolConfig(max_blocks=64)
        pool = HardwareAwareKVPool(config)
        pool.initialize()
        
        blocks = pool.allocate_for_task("task_1", 4)
        
        freed = pool.free_task("task_1")
        assert len(freed) == 4
        assert set(freed) == set(blocks)
        
        stats = pool.get_statistics()
        assert stats["block_manager"]["allocated_blocks"] == 0
        assert stats["block_manager"]["free_blocks"] == 64
        
        pool.shutdown()
    
    def test_oom_handling(self):
        """测试 OOM 处理"""
        from zulong.adapters.memory_backend import HardwareAwareKVPool, KVPoolConfig
        
        config = KVPoolConfig(max_blocks=8)
        pool = HardwareAwareKVPool(config)
        pool.initialize()
        
        pool.allocate_for_task("task_1", 4)
        pool.allocate_for_task("task_2", 4)
        
        with pytest.raises(RuntimeError) as excinfo:
            pool.allocate_for_task("task_3", 4)
        
        assert "OOM" in str(excinfo.value)
        
        pool.shutdown()
    
    def test_get_block_view(self):
        """测试获取块视图"""
        from zulong.adapters.memory_backend import HardwareAwareKVPool, KVPoolConfig
        
        config = KVPoolConfig(
            max_blocks=16,
            block_size=4,
            hidden_size=64,
            num_layers=2,
        )
        pool = HardwareAwareKVPool(config)
        pool.initialize()
        
        view = pool.get_block_view(0)
        
        assert view.shape == (2, 2, 4, 64)
        
        pool.shutdown()


class TestBlockTableManager:
    """测试块表管理器"""
    
    def test_block_table_manager_init(self):
        """测试块表管理器初始化"""
        from zulong.adapters.memory_backend import BlockTableManager
        
        manager = BlockTableManager(max_blocks=100)
        
        assert manager.max_blocks == 100
        assert len(manager.free_blocks) == 100
        assert len(manager.allocated_blocks) == 0
    
    def test_allocate_and_free(self):
        """测试分配和释放"""
        from zulong.adapters.memory_backend import BlockTableManager
        
        manager = BlockTableManager(max_blocks=100)
        
        blocks = manager.allocate_blocks("task_1", 10)
        assert len(blocks) == 10
        assert len(manager.free_blocks) == 90
        assert "task_1" in manager.allocated_blocks
        
        freed = manager.free_blocks("task_1")
        assert len(freed) == 10
        assert len(manager.free_blocks) == 100
        assert "task_1" not in manager.allocated_blocks
    
    def test_statistics(self):
        """测试统计信息"""
        from zulong.adapters.memory_backend import BlockTableManager
        
        manager = BlockTableManager(max_blocks=100)
        
        manager.allocate_blocks("task_1", 10)
        manager.allocate_blocks("task_2", 20)
        
        stats = manager.get_statistics()
        
        assert stats["total_blocks"] == 100
        assert stats["free_blocks"] == 70
        assert stats["allocated_blocks"] == 2
        assert stats["utilization"] == 0.3
        assert "task_1" in stats["tasks"]
        assert "task_2" in stats["tasks"]


class TestL1BHotSwapScheduler:
    """测试热交换调度器"""
    
    def test_scheduler_initialization(self):
        """测试调度器初始化"""
        from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
        from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig
        
        config = KVPoolConfig(max_blocks=64)
        pool = MockKVPool(config)
        pool.initialize()
        
        scheduler_config = HotSwapConfig(switch_threshold=2)
        scheduler = L1B_HotSwapScheduler(pool, scheduler_config)
        
        assert not scheduler._initialized
        
        success = scheduler.initialize()
        assert success
        assert scheduler._initialized
        assert scheduler.L2_PRIME is not None
        assert scheduler.L2_BACKUP is not None
        
        scheduler.shutdown()
        pool.shutdown()
    
    def test_single_task_processing(self):
        """测试单任务处理"""
        from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
        from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig
        
        config = KVPoolConfig(max_blocks=64)
        pool = MockKVPool(config)
        pool.initialize()
        
        scheduler = L1B_HotSwapScheduler(pool, HotSwapConfig(switch_threshold=2))
        scheduler.initialize()
        
        response = scheduler.on_user_input("你好")
        
        assert response is not None
        assert len(scheduler.active_tasks) == 1
        assert scheduler.task_counter == 1
        
        scheduler.shutdown()
        pool.shutdown()
    
    def test_hot_swap_trigger(self):
        """测试热交换触发"""
        from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
        from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig, TaskStatus
        
        config = KVPoolConfig(max_blocks=64)
        pool = MockKVPool(config)
        pool.initialize()
        
        scheduler = L1B_HotSwapScheduler(pool, HotSwapConfig(switch_threshold=2, block_per_task=4))
        scheduler.initialize()
        
        scheduler.on_user_input("任务1")
        assert len(scheduler.active_tasks) == 1
        assert scheduler._hot_swap_count == 0
        
        scheduler.on_user_input("任务2")
        assert len(scheduler.active_tasks) == 2
        assert scheduler._hot_swap_count == 0
        
        scheduler.on_user_input("任务3")
        
        assert scheduler._hot_swap_count == 1
        assert len(scheduler.active_tasks) == 1
        assert len(scheduler.freeze_stack) == 2
        
        for ctx in scheduler.freeze_stack:
            assert ctx.status == TaskStatus.MIGRATED
        
        scheduler.shutdown()
        pool.shutdown()
    
    def test_task_resume(self):
        """测试任务恢复"""
        from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
        from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig, TaskStatus
        
        config = KVPoolConfig(max_blocks=64)
        pool = MockKVPool(config)
        pool.initialize()
        
        scheduler = L1B_HotSwapScheduler(pool, HotSwapConfig(switch_threshold=2, block_per_task=4))
        scheduler.initialize()
        
        scheduler.on_user_input("任务1")
        scheduler.on_user_input("任务2")
        scheduler.on_user_input("任务3")
        
        assert len(scheduler.freeze_stack) == 2
        
        task_id = scheduler.freeze_stack[0].task_id
        response = scheduler.resume_task(task_id)
        
        assert response is not None
        assert len(scheduler.freeze_stack) == 1
        assert len(scheduler.active_tasks) == 2
        
        scheduler.shutdown()
        pool.shutdown()
    
    def test_task_complete(self):
        """测试任务完成"""
        from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
        from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig
        
        config = KVPoolConfig(max_blocks=64)
        pool = MockKVPool(config)
        pool.initialize()
        
        scheduler = L1B_HotSwapScheduler(pool, HotSwapConfig(switch_threshold=2, block_per_task=4))
        scheduler.initialize()
        
        scheduler.on_user_input("任务1")
        
        stats_before = pool.get_statistics()
        assert stats_before["block_manager"]["allocated_blocks"] == 1
        
        task_id = list(scheduler.active_tasks.keys())[0]
        scheduler.complete_task(task_id)
        
        assert len(scheduler.active_tasks) == 0
        
        stats_after = pool.get_statistics()
        assert stats_after["block_manager"]["allocated_blocks"] == 0
        
        scheduler.shutdown()
        pool.shutdown()
    
    def test_statistics(self):
        """测试统计信息"""
        from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
        from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig
        
        config = KVPoolConfig(max_blocks=64)
        pool = MockKVPool(config)
        pool.initialize()
        
        scheduler = L1B_HotSwapScheduler(pool, HotSwapConfig(switch_threshold=2))
        scheduler.initialize()
        
        scheduler.on_user_input("任务1")
        scheduler.on_user_input("任务2")
        scheduler.on_user_input("任务3")
        
        stats = scheduler.get_statistics()
        
        assert stats["initialized"]
        assert stats["task_counter"] == 3
        assert stats["hot_swap_count"] == 1
        assert stats["switch_threshold"] == 2
        assert stats["l2_prime_status"] is not None
        assert stats["l2_backup_status"] is not None
        
        scheduler.shutdown()
        pool.shutdown()


class TestModelLoader:
    """测试模型加载器"""
    
    def test_hardware_detection(self):
        """测试硬件检测"""
        from zulong.adapters.model_loader import detect_hardware, HardwareType
        
        hw_type = detect_hardware()
        
        assert hw_type in HardwareType
    
    def test_auto_select_model(self):
        """测试自动选择模型"""
        from zulong.adapters.model_loader import auto_select_model
        
        config = auto_select_model()
        
        assert config.model_id is not None
        assert config.model_name is not None
        assert config.tensor_parallel_size >= 1
        assert 0 < config.gpu_memory_utilization <= 1
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        from zulong.adapters.model_loader import get_model_info
        
        info = get_model_info()
        
        assert "hardware_type" in info
        assert "model_id" in info
        assert "model_name" in info
    
    def test_list_available_models(self):
        """测试列出可用模型"""
        from zulong.adapters.model_loader import list_available_models
        
        models = list_available_models()
        
        assert len(models) > 0
        assert "nvidia_low_end" in models
        assert "apu" in models
    
    def test_init_mock_engines(self):
        """测试初始化 Mock 引擎"""
        from zulong.adapters.model_loader import init_l2_engines
        
        prime, backup = init_l2_engines(use_vllm=False)
        
        assert prime is not None
        assert backup is not None
        assert prime.engine_id == "PRIME"
        assert backup.engine_id == "BACKUP"


class TestIntegration:
    """集成测试"""
    
    def test_full_hotswap_flow(self):
        """测试完整的热交换流程"""
        from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
        from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig
        
        config = KVPoolConfig(max_blocks=128)
        pool = MockKVPool(config)
        assert pool.initialize()
        
        scheduler = L1B_HotSwapScheduler(
            pool, 
            HotSwapConfig(
                switch_threshold=2,
                block_per_task=8,
                max_freeze_stack_size=5,
            )
        )
        assert scheduler.initialize()
        
        scheduler.on_user_input("第一个任务：今天天气怎么样？")
        assert len(scheduler.active_tasks) == 1
        
        scheduler.on_user_input("第二个任务：讲个笑话")
        assert len(scheduler.active_tasks) == 2
        
        response = scheduler.on_user_input("第三个任务：救命！")
        assert response is not None
        
        assert scheduler._hot_swap_count == 1
        assert len(scheduler.freeze_stack) == 2
        assert len(scheduler.active_tasks) == 1
        
        frozen_task_id = scheduler.freeze_stack[0].task_id
        resume_response = scheduler.resume_task(frozen_task_id)
        assert resume_response is not None
        assert len(scheduler.freeze_stack) == 1
        
        stats = scheduler.get_statistics()
        assert stats["task_counter"] == 3
        assert stats["hot_swap_count"] == 1
        
        scheduler.shutdown()
        pool.shutdown()
    
    def test_multiple_hotswaps(self):
        """测试多次热交换"""
        from zulong.adapters.memory_backend import MockKVPool, KVPoolConfig
        from zulong.l1b.hotswap_scheduler import L1B_HotSwapScheduler, HotSwapConfig
        
        config = KVPoolConfig(max_blocks=256)
        pool = MockKVPool(config)
        pool.initialize()
        
        scheduler = L1B_HotSwapScheduler(
            pool,
            HotSwapConfig(
                switch_threshold=2,
                block_per_task=8,
                max_freeze_stack_size=10,
            )
        )
        scheduler.initialize()
        
        for i in range(6):
            scheduler.on_user_input(f"任务 {i+1}")
        
        assert scheduler._hot_swap_count == 2
        assert len(scheduler.freeze_stack) == 4
        
        scheduler.shutdown()
        pool.shutdown()


def run_tests():
    """运行所有测试"""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
