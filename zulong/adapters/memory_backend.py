# File: zulong/adapters/memory_backend.py
"""
原子任务 1: 硬件感知的内存后端
目标: 实现一套代码，两套存储策略。
TSD v1.9: KV Cache 热切换机制 - 硬件抽象层
"""

import torch
import os
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KVPoolConfig:
    max_blocks: int = 8192
    block_size: int = 16
    hidden_size: int = 2048
    num_layers: int = 32
    dtype: torch.dtype = torch.float16


@dataclass
class BlockInfo:
    block_id: int
    task_id: Optional[str] = None
    is_allocated: bool = False
    last_access_time: float = 0.0


class BlockTableManager:
    """
    块表管理器 (类似 vLLM)。
    管理逻辑 Block ID 到 物理 Block ID 的映射。
    这是实现 "热交换" 的关键：我们交换的是 Block Table，而不是数据。
    """
    
    def __init__(self, max_blocks: int):
        self.max_blocks = max_blocks
        self._free_blocks: List[int] = list(range(max_blocks))
        self.allocated_blocks: Dict[str, List[int]] = {}
        self.block_info: Dict[int, BlockInfo] = {
            i: BlockInfo(block_id=i) for i in range(max_blocks)
        }
        self._lock = None
        
    def allocate_blocks(self, task_id: str, num_blocks: int) -> List[int]:
        """
        为任务分配物理块
        
        Args:
            task_id: 任务唯一标识
            num_blocks: 需要分配的块数量
            
        Returns:
            分配的物理块 ID 列表
            
        Raises:
            RuntimeError: 显存不足时抛出
        """
        if len(self._free_blocks) < num_blocks:
            raise RuntimeError(
                f"KV Cache OOM! 请求 {num_blocks} 块，仅剩 {len(self._free_blocks)} 块。"
                "请先驱逐旧任务或增加 max_blocks。"
            )
        
        allocated = []
        import time
        current_time = time.time()
        
        for _ in range(num_blocks):
            block_id = self._free_blocks.pop()
            allocated.append(block_id)
            
            self.block_info[block_id].task_id = task_id
            self.block_info[block_id].is_allocated = True
            self.block_info[block_id].last_access_time = current_time
        
        self.allocated_blocks[task_id] = allocated
        logger.info(f"[BlockTable] 为任务 {task_id} 分配了 {num_blocks} 个块: {allocated}")
        
        return allocated
    
    def release_blocks(self, task_id: str) -> List[int]:
        """
        释放任务占用的物理块
        
        Args:
            task_id: 任务唯一标识
            
        Returns:
            释放的物理块 ID 列表
        """
        if task_id not in self.allocated_blocks:
            logger.warning(f"[BlockTable] 任务 {task_id} 没有分配的块")
            return []
        
        freed_blocks = self.allocated_blocks[task_id]
        
        for block_id in freed_blocks:
            self.block_info[block_id].task_id = None
            self.block_info[block_id].is_allocated = False
        
        self._free_blocks.extend(freed_blocks)
        del self.allocated_blocks[task_id]
        
        logger.info(f"[BlockTable] 释放了任务 {task_id} 的 {len(freed_blocks)} 个块")
        return freed_blocks
    
    @property
    def free_blocks(self) -> List[int]:
        """获取空闲块列表"""
        return self._free_blocks
    
    def get_task_blocks(self, task_id: str) -> Optional[List[int]]:
        """获取任务占用的块列表"""
        return self.allocated_blocks.get(task_id)
    
    def get_statistics(self) -> Dict:
        """获取块管理统计信息"""
        return {
            "total_blocks": self.max_blocks,
            "free_blocks": len(self._free_blocks),
            "allocated_blocks": len(self.allocated_blocks),
            "utilization": (self.max_blocks - len(self._free_blocks)) / self.max_blocks,
            "tasks": list(self.allocated_blocks.keys()),
        }


class HardwareAwareKVPool:
    """
    硬件感知的 KV Cache 池。
    策略:
      - 如果 GPU 显存 > 48GB (APU): 使用 Unified Memory (Base = CUDA, resizable)
      - 否则 (Discrete GPU): 严格限制在 GPU 显存内，预分配 Block。
    
    TSD v1.9: 支持 KV Cache 热切换
    """
    
    def __init__(self, config: Optional[KVPoolConfig] = None):
        self.config = config or KVPoolConfig()
        
        self.block_size = self.config.block_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_layers
        self.max_blocks = self.config.max_blocks
        
        self.is_unified_memory = self._detect_unified_memory()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.is_unified_memory:
            logger.info("[HardwareAwareKVPool] 检测到统一内存架构 (APU)，启用 Zero-Copy 模式")
        else:
            logger.info(f"[HardwareAwareKVPool] 检测到独立显卡，启用 PagedAttention 模式")
        
        self.physical_pool: Optional[torch.Tensor] = None
        self.block_manager = BlockTableManager(self.max_blocks)
        
        self._initialized = False
    
    def _detect_unified_memory(self) -> bool:
        """
        检测是否为统一内存架构 (APU)
        
        Returns:
            True 如果是 APU 统一内存架构，False 如果是独立显卡
        """
        if not torch.cuda.is_available():
            logger.warning("[HardwareAwareKVPool] CUDA 不可用，将使用 CPU")
            return False
        
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            total_gb = total_mem / (1024 ** 3)
            
            if total_gb > 64:
                logger.info(f"[HardwareAwareKVPool] 检测到大内存 ({total_gb:.1f}GB)，判定为 APU")
                return True
            else:
                logger.info(f"[HardwareAwareKVPool] 检测到标准显存 ({total_gb:.1f}GB)，判定为独立显卡")
                return False
        except Exception as e:
            logger.warning(f"[HardwareAwareKVPool] 内存检测失败: {e}，默认使用独立显卡模式")
            return False
    
    def initialize(self) -> bool:
        """
        初始化物理 KV Cache 池
        
        Returns:
            True 如果初始化成功
        """
        if self._initialized:
            logger.warning("[HardwareAwareKVPool] 已经初始化，跳过")
            return True
        
        try:
            shape = (
                self.max_blocks,
                self.num_layers,
                2,
                self.block_size,
                self.hidden_size,
            )
            
            self.physical_pool = torch.empty(
                shape,
                dtype=self.config.dtype,
                device=self.device,
            )
            
            self._initialized = True
            
            mem_size_mb = self.physical_pool.numel() * self.physical_pool.element_size() / (1024 ** 2)
            logger.info(
                f"[HardwareAwareKVPool] 初始化完成: "
                f"shape={shape}, dtype={self.config.dtype}, "
                f"device={self.device}, size={mem_size_mb:.1f}MB"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"[HardwareAwareKVPool] 初始化失败: {e}")
            return False
    
    def get_block_view(self, block_id: int) -> torch.Tensor:
        """
        获取物理块的视图 (View)。
        在 APU 上是真正的 View。
        在独立显卡上也是 View，但物理上在 GPU。
        
        Args:
            block_id: 物理 Block ID
            
        Returns:
            该 Block 的 Tensor 视图
        """
        if not self._initialized:
            raise RuntimeError("KV Pool 未初始化，请先调用 initialize()")
        
        if block_id < 0 or block_id >= self.max_blocks:
            raise ValueError(f"无效的 block_id: {block_id}，范围: [0, {self.max_blocks})")
        
        return self.physical_pool[block_id]
    
    def get_blocks_view(self, block_ids: List[int]) -> torch.Tensor:
        """
        获取多个物理块的视图
        
        Args:
            block_ids: 物理 Block ID 列表
            
        Returns:
            多个 Block 的 Tensor 视图 (拼接)
        """
        if not self._initialized:
            raise RuntimeError("KV Pool 未初始化，请先调用 initialize()")
        
        views = [self.get_block_view(bid) for bid in block_ids]
        return torch.stack(views, dim=0)
    
    def allocate_for_task(self, task_id: str, num_blocks: int) -> List[int]:
        """
        为任务分配 KV Cache 块
        
        Args:
            task_id: 任务唯一标识
            num_blocks: 需要的块数量
            
        Returns:
            分配的物理块 ID 列表
        """
        return self.block_manager.allocate_blocks(task_id, num_blocks)
    
    def free_task(self, task_id: str) -> List[int]:
        """
        释放任务的 KV Cache 块
        
        Args:
            task_id: 任务唯一标识
            
        Returns:
            释放的物理块 ID 列表
        """
        return self.block_manager.release_blocks(task_id)
    
    def get_statistics(self) -> Dict:
        """获取 KV Pool 统计信息"""
        stats = {
            "initialized": self._initialized,
            "is_unified_memory": self.is_unified_memory,
            "device": self.device,
            "config": {
                "max_blocks": self.max_blocks,
                "block_size": self.block_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            "block_manager": self.block_manager.get_statistics(),
        }
        
        if self._initialized and self.physical_pool is not None:
            stats["pool_shape"] = list(self.physical_pool.shape)
            stats["pool_dtype"] = str(self.physical_pool.dtype)
            stats["pool_size_mb"] = (
                self.physical_pool.numel() * self.physical_pool.element_size() / (1024 ** 2)
            )
        
        return stats
    
    def shutdown(self):
        """关闭 KV Pool，释放资源"""
        if self.physical_pool is not None:
            del self.physical_pool
            self.physical_pool = None
        
        self._initialized = False
        logger.info("[HardwareAwareKVPool] 已关闭")


class MockKVPool(HardwareAwareKVPool):
    """
    Mock KV Pool，用于测试环境（无 GPU）
    """
    
    def __init__(self, config: Optional[KVPoolConfig] = None):
        super().__init__(config)
        self.device = "cpu"
        self.is_unified_memory = False
    
    def _detect_unified_memory(self) -> bool:
        return False
    
    def initialize(self) -> bool:
        self._initialized = True
        logger.info("[MockKVPool] 初始化完成 (Mock 模式)")
        return True
    
    def get_block_view(self, block_id: int) -> torch.Tensor:
        return torch.zeros(
            (self.num_layers, 2, self.block_size, self.hidden_size),
            dtype=self.config.dtype,
        )
