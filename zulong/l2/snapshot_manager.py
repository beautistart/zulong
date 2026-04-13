# File: zulong/l2/snapshot_manager.py
# 任务快照管理器 - 管理多任务快照的冻结、恢复与切换
# 对应 TSD v1.7: 上下文隔离与任务切换机制

from .task_snapshot import TaskSnapshot, TaskStatus, KVCacheSnapshot
from typing import Dict, Optional, List
from collections import OrderedDict
import threading
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='[SnapshotManager] %(message)s')
logger = logging.getLogger(__name__)


class SnapshotManager:
    """任务快照管理器（单例模式）
    
    类比：就像一个多存档槽的游戏系统
    - 可以同时保存多个任务的进度
    - 支持快速切换不同任务
    - 自动管理内存，淘汰最久未使用的快照
    
    核心功能：
    1. 创建快照 (create_snapshot)
    2. 冻结任务 (freeze) - 保存状态，释放 KV Cache
    3. 恢复任务 (thaw/resume) - 重建 KV Cache，恢复执行
    4. 切换任务 (switch_to) - 上下文隔离的关键
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SnapshotManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """初始化快照管理器"""
        # 使用 OrderedDict 实现 LRU 缓存
        # key: task_id, value: TaskSnapshot
        self._snapshots: OrderedDict[str, TaskSnapshot] = OrderedDict()
        
        # 当前活跃的任务 ID
        self._active_task_id: Optional[str] = None
        
        # 配置参数
        self._max_snapshots = 5              # 最大快照数量
        self._max_memory_mb = 512            # 最大内存占用 (MB)
        self._current_memory_mb = 0.0        # 当前内存占用
        
        # 线程锁
        self._snapshot_lock = threading.RLock()
        
        logger.info(f"SnapshotManager initialized (max_snapshots={self._max_snapshots})")
    
    def create_snapshot(
        self,
        task_id: str,
        task_name: str,
        context_window: List[Dict] = None,
        working_memory: Dict = None,
        intent_stack: List = None,
        kv_cache_tokens: int = 0,
        metadata: Dict = None
    ) -> TaskSnapshot:
        """创建新的任务快照
        
        Args:
            task_id: 任务唯一标识
            task_name: 任务名称（人类可读）
            context_window: 上下文窗口（对话历史）
            working_memory: 工作记忆
            intent_stack: 意图栈
            kv_cache_tokens: KV Cache token 数量
            metadata: 额外元数据
            
        Returns:
            TaskSnapshot: 新创建的快照
        """
        with self._snapshot_lock:
            # 如果已存在同名任务，先移除旧快照
            if task_id in self._snapshots:
                logger.info(f"Removing existing snapshot for task: {task_id}")
                self._remove_snapshot(task_id)
            
            # 检查是否需要淘汰旧快照
            self._evict_if_needed()
            
            # 创建 KV Cache 快照元数据
            kv_cache = None
            if kv_cache_tokens > 0:
                # 估算内存占用: 每个 token 约 0.5MB (简化计算)
                estimated_memory = kv_cache_tokens * 0.5
                kv_cache = KVCacheSnapshot(
                    num_tokens=kv_cache_tokens,
                    memory_size_mb=estimated_memory
                )
                self._current_memory_mb += estimated_memory
            
            # 创建任务快照
            snapshot = TaskSnapshot(
                task_id=task_id,
                task_name=task_name,
                context_window=context_window or [],
                working_memory=working_memory or {},
                intent_stack=intent_stack or [],
                kv_cache=kv_cache,
                metadata=metadata or {}
            )
            
            # 保存到有序字典（最新使用的放在末尾）
            self._snapshots[task_id] = snapshot
            self._active_task_id = task_id
            
            logger.info(f"Created snapshot: {snapshot.get_summary()}")
            logger.info(f"Memory usage: {self._current_memory_mb:.2f}/{self._max_memory_mb}MB")
            
            return snapshot
    
    def freeze(self, task_id: str) -> Optional[TaskSnapshot]:
        """冻结任务 - 保存状态并释放 KV Cache
        
        TSD v1.7 对应:
        - 5.2 任务冻结机制：保存 KV Cache 元数据到堆栈
        - 保存环境快照用于重评估
        
        类比：就像把游戏存档保存到硬盘，然后关闭游戏释放内存
        
        Args:
            task_id: 要冻结的任务 ID
            
        Returns:
            TaskSnapshot: 冻结后的快照，如果不存在则返回 None
        """
        with self._snapshot_lock:
            if task_id not in self._snapshots:
                logger.warning(f"Cannot freeze: task {task_id} not found")
                return None
            
            snapshot = self._snapshots[task_id]
            
            # 🎯 TSD v1.7: 保存环境快照（用于恢复时重评估）
            from .environment_snapshot import environment_snapshot_manager
            snapshot.environment_snapshot = environment_snapshot_manager.create_snapshot(task_id)
            logger.info(f"Saved environment snapshot for task: {task_id}")
            
            # 标记为冻结状态
            snapshot.freeze()
            
            # 释放 KV Cache 内存（实际释放由 ModelContainer 执行）
            if snapshot.kv_cache:
                released_memory = snapshot.kv_cache.memory_size_mb
                self._current_memory_mb -= released_memory
                logger.info(f"Released KV Cache: {released_memory:.2f}MB")
                # 保留元数据，但标记为已释放
                snapshot.kv_cache.memory_size_mb = 0
            
            # 如果不是活跃任务，移动到有序字典开头（表示最近最少使用）
            if self._active_task_id != task_id:
                self._snapshots.move_to_end(task_id, last=False)
            
            logger.info(f"Frozen task: {task_id}")
            return snapshot
    
    def thaw(self, task_id: str, kv_cache_tokens: int = 0) -> Optional[TaskSnapshot]:
        """解冻任务 - 恢复状态并重建 KV Cache
        
        类比：就像从硬盘加载游戏存档，重新开始游戏
        
        Args:
            task_id: 要解冻的任务 ID
            kv_cache_tokens: 新的 KV Cache token 数量（用于重建）
            
        Returns:
            TaskSnapshot: 解冻后的快照，如果不存在则返回 None
        """
        with self._snapshot_lock:
            if task_id not in self._snapshots:
                logger.warning(f"Cannot thaw: task {task_id} not found")
                return None
            
            snapshot = self._snapshots[task_id]
            
            # 检查内存是否足够
            if kv_cache_tokens > 0:
                estimated_memory = kv_cache_tokens * 0.5
                if self._current_memory_mb + estimated_memory > self._max_memory_mb:
                    logger.warning("Not enough memory to thaw task, evicting...")
                    self._evict_lru_snapshot()
            
            # 重建 KV Cache 元数据
            if kv_cache_tokens > 0:
                estimated_memory = kv_cache_tokens * 0.5
                snapshot.kv_cache = KVCacheSnapshot(
                    num_tokens=kv_cache_tokens,
                    memory_size_mb=estimated_memory
                )
                self._current_memory_mb += estimated_memory
                logger.info(f"Rebuilt KV Cache: {estimated_memory:.2f}MB")
            
            # 标记为运行状态
            snapshot.thaw()
            
            # 设为活跃任务
            self._active_task_id = task_id
            
            # 移动到有序字典末尾（表示最近使用）
            self._snapshots.move_to_end(task_id)
            
            logger.info(f"Thawed task: {task_id}")
            logger.info(f"Memory usage: {self._current_memory_mb:.2f}/{self._max_memory_mb}MB")
            
            return snapshot
    
    def switch_to(self, task_id: str) -> Optional[TaskSnapshot]:
        """切换到指定任务 - 上下文隔离的核心
        
        类比：就像在游戏中切换不同的存档槽
        
        Args:
            task_id: 要切换到的任务 ID
            
        Returns:
            TaskSnapshot: 切换后的快照，如果不存在则返回 None
        """
        with self._snapshot_lock:
            if task_id not in self._snapshots:
                logger.warning(f"Cannot switch: task {task_id} not found")
                return None
            
            # 1. 冻结当前活跃任务
            if self._active_task_id and self._active_task_id != task_id:
                logger.info(f"Freezing current task: {self._active_task_id}")
                self.freeze(self._active_task_id)
            
            # 2. 解冻目标任务
            logger.info(f"Switching to task: {task_id}")
            snapshot = self.thaw(task_id)
            
            return snapshot
    
    def get_snapshot(self, task_id: str) -> Optional[TaskSnapshot]:
        """获取指定任务的快照"""
        with self._snapshot_lock:
            return self._snapshots.get(task_id)
    
    def get_active_snapshot(self) -> Optional[TaskSnapshot]:
        """获取当前活跃任务的快照"""
        with self._snapshot_lock:
            if self._active_task_id:
                return self._snapshots.get(self._active_task_id)
            return None
    
    def get_active_task_id(self) -> Optional[str]:
        """获取当前活跃任务 ID"""
        with self._snapshot_lock:
            return self._active_task_id
    
    def list_snapshots(self) -> List[Dict]:
        """列出所有快照的摘要信息"""
        with self._snapshot_lock:
            return [
                {
                    "task_id": task_id,
                    "task_name": snapshot.task_name,
                    "status": snapshot.status.name,
                    "is_active": task_id == self._active_task_id,
                    "progress": snapshot.execution_pointer.progress_percentage,
                    "memory_mb": snapshot.kv_cache.memory_size_mb if snapshot.kv_cache else 0
                }
                for task_id, snapshot in self._snapshots.items()
            ]
    
    def update_snapshot(
        self,
        task_id: str,
        context_window: List[Dict] = None,
        working_memory: Dict = None,
        execution_pointer: Dict = None
    ) -> bool:
        """更新快照状态（用于任务执行过程中）
        
        Args:
            task_id: 任务 ID
            context_window: 新的上下文窗口（可选）
            working_memory: 新的工作记忆（可选）
            execution_pointer: 新的执行指针（可选）
            
        Returns:
            bool: 是否成功更新
        """
        with self._snapshot_lock:
            if task_id not in self._snapshots:
                return False
            
            snapshot = self._snapshots[task_id]
            
            if context_window is not None:
                snapshot.context_window = context_window
            if working_memory is not None:
                snapshot.working_memory.update(working_memory)
            if execution_pointer is not None:
                snapshot.execution_pointer.current_step = execution_pointer.get("current_step", snapshot.execution_pointer.current_step)
                snapshot.execution_pointer.step_description = execution_pointer.get("step_description", snapshot.execution_pointer.step_description)
                snapshot.execution_pointer.generated_content = execution_pointer.get("generated_content", snapshot.execution_pointer.generated_content)
            
            # 移动到末尾（表示最近使用）
            self._snapshots.move_to_end(task_id)
            
            return True
    
    def _remove_snapshot(self, task_id: str) -> bool:
        """移除指定快照"""
        if task_id not in self._snapshots:
            return False
        
        snapshot = self._snapshots[task_id]
        
        # 释放内存
        if snapshot.kv_cache:
            self._current_memory_mb -= snapshot.kv_cache.memory_size_mb
        
        # 如果移除的是活跃任务，清空活跃任务 ID
        if self._active_task_id == task_id:
            self._active_task_id = None
        
        del self._snapshots[task_id]
        logger.info(f"Removed snapshot: {task_id}")
        return True
    
    def _evict_if_needed(self):
        """检查是否需要淘汰旧快照"""
        # 检查数量限制
        while len(self._snapshots) >= self._max_snapshots:
            self._evict_lru_snapshot()
        
        # 检查内存限制
        while self._current_memory_mb > self._max_memory_mb and len(self._snapshots) > 0:
            self._evict_lru_snapshot()
    
    def _evict_lru_snapshot(self):
        """淘汰最近最少使用的快照（有序字典的第一个）"""
        if not self._snapshots:
            return
        
        # 获取第一个（最旧的）快照
        lru_task_id = next(iter(self._snapshots))
        
        # 不能淘汰活跃任务
        if lru_task_id == self._active_task_id and len(self._snapshots) > 1:
            # 尝试第二个
            lru_task_id = list(self._snapshots.keys())[1]
        
        if lru_task_id == self._active_task_id:
            logger.warning("Cannot evict active task, memory limit reached!")
            return
        
        logger.info(f"Evicting LRU snapshot: {lru_task_id}")
        self._remove_snapshot(lru_task_id)
    
    def get_memory_stats(self) -> Dict:
        """获取内存统计信息"""
        with self._snapshot_lock:
            return {
                "current_mb": self._current_memory_mb,
                "max_mb": self._max_memory_mb,
                "usage_percentage": (self._current_memory_mb / self._max_memory_mb * 100) if self._max_memory_mb > 0 else 0,
                "snapshot_count": len(self._snapshots),
                "max_snapshots": self._max_snapshots
            }
    
    def clear_all(self):
        """清除所有快照（用于测试或重置）"""
        with self._snapshot_lock:
            self._snapshots.clear()
            self._active_task_id = None
            self._current_memory_mb = 0.0
            logger.info("All snapshots cleared")


# 全局快照管理器实例
snapshot_manager = SnapshotManager()
