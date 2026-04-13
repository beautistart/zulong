# File: zulong/l2/l2_snapshot_interface.py
# L2 快照接口 - 为 L1-B 注意力控制器提供快照与恢复功能

"""
祖龙 (ZULONG) 系统 - L2 快照接口层

核心功能:
1. 创建任务快照 (供 L1-B 冻结时调用)
2. 恢复任务快照 (供 L1-B 恢复时调用)
3. 获取当前任务状态
4. 清理已完成的快照

TSD v1.8 对应:
- 2.4 任务冻结与重组算法
- 3.3 上下文快照管理
- 4.3.2 L2 输出格式
"""

import time
import logging
from typing import Optional, Dict, Any, List
from threading import Lock

from zulong.l2.snapshot_manager import SnapshotManager
from zulong.l2.task_snapshot import TaskSnapshot
from zulong.core.attention_atoms import ContextSnapshot

logger = logging.getLogger("L2SnapshotInterface")


class L2SnapshotInterface:
    """
    L2 快照接口层
    
    职责:
    - 作为 L1-B 与 L2 快照管理器之间的桥梁
    - 提供简化的 API 供 L1-B 调用
    - 处理 KV Cache 的保存与恢复
    
    TSD v1.8 对应:
    - 3.3.1 上下文快照数据结构
    - 2.4.2 任务冻结机制
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化 L2 快照接口"""
        if self._initialized:
            return
            
        self.snapshot_manager = SnapshotManager()
        self._initialized = True
        
        logger.info("🧩 [L2SnapshotInterface] 初始化完成")
        logger.info(f"   - 快照管理器：已连接")
        logger.info(f"   - 最大快照数：{self.snapshot_manager._max_snapshots}")
    
    def create_context_snapshot(self, task_id: Optional[str] = None) -> Optional[ContextSnapshot]:
        """
        创建上下文快照 (供 L1-B 冻结时调用)
        
        TSD v1.8 对应:
        - 3.3.1 上下文快照数据结构
        
        Args:
            task_id: 任务 ID，如果为 None 则使用当前活跃任务
            
        Returns:
            ContextSnapshot: 上下文快照，如果任务不存在则返回 None
        """
        try:
            # 如果未指定 task_id，使用当前活跃任务
            if task_id is None:
                task_id = self.snapshot_manager.get_active_task_id()
                if task_id is None:
                    logger.warning("⚠️ [L2SnapshotInterface] 无活跃任务，无法创建快照")
                    return None
            
            # 获取任务快照
            task_snapshot = self.snapshot_manager.get_snapshot(task_id)
            
            if task_snapshot is None:
                logger.warning(f"⚠️ [L2SnapshotInterface] 任务 {task_id} 不存在")
                return None
            
            # 转换为 ContextSnapshot (L1-B 使用的格式)
            context_snapshot = ContextSnapshot(
                task_id=task_snapshot.task_id,
                summary=task_snapshot.task_name,  # 使用任务名作为摘要
                full_history=task_snapshot.context_window.copy(),
                kv_cache_ptr=task_snapshot.kv_cache,  # KV Cache 指针（元数据）
                generation_state={
                    "execution_pointer": task_snapshot.execution_pointer,
                    "working_memory": task_snapshot.working_memory,
                    "intent_stack": task_snapshot.intent_stack,
                    "environment_snapshot": task_snapshot.environment_snapshot
                },
                pause_reason="emergency_interrupt"
            )
            
            logger.info(f"📸 [L2SnapshotInterface] 创建上下文快照：task_id={task_id}")
            logger.info(f"   - 上下文长度：{len(context_snapshot.full_history)}")
            logger.info(f"   - KV Cache tokens: {context_snapshot.kv_cache_ptr.num_tokens if context_snapshot.kv_cache_ptr else 0}")
            
            return context_snapshot
            
        except Exception as e:
            logger.error(f"❌ [L2SnapshotInterface] 创建快照失败：{e}", exc_info=True)
            return None
    
    def restore_context_snapshot(
        self, 
        snapshot: ContextSnapshot,
        kv_cache_tokens: int = 0
    ) -> bool:
        """
        恢复上下文快照 (供 L1-B 恢复时调用)
        
        TSD v1.8 对应:
        - 3.3.1 上下文快照恢复
        - 2.4.3 任务恢复机制
        
        Args:
            snapshot: 上下文快照
            kv_cache_tokens: KV Cache token 数量（用于重建）
            
        Returns:
            bool: 是否成功恢复
        """
        try:
            task_id = snapshot.task_id
            
            # 恢复快照（KV Cache token 数量从快照中获取）
            kv_cache_tokens = snapshot.kv_cache_ptr.num_tokens if snapshot.kv_cache_ptr else 0
            
            # 检查快照是否已存在
            existing = self.snapshot_manager.get_snapshot(task_id)
            
            if existing is None:
                # 从 ContextSnapshot 创建新的 TaskSnapshot
                logger.info(f"📥 [L2SnapshotInterface] 创建新任务：task_id={task_id}")
                
                self.snapshot_manager.create_snapshot(
                    task_id=task_id,
                    task_name=snapshot.summary,
                    context_window=snapshot.full_history.copy(),
                    working_memory=snapshot.generation_state.get("working_memory", {}),
                    intent_stack=snapshot.generation_state.get("intent_stack", []),
                    kv_cache_tokens=kv_cache_tokens,
                    metadata={
                        "pause_reason": snapshot.pause_reason,
                        "created_from": "l1b_restore"
                    }
                )
            else:
                # 恢复已存在的快照
                logger.info(f"📥 [L2SnapshotInterface] 恢复任务：task_id={task_id}")
                
                # 更新快照数据
                self.snapshot_manager.update_snapshot(
                    task_id=task_id,
                    context_window=snapshot.full_history.copy(),
                    working_memory=snapshot.generation_state.get("working_memory", {})
                    # execution_pointer 是对象，不需要更新
                )
                
                # 解冻任务（使用更新后的 kv_cache_tokens）
                self.snapshot_manager.thaw(task_id, kv_cache_tokens)
            
            # 切换到恢复的任务
            self.snapshot_manager.switch_to(task_id)
            
            logger.info(f"✅ [L2SnapshotInterface] 恢复成功：task_id={task_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [L2SnapshotInterface] 恢复快照失败：{e}", exc_info=True)
            return False
    
    def freeze_current_task(self) -> Optional[ContextSnapshot]:
        """
        冻结当前任务 (供 L1-B 中断时调用)
        
        TSD v1.8 对应:
        - 2.4.2 任务冻结机制
        
        Returns:
            ContextSnapshot: 冻结的快照，如果无活跃任务则返回 None
        """
        try:
            # 获取当前活跃任务 ID
            task_id = self.snapshot_manager.get_active_task_id()
            
            if task_id is None:
                logger.warning("⚠️ [L2SnapshotInterface] 无活跃任务可冻结")
                return None
            
            # 冻结任务
            frozen_snapshot = self.snapshot_manager.freeze(task_id)
            
            if frozen_snapshot is None:
                logger.warning(f"⚠️ [L2SnapshotInterface] 冻结任务 {task_id} 失败")
                return None
            
            # 转换为 ContextSnapshot
            context_snapshot = ContextSnapshot(
                task_id=frozen_snapshot.task_id,
                summary=frozen_snapshot.task_name,
                full_history=frozen_snapshot.context_window.copy(),
                kv_cache_ptr=frozen_snapshot.kv_cache,
                generation_state={
                    "execution_pointer": frozen_snapshot.execution_pointer,
                    "working_memory": frozen_snapshot.working_memory,
                    "intent_stack": frozen_snapshot.intent_stack,
                    "environment_snapshot": frozen_snapshot.environment_snapshot
                },
                pause_reason="emergency_interrupt"
            )
            
            logger.info(f"🧊 [L2SnapshotInterface] 冻结任务：task_id={task_id}")
            return context_snapshot
            
        except Exception as e:
            logger.error(f"❌ [L2SnapshotInterface] 冻结任务失败：{e}", exc_info=True)
            return None
    
    def get_current_task_id(self) -> Optional[str]:
        """
        获取当前任务 ID
        
        Returns:
            Optional[str]: 当前任务 ID
        """
        return self.snapshot_manager.get_active_task_id()
    
    def is_task_active(self, task_id: str) -> bool:
        """
        检查任务是否活跃
        
        Args:
            task_id: 任务 ID
            
        Returns:
            bool: 是否活跃
        """
        return self.snapshot_manager.get_active_task_id() == task_id
    
    def cleanup_completed_tasks(self, max_age_seconds: float = 300.0) -> int:
        """
        清理已完成的任务快照
        
        Args:
            max_age_seconds: 最大保留时间（秒），默认 5 分钟
            
        Returns:
            int: 清理的任务数量
        """
        try:
            current_time = time.time()
            cleaned_count = 0
            
            snapshots = self.snapshot_manager.list_snapshots()
            
            for snapshot_info in snapshots:
                task_id = snapshot_info["task_id"]
                
                # 检查任务是否已完成（简化：假设所有任务都未完成）
                # 未来可以添加任务完成状态标记
                task_snapshot = self.snapshot_manager.get_snapshot(task_id)
                
                if task_snapshot:
                    # 检查创建时间
                    created_at = task_snapshot.metadata.get("created_at", current_time)
                    age = current_time - created_at
                    
                    if age > max_age_seconds and not snapshot_info["is_active"]:
                        logger.info(f"🗑️ [L2SnapshotInterface] 清理过期任务：task_id={task_id}, age={age:.1f}s")
                        # 这里可以添加删除逻辑
                        cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ [L2SnapshotInterface] 清理任务失败：{e}", exc_info=True)
            return 0


# 全局单例
_l2_snapshot_interface: Optional[L2SnapshotInterface] = None


def get_l2_snapshot_interface() -> L2SnapshotInterface:
    """获取 L2 快照接口单例"""
    global _l2_snapshot_interface
    if _l2_snapshot_interface is None:
        _l2_snapshot_interface = L2SnapshotInterface()
    return _l2_snapshot_interface
