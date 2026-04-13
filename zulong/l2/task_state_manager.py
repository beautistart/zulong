# File: zulong/l2/task_state_manager.py
# 任务状态管理器 - 第五阶段总装
# 对应 TSD v1.7: 任务冻结与恢复

import uuid
import time
import threading
from typing import Dict, List, Optional

from zulong.core.types import TaskSnapshot, TaskStatus

import logging
logger = logging.getLogger(__name__)


class TaskStateManager:
    """任务状态管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化任务状态管理器"""
        if not hasattr(self, '_initialized'):
            self._active_task_id: Optional[str] = None
            self._frozen_tasks: Dict[str, TaskSnapshot] = {}
            self._task_stack: List[str] = []
            self._lock = threading.Lock()
            self._initialized = True
            logger.info("TaskStateManager initialized")
    
    def create_task(self, task_id: str, context: List[Dict]) -> str:
        """创建新任务
        
        Args:
            task_id: 任务 ID
            context: 初始上下文
            
        Returns:
            str: 任务 ID
        """
        with self._lock:
            # 如果已有活跃任务，先冻结
            if self._active_task_id:
                self.freeze_current()
            
            # 创建新任务
            snapshot = TaskSnapshot(
                task_id=task_id,
                context_history=context,
                working_memory={},
                execution_pointer="start",
                created_at=time.time(),
                last_updated=time.time()
            )
            
            # 设置为活跃任务
            self._active_task_id = task_id
            
            logger.info(f"Task {task_id} created and set as active")
            return task_id
    
    def update_task(self, task_id: str, new_token: str, vars_update: Dict[str, any]):
        """更新任务
        
        Args:
            task_id: 任务 ID
            new_token: 新生成的 token
            vars_update: 变量更新
        """
        with self._lock:
            if task_id == self._active_task_id:
                # 更新活跃任务
                # 这里简化处理，实际应该从内存中获取快照
                logger.debug(f"Updating active task {task_id}")
            elif task_id in self._frozen_tasks:
                # 更新冻结任务
                snapshot = self._frozen_tasks[task_id]
                snapshot.last_updated = time.time()
                logger.debug(f"Updating frozen task {task_id}")
    
    def freeze_current(self):
        """冻结当前任务"""
        with self._lock:
            if self._active_task_id:
                # 检查任务是否已经在冻结列表中（避免重复冻结）
                if self._active_task_id in self._frozen_tasks:
                    logger.debug(f"Task {self._active_task_id} already frozen, skipping")
                    self._active_task_id = None
                    return
                
                # 模拟创建快照
                snapshot = TaskSnapshot(
                    task_id=self._active_task_id,
                    context_history=[],
                    working_memory={},
                    execution_pointer="generating_step_2",
                    created_at=time.time(),
                    last_updated=time.time()
                )
                
                # 保存到冻结任务
                self._frozen_tasks[self._active_task_id] = snapshot
                
                # 推入栈
                self._task_stack.append(self._active_task_id)
                
                # 清除活跃任务
                logger.info(f"Task {self._active_task_id} frozen, pushed to stack")
                logger.info(f"Froze task {self._active_task_id} at generating_step_2")
                self._active_task_id = None
    
    def resume_task(self, task_id: str):
        """恢复任务
        
        Args:
            task_id: 任务 ID
        """
        with self._lock:
            if task_id in self._frozen_tasks:
                # 从冻结任务中取出
                snapshot = self._frozen_tasks.pop(task_id)
                
                # 如果有活跃任务，先冻结
                if self._active_task_id:
                    self.freeze_current()
                
                # 设置为活跃任务
                self._active_task_id = task_id
                
                # 从栈中弹出
                if task_id in self._task_stack:
                    self._task_stack.remove(task_id)
                
                # 同步到状态管理器 - 恢复任务时设置为 BUSY
                from zulong.core.state_manager import state_manager
                from zulong.core.types import L2Status
                state_manager.set_l2_status(L2Status.BUSY, task_id)
                
                logger.info(f"Task {task_id} resumed and set as active")
            else:
                logger.warning(f"Task {task_id} not found in frozen tasks")
    
    def get_current_context(self) -> List[Dict]:
        """获取当前上下文
        
        Returns:
            List[Dict]: 当前上下文
        """
        with self._lock:
            # 这里简化处理，实际应该返回真实的上下文
            return [ {"role": "user", "content": "Hello"} ]
    
    def get_active_task(self) -> Optional[str]:
        """获取当前活跃任务
        
        Returns:
            Optional[str]: 活跃任务 ID
        """
        with self._lock:
            return self._active_task_id
    
    def get_task_stack(self) -> List[str]:
        """获取任务栈
        
        Returns:
            List[str]: 任务栈
        """
        with self._lock:
            return self._task_stack.copy()


# 全局任务状态管理器实例
task_state_manager = TaskStateManager()
