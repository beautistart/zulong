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
            self._active_snapshot: Optional[TaskSnapshot] = None  # 活跃任务快照
            self._frozen_tasks: Dict[str, TaskSnapshot] = {}
            self._task_stack: List[str] = []
            self._lock = threading.RLock()  # 使用 RLock 避免嵌套调用死锁
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
            self._active_snapshot = snapshot
            
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
            if task_id == self._active_task_id and self._active_snapshot:
                # 更新活跃任务快照
                self._active_snapshot.last_updated = time.time()
                if vars_update:
                    self._active_snapshot.working_memory.update(vars_update)
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
                    self._active_snapshot = None
                    return
                
                # 使用活跃快照（如果存在），否则创建新快照
                if self._active_snapshot:
                    snapshot = self._active_snapshot
                    snapshot.last_updated = time.time()
                else:
                    snapshot = TaskSnapshot(
                        task_id=self._active_task_id,
                        context_history=[],
                        working_memory={},
                        execution_pointer="frozen",
                        created_at=time.time(),
                        last_updated=time.time()
                    )
                
                # 保存到冻结任务
                self._frozen_tasks[self._active_task_id] = snapshot
                
                # 推入栈
                self._task_stack.append(self._active_task_id)
                
                # 清除活跃任务
                logger.info(f"Task {self._active_task_id} frozen, pushed to stack")
                logger.info(f"Froze task {self._active_task_id} at {snapshot.execution_pointer}")
                self._active_task_id = None
                self._active_snapshot = None
    
    def resume_task(self, task_id: str, task_graph=None):
        """恢复任务
        
        Args:
            task_id: 任务 ID
            task_graph: 关联的 TaskGraph（可选，恢复时传入）
        """
        with self._lock:
            if task_id in self._frozen_tasks:
                # 从冻结任务中取出
                snapshot = self._frozen_tasks.pop(task_id)
                
                # 如果提供了 TaskGraph，关联到快照并同步到 MemoryGraph
                if task_graph:
                    snapshot.working_memory['task_graph'] = task_graph
                    self._sync_to_memory_graph(task_graph, task_id)
                    task_graph_id = getattr(task_graph, 'id', task_id)
                    logger.info(f"[TaskStateManager] 任务 {task_id} 已关联 TaskGraph: {task_graph_id}")
                    # 同步到 task_tools 的活跃 TaskGraph，保持两套状态一致
                    try:
                        from zulong.tools.task_tools import set_active_task_graph
                        _ws = task_graph.metadata.get("workspace_dir", "") if hasattr(task_graph, 'metadata') else ""
                        set_active_task_graph(task_graph, task_graph_id, workspace_dir=_ws)
                        logger.info(
                            f"[TaskStateManager] 已同步 TaskGraph {task_graph_id} 到 task_tools"
                        )
                    except Exception as e:
                        logger.warning(f"[TaskStateManager] task_tools 同步失败: {e}")
                
                # 如果有活跃任务，先冻结
                if self._active_task_id:
                    self.freeze_current()
                
                # 设置为活跃任务
                self._active_task_id = task_id
                self._active_snapshot = snapshot
                
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
    
    def _sync_to_memory_graph(self, task_graph, task_id):
        """同步 TaskGraph 到 MemoryGraph"""
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.memory.graph_adapters import TaskGraphAdapter
            from zulong.memory.memory_graph import EdgeType
            
            memory_graph = get_memory_graph()
            adapter = TaskGraphAdapter()
            adapter.sync(memory_graph, task_graph)
            
            # 建立对话与任务的关联边
            dialogue_id = f"dialogue:resume_{task_id}"
            task_root_id = f"task:{task_graph.id}" if hasattr(task_graph, 'id') else f"task:{task_id}"
            if memory_graph.has_node(dialogue_id) and memory_graph.has_node(task_root_id):
                memory_graph.add_edge(
                    dialogue_id, task_root_id,
                    EdgeType.REFERENCE, weight=0.9,
                    metadata={"binding_type": "resume_task"}
                )
            logger.info(f"[TaskStateManager] TaskGraph {task_graph.id} 已同步到 MemoryGraph")
        except Exception as e:
            logger.warning(f"[TaskStateManager] 同步到 MemoryGraph 失败: {e}")
    
    def get_current_context(self) -> List[Dict]:
        """获取当前上下文
        
        Returns:
            List[Dict]: 当前上下文（活跃任务的 context_history，无活跃任务返回空列表）
        """
        with self._lock:
            if self._active_snapshot and self._active_snapshot.context_history:
                return self._active_snapshot.context_history.copy()
            return []
    
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
