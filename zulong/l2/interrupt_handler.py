# File: zulong/l2/interrupt_handler.py
# L2 中断处理器 - 处理生成过程中的中断信号
# 对应 TSD v1.7: L2 生成循环中监听 interrupt_flag，触发时保存快照并停止生成

from .snapshot_manager import snapshot_manager
from .task_snapshot import TaskSnapshot, TaskStatus
from .environment_snapshot import environment_snapshot_manager, EnvironmentChange
from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent
from zulong.core.state_manager import state_manager
from typing import Dict, Any, Optional, Callable, List
import threading
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='[L2-Interrupt] %(message)s')
logger = logging.getLogger(__name__)


class InterruptHandler:
    """L2 中断处理器
    
    类比：就像一位正在讲故事的讲述者，当有人紧急打断时：
    1. 记住讲到哪里（保存执行指针）
    2. 暂停讲述（停止生成）
    3. 处理打断者的问题（处理新任务）
    4. 决定：继续原来的故事，还是重新开始（重评估）
    
    核心职责：
    1. 监听中断信号 (interrupt_flag)
    2. 触发时保存当前任务快照 (freeze)
    3. 停止当前生成过程
    4. 协调恢复流程
    """
    
    def __init__(self):
        """初始化中断处理器"""
        self._interrupt_callbacks: List[Callable] = []
        self._resume_callbacks: List[Callable] = []
        self._is_generating = False
        self._current_task_id: Optional[str] = None
        self._lock = threading.RLock()
        
        # 注册事件监听
        self._register_event_handlers()
        
        logger.info("L2 InterruptHandler initialized")
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        event_bus.subscribe(EventType.SYSTEM_INTERRUPT, self._on_interrupt_event, "L2")
    
    def _on_interrupt_event(self, event: ZulongEvent):
        """处理系统中断事件"""
        logger.info(f"Received interrupt event: {event.payload}")
        reason = event.payload.get("reason", "unknown")
        
        # 触发中断处理
        self.handle_interrupt(reason)
    
    def register_interrupt_callback(self, callback: Callable):
        """注册中断回调函数
        
        Args:
            callback: 中断时调用的函数，接收 reason 参数
        """
        self._interrupt_callbacks.append(callback)
    
    def register_resume_callback(self, callback: Callable):
        """注册恢复回调函数
        
        Args:
            callback: 恢复时调用的函数，接收 snapshot 参数
        """
        self._resume_callbacks.append(callback)
    
    def start_generation(self, task_id: str) -> bool:
        """开始生成 - 标记进入生成状态
        
        Args:
            task_id: 当前任务 ID
            
        Returns:
            bool: 是否成功开始
        """
        with self._lock:
            if self._is_generating:
                logger.warning(f"Already generating for task: {self._current_task_id}")
                return False
            
            self._is_generating = True
            self._current_task_id = task_id
            
            # 清除中断标志
            state_manager.set_interrupt_flag(False)
            
            logger.info(f"Started generation for task: {task_id}")
            return True
    
    def stop_generation(self) -> Optional[str]:
        """停止生成 - 标记退出生成状态
        
        Returns:
            str: 停止的任务 ID，如果没有正在生成的任务则返回 None
        """
        with self._lock:
            if not self._is_generating:
                return None
            
            task_id = self._current_task_id
            self._is_generating = False
            self._current_task_id = None
            
            logger.info(f"Stopped generation for task: {task_id}")
            return task_id
    
    def check_interrupt(self) -> bool:
        """检查是否收到中断信号
        
        在生成循环中定期调用此函数来检查中断
        
        Returns:
            bool: 是否收到中断信号
        """
        # 检查全局中断标志
        if state_manager.get_interrupt_flag():
            return True
        
        # 也可以检查其他中断条件
        # 例如：检查是否有高优先级事件
        
        return False
    
    def handle_interrupt(self, reason: str = "unknown") -> Optional[TaskSnapshot]:
        """处理中断 - 核心方法
        
        流程：
        1. 设置中断标志
        2. 停止生成
        3. 冻结当前任务快照
        4. 通知回调函数
        
        Args:
            reason: 中断原因
            
        Returns:
            TaskSnapshot: 冻结后的快照，如果没有正在生成的任务则返回 None
        """
        with self._lock:
            # 1. 设置中断标志
            state_manager.set_interrupt_flag(True)
            
            # 2. 检查是否有正在生成的任务
            if not self._is_generating or not self._current_task_id:
                logger.info("No active generation to interrupt")
                return None
            
            task_id = self._current_task_id
            logger.info(f"Handling interrupt for task: {task_id}, reason: {reason}")
            
            # 3. 停止生成
            self._is_generating = False
            
            # 4. 冻结当前任务快照
            snapshot = snapshot_manager.freeze(task_id)
            
            if snapshot:
                logger.info(f"Task frozen: {snapshot.get_summary()}")
                
                # 5. 通知中断回调
                for callback in self._interrupt_callbacks:
                    try:
                        callback(reason, snapshot)
                    except Exception as e:
                        logger.error(f"Error in interrupt callback: {e}")
            
            self._current_task_id = None
            
            return snapshot
    
    def resume_generation(self, task_id: str) -> Optional[TaskSnapshot]:
        """恢复生成 - 从冻结状态恢复
        
        TSD v1.7 对应:
        - 5.3 重评估机制：恢复任务前对比环境快照
        - 5.4 执行前确认：检查物体状态、用户位置、任务条件
        
        流程:
        1. 解冻任务
        2. 创建当前环境快照
        3. 对比冻结时的环境快照
        4. 根据变化程度决定：CONTINUE / REPLAN / ABORT
        5. 如果 CONTINUE，恢复生成
        
        Args:
            task_id: 要恢复的任务 ID
            
        Returns:
            TaskSnapshot: 恢复后的快照，如果任务不存在则返回 None
        """
        with self._lock:
            # 1. 解冻任务
            snapshot = snapshot_manager.thaw(task_id)
            
            if not snapshot:
                logger.warning(f"Cannot resume: task {task_id} not found")
                return None
            
            logger.info(f"Resuming generation for task: {task_id}")
            
            # 2. 🎯 重评估：对比环境快照
            re_eval_result = self._re_evaluate_environment(task_id, snapshot)
            
            if re_eval_result.recommendation == "ABORT":
                logger.warning(f"环境变化过大，任务中止：{re_eval_result.changes}")
                # 发布任务中止通知
                event_bus.publish(ZulongEvent(
                    type=EventType.TASK_ABORTED,
                    priority=EventPriority.HIGH,
                    source="L2/InterruptHandler",
                    payload={
                        "task_id": task_id,
                        "reason": "environment_changed",
                        "changes": re_eval_result.changes
                    }
                ))
                return None
            
            elif re_eval_result.recommendation == "REPLAN":
                logger.info(f"环境发生变化，需要重新规划：{re_eval_result.changes}")
                # 发布重新规划请求
                event_bus.publish(ZulongEvent(
                    type=EventType.REPLAN_REQUEST,
                    priority=EventPriority.NORMAL,
                    source="L2/InterruptHandler",
                    payload={
                        "task_id": task_id,
                        "reason": "environment_changed",
                        "changes": re_eval_result.changes,
                        "severity": re_eval_result.severity
                    }
                ))
                # 继续恢复，但 L2 应该重新规划
            
            else:
                logger.info(f"环境无重大变化，继续执行")
            
            # 3. 设置当前任务
            self._current_task_id = task_id
            self._is_generating = True
            
            # 4. 清除中断标志
            state_manager.set_interrupt_flag(False)
            
            # 5. 通知恢复回调
            for callback in self._resume_callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    logger.error(f"Error in resume callback: {e}")
            
            return snapshot
    
    def _re_evaluate_environment(self, task_id: str, snapshot: TaskSnapshot) -> EnvironmentChange:
        """重评估环境 - 对比冻结时和当前的环境状态
        
        TSD v1.7 对应:
        - 5.3 重评估机制
        - 检测：物体掉落、用户移动、任务条件消失
        
        Args:
            task_id: 任务 ID
            snapshot: 任务快照
            
        Returns:
            EnvironmentChange: 环境变化检测结果
        """
        logger.info(f"Re-evaluating environment for task: {task_id}")
        
        # 1. 获取冻结时的环境快照（如果有）
        old_snapshot = snapshot.environment_snapshot if hasattr(snapshot, 'environment_snapshot') else None
        
        if not old_snapshot:
            logger.warning("No environment snapshot found, assuming no changes")
            # 创建一个新的快照作为基准
            old_snapshot = environment_snapshot_manager.create_snapshot(task_id)
        
        # 2. 🎯 创建当前环境快照
        # TODO: 实际实现应该从视觉/听觉系统获取数据
        new_snapshot = environment_snapshot_manager.create_snapshot(task_id)
        
        # 示例：填充实际数据（后续会由视觉系统提供）
        # from zulong.l1a.vision_processor import vision_processor
        # if vision_processor.get_latest_frame() is not None:
        #     new_snapshot.scene_description = "active"
        #     new_snapshot.add_object(...)
        
        # 3. 对比两个快照
        change_result = environment_snapshot_manager.compare_snapshots(old_snapshot, new_snapshot)
        
        logger.info(f"Environment changes detected: {change_result.changes}")
        logger.info(f"Severity: {change_result.severity}, Recommendation: {change_result.recommendation}")
        
        return change_result
    
    def is_generating(self) -> bool:
        """检查是否正在生成"""
        with self._lock:
            return self._is_generating
    
    def get_current_task_id(self) -> Optional[str]:
        """获取当前任务 ID"""
        with self._lock:
            return self._current_task_id


class GenerationLoop:
    """生成循环包装器 - 集成中断检查的生成循环
    
    使用示例：
        loop = GenerationLoop(interrupt_handler)
        for output in loop.generate(task_id, prompt, model):
            print(output)
    """
    
    def __init__(self, interrupt_handler: InterruptHandler):
        """初始化生成循环
        
        Args:
            interrupt_handler: 中断处理器实例
        """
        self.interrupt_handler = interrupt_handler
        self._interrupted = False
    
    def generate(
        self,
        task_id: str,
        prompt: str,
        model,
        max_tokens: int = 100,
        update_interval: int = 5
    ):
        """生成文本，支持中断检查
        
        Args:
            task_id: 任务 ID
            prompt: 输入提示
            model: 模型实例（需要有 generate_stream 方法）
            max_tokens: 最大生成 token 数
            update_interval: 检查中断的间隔（每多少 token 检查一次）
            
        Yields:
            dict: 生成结果，包含 text 和 interrupted 字段
        """
        # 标记开始生成
        if not self.interrupt_handler.start_generation(task_id):
            yield {"text": "", "interrupted": False, "error": "Failed to start generation"}
            return
        
        self._interrupted = False
        generated_text = ""
        token_count = 0
        
        try:
            # 模拟流式生成
            for i in range(max_tokens):
                # 定期检查中断
                if i % update_interval == 0:
                    if self.interrupt_handler.check_interrupt():
                        self._interrupted = True
                        logger.info(f"Generation interrupted at token {i}")
                        break
                
                # 模拟生成一个 token
                # 实际应用中，这里调用模型的 generate_stream 方法
                token = self._simulate_token_generation(prompt, i)
                generated_text += token
                token_count += 1
                
                # 更新任务快照中的进度
                snapshot_manager.update_snapshot(
                    task_id=task_id,
                    execution_pointer={
                        "current_step": token_count,
                        "step_description": f"Generating token {token_count}/{max_tokens}",
                        "generated_content": generated_text
                    }
                )
                
                yield {
                    "text": token,
                    "interrupted": False,
                    "token_count": token_count
                }
        
        finally:
            # 标记停止生成
            self.interrupt_handler.stop_generation()
            
            # 如果被打断，更新最终状态
            if self._interrupted:
                yield {
                    "text": "",
                    "interrupted": True,
                    "token_count": token_count,
                    "final_text": generated_text
                }
    
    def _simulate_token_generation(self, prompt: str, index: int) -> str:
        """模拟 token 生成（用于测试）"""
        # 实际应用中，这里调用真实模型
        words = ["Hello", " ", "world", ".", " ", "This", " ", "is", " ", "a", " ", "test", "."]
        return words[index % len(words)]
    
    def was_interrupted(self) -> bool:
        """检查生成是否被中断"""
        return self._interrupted


# 全局中断处理器实例
interrupt_handler = InterruptHandler()
