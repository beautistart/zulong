# File: zulong/l1b/hotswap_scheduler.py
"""
原子任务 2: L1-B 增强调度器 (双实例热备)
TSD v1.9: KV Cache 热切换机制 - 核心调度器

架构优化 v2.0:
1. 输出归一化：L2_BACKUP 结果必须移交 L2_PRIME 后再输出
2. L1-B 流式代理：所有输出经过 L1-B 中转
3. 紧急中断支持：毫秒级切断输出流
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Generator, Iterator
from enum import Enum
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    FROZEN = "FROZEN"
    MIGRATED = "MIGRATED"
    COMPLETED = "COMPLETED"
    WAITING_OUTPUT = "WAITING_OUTPUT"


class EmergencySignal(Enum):
    NONE = "NONE"
    STOP = "STOP"
    PAUSE = "PAUSE"
    RESUME = "RESUME"


@dataclass
class TaskContext:
    """任务上下文"""
    task_id: str
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    block_table: List[int] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)
    response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    engine_id: str = "PRIME"
    intermediate_result: Optional[str] = None
    needs_handover: bool = False


@dataclass
class HotSwapConfig:
    """热交换配置"""
    switch_threshold: int = 3
    max_active_tasks: int = 4
    max_freeze_stack_size: int = 10
    auto_resume_enabled: bool = True
    block_per_task: int = 4
    enable_parallel_compute: bool = True
    enable_streaming_proxy: bool = True


class MiniL2Engine:
    """模拟 L2 引擎 (用于测试逻辑)"""
    
    def __init__(self, device: str = "cuda", engine_id: str = "PRIME"):
        self.device = device
        self.engine_id = engine_id
        self.loaded_block_tables: List[List[int]] = []
        self.is_busy = False
        self._interrupt_flag = False
        
    def register_context_blocks(self, blocks: List[int]):
        self.loaded_block_tables.append(blocks)
        logger.info(f"[L2_{self.engine_id}] 注册了 {len(blocks)} 个 KV Cache 块: {blocks}")
    
    def unregister_context_blocks(self, blocks: List[int]):
        if blocks in self.loaded_block_tables:
            self.loaded_block_tables.remove(blocks)
            logger.info(f"[L2_{self.engine_id}] 注销了 {len(blocks)} 个 KV Cache 块")
    
    def generate(self, input_text: str, past_key_values: Optional[List[int]] = None) -> str:
        self.is_busy = True
        self._interrupt_flag = False
        try:
            blocks_info = f"使用 KV Blocks: {past_key_values}" if past_key_values else "无历史上下文"
            logger.info(f"[L2_{self.engine_id}] 正在生成回复... {blocks_info}")
            
            time.sleep(0.1)
            
            if self._interrupt_flag:
                logger.info(f"[L2_{self.engine_id}] 被中断")
                return f"[L2_{self.engine_id}] 任务被中断: {input_text[:30]}..."
            
            response = f"[L2_{self.engine_id}] 收到: {input_text[:50]}..."
            return response
        finally:
            self.is_busy = False
    
    def stream_generate(self, input_text: str, past_key_values: Optional[List[int]] = None) -> Iterator[str]:
        """流式生成 - 返回 Token 迭代器"""
        self.is_busy = True
        self._interrupt_flag = False
        
        try:
            logger.info(f"[L2_{self.engine_id}] 开始流式生成...")
            
            tokens = [
                f"[L2_{self.engine_id}] ",
                "收到: ",
                input_text[:20],
                "...",
                " 完成!"
            ]
            
            for token in tokens:
                if self._interrupt_flag:
                    logger.info(f"[L2_{self.engine_id}] 流式中断")
                    yield f"[中断]"
                    return
                
                time.sleep(0.05)
                yield token
                
        finally:
            self.is_busy = False
    
    def interrupt(self):
        """中断当前计算"""
        self._interrupt_flag = True
    
    def get_status(self) -> Dict:
        return {
            "engine_id": self.engine_id,
            "device": self.device,
            "is_busy": self.is_busy,
            "loaded_contexts": len(self.loaded_block_tables),
        }


class OutputGateway:
    """
    输出网关 - L1-B 的输出控制中心
    
    职责:
    1. 统一管理所有输出流
    2. 支持毫秒级紧急中断
    3. 将 Token 流转发给 L0 层
    """
    
    def __init__(self):
        self._emergency_signal = EmergencySignal.NONE
        self._output_buffer: List[str] = []
        self._lock = threading.Lock()
        self._output_callbacks: List[Callable[[str], None]] = []
        self._is_paused = False
    
    def register_output_callback(self, callback: Callable[[str], None]):
        """注册输出回调函数"""
        self._output_callbacks.append(callback)
    
    def emit(self, token: str) -> bool:
        """
        发射 Token 到输出流
        
        Returns:
            True: Token 已发送
            False: 被紧急信号切断
        """
        with self._lock:
            if self._emergency_signal == EmergencySignal.STOP:
                logger.info("[OutputGateway] 紧急停止信号，切断输出")
                return False
            
            if self._is_paused:
                self._output_buffer.append(token)
                return True
            
            self._output_buffer.append(token)
            
            for callback in self._output_callbacks:
                try:
                    callback(token)
                except Exception as e:
                    logger.error(f"[OutputGateway] 回调执行失败: {e}")
            
            return True
    
    def trigger_emergency(self, signal: EmergencySignal):
        """触发紧急信号"""
        with self._lock:
            self._emergency_signal = signal
            logger.info(f"[OutputGateway] 紧急信号: {signal.value}")
            
            if signal == EmergencySignal.STOP:
                self._output_buffer.clear()
    
    def clear_emergency(self):
        """清除紧急信号"""
        with self._lock:
            self._emergency_signal = EmergencySignal.NONE
    
    def pause(self):
        """暂停输出"""
        with self._lock:
            self._is_paused = True
    
    def resume(self):
        """恢复输出"""
        with self._lock:
            self._is_paused = False
            for token in self._output_buffer:
                for callback in self._output_callbacks:
                    callback(token)
            self._output_buffer.clear()
    
    def get_status(self) -> Dict:
        return {
            "emergency_signal": self._emergency_signal.value,
            "is_paused": self._is_paused,
            "buffer_size": len(self._output_buffer),
        }


class L1B_HotSwapScheduler:
    """
    TSD v1.9 L1-B Plugin
    KV Cache 热交换调度器 - 流式代理架构
    
    架构优化 v2.0:
    - L2_PRIME: 负责思考（生成 Token）
    - L2_BACKUP: 冷存储（不直接输出）
    - L1-B: 流式代理（控制输出，支持紧急中断）
    """
    
    def __init__(
        self,
        kv_pool,
        config: Optional[HotSwapConfig] = None,
    ):
        self.kv_pool = kv_pool
        self.config = config or HotSwapConfig()
        
        self.L2_PRIME: Optional[MiniL2Engine] = None
        self.L2_BACKUP: Optional[MiniL2Engine] = None
        
        self.active_tasks: Dict[str, TaskContext] = {}
        self.freeze_stack: List[TaskContext] = []
        self.task_counter = 0
        
        self._hot_swap_count = 0
        self._initialized = False
        
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()
        
        self.backup_task_queue: List[TaskContext] = []
        self._backup_running = False
        
        self.output_gateway = OutputGateway()
        
        self._pending_handover_tasks: List[TaskContext] = []
    
    def initialize(self) -> bool:
        if self._initialized:
            logger.warning("[L1B_HotSwapScheduler] 已经初始化，跳过")
            return True
        
        try:
            device = "cuda" if hasattr(self.kv_pool, 'device') else "cpu"
            
            self.L2_PRIME = MiniL2Engine(device=device, engine_id="PRIME")
            self.L2_BACKUP = MiniL2Engine(device=device, engine_id="BACKUP")
            
            self._initialized = True
            logger.info(
                f"[L1B_HotSwapScheduler] 初始化完成: "
                f"阈值={self.config.switch_threshold}, "
                f"流式代理={self.config.enable_streaming_proxy}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"[L1B_HotSwapScheduler] 初始化失败: {e}")
            return False
    
    def set_real_engine(self, engine, engine_id: str = "PRIME"):
        if engine_id == "PRIME":
            self.L2_PRIME = engine
            logger.info("[L1B_HotSwapScheduler] 设置了真实的 L2_PRIME 引擎")
        elif engine_id == "BACKUP":
            self.L2_BACKUP = engine
            logger.info("[L1B_HotSwapScheduler] 设置了真实的 L2_BACKUP 引擎")
    
    def register_output_callback(self, callback: Callable[[str], None]):
        """注册输出回调 - 用于将 Token 发送到 L0 层"""
        self.output_gateway.register_output_callback(callback)
    
    def trigger_emergency_stop(self):
        """触发紧急停止 - 毫秒级切断输出"""
        self.output_gateway.trigger_emergency(EmergencySignal.STOP)
        
        if self.L2_PRIME:
            self.L2_PRIME.interrupt()
        if self.L2_BACKUP:
            self.L2_BACKUP.interrupt()
        
        logger.info("[L1B_HotSwapScheduler] 紧急停止已触发")
    
    def on_user_input_stream(self, prompt: str, metadata: Optional[Dict] = None) -> Iterator[str]:
        """
        流式接口：统一由 L1-B 掌控输出
        
        架构优化 v2.0:
        - L1-B 作为流式代理
        - 支持毫秒级紧急中断
        - 所有输出经过 L1-B 中转
        """
        if not self._initialized:
            raise RuntimeError("调度器未初始化，请先调用 initialize()")
        
        self.output_gateway.clear_emergency()
        
        with self._lock:
            self.task_counter += 1
            task_id = f"TASK_{self.task_counter}"
            
            if self.L2_PRIME and self.L2_PRIME.is_busy:
                logger.info(f"[L1B_HotSwapScheduler] 中断当前任务，接收新任务 {task_id}")
                self.L2_PRIME.interrupt()
                
                for tid, ctx in self.active_tasks.items():
                    if ctx.status == TaskStatus.RUNNING:
                        ctx.status = TaskStatus.INTERRUPTED
            
            new_context = TaskContext(
                task_id=task_id,
                prompt=prompt,
                status=TaskStatus.PENDING,
                metadata=metadata or {},
            )
            
            if len(self.active_tasks) >= self.config.switch_threshold:
                self._trigger_hot_swap()
        
        try:
            blocks = self.kv_pool.allocate_for_task(
                task_id, 
                self.config.block_per_task
            )
            new_context.block_table = blocks
        except RuntimeError as e:
            logger.error(f"[L1B_HotSwapScheduler] KV Cache 分配失败: {e}")
            with self._lock:
                self.task_counter -= 1
            yield f"[系统] 资源不足，请稍后重试: {e}"
            return
        
        with self._lock:
            new_context.status = TaskStatus.RUNNING
            new_context.engine_id = "PRIME"
            self.active_tasks[task_id] = new_context
        
        full_response = []
        
        if hasattr(self.L2_PRIME, 'stream_generate'):
            token_generator = self.L2_PRIME.stream_generate(
                input_text=prompt,
                past_key_values=blocks,
            )
            
            for token in token_generator:
                if not self.output_gateway.emit(token):
                    logger.info("[L1B_HotSwapScheduler] 输出被紧急信号切断")
                    full_response.append("[被中断]")
                    break
                
                full_response.append(token)
                yield token
        else:
            response = self.L2_PRIME.generate(
                input_text=prompt,
                past_key_values=blocks,
            )
            full_response.append(response)
            self.output_gateway.emit(response)
            yield response
        
        with self._lock:
            if new_context.status == TaskStatus.INTERRUPTED:
                new_context.response = "".join(full_response)
            else:
                new_context.status = TaskStatus.COMPLETED
                new_context.response = "".join(full_response)
    
    def on_user_input(self, prompt: str, metadata: Optional[Dict] = None) -> str:
        """
        非流式接口 - 兼容旧代码
        """
        full_response = []
        for token in self.on_user_input_stream(prompt, metadata):
            full_response.append(token)
        return "".join(full_response)
    
    def _trigger_hot_swap(self):
        """
        热交换 - 输出归一化版本
        
        关键改动:
        - L2_BACKUP 计算完成后，结果标记为 WAITING_OUTPUT
        - 不直接输出，等待 L2_PRIME 接管
        """
        logger.info("[L1B_HotSwapScheduler] 执行 L2 实例热交换...")
        start_time = time.time()
        
        migrated_count = 0
        tasks_to_migrate = []
        
        for task_id, context in list(self.active_tasks.items()):
            context.status = TaskStatus.MIGRATED
            context.engine_id = "BACKUP"
            context.needs_handover = True
            self._bind_to_backup(context.block_table)
            tasks_to_migrate.append(context)
            migrated_count += 1
        
        if len(self.freeze_stack) + migrated_count > self.config.max_freeze_stack_size:
            evict_count = len(self.freeze_stack) + migrated_count - self.config.max_freeze_stack_size
            self._evict_oldest_frozen(evict_count)
        
        self.freeze_stack.extend(tasks_to_migrate)
        self.active_tasks.clear()
        
        if self.config.enable_parallel_compute and tasks_to_migrate:
            self._start_backup_computation(tasks_to_migrate)
        
        self._hot_swap_count += 1
        elapsed = (time.time() - start_time) * 1000
        
        logger.info(
            f"[L1B_HotSwapScheduler] 热交换完成。"
            f"迁移了 {migrated_count} 个任务，耗时 {elapsed:.2f}ms。"
            f"结果将移交 L2_PRIME 后输出。"
        )
    
    def _start_backup_computation(self, tasks: List[TaskContext]):
        """
        启动 L2_BACKUP 计算 - 输出归一化版本
        
        关键改动:
        - 计算结果存储到 intermediate_result
        - 状态设为 WAITING_OUTPUT
        - 不直接输出
        """
        logger.info(f"[L1B_HotSwapScheduler] 启动 L2_BACKUP 计算 {len(tasks)} 个冻结任务 (结果不直接输出)")
        
        self.backup_task_queue.extend(tasks)
        
        if not self._backup_running:
            self._backup_running = True
            self._executor.submit(self._backup_compute_loop_with_handover)
    
    def _backup_compute_loop_with_handover(self):
        """
        L2_BACKUP 计算循环 - 输出归一化版本
        
        关键改动:
        - 结果存储到 intermediate_result
        - 标记为 WAITING_OUTPUT
        - 等待 L2_PRIME 接管后输出
        """
        logger.info("[L2_BACKUP] 开始后台计算 (结果不直接输出)")
        
        while self.backup_task_queue:
            with self._lock:
                if not self.backup_task_queue:
                    break
                task = self.backup_task_queue.pop(0)
            
            logger.info(f"[L2_BACKUP] 计算任务: {task.task_id} (结果将移交 L2_PRIME)")
            
            try:
                response = self.L2_BACKUP.generate(
                    input_text=task.prompt,
                    past_key_values=task.block_table,
                )
                
                with self._lock:
                    task.intermediate_result = response
                    task.status = TaskStatus.WAITING_OUTPUT
                    self._pending_handover_tasks.append(task)
                
                logger.info(f"[L2_BACKUP] 任务 {task.task_id} 计算完成，等待 L2_PRIME 接管输出")
                
            except Exception as e:
                logger.error(f"[L2_BACKUP] 计算任务 {task.task_id} 失败: {e}")
        
        self._backup_running = False
        logger.info("[L2_BACKUP] 后台计算完成")
    
    def get_pending_handover_tasks(self) -> List[TaskContext]:
        """获取等待移交的任务列表"""
        with self._lock:
            return list(self._pending_handover_tasks)
    
    def handover_and_output(self, task: TaskContext) -> Iterator[str]:
        """
        移交任务并输出 - 输出归一化
        
        TSD v2.0 规范:
        - L2_BACKUP 的结果直接输出
        - 不需要 L2_PRIME 重新生成
        - 通过 OutputGateway 统一输出
        """
        if task.intermediate_result is None:
            yield "[错误] 任务无中间结果"
            return
        
        logger.info(f"[L1B_HotSwapScheduler] 移交任务 {task.task_id} 到 L2_PRIME 输出")
        
        full_response = []
        
        for char in task.intermediate_result:
            if self.output_gateway._emergency_signal == EmergencySignal.STOP:
                logger.info("[L1B_HotSwapScheduler] 输出被紧急信号切断")
                full_response.append("[被中断]")
                break
            
            if not self.output_gateway.emit(char):
                logger.info("[L1B_HotSwapScheduler] 输出被紧急信号切断")
                full_response.append("[被中断]")
                break
            
            full_response.append(char)
            yield char
        
        with self._lock:
            task.status = TaskStatus.COMPLETED
            task.response = "".join(full_response)
            if task in self._pending_handover_tasks:
                self._pending_handover_tasks.remove(task)
    
    def _bind_to_backup(self, block_table: List[int]):
        if self.L2_BACKUP is None:
            logger.warning("[L1B_HotSwapScheduler] L2_BACKUP 未初始化")
            return
        self.L2_BACKUP.register_context_blocks(block_table)
    
    def _evict_oldest_frozen(self, count: int):
        for _ in range(count):
            if not self.freeze_stack:
                break
            
            oldest = self.freeze_stack.pop(0)
            self.kv_pool.free_task(oldest.task_id)
            
            if self.L2_BACKUP:
                self.L2_BACKUP.unregister_context_blocks(oldest.block_table)
            
            logger.info(f"[L1B_HotSwapScheduler] 驱逐冻结任务: {oldest.task_id}")
    
    def wait_for_all_complete(self, timeout: float = 60.0) -> Dict[str, str]:
        """等待所有任务完成"""
        logger.info("[L1B_HotSwapScheduler] 等待所有任务完成...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                all_completed = all(
                    ctx.status in [TaskStatus.COMPLETED, TaskStatus.WAITING_OUTPUT]
                    for ctx in self.freeze_stack
                )
                
                if all_completed and not self._backup_running:
                    break
            
            time.sleep(0.1)
        
        results = {}
        with self._lock:
            for ctx in self.freeze_stack:
                if ctx.response:
                    results[ctx.task_id] = ctx.response
                elif ctx.intermediate_result:
                    results[ctx.task_id] = ctx.intermediate_result
            for ctx in self.active_tasks.values():
                if ctx.response:
                    results[ctx.task_id] = ctx.response
        
        logger.info(f"[L1B_HotSwapScheduler] 所有任务完成，共 {len(results)} 个结果")
        return results
    
    def resume_frozen_task(self, keyword: str) -> Dict:
        """恢复冻结的任务"""
        result = {
            "found": False,
            "task_id": None,
            "resume_delay_ms": 0.0,
            "response": None,
        }
        
        resume_start = time.time()
        
        with self._lock:
            for ctx in self.freeze_stack:
                if keyword.lower() in ctx.prompt.lower():
                    result["found"] = True
                    result["task_id"] = ctx.task_id
                    
                    resume_delay = (time.time() - resume_start) * 1000
                    result["resume_delay_ms"] = resume_delay
                    
                    logger.info(
                        f"[L1B_HotSwapScheduler] 找到冻结任务: {ctx.task_id}, "
                        f"恢复延迟: {resume_delay:.2f}ms"
                    )
                    
                    if ctx.response:
                        result["response"] = ctx.response
                    elif ctx.intermediate_result:
                        result["response"] = ctx.intermediate_result
                    
                    return result
        
        resume_delay = (time.time() - resume_start) * 1000
        result["resume_delay_ms"] = resume_delay
        
        return result
    
    def get_frozen_task_by_keyword(self, keyword: str) -> Optional[TaskContext]:
        """通过关键词查找冻结任务"""
        with self._lock:
            for ctx in self.freeze_stack:
                if keyword.lower() in ctx.prompt.lower():
                    return ctx
        return None
    
    def get_statistics(self) -> Dict:
        with self._lock:
            return {
                "initialized": self._initialized,
                "task_counter": self.task_counter,
                "active_tasks": len(self.active_tasks),
                "frozen_tasks": len(self.freeze_stack),
                "hot_swap_count": self._hot_swap_count,
                "switch_threshold": self.config.switch_threshold,
                "backup_queue_size": len(self.backup_task_queue),
                "backup_running": self._backup_running,
                "pending_handover": len(self._pending_handover_tasks),
                "l2_prime_status": self.L2_PRIME.get_status() if self.L2_PRIME else None,
                "l2_backup_status": self.L2_BACKUP.get_status() if self.L2_BACKUP else None,
                "output_gateway": self.output_gateway.get_status(),
                "active_task_ids": list(self.active_tasks.keys()),
                "frozen_task_ids": [t.task_id for t in self.freeze_stack],
            }
    
    def shutdown(self):
        with self._lock:
            for task_id in list(self.active_tasks.keys()):
                self.kv_pool.free_task(task_id)
            
            for context in self.freeze_stack:
                self.kv_pool.free_task(context.task_id)
            
            self.active_tasks.clear()
            self.freeze_stack.clear()
            self.backup_task_queue.clear()
            self._pending_handover_tasks.clear()
        
        self._executor.shutdown(wait=True)
        
        self.L2_PRIME = None
        self.L2_BACKUP = None
        
        self._initialized = False
        logger.info("[L1B_HotSwapScheduler] 已关闭")
