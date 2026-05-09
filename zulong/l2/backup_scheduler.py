# File: zulong/l2/backup_scheduler.py
"""
L2-BACKUP 智能调度器 - TSD v2.4

功能：
1. 监听 L2-PRIME 状态
2. 在 L2-PRIME 空闲时触发后台复盘
3. 智能管理复盘任务队列
4. 支持 Map-Reduce 分步摘要

对应 TSD v2.4: L2-BACKUP 智能调度、资源优化
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from zulong.core.types import L2Status
import queue

logger = logging.getLogger(__name__)


class SummarizationTask:
    """复盘任务"""
    
    def __init__(self, 
                 task_id: str,
                 conversation_turns: List[Dict[str, str]],
                 priority: int = 1,
                 created_at: float = None):
        """
        初始化复盘任务
        
        Args:
            task_id: 任务 ID
            conversation_turns: 对话轮次列表
            priority: 优先级（0-9，0 最高）
            created_at: 创建时间
        """
        self.task_id = task_id
        self.conversation_turns = conversation_turns
        self.priority = priority
        self.created_at = created_at or time.time()
        self.status = "pending"  # pending, running, completed, failed
        self.result = None
        self.error = None
    
    def __lt__(self, other):
        """用于优先级队列排序"""
        return self.priority < other.priority


class L2BackupScheduler:
    """
    L2-BACKUP 智能调度器
    
    核心职责：
    1. 监听 L2-PRIME 状态
    2. 在空闲时触发后台复盘
    3. 管理复盘任务队列
    4. 支持 Map-Reduce 分步摘要
    """
    
    def __init__(self, 
                 l2_prime_endpoint: str = "http://localhost:8000",
                 l2_backup_endpoint: str = "http://localhost:8001"):
        """
        初始化调度器
        
        Args:
            l2_prime_endpoint: L2-PRIME 端点
            l2_backup_endpoint: L2-BACKUP 端点
        """
        self.l2_prime_endpoint = l2_prime_endpoint
        self.l2_backup_endpoint = l2_backup_endpoint
        
        # L2 状态监控
        self.l2_prime_status = L2Status.IDLE
        self.l2_backup_status = L2Status.IDLE
        
        # 任务队列
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._running_tasks: Dict[str, SummarizationTask] = {}
        self._completed_tasks: List[SummarizationTask] = []
        
        # 调度器状态
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # 🔥 新增：任务中断控制
        self._current_task: Optional[asyncio.Task] = None  # 当前正在执行的异步任务
        self._interrupt_flag = False  # 中断标志
        
        # 回调函数
        self._on_summarization_complete = None
        
        # 统计信息
        self._stats = {
            "total_tasks_received": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "total_tasks_interrupted": 0,  # 🔥 新增：被中断的任务数
            "avg_processing_time": 0.0,
            "current_queue_size": 0
        }
        
        # Map-Reduce 状态
        self._map_reduce_buffer: List[Dict] = []
        self._map_reduce_threshold = 10  # 累积 10 轮触发 Reduce
        
        logger.info("[L2BackupScheduler] 初始化完成")
    
    def update_l2_prime_status(self, status: L2Status):
        """更新 L2-PRIME 状态"""
        old_status = self.l2_prime_status
        self.l2_prime_status = status
        
        if old_status != status:
            logger.info(f"[L2BackupScheduler] L2-PRIME 状态变更：{old_status.value} → {status.value}")
    
    def update_l2_backup_status(self, status: L2Status):
        """更新 L2-BACKUP 状态"""
        old_status = self.l2_backup_status
        self.l2_backup_status = status
        
        if old_status != status:
            logger.info(f"[L2BackupScheduler] L2-BACKUP 状态变更：{old_status.value} → {status.value}")
    
    async def submit_summarization_task(self, 
                                       conversation_turns: List[Dict[str, str]],
                                       priority: int = 1) -> str:
        """
        提交复盘任务
        
        Args:
            conversation_turns: 对话轮次列表
            priority: 优先级（0-9，0 最高）
            
        Returns:
            str: 任务 ID
        """
        task_id = f"task_{int(time.time() * 1000)}"
        task = SummarizationTask(
            task_id=task_id,
            conversation_turns=conversation_turns,
            priority=priority
        )
        
        # 加入队列
        self._task_queue.put(task)
        self._running_tasks[task_id] = task
        self._stats["total_tasks_received"] += 1
        self._stats["current_queue_size"] = self._task_queue.qsize()
        
        logger.info(f"[L2BackupScheduler] 提交复盘任务：{task_id}, 优先级：{priority}, 轮次：{len(conversation_turns)}")
        
        return task_id
    
    async def _process_task(self, task: SummarizationTask):
        """
        处理复盘任务
        
        Args:
            task: 复盘任务
        """
        try:
            task.status = "running"
            start_time = time.time()
            
            logger.info(f"[L2BackupScheduler] 开始处理任务：{task.task_id}")
            
            # 🔥 调用 L2-BACKUP 进行复盘（支持中断检查）
            summary = await self._call_l2_backup_with_interrupt_check(task.conversation_turns)
            
            # 🔥 检查是否被中断
            if self._interrupt_flag:
                logger.info(f"[L2BackupScheduler] 任务被中断：{task.task_id}")
                task.status = "interrupted"
                self._stats["total_tasks_interrupted"] += 1
                return
            
            # 更新任务状态
            task.status = "completed"
            task.result = summary
            processing_time = time.time() - start_time
            
            # 更新统计
            self._stats["total_tasks_completed"] += 1
            self._stats["avg_processing_time"] = (
                self._stats["avg_processing_time"] * (self._stats["total_tasks_completed"] - 1) 
                + processing_time
            ) / self._stats["total_tasks_completed"]
            
            logger.info(f"[L2BackupScheduler] 任务完成：{task.task_id}, 耗时：{processing_time:.2f}秒")
            
            # 🔥 修复：任务完成后立即回写审查结果
            if task.conversation_turns:
                meta = task.conversation_turns[0].get("_meta", {}) if task.conversation_turns else {}
                review_task_id = meta.get("review_task_id")
                if review_task_id and task.result:
                    try:
                        from zulong.memory.llm_memory_reviewer import get_llm_memory_reviewer
                        reviewer = get_llm_memory_reviewer()
                        if reviewer:
                            original_memories = meta.get("original_memories", [])
                            context = meta.get("context", {})
                            import asyncio as _aio
                            # 🔥 增强日志
                            logger.info(f"[L2BackupScheduler] 开始回写审查结果: {review_task_id}, 原始记忆数={len(original_memories)}, context={context}")
                            await reviewer.process_review_result(
                                task_id=review_task_id,
                                llm_response=task.result,
                                original_memories=original_memories,
                                context=context,
                            )
                            logger.info(f"[L2BackupScheduler] ✅ 审查结果已回写: {review_task_id}")
                    except Exception as re:
                        logger.warning(f"[L2BackupScheduler] 审查结果回写失败: {re}", exc_info=True)
            
            # 回调
            if self._on_summarization_complete:
                await self._on_summarization_complete(task)
                
        except asyncio.CancelledError:
            task.status = "interrupted"
            self._stats["total_tasks_interrupted"] += 1
            logger.info(f"[L2BackupScheduler] 任务被取消：{task.task_id}")
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self._stats["total_tasks_failed"] += 1
            logger.error(f"[L2BackupScheduler] 任务失败：{task.task_id}, 错误：{e}", exc_info=True)
        
        finally:
            # 从运行中移除
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]
            
            # 加入已完成列表
            self._completed_tasks.append(task)
            
            # 限制已完成列表大小
            if len(self._completed_tasks) > 100:
                self._completed_tasks.pop(0)
            
            # 🔥 修复：任务完成后重置L2-BACKUP状态为IDLE
            self.update_l2_backup_status(L2Status.IDLE)
            logger.info(f"[L2BackupScheduler] L2-BACKUP状态重置为IDLE，可处理下一个任务")
    
    async def _call_l2_backup(self, conversation_turns: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        调用 L2-BACKUP 进行复盘
        
        Args:
            conversation_turns: 对话轮次列表
            
        Returns:
            Dict: 复盘结果（摘要、关键词等）
        """
        try:
            # 🔥 方案 1: 使用 HTTP 调用 L2-BACKUP API
            # import aiohttp
            # async with aiohttp.ClientSession() as session:
            #     async with session.post(
            #         f"{self.l2_backup_endpoint}/summarize",
            #         json={"conversation": conversation_turns}
            #     ) as response:
            #         result = await response.json()
            #         return result
            
            # 🔥 方案 2: 使用项目现有的推理引擎
            from zulong.l2.inference_engine import L2InferenceEngine
            engine = L2InferenceEngine()
            
            # 构建复盘 Prompt
            prompt = self._build_summarization_prompt(conversation_turns)
            
            # 调用 L2-BACKUP
            summary = await engine.generate(prompt, use_backup=True)
            
            return {
                "summary": summary,
                "turns_count": len(conversation_turns),
                "timestamp": time.time()
            }
            
        except asyncio.CancelledError:
            logger.info("[L2BackupScheduler] L2-BACKUP 调用被取消")
            raise
        except Exception as e:
            logger.error(f"[L2BackupScheduler] 调用 L2-BACKUP 失败：{e}")
            raise
    
    async def _call_l2_backup_with_interrupt_check(self, conversation_turns: List[Dict[str, str]]) -> Dict[str, Any]:
        """🔥 调用 L2-BACKUP（支持中断检查）
        
        在执行过程中检查中断标志，如果被中断则抛出CancelledError
        """
        # 创建异步任务执行实际的调用
        task = asyncio.create_task(self._call_l2_backup(conversation_turns))
        
        # 等待完成，但期间检查中断标志
        while not task.done():
            # 检查中断
            if self._interrupt_flag:
                logger.info("[L2BackupScheduler] 检测到中断标志，取消L2-BACKUP调用")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise asyncio.CancelledError("任务被中断")
            
            # 短暂等待后继续检查
            await asyncio.sleep(0.1)
        
        # 返回结果
        return await task
    
    def _build_summarization_prompt(self, conversation_turns: List[Dict[str, str]]) -> str:
        """
        构建复盘 Prompt
        
        Args:
            conversation_turns: 对话轮次列表
            
        Returns:
            str: 复盘 Prompt
        """
        turns_text = ""
        for i, turn in enumerate(conversation_turns):
            turns_text += f"第{i+1}轮:\n用户：{turn.get('user', '')}\nAI: {turn.get('assistant', '')}\n\n"
        
        prompt = f"""请对以下对话进行摘要，提取关键信息和主题：

{turns_text}

请输出：
1. 对话主题
2. 关键信息摘要（200 字以内）
3. 用户意图
4. 待办事项（如有）

格式：JSON"""
        
        return prompt
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        logger.info("[L2BackupScheduler] 调度器启动")
        
        while self._running:
            try:
                # 1. 检查 L2-PRIME 状态
                # 🔥 修复：如果状态未知或忙碌超过5分钟，强制设为IDLE允许后台任务执行
                if self.l2_prime_status in (L2Status.IDLE, L2Status.ERROR):
                    # 2. L2-PRIME 空闲或错误，可以触发复盘
                    if not self._task_queue.empty():
                        # 3. 获取最高优先级任务
                        task = self._task_queue.get()
                        
                        # 4. 检查 L2-BACKUP 状态
                        if self.l2_backup_status == L2Status.IDLE:
                            # 5. 启动后台任务
                            self.update_l2_backup_status(L2Status.PROCESSING)
                            
                            # 异步处理任务
                            self._current_task = asyncio.create_task(self._process_task(task))
                            
                            # 重置 BACKUP 状态（等待任务完成）
                            # 🔥 修复：不要立即重置，让_process_task完成后再重置
                            # await asyncio.sleep(0.5)
                            # self.update_l2_backup_status(L2Status.IDLE)
                        else:
                            # BACKUP 忙碌，稍后重试
                            self._task_queue.put(task)
                            await asyncio.sleep(2.0)
                    else:
                        # 队列为空，等待
                        await asyncio.sleep(1.0)
                else:
                    # L2-PRIME 忙碌，等待
                    await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"[L2BackupScheduler] 调度循环错误：{e}")
                await asyncio.sleep(1.0)
        
        logger.info("[L2BackupScheduler] 调度器已停止")
    
    def start(self):
        """启动调度器"""
        if not self._running:
            self._running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("[L2BackupScheduler] ✅ 调度器已启动")
    
    def stop(self):
        """停止调度器"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            logger.info("[L2BackupScheduler] 调度器已停止")
    
    def interrupt_current_task(self):
        """🔥 中断当前正在执行的任务
        
        当有新任务或用户输入时调用，中断正在执行的审查任务
        """
        if self._current_task and not self._current_task.done():
            logger.info("[L2BackupScheduler] 🔥 中断当前正在执行的任务")
            self._interrupt_flag = True
            self._current_task.cancel()
            self._current_task = None
    
    def clear_interrupt_flag(self):
        """🔥 清除中断标志
        
        在开始新任务前调用
        """
        self._interrupt_flag = False
    
    def register_completion_callback(self, callback):
        """注册完成回调"""
        self._on_summarization_complete = callback
        logger.info("[L2BackupScheduler] 完成回调已注册")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "l2_prime_status": self.l2_prime_status.value,
            "l2_backup_status": self.l2_backup_status.value,
            "running_tasks": len(self._running_tasks),
            "queue_size": self._task_queue.qsize(),
            "interrupt_flag": self._interrupt_flag  # 🔥 新增：中断标志状态
        }
    
    async def map_reduce_summarization(self, 
                                      conversation_turns: List[Dict[str, str]],
                                      chunk_size: int = 5) -> Dict[str, Any]:
        """
        Map-Reduce 分步摘要
        
        Args:
            conversation_turns: 对话轮次列表
            chunk_size: 每块大小
            
        Returns:
            Dict: 最终摘要
        """
        logger.info(f"[L2BackupScheduler] 启动 Map-Reduce 摘要，轮次：{len(conversation_turns)}")
        
        # Map 阶段：分块摘要
        chunks = []
        for i in range(0, len(conversation_turns), chunk_size):
            chunk = conversation_turns[i:i + chunk_size]
            chunks.append(chunk)
        
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"[L2BackupScheduler] Map 阶段 {i+1}/{len(chunks)}")
            summary = await self._call_l2_backup(chunk)
            partial_summaries.append(summary)
        
        # Reduce 阶段：合并摘要
        logger.info(f"[L2BackupScheduler] Reduce 阶段：合并 {len(partial_summaries)} 个摘要")
        
        final_summary = await self._call_l2_backup([
            {"user": "请合并以下摘要", "assistant": str(partial_summaries)}
        ])
        
        return final_summary


# 全局单例
_l2_backup_scheduler: Optional[L2BackupScheduler] = None


def get_l2_backup_scheduler(
    l2_prime_endpoint: str = "http://localhost:8000",
    l2_backup_endpoint: str = "http://localhost:8001"
) -> L2BackupScheduler:
    """获取 L2-BACKUP 调度器单例"""
    global _l2_backup_scheduler
    if _l2_backup_scheduler is None:
        _l2_backup_scheduler = L2BackupScheduler(
            l2_prime_endpoint=l2_prime_endpoint,
            l2_backup_endpoint=l2_backup_endpoint
        )
    return _l2_backup_scheduler
