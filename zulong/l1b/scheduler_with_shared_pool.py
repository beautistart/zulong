# File: zulong/l1b/scheduler_with_shared_pool.py
# L1-B 调度器 (集成共享池架构 TSD v2.5)
# 使用 SharedMemoryPool 和 DataIngestion 替代旧的存储方式

import asyncio
import json
import logging
import queue
import re
import time
import threading
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent
from zulong.infrastructure.shared_memory_pool import get_shared_memory_pool, DataEnvelope, ZoneType, DataType
from zulong.infrastructure.data_ingestion import data_ingestion
from zulong.memory.three_libraries import (
    get_three_library_manager,
    ThreeLibraryManager
)
from zulong.memory.tagging_engine import TaggingEngine
from zulong.storage.hot_storage import HotStorage
from zulong.memory.short_term_memory import get_short_term_memory, ShortTermMemory

logger = logging.getLogger(__name__)


@dataclass
class TaskItem:
    """任务项"""
    task_id: str
    prompt: str
    status: str  # "PENDING", "READY", "EXECUTING", "COMPLETED"
    raw_input: str
    user_trace_id: Optional[str] = None  # 🔥 新增：用户输入的 trace_id
    ai_trace_id: Optional[str] = None    # 🔥 新增：AI 回复的 trace_id
    vision_trace_ids: List[str] = field(default_factory=list)  # 🔥 新增：视觉 trace_ids
    audio_trace_ids: List[str] = field(default_factory=list)   # 🔥 新增：听觉 trace_ids
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    ready_at: Optional[float] = None


@dataclass
class ComplexTaskContext:
    """复杂任务上下文 (用于 L1-B 接管执行)"""
    task_id: str                              # 父任务 ID
    original_goal: str                        # 原始目标
    subtasks: List[Dict[str, Any]]           # 子任务列表 [{"id", "description", "status"}]
    dependencies: Dict[str, List[str]]        # 依赖关系 {"subtask_id": ["dep1", "dep2"]}
    parallel_groups: List[List[str]]          # 并行组 [["t1", "t2"], ["t3"]]
    results: Dict[str, Any] = field(default_factory=dict)  # 子任务执行结果
    current_index: int = 0                    # 当前执行的子任务索引
    status: str = "PENDING"                   # PENDING, EXECUTING, COMPLETED, FAILED
    progress: float = 0.0                     # 进度 0.0-1.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class AsyncL1BSchedulerWithSharedPool:
    """L1-B 异步非阻塞调度器 (集成共享池架构)
    
    架构升级 (TSD v2.5):
    - ✅ 使用 SharedMemoryPool 替代局部存储
    - ✅ 使用 DataIngestion 作为统一数据入口
    - ✅ 多模态数据关联 (视觉/听觉/文本)
    - ✅ 上下文打包 (过去 30 秒窗口)
    - ✅ 记忆固化到 Memory Zone
    
    数据流:
    1. 用户输入 → DataIngestion → Raw Zone
    2. 从共享池打包上下文 (过去 30s 视听流 + 系统状态)
    3. 构建 Prompt (包含多模态上下文)
    4. 发送到 L2
    5. 接收 AI 回复
    6. AI 回复 → DataIngestion → Raw Zone
    7. 记忆固化 → Memory Zone (通过 ShortTermMemory)
    
    对应 TSD v2.5 第 4.2 节
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化异步调度器"""
        if not hasattr(self, '_initialized'):
            self._initialized = False
            self._running = False
            
            self.task_queue: queue.Queue = queue.Queue()
            self.ready_tasks: Dict[str, TaskItem] = {}
            self.current_task: Optional[TaskItem] = None
            
            # 🔥 复杂任务上下文存储 (Phase 3.4)
            self.complex_tasks: Dict[str, ComplexTaskContext] = {}
            
            # 🔥 共享池架构组件 (懒加载，在异步方法中初始化)
            self._shared_memory_pool: Optional[Any] = None
            self.data_ingestion = data_ingestion
            
            # 🔥 传统组件 (保留用于 RAG 检索)
            self.library_manager: Optional[ThreeLibraryManager] = None
            self.tagging_engine: Optional[TaggingEngine] = None
            self.hot_storage: Optional[HotStorage] = None
            self.short_term_memory: Optional[ShortTermMemory] = None
            
            self._executor = ThreadPoolExecutor(max_workers=4)
            self._loop: Optional[asyncio.AbstractEventLoop] = None
            self._thread: Optional[threading.Thread] = None
            
            self._interrupt_handlers: List[Callable] = []
            self._query_rewrite_rules: Dict[str, str] = {}
            
            self._load_query_rewrite_rules()
            
            logger.info("[AsyncL1BSchedulerWithSharedPool] 初始化完成")
    
    async def _get_shared_memory_pool(self):
        """获取共享池实例 (异步懒加载)"""
        if self._shared_memory_pool is None:
            self._shared_memory_pool = await get_shared_memory_pool()
        return self._shared_memory_pool
    
    def initialize(self):
        """初始化组件"""
        if self._initialized:
            return
        
        # 🔥 共享池架构组件已初始化 (单例模式)
        logger.info("[AsyncL1BSchedulerWithSharedPool] 共享池组件已就绪")
        
        # 🔥 传统组件初始化 (用于 RAG)
        self.library_manager = get_three_library_manager()
        self.tagging_engine = TaggingEngine()
        self.hot_storage = HotStorage()
        self.short_term_memory = get_short_term_memory()
        
        self._initialized = True
        logger.info("[AsyncL1BSchedulerWithSharedPool] 所有组件初始化完成")
    
    def start(self):
        """启动调度器"""
        if self._running:
            return
        
        self.initialize()
        self._running = True
        
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        
        logger.info("[AsyncL1BSchedulerWithSharedPool] 调度器已启动")
    
    def stop(self):
        """停止调度器"""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._executor.shutdown(wait=False)
        logger.info("[AsyncL1BSchedulerWithSharedPool] 调度器已停止")
    
    def _run_event_loop(self):
        """运行事件循环"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._main_loop())
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.close()
    
    async def _main_loop(self):
        """主循环"""
        # 导入复盘触发器
        from zulong.review.trigger import ReviewTrigger, TriggerType
        
        # 获取 ReviewTrigger 实例
        try:
            self.review_trigger = ReviewTrigger()
            await self.review_trigger.start()
            
            # 🔥 修复：注册回调函数
            from zulong.review.integration import get_replay_integration
            replay_integration = get_replay_integration()
            
            self.review_trigger.register_callback(
                TriggerType.USER_ACTIVE,
                replay_integration.on_replay_triggered
            )
            self.review_trigger.register_callback(
                TriggerType.QUIET_MODE,
                replay_integration.on_replay_triggered
            )
            self.review_trigger.register_callback(
                TriggerType.NIGHT_SCHEDULE,
                replay_integration.on_replay_triggered
            )
            
            logger.info("[AsyncL1BSchedulerWithSharedPool] ReviewTrigger 已启动并注册回调")
        except Exception as e:
            logger.error(f"[AsyncL1BSchedulerWithSharedPool] 初始化 ReviewTrigger 失败：{e}")
            self.review_trigger = None
        
        while self._running:
            try:
                await asyncio.sleep(0.01)
                self._check_for_emergency_interrupts()
                
                # 🔥 检测复盘关键词并触发
                await self._check_review_keyword()
                
            except Exception as e:
                logger.error(f"[AsyncL1BSchedulerWithSharedPool] 主循环错误：{e}")
    
    def _load_query_rewrite_rules(self):
        """加载查询重写规则"""
        self._query_rewrite_rules = {
            "切蛋糕": "任务分片",
            "像切蛋糕": "任务分片",
            "蛋糕分": "任务分片",
            "一刀切": "统一处理",
            "滚雪球": "迭代积累",
            "搭积木": "模块化构建",
            "拼图": "组合集成",
            "剥洋葱": "逐层深入",
            "走迷宫": "路径搜索",
            "下棋": "策略规划",
            "打太极": "柔性处理",
            "拔河": "资源竞争",
            "接力棒": "任务传递",
            "多米诺": "连锁反应",
            "蝴蝶效应": "敏感依赖"
        }
    
    def rewrite_query(self, raw_input: str) -> str:
        """查询重写：将口语转化为技术术语"""
        tech_query = raw_input
        for colloquial, technical in self._query_rewrite_rules.items():
            if colloquial in raw_input:
                tech_query = tech_query.replace(colloquial, technical)
                logger.info(f"[AsyncL1BSchedulerWithSharedPool] 查询重写：'{colloquial}' -> '{technical}'")
        return tech_query
    
    async def _check_review_keyword(self):
        """检测复盘关键词并触发复盘机制
        
        精确匹配"启动复盘"，避免误触
        """
        try:
            # 从共享池读取最新的用户输入
            pool = await self._get_shared_memory_pool()
            latest_inputs = pool.read_recent(
                zone_type=ZoneType.RAW,
                data_type=DataType.TEXT,
                limit=1
            )
            
            if latest_inputs:
                latest_input = latest_inputs[0]
                user_text = latest_input.data.get('text', '') if hasattr(latest_input, 'data') else str(latest_input)
                
                # 精确匹配"启动复盘"
                if user_text.strip() == "启动复盘":
                    logger.info(f"[AsyncL1BSchedulerWithSharedPool] 🔍 检测到复盘关键词：'{user_text}'")
                    
                    # 触发复盘机制
                    if self.review_trigger:
                        from zulong.review.trigger import TriggerType
                        
                        logger.info("[AsyncL1BSchedulerWithSharedPool] 🚀 触发复盘机制...")
                        
                        result = await self.review_trigger.trigger_user_active(
                            context={
                                'trigger_keyword': '启动复盘',
                                'user_input': user_text,
                                'trigger_source': 'L1B_scheduler',
                                'trace_id': latest_input.trace_id if hasattr(latest_input, 'trace_id') else None
                            }
                        )
                        
                        if result:
                            logger.info("[AsyncL1BSchedulerWithSharedPool] ✅ 复盘机制触发成功")
                        else:
                            logger.warning("[AsyncL1BSchedulerWithSharedPool] ⚠️ 复盘机制触发返回 False")
                    else:
                        logger.warning("[AsyncL1BSchedulerWithSharedPool] ⚠️ ReviewTrigger 未初始化")
                        
        except Exception as e:
            logger.debug(f"[AsyncL1BSchedulerWithSharedPool] 检测复盘关键词异常：{e}")
    
    async def _build_context_pack(self, time_window_sec: float = 30.0,
                           vision_trace_ids: Optional[List[str]] = None,
                           audio_trace_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        构建上下文包 (TSD v2.5 核心方法)
        
        Args:
            time_window_sec: 时间窗口 (默认 30 秒)
            vision_trace_ids: 关联的视觉 trace_id 列表
            audio_trace_ids: 关联的听觉 trace_id 列表
            
        Returns:
            Dict: 上下文包
        """
        # 🔥 从共享池读取数据
        pool = await self._get_shared_memory_pool()
        context_pack = pool.build_context_pack(
            time_window_sec=time_window_sec
        )
        
        # 🔥 如果有指定的 trace_ids，添加特定数据
        if vision_trace_ids:
            for trace_id in vision_trace_ids:
                envelope = pool.read(trace_id)
                if envelope:
                    if 'vision' not in context_pack:
                        context_pack['vision'] = []
                    context_pack['vision'].append(envelope.to_dict())
        
        if audio_trace_ids:
            for trace_id in audio_trace_ids:
                envelope = pool.read(trace_id)
                if envelope:
                    if 'audio' not in context_pack:
                        context_pack['audio'] = []
                    context_pack['audio'].append(envelope.to_dict())
        
        logger.info(f"[AsyncL1BSchedulerWithSharedPool] 📦 构建上下文包："
                   f"{len(context_pack.get('vision', []))}视觉 + "
                   f"{len(context_pack.get('audio', []))}听觉 + "
                   f"{len(context_pack.get('text', []))}文本")
        
        return context_pack
    
    def _build_prompt(self, user_input: str, context_pack: Dict[str, Any], 
                     turn_id: int) -> str:
        """
        构建 Prompt (包含多模态上下文)
        
        Args:
            user_input: 用户输入
            context_pack: 上下文包
            turn_id: 对话轮数
            
        Returns:
            str: 完整的 Prompt
        """
        prompt_parts = []
        
        # 1. 系统角色
        prompt_parts.append("你是祖龙 (ZULONG) 机器人系统，一个智能家庭助手。")
        
        # 2. 添加短期记忆 (最近的对话历史)
        if self.short_term_memory:
            recent_memories = self.short_term_memory.get_recent(limit=3, include_context=True)
            if recent_memories:
                prompt_parts.append("\n【最近的对话历史】")
                for memory in recent_memories:
                    prompt_parts.append(f"用户：{memory['user_input']}")
                    prompt_parts.append(f"AI: {memory['ai_response']}")
        
        # 3. 添加多模态上下文
        if context_pack.get('text'):
            prompt_parts.append("\n【最近的文本上下文】")
            for text_item in context_pack['text'][-5:]:  # 最近 5 条
                prompt_parts.append(f"- {text_item.get('payload', '')}")
        
        if context_pack.get('vision'):
            prompt_parts.append("\n【视觉上下文】")
            for vision_item in context_pack['vision'][-3:]:  # 最近 3 条
                metadata = vision_item.get('metadata', {})
                prompt_parts.append(f"- 视觉数据：{metadata.get('description', '无描述')}")
        
        if context_pack.get('audio'):
            prompt_parts.append("\n【听觉上下文】")
            for audio_item in context_pack['audio'][-3:]:  # 最近 3 条
                metadata = audio_item.get('metadata', {})
                prompt_parts.append(f"- 听觉数据：{metadata.get('description', '无描述')}")
        
        # 4. 添加当前用户输入
        prompt_parts.append(f"\n【当前用户输入】\n{user_input}")
        
        # 5. 添加对话轮数
        prompt_parts.append(f"\n【对话轮数】第 {turn_id} 轮")
        
        full_prompt = "\n".join(prompt_parts)
        logger.info(f"[AsyncL1BSchedulerWithSharedPool] 📝 Prompt 构建完成 ({len(full_prompt)} 字符)")
        
        return full_prompt
    
    async def _send_to_l2(self, prompt: str, task_id: str) -> str:
        """
        发送到 L2 推理引擎
        
        Args:
            prompt: 构建好的 Prompt
            task_id: 任务 ID
            
        Returns:
            str: AI 回复
        """
        # 🔥 此处简化，实际应调用 L2 接口
        # TODO: 实现 L2 调用
        logger.info(f"[AsyncL1BSchedulerWithSharedPool] 🧠 发送任务到 L2: {task_id}")
        
        # 模拟 L2 回复
        await asyncio.sleep(0.1)  # 模拟推理延迟
        ai_response = "[L2 回复：这是一个模拟回复]"
        
        return ai_response
    
    async def handle_request_async(self, user_input: str,
                                   vision_trace_ids: Optional[List[str]] = None,
                                   audio_trace_ids: Optional[List[str]] = None,
                                   file_trace_ids: Optional[List[str]] = None) -> str:
        """
        处理用户请求 (异步版本，TSD v2.5 核心方法)
        
        数据流:
        1. 写入用户输入到共享池
        2. 从共享池打包上下文 (过去 30s 视听流 + 系统状态)
        3. 构建 Prompt (包含多模态上下文)
        4. 发送到 L2
        5. 接收 AI 回复
        6. 写入 AI 回复到共享池
        7. 记忆固化 (ShortTermMemory.store)
        
        Args:
            user_input: 用户输入
            vision_trace_ids: 关联的视觉 trace_id 列表
            audio_trace_ids: 关联的听觉 trace_id 列表
            file_trace_ids: 关联的文件 trace_id 列表
            
        Returns:
            str: AI 回复
        """
        try:
            # 🔥 1. 写入用户输入到共享池 (TSD v2.5)
            timestamp = time.time()
            user_trace = await self.data_ingestion.ingest_text(
                text=user_input,
                source="user",
                timestamp=timestamp,
                metadata={
                    "vision_trace_ids": vision_trace_ids or [],
                    "audio_trace_ids": audio_trace_ids or [],
                    "file_trace_ids": file_trace_ids or []
                }
            )
            logger.info(f"[AsyncL1BSchedulerWithSharedPool] ✅ 用户输入入池：{user_trace[:20]}")
            
            # 🔥 2. 从共享池打包上下文 (TSD v2.5 核心：多模态上下文)
            context_pack = self._build_context_pack(
                time_window_sec=30.0,  # 过去 30 秒
                vision_trace_ids=vision_trace_ids or [],
                audio_trace_ids=audio_trace_ids or []
            )
            
            # 🔥 3. 构建 Prompt
            prompt = self._build_prompt(
                user_input=user_input,
                context_pack=context_pack,
                turn_id=self.short_term_memory.get_current_turn() if self.short_term_memory else 0
            )
            
            # 🔥 4. 发送到 L2
            ai_response = await self._send_to_l2(prompt, task_id=user_trace[:8])
            
            # 🔥 5. 写入 AI 回复到共享池
            ai_trace = await self.data_ingestion.ingest_text(
                text=ai_response,
                source="assistant",
                timestamp=time.time(),
                metadata={
                    "user_trace": user_trace,
                    "vision_trace_ids": vision_trace_ids or [],
                    "audio_trace_ids": audio_trace_ids or [],
                    "file_trace_ids": file_trace_ids or []
                }
            )
            logger.info(f"[AsyncL1BSchedulerWithSharedPool] ✅ AI 回复入池：{ai_trace[:20]}")
            
            # 🔥 6. 记忆固化 (TSD v2.5: 写入 Memory Zone) - 🔥 异步版本
            if self.short_term_memory:
                try:
                    # 🔥 使用异步接口
                    success = await self.short_term_memory.store(
                        user_input=user_input,
                        ai_response=ai_response,
                        metadata={
                            "user_trace": user_trace,
                            "ai_trace": ai_trace,
                            "vision_trace_ids": vision_trace_ids or [],
                            "audio_trace_ids": audio_trace_ids or [],
                            "context_pack": context_pack
                        }
                    )
                    if success:
                        logger.info(f"[AsyncL1BSchedulerWithSharedPool] ✅ 记忆固化完成 (turn={self.short_term_memory.get_current_turn()})")
                    else:
                        logger.warning(f"[AsyncL1BSchedulerWithSharedPool] ⚠️ 记忆固化失败")
                except Exception as e:
                    logger.error(f"[AsyncL1BSchedulerWithSharedPool] ❌ 记忆固化失败：{e}")
            
            return ai_response
            
        except Exception as e:
            logger.error(f"[AsyncL1BSchedulerWithSharedPool] ❌ 处理请求失败：{e}", exc_info=True)
            return f"[错误：{str(e)}]"
    
    def handle_request(self, user_input: str,
                       vision_trace_ids: Optional[List[str]] = None,
                       audio_trace_ids: Optional[List[str]] = None,
                       file_trace_ids: Optional[List[str]] = None) -> str:
        """
        处理用户请求 (同步包装器)
        
        Args:
            user_input: 用户输入
            vision_trace_ids: 关联的视觉 trace_id 列表
            audio_trace_ids: 关联的听觉 trace_id 列表
            file_trace_ids: 关联的文件 trace_id 列表
            
        Returns:
            str: AI 回复
        """
        # 🔥 在异步环境中直接调用异步方法
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 🔥 事件循环正在运行，使用 create_task
                future = asyncio.run_coroutine_threadsafe(
                    self.handle_request_async(
                        user_input,
                        vision_trace_ids,
                        audio_trace_ids,
                        file_trace_ids
                    ),
                    self._loop
                )
                return future.result(timeout=5.0)
            else:
                # 🔥 事件循环未运行，直接运行
                return loop.run_until_complete(
                    self.handle_request_async(
                        user_input,
                        vision_trace_ids,
                        audio_trace_ids,
                        file_trace_ids
                    )
                )
        except Exception as e:
            logger.error(f"[AsyncL1BSchedulerWithSharedPool] ❌ 同步调用失败：{e}")
            return f"[错误：{str(e)}]"
    
    # ========== Phase 3.4: 复杂任务 L1-B 调度接管 ==========
    
    async def execute_subtasks_async(
        self,
        original_goal: str,
        subtasks: List[Dict[str, Any]],
        dependencies: Optional[Dict[str, List[str]]] = None,
        parallel_groups: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        执行复杂任务 (L2 委托给 L1-B)
        
        当 L2 分解的子任务数量 > 3 时，调用此方法让 L1-B 接管执行
        
        Args:
            original_goal: 原始目标描述
            subtasks: 子任务列表 [{"id": "t1", "description": "搜索AI新闻"}, ...]
            dependencies: 依赖关系 {"t2": ["t1"], "t3": ["t1", "t2"]}
            parallel_groups: 并行组 [["t1", "t2"], ["t3"]] (可选，如果不提供则自动计算)
            
        Returns:
            Dict: 执行结果 {"task_id": "...", "status": "COMPLETED", "results": {...}, "progress": 1.0}
        """
        if not subtasks:
            return {"status": "FAILED", "error": "子任务列表为空"}
        
        # 1. 创建复杂任务上下文
        task_id = f"complex_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # 2. 如果没有提供并行组，根据依赖关系自动计算
        if parallel_groups is None:
            parallel_groups = self._compute_parallel_groups(subtasks, dependencies or {})
        
        # 3. 初始化上下文
        ctx = ComplexTaskContext(
            task_id=task_id,
            original_goal=original_goal,
            subtasks=subtasks,
            dependencies=dependencies or {},
            parallel_groups=parallel_groups,
            status="EXECUTING"
        )
        
        self.complex_tasks[task_id] = ctx
        
        # 4. 写入共享池 (Memory Zone)
        await self._save_task_context_to_pool(ctx)
        
        logger.info(f"[AsyncL1BSchedulerWithSharedPool] 🚀 开始执行复杂任务: {task_id}")
        logger.info(f"  - 原始目标: {original_goal}")
        logger.info(f"  - 子任务数: {len(subtasks)}")
        logger.info(f"  - 并行组数: {len(parallel_groups)}")
        
        # 5. 按依赖顺序执行子任务
        try:
            for group_index, group in enumerate(parallel_groups):
                logger.info(f"[AsyncL1BSchedulerWithSharedPool] 📦 执行并行组 {group_index + 1}/{len(parallel_groups)}: {group}")
                
                # 并行执行当前组的所有子任务
                tasks = [self._execute_single_subtask(ctx, subtask_id) for subtask_id in group]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 检查结果
                for subtask_id, result in zip(group, results):
                    if isinstance(result, Exception):
                        logger.error(f"[AsyncL1BSchedulerWithSharedPool] ❌ 子任务 {subtask_id} 失败: {result}")
                        ctx.results[subtask_id] = {"status": "FAILED", "error": str(result)}
                    else:
                        ctx.results[subtask_id] = result
                        logger.info(f"[AsyncL1BSchedulerWithSharedPool] ✅ 子任务 {subtask_id} 完成")
                
                # 更新进度
                ctx.current_index = group_index + 1
                ctx.progress = (group_index + 1) / len(parallel_groups)
                await self._update_task_context_in_pool(ctx)
            
            # 6. 所有任务完成
            ctx.status = "COMPLETED"
            ctx.progress = 1.0
            ctx.completed_at = time.time()
            await self._update_task_context_in_pool(ctx)
            
            logger.info(f"[AsyncL1BSchedulerWithSharedPool] 🎉 复杂任务完成: {task_id}")
            
            return {
                "task_id": task_id,
                "status": "COMPLETED",
                "original_goal": original_goal,
                "results": ctx.results,
                "progress": 1.0,
                "completed_at": ctx.completed_at
            }
            
        except Exception as e:
            ctx.status = "FAILED"
            ctx.completed_at = time.time()
            await self._update_task_context_in_pool(ctx)
            
            logger.error(f"[AsyncL1BSchedulerWithSharedPool] ❌ 复杂任务失败: {task_id}, 错误: {e}")
            
            return {
                "task_id": task_id,
                "status": "FAILED",
                "error": str(e),
                "results": ctx.results,
                "progress": ctx.progress
            }
    
    def _compute_parallel_groups(
        self,
        subtasks: List[Dict[str, Any]],
        dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """
        根据依赖关系计算并行组 (拓扑排序)
        
        Returns:
            List[List[str]]: 并行组列表，每个组内的任务可以并行执行
            例如: [["t1"], ["t2", "t3"], ["t4"]]
        """
        if not dependencies:
            # 没有依赖关系，所有任务可以并行
            return [[task["id"] for task in subtasks]]
        
        # 构建入度表
        task_ids = {task["id"] for task in subtasks}
        in_degree = {task_id: 0 for task_id in task_ids}
        adj_list = {task_id: [] for task_id in task_ids}
        
        for task_id, deps in dependencies.items():
            if task_id not in task_ids:
                continue
            for dep in deps:
                if dep in task_ids:
                    adj_list[dep].append(task_id)
                    in_degree[task_id] += 1
        
        # Kahn 算法进行拓扑排序
        groups = []
        current_group = [task_id for task_id, degree in in_degree.items() if degree == 0]
        
        while current_group:
            groups.append(current_group)
            next_group = []
            
            for task_id in current_group:
                for neighbor in adj_list[task_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_group.append(neighbor)
            
            current_group = next_group
        
        # 检查是否有环
        if sum(len(g) for g in groups) != len(task_ids):
            logger.warning("[AsyncL1BSchedulerWithSharedPool] ⚠️ 检测到依赖环，退化为顺序执行")
            return [[task["id"] for task in subtasks]]
        
        return groups
    
    async def _execute_single_subtask(
        self,
        ctx: ComplexTaskContext,
        subtask_id: str
    ) -> Dict[str, Any]:
        """
        执行单个子任务
        
        Args:
            ctx: 复杂任务上下文
            subtask_id: 子任务 ID
            
        Returns:
            Dict: 执行结果 {"id": "...", "status": "COMPLETED", "result": "..."}
        """
        # 查找子任务描述
        subtask = next((t for t in ctx.subtasks if t["id"] == subtask_id), None)
        if not subtask:
            return {"id": subtask_id, "status": "FAILED", "error": "子任务不存在"}
        
        description = subtask.get("description", subtask.get("name", ""))
        
        # 更新状态为 EXECUTING
        subtask["status"] = "EXECUTING"
        logger.info(f"[AsyncL1BSchedulerWithSharedPool] 🔨 执行子任务 {subtask_id}: {description}")
        
        try:
            # 构建上下文 (包含之前子任务的结果)
            context_with_results = {
                "original_goal": ctx.original_goal,
                "current_subtask": description,
                "previous_results": {
                    k: v for k, v in ctx.results.items()
                    if k in ctx.dependencies.get(subtask_id, [])
                }
            }
            
            # 调用 L2 执行子任务
            prompt = json.dumps(context_with_results, ensure_ascii=False)
            result = await self._send_to_l2(prompt, task_id=subtask_id)
            
            # 更新状态
            subtask["status"] = "COMPLETED"
            
            return {
                "id": subtask_id,
                "status": "COMPLETED",
                "description": description,
                "result": result
            }
            
        except Exception as e:
            subtask["status"] = "FAILED"
            logger.error(f"[AsyncL1BSchedulerWithSharedPool] ❌ 子任务 {subtask_id} 执行失败: {e}")
            return {
                "id": subtask_id,
                "status": "FAILED",
                "description": description,
                "error": str(e)
            }
    
    async def _save_task_context_to_pool(self, ctx: ComplexTaskContext):
        """将任务上下文保存到共享池 (Memory Zone)"""
        try:
            envelope = DataEnvelope(
                trace_id=ctx.task_id,
                timestamp=time.time(),
                data_type=DataType.CONTEXT_PACK,
                zone=ZoneType.MEMORY,
                payload=ctx.__dict__,
                metadata={
                    "type": "complex_task",
                    "original_goal": ctx.original_goal,
                    "subtask_count": len(ctx.subtasks)
                }
            )
            
            pool = await self._get_shared_memory_pool()
            await pool.write(envelope)
            logger.debug(f"[AsyncL1BSchedulerWithSharedPool] 💾 保存任务上下文到 Memory Zone: {ctx.task_id}")
            
        except Exception as e:
            logger.error(f"[AsyncL1BSchedulerWithSharedPool] ❌ 保存任务上下文失败: {e}")
    
    async def _update_task_context_in_pool(self, ctx: ComplexTaskContext):
        """更新共享池中的任务上下文 (通过覆盖写入)"""
        try:
            # SharedMemoryPool 没有 update 方法，直接重新写入（覆盖）
            envelope = DataEnvelope(
                trace_id=ctx.task_id,
                timestamp=time.time(),
                data_type=DataType.CONTEXT_PACK,
                zone=ZoneType.MEMORY,
                payload=ctx.__dict__,
                metadata={
                    "type": "complex_task",
                    "progress": ctx.progress,
                    "status": ctx.status,
                    "current_index": ctx.current_index,
                    "original_goal": ctx.original_goal,
                    "subtask_count": len(ctx.subtasks)
                }
            )
            
            pool = await self._get_shared_memory_pool()
            await pool.write(envelope)
            logger.debug(f"[AsyncL1BSchedulerWithSharedPool] 📊 更新任务进度: {ctx.task_id} - {ctx.progress:.1%}")
            
        except Exception as e:
            logger.error(f"[AsyncL1BSchedulerWithSharedPool] ❌ 更新任务上下文失败: {e}")
    
    async def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        查询复杂任务进度 (供 L2 查询)
        
        Args:
            task_id: 任务 ID
            
        Returns:
            Dict: 进度信息 {"task_id": "...", "progress": 0.5, "status": "EXECUTING", ...}
        """
        # 优先从内存查找
        ctx = self.complex_tasks.get(task_id)
        if ctx:
            return {
                "task_id": ctx.task_id,
                "original_goal": ctx.original_goal,
                "status": ctx.status,
                "progress": ctx.progress,
                "current_index": ctx.current_index,
                "total_subtasks": len(ctx.subtasks),
                "completed_subtasks": sum(1 for t in ctx.subtasks if t.get("status") == "COMPLETED"),
                "results": ctx.results
            }
        
        # 从共享池查找
        try:
            pool = await self._get_shared_memory_pool()
            envelope = await pool.read(task_id)
            if envelope:
                return envelope.payload
        except Exception as e:
            logger.error(f"[AsyncL1BSchedulerWithSharedPool] ❌ 查询任务进度失败: {e}")
        
        return None
    
    # ========== 结束 Phase 3.4 ==========
    
    def _check_for_emergency_interrupts(self):
        """检查紧急中断"""
        # TODO: 实现紧急中断逻辑
        pass


# 🔥 全局单例
scheduler_with_shared_pool = AsyncL1BSchedulerWithSharedPool()
