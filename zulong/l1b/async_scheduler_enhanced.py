# File: zulong/l1b/async_scheduler_enhanced.py
# L1-B 异步调度器 (TSD v2.5 共享池增强版)
# 对应文档：数据统一共享池化以及增强记忆共享

"""
L1-B 异步调度器 (共享池增强版)

核心增强:
1. 从共享池读取多模态上下文 (视觉、听觉、系统状态)
2. 使用 DataIngestion 写入对话到共享池
3. 支持文件上传处理的 trace_id 关联
4. 完整的上下文打包 (包含感知快照)

数据流:
1. 用户输入 → DataIngestion → Raw Zone
2. 发布事件 → L1-B 订阅
3. 从共享池打包上下文 (过去 30s 视听流 + 系统状态)
4. 构建 Prompt → 发送到 L2
5. AI 回复 → DataIngestion → Raw Zone
6. 记忆固化 → ShortTermMemory.store() → Memory Zone
"""

import asyncio
import json
import logging
import queue
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent
from zulong.infrastructure.shared_memory_pool import (
    shared_memory_pool, ZoneType, DataType, DataEnvelope
)
from zulong.infrastructure.data_ingestion import data_ingestion
from zulong.memory.short_term_memory import get_short_term_memory, ShortTermMemory
from zulong.memory.three_libraries import get_three_library_manager, ThreeLibraryManager
from zulong.memory.tagging_engine import TaggingEngine
from zulong.storage.hot_storage import HotStorage

logger = logging.getLogger(__name__)


@dataclass
class TaskItem:
    """任务项 (TSD v2.5 增强版)"""
    task_id: str
    prompt: str
    status: str  # "PENDING", "READY", "EXECUTING", "COMPLETED"
    raw_input: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    # TSD v2.5 新增：关联的 trace_id
    user_trace_id: Optional[str] = None       # 用户输入 trace_id
    vision_trace_ids: List[str] = field(default_factory=list)  # 关联的视觉 trace_id
    audio_trace_ids: List[str] = field(default_factory=list)   # 关联的听觉 trace_id
    file_trace_ids: List[str] = field(default_factory=list)    # 关联的文件 trace_id
    
    created_at: float = field(default_factory=time.time)
    ready_at: Optional[float] = None


class AsyncL1BSchedulerEnhanced:
    """
    L1-B 异步调度器 (TSD v2.5 共享池增强版)
    
    相比旧版 AsyncL1BScheduler 的增强:
    1. ✅ 使用 shared_memory_pool 替代本地缓存
    2. ✅ 使用 data_ingestion 统一数据入口
    3. ✅ 支持多模态上下文打包 (视觉 + 听觉 + 文本)
    4. ✅ 支持文件上传关联
    5. ✅ 完整的记忆固化流程
    
    使用示例:
    ```python
    scheduler = AsyncL1BSchedulerEnhanced()
    scheduler.initialize()
    scheduler.start()
    
    # 处理用户请求
    response = await scheduler.handle_request("今天下午我要出去量房")
    ```
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化调度器"""
        if not hasattr(self, '_initialized'):
            self._initialized = False
            self._running = False
            
            # 任务队列
            self.task_queue: queue.Queue = queue.Queue()
            self.ready_tasks: Dict[str, TaskItem] = {}
            self.current_task: Optional[TaskItem] = None
            
            # 核心组件
            self.pool = shared_memory_pool
            self.data_ingestion = data_ingestion
            self.library_manager: Optional[ThreeLibraryManager] = None
            self.tagging_engine: Optional[TaggingEngine] = None
            self.hot_storage: Optional[HotStorage] = None
            self.short_term_memory: Optional[ShortTermMemory] = None
            
            # 线程池和事件循环
            self._executor = ThreadPoolExecutor(max_workers=4)
            self._loop: Optional[asyncio.AbstractEventLoop] = None
            self._thread: Optional[threading.Thread] = None
            
            # 中断处理
            self._interrupt_handlers: List[Callable] = []
            
            # 对话轮数计数器
            self._current_turn = 0
            self._turn_lock = threading.Lock()
            
            logger.info("[AsyncL1BSchedulerEnhanced] 初始化完成 (TSD v2.5 共享池架构)")
    
    def initialize(self):
        """初始化组件"""
        if self._initialized:
            return
        
        self.library_manager = get_three_library_manager()
        self.tagging_engine = TaggingEngine()
        self.hot_storage = HotStorage()
        self.short_term_memory = get_short_term_memory()
        
        self._initialized = True
        logger.info("[AsyncL1BSchedulerEnhanced] 组件初始化完成")
        logger.info(f"   - SharedMemoryPool: ✅")
        logger.info(f"   - DataIngestion: ✅")
        logger.info(f"   - ShortTermMemory: ✅")
        logger.info(f"   - ThreeLibraries: ✅")
    
    def start(self):
        """启动调度器"""
        if self._running:
            return
        
        self.initialize()
        self._running = True
        
        # 启动事件循环线程
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        
        # 订阅用户输入事件
        event_bus.subscribe(
            EventType.USER_TEXT,
            self._on_user_input,
            subscriber="AsyncL1BSchedulerEnhanced"
        )
        
        logger.info("[AsyncL1BSchedulerEnhanced] 调度器已启动")
    
    def stop(self):
        """停止调度器"""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._executor.shutdown(wait=False)
        logger.info("[AsyncL1BSchedulerEnhanced] 调度器已停止")
    
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
        while self._running:
            try:
                await asyncio.sleep(0.01)
                self._check_for_emergency_interrupts()
            except Exception as e:
                logger.error(f"[AsyncL1BSchedulerEnhanced] 主循环错误：{e}")
    
    async def _on_user_input(self, event: ZulongEvent):
        """
        用户输入事件处理
        
        Args:
            event: USER_TEXT 事件
        """
        try:
            user_input = event.payload.get("text")
            trace_id = event.payload.get("trace_id")
            
            if not user_input or not trace_id:
                logger.warning(f"⚠️ 事件缺少必要数据：{event.payload}")
                return
            
            # 将任务加入队列
            task = TaskItem(
                task_id=f"task_{trace_id[6:]}",  # trace_xxx → task_xxx
                prompt="",  # 稍后构建
                status="PENDING",
                raw_input=user_input,
                user_trace_id=trace_id
            )
            
            self.task_queue.put(task)
            logger.info(f"📥 [AsyncL1BSchedulerEnhanced] 接收用户输入：{user_input[:30]}... (trace_id={trace_id[:15]})")
        
        except Exception as e:
            logger.error(f"❌ [AsyncL1BSchedulerEnhanced] 处理用户输入失败：{e}", exc_info=True)
    
    async def handle_request(self, user_input: str,
                            vision_trace_ids: Optional[List[str]] = None,
                            audio_trace_ids: Optional[List[str]] = None,
                            file_trace_ids: Optional[List[str]] = None) -> str:
        """
        处理用户请求 (TSD v2.5 核心方法)
        
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
            # 1. 写入用户输入到共享池
            with self._turn_lock:
                self._current_turn += 1
                turn_id = self._current_turn
            
            timestamp = time.time()
            user_trace = await self.data_ingestion.ingest_text(
                text=user_input,
                source="user",
                timestamp=timestamp,
                metadata={
                    "turn_id": turn_id,
                    "vision_trace_ids": vision_trace_ids or [],
                    "audio_trace_ids": audio_trace_ids or [],
                    "file_trace_ids": file_trace_ids or []
                }
            )
            
            logger.info(f"📝 [AsyncL1BSchedulerEnhanced] 用户输入入池：turn={turn_id}, trace={user_trace[:15]}")
            
            # 2. 从共享池打包上下文 (TSD v2.5 核心：多模态上下文)
            context_pack = self._build_context_pack(
                time_window_sec=30.0,  # 过去 30 秒
                vision_trace_ids=vision_trace_ids or [],
                audio_trace_ids=audio_trace_ids or []
            )
            
            logger.info(f"📦 [AsyncL1BSchedulerEnhanced] 上下文打包完成："
                       f"{len(context_pack.get('vision', []))}视觉 + "
                       f"{len(context_pack.get('audio', []))}听觉 + "
                       f"{len(context_pack.get('recent_memory', []))}记忆")
            
            # 3. 构建 Prompt
            prompt = self._build_prompt(
                user_input=user_input,
                context_pack=context_pack,
                turn_id=turn_id
            )
            
            # 4. 发送到 L2 (此处简化，实际应调用 L2 接口)
            # ai_response = await self._send_to_l2(prompt)
            ai_response = "[L2 回复]"  # TODO: 实现 L2 调用
            
            # 5. 写入 AI 回复到共享池
            ai_trace = await self.data_ingestion.ingest_text(
                text=ai_response,
                source="assistant",
                timestamp=time.time(),
                metadata={
                    "turn_id": turn_id,
                    "user_trace": user_trace
                }
            )
            
            logger.info(f"📝 [AsyncL1BSchedulerEnhanced] AI 回复入池：turn={turn_id}, trace={ai_trace[:15]}")
            
            # 6. 记忆固化 (TSD v2.5: 写入 Memory Zone)
            self.short_term_memory.store(
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
            
            logger.info(f"🧠 [AsyncL1BSchedulerEnhanced] 记忆固化完成：turn={turn_id}")
            
            return ai_response
        
        except Exception as e:
            logger.error(f"❌ [AsyncL1BSchedulerEnhanced] 处理请求失败：{e}", exc_info=True)
            return f"[错误：{str(e)}]"
    
    def _build_context_pack(self, time_window_sec: float = 30.0,
                           vision_trace_ids: Optional[List[str]] = None,
                           audio_trace_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        构建上下文包 (TSD v2.5 核心)
        
        Args:
            time_window_sec: 时间窗口 (秒)
            vision_trace_ids: 指定的视觉 trace_id 列表
            audio_trace_ids: 指定的听觉 trace_id 列表
        
        Returns:
            Dict[str, Any]: 上下文包
        """
        context = {
            "timestamp": time.time(),
            "time_window_sec": time_window_sec,
            "vision": [],
            "audio": [],
            "recent_memory": [],
            "system_state": {}
        }
        
        # 1. 获取最近的感知数据 (从共享池)
        recent_data = self.pool.get_recent(time_window_sec)
        
        for envelope in recent_data:
            if envelope.data_type == DataType.VIDEO_FRAME:
                context["vision"].append(envelope.to_dict())
            elif envelope.data_type == DataType.AUDIO_FEATURE:
                context["audio"].append(envelope.to_dict())
        
        # 2. 如果指定了 trace_id，强制包含
        if vision_trace_ids:
            for trace_id in vision_trace_ids:
                envelope = self.pool.read_feature(trace_id)
                if envelope:
                    context["vision"].append(envelope.to_dict())
        
        if audio_trace_ids:
            for trace_id in audio_trace_ids:
                envelope = self.pool.read_feature(trace_id)
                if envelope:
                    context["audio"].append(envelope.to_dict())
        
        # 3. 获取最近记忆 (从 ShortTermMemory)
        recent_memory = self.short_term_memory.get_recent(limit=5, include_context=True)
        context["recent_memory"] = recent_memory
        
        # 4. 获取系统状态
        context["system_state"] = {
            "power_state": "ACTIVE",  # TODO: 从 StateManager 获取
            "l2_status": "IDLE"       # TODO: 从 StateManager 获取
        }
        
        return context
    
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
        # 1. 构建系统提示
        system_prompt = f"""你是祖龙 (ZULONG) 机器人，当前是第 {turn_id} 轮对话。

## 上下文信息
- 视觉信息：{len(context_pack['vision'])} 条
- 听觉信息：{len(context_pack['audio'])} 条
- 历史记忆：{len(context_pack['recent_memory'])} 轮

"""
        
        # 2. 添加历史记忆
        if context_pack['recent_memory']:
            system_prompt += "\n## 历史对话\n"
            for memory in context_pack['recent_memory']:
                user_text = memory.get('user', {}).get('text', '')
                ai_text = memory.get('assistant', {}).get('text', '')
                system_prompt += f"用户：{user_text}\nAI: {ai_text}\n"
        
        # 3. 添加视觉上下文
        if context_pack['vision']:
            system_prompt += "\n## 视觉信息\n"
            for vision in context_pack['vision'][-3:]:  # 最近 3 条
                payload = vision.get('payload', {})
                if isinstance(payload, dict):
                    vision_target_pos = payload.get('vision_target_pos')
                    if vision_target_pos:
                        system_prompt += f"- 检测到运动目标：位置={vision_target_pos}\n"
        
        # 4. 添加听觉上下文
        if context_pack['audio']:
            system_prompt += "\n## 听觉信息\n"
            for audio in context_pack['audio'][-3:]:  # 最近 3 条
                payload = audio.get('payload', {})
                if isinstance(payload, dict):
                    is_speech = payload.get('is_speech')
                    if is_speech:
                        system_prompt += f"- 检测到语音\n"
        
        # 5. 用户输入
        system_prompt += f"\n## 当前输入\n用户：{user_input}\n\nAI:"
        
        return system_prompt
    
    def _check_for_emergency_interrupts(self):
        """检查紧急中断"""
        # TODO: 实现紧急中断检测
        pass


# 全局单例
async_l1b_scheduler_enhanced = AsyncL1BSchedulerEnhanced()
