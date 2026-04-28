# File: zulong/l2/event_handler.py
# L2 专家调用事件处理器 - 事件驱动架构

import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import time
import threading

from ..core.event_bus import EventBus
from ..core.types import ZulongEvent, EventType, PowerState, L2Status
from .expert_invoker import ExpertInvoker, ExpertCallResult

logger = logging.getLogger(__name__)


@dataclass
class ExpertCallRequest:
    """专家调用请求
    
    TSD v1.7 对应规则:
    - 结构化请求格式
    - 支持优先级
    - 支持超时控制
    """
    query: str
    context: Optional[Dict[str, Any]] = None
    use_rag: bool = True
    use_tools: bool = True
    priority: int = 5  # 1-10
    timeout: float = 60.0
    callback: Optional[Callable[[ExpertCallResult], None]] = None
    request_id: str = field(default_factory=lambda: f"req_{int(time.time() * 1000)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "query": self.query,
            "context": self.context,
            "use_rag": self.use_rag,
            "use_tools": self.use_tools,
            "priority": self.priority,
            "timeout": self.timeout,
            "has_callback": self.callback is not None
        }


class ExpertEventHandler:
    """L2 专家调用事件处理器
    
    TSD v1.7 对应规则:
    - 事件驱动架构
    - 订阅专家调用相关事件
    - 异步处理
    - 支持中断
    
    功能:
    - 订阅 USER_* 事件
    - 调用专家处理器
    - 发布结果事件
    - 支持中断机制
    """
    
    def __init__(
        self,
        expert_invoker: Optional[ExpertInvoker] = None,
        event_bus: Optional[EventBus] = None,
        auto_subscribe: bool = True
    ):
        """初始化事件处理器
        
        Args:
            expert_invoker: 专家调用器
            event_bus: 事件总线
            auto_subscribe: 是否自动订阅事件
        """
        self.expert_invoker = expert_invoker or ExpertInvoker()
        self.event_bus = event_bus or EventBus()
        
        # 当前调用状态
        self.current_request: Optional[ExpertCallRequest] = None
        self.is_processing = False
        self._interrupt_flag = False
        self._lock = threading.Lock()
        
        # 订阅事件
        if auto_subscribe:
            self._subscribe_events()
        
        logger.info("[ExpertEventHandler] Initialized")
    
    def _subscribe_events(self):
        """订阅事件"""
        # 订阅用户事件
        self.event_bus.subscribe(
            event_type=EventType.USER_SPEECH,
            handler=self._handle_user_speech,
            subscriber="ExpertEventHandler"
        )
        
        self.event_bus.subscribe(
            event_type=EventType.USER_COMMAND,
            handler=self._handle_user_command,
            subscriber="ExpertEventHandler"
        )
        
        # 订阅专家调用事件
        self.event_bus.subscribe(
            event_type=EventType.EXPERT_CALL,
            handler=self._handle_expert_call,
            subscriber="ExpertEventHandler"
        )
        
        # 订阅中断事件
        self.event_bus.subscribe(
            event_type=EventType.INTERRUPT,
            handler=self._handle_interrupt,
            subscriber="ExpertEventHandler"
        )
        
        logger.info("[ExpertEventHandler] Subscribed to events")
    
    def _handle_user_speech(self, event: ZulongEvent):
        """处理用户语音事件
        
        Args:
            event: 语音事件
        """
        logger.info(f"[ExpertEventHandler] User speech: {event.payload.get('text', '')[:50]}...")
        
        # 提取语音文本
        text = event.payload.get("text", "")
        if not text:
            logger.warning("[ExpertEventHandler] Empty speech text")
            return
        
        # 创建调用请求
        request = ExpertCallRequest(
            query=text,
            context=event.payload.get("context"),
            priority=event.payload.get("priority", 5),
            timeout=event.payload.get("timeout", 60.0)
        )
        
        # 异步处理
        self._enqueue_request(request)
    
    def _handle_user_command(self, event: ZulongEvent):
        """处理用户命令事件
        
        Args:
            event: 命令事件
        """
        logger.info(f"[ExpertEventHandler] User command: {event.payload.get('command', '')[:50]}...")
        
        # 提取命令
        command = event.payload.get("command", "")
        if not command:
            logger.warning("[ExpertEventHandler] Empty command")
            return
        
        # 创建调用请求
        request = ExpertCallRequest(
            query=command,
            context=event.payload.get("context"),
            priority=event.payload.get("priority", 5),
            timeout=event.payload.get("timeout", 60.0)
        )
        
        # 异步处理
        self._enqueue_request(request)
    
    def _handle_expert_call(self, event: ZulongEvent):
        """处理专家调用事件
        
        Args:
            event: 专家调用事件
        """
        logger.info(f"[ExpertEventHandler] Expert call request")
        
        # 从事件中提取请求
        payload = event.payload
        request = ExpertCallRequest(
            query=payload.get("query", ""),
            context=payload.get("context"),
            use_rag=payload.get("use_rag", True),
            use_tools=payload.get("use_tools", True),
            priority=payload.get("priority", 5),
            timeout=payload.get("timeout", 60.0),
            callback=payload.get("callback")
        )
        
        # 异步处理
        self._enqueue_request(request)
    
    def _handle_interrupt(self, event: ZulongEvent):
        """处理中断事件
        
        Args:
            event: 中断事件
        """
        logger.warning(f"[ExpertEventHandler] INTERRUPT received!")
        
        # 设置中断标志
        with self._lock:
            self._interrupt_flag = True
        
        # 发布中断确认事件
        self.event_bus.publish(ZulongEvent(
            type=EventType.INTERRUPT_ACK,
            source="ExpertEventHandler",
            payload={
                "reason": event.payload.get("reason", "user_request"),
                "timestamp": time.time()
            }
        ))
    
    def _enqueue_request(self, request: ExpertCallRequest):
        """将请求加入队列
        
        Args:
            request: 调用请求
        """
        # 检查是否正在处理
        if self.is_processing:
            logger.info(f"[ExpertEventHandler] Already processing, queuing request {request.request_id}")
            # 这里可以添加更复杂的队列管理逻辑
            return
        
        # 设置当前请求
        with self._lock:
            self.current_request = request
            self.is_processing = True
            self._interrupt_flag = False
        
        # 异步处理
        thread = threading.Thread(
            target=self._process_request,
            args=(request,),
            daemon=True
        )
        thread.start()
    
    def _process_request(self, request: ExpertCallRequest):
        """处理调用请求
        
        Args:
            request: 调用请求
        """
        start_time = time.time()
        logger.info(f"[ExpertEventHandler] Processing request {request.request_id}")
        
        try:
            # 调用专家
            result = self.expert_invoker.invoke(
                query=request.query,
                context=request.context,
                use_rag=request.use_rag,
                use_tools=request.use_tools,
                timeout=request.timeout
            )
            
            # 检查是否被中断
            if self._interrupt_flag:
                logger.warning(f"[ExpertEventHandler] Request {request.request_id} interrupted")
                result = ExpertCallResult(
                    success=False,
                    response="Interrupted by user",
                    execution_time=time.time() - start_time,
                    metadata={"interrupted": True}
                )
            
            # 发布结果事件
            self.event_bus.publish(ZulongEvent(
                type=EventType.EXPERT_RESULT,
                source="ExpertEventHandler",
                payload={
                    "request_id": request.request_id,
                    "result": result.to_dict(),
                    "execution_time": time.time() - start_time
                }
            ))
            
            # 回调
            if request.callback:
                try:
                    request.callback(result)
                except Exception as e:
                    logger.error(f"[ExpertEventHandler] Callback error: {e}")
            
            logger.info(f"[ExpertEventHandler] Request {request.request_id} completed in {time.time() - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"[ExpertEventHandler] Processing error: {e}")
            
            # 发布错误事件
            self.event_bus.publish(ZulongEvent(
                type=EventType.EXPERT_ERROR,
                source="ExpertEventHandler",
                payload={
                    "request_id": request.request_id,
                    "error": str(e),
                    "timestamp": time.time()
                }
            ))
        
        finally:
            # 清理状态
            with self._lock:
                self.is_processing = False
                self.current_request = None
    
    def call_expert(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_rag: bool = True,
        use_tools: bool = True,
        priority: int = 5,
        timeout: float = 60.0,
        callback: Optional[Callable[[ExpertCallResult], None]] = None
    ) -> str:
        """调用专家（通过事件）
        
        Args:
            query: 查询
            context: 上下文
            use_rag: 是否使用 RAG
            use_tools: 是否使用工具
            priority: 优先级
            timeout: 超时
            callback: 回调
            
        Returns:
            str: 请求 ID
        """
        request = ExpertCallRequest(
            query=query,
            context=context,
            use_rag=use_rag,
            use_tools=use_tools,
            priority=priority,
            timeout=timeout,
            callback=callback
        )
        
        # 发布专家调用事件
        self.event_bus.publish(ZulongEvent(
            type=EventType.EXPERT_CALL,
            source="ExpertEventHandler",
            payload={
                "query": query,
                "context": context,
                "use_rag": use_rag,
                "use_tools": use_tools,
                "priority": priority,
                "timeout": timeout,
                "callback": callback
            }
        ))
        
        logger.info(f"[ExpertEventHandler] Expert call published: {request.request_id}")
        
        return request.request_id
    
    def interrupt(self, reason: str = "user_request") -> None:
        """中断当前处理
        
        Args:
            reason: 中断原因
        """
        logger.warning(f"[ExpertEventHandler] Interrupting: {reason}")
        
        # 设置中断标志
        with self._lock:
            self._interrupt_flag = True
        
        # 发布中断事件
        self.event_bus.publish(ZulongEvent(
            type=EventType.INTERRUPT,
            source="ExpertEventHandler",
            payload={
                "reason": reason,
                "timestamp": time.time()
            }
        ))
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "is_processing": self.is_processing,
            "current_request": self.current_request.to_dict() if self.current_request else None,
            "interrupt_flag": self._interrupt_flag,
            "expert_invoker_stats": self.expert_invoker.get_statistics()
        }
    
    def print_status(self):
        """打印状态"""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("L2 专家调用事件处理器状态")
        print("=" * 60)
        print(f"正在处理：{status['is_processing']}")
        
        if status['current_request']:
            print(f"当前请求：{status['current_request']['request_id']}")
            print(f"查询：{status['current_request']['query'][:50]}...")
            print(f"优先级：{status['current_request']['priority']}")
        
        print(f"中断标志：{status['interrupt_flag']}")
        print("=" * 60 + "\n")
