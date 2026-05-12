# 复盘机制:三重触发器
"""
功能:
- 用户主动触发 (高优先级)
- 安静模式触发 (中优先级)
- 夜间定时触发 (低优先级)
- 优先级调度
- 防冲突机制

对应 TSD v2.3 第 11.1 节
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """触发器类型"""
    USER_ACTIVE = "user_active"  # 用户主动
    QUIET_MODE = "quiet_mode"    # 安静模式
    NIGHT_SCHEDULE = "night_schedule"  # 夜间定时


class TriggerPriority(Enum):
    """触发优先级"""
    HIGH = 1      # 用户主动
    MEDIUM = 2    # 安静模式
    LOW = 3       # 夜间定时


class ReviewTrigger:
    """复盘触发器"""
    
    def __init__(self,
                 quiet_mode_timeout_minutes: int = 30,
                 night_trigger_hour: int = 2,
                 night_trigger_minute: int = 0,
                 max_concurrent_triggers: int = 1):
        """初始化触发器
        
        Args:
            quiet_mode_timeout_minutes: 安静模式超时时间(分钟)
            night_trigger_hour: 夜间触发小时(0-23)
            night_trigger_minute: 夜间触发分钟(0-59)
            max_concurrent_triggers: 最大并发触发数
        """
        self.quiet_mode_timeout = timedelta(minutes=quiet_mode_timeout_minutes)
        self.night_trigger_hour = night_trigger_hour
        self.night_trigger_minute = night_trigger_minute
        self.max_concurrent_triggers = max_concurrent_triggers
        
        # 回调函数
        self._trigger_callbacks: Dict[TriggerType, Callable] = {}
        
        # 状态跟踪
        self._last_user_activity: Optional[datetime] = None
        self._is_quiet_mode_enabled: bool = True
        self._active_triggers: int = 0
        self._trigger_queue: asyncio.Queue = asyncio.Queue()
        
        # 后台任务
        self._monitor_task: Optional[asyncio.Task] = None
        self._night_scheduler_task: Optional[asyncio.Task] = None
        self._running: bool = False
        
        # 统计信息
        self.stats = {
            'total_triggers': 0,
            'user_active_count': 0,
            'quiet_mode_count': 0,
            'night_schedule_count': 0,
            'failed_triggers': 0,
            'last_trigger_time': None
        }
        
        logger.info(f"[ReviewTrigger] 初始化完成:"
                   f"quiet_timeout={quiet_mode_timeout_minutes}min, "
                   f"night_trigger={night_trigger_hour:02d}:{night_trigger_minute:02d}")
    
    def register_callback(self,
                          trigger_type: TriggerType,
                          callback: Callable):
        """注册触发回调
        
        Args:
            trigger_type: 触发器类型
            callback: 回调函数(异步)
        """
        self._trigger_callbacks[trigger_type] = callback
        logger.info(f"[ReviewTrigger] 已注册 {trigger_type.value} 回调")
    
    async def start(self):
        """启动触发器监控"""
        if self._running:
            logger.warning(f"[ReviewTrigger] 已经在运行中")
            return
        
        self._running = True
        
        # 启动安静模式监控
        self._monitor_task = asyncio.create_task(self._quiet_mode_monitor())
        
        # 启动夜间定时调度
        self._night_scheduler_task = asyncio.create_task(self._night_scheduler())
        
        # 启动触发器处理
        asyncio.create_task(self._trigger_processor())
        
        logger.info(f"[ReviewTrigger] 已启动")
    
    async def stop(self):
        """停止触发器监控"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消后台任务
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._night_scheduler_task:
            self._night_scheduler_task.cancel()
            try:
                await self._night_scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"[ReviewTrigger] 已停止")
    
    def record_user_activity(self):
        """记录用户活动(由 L1-B 调用)"""
        self._last_user_activity = datetime.now()
        
        # 如果安静模式启用,用户活动后重置
        if self._is_quiet_mode_enabled:
            self._is_quiet_mode_enabled = False
            logger.debug(f"[ReviewTrigger] 用户活动检测到,退出安静模式")
    
    def enable_quiet_mode(self):
        """启用安静模式(由用户或系统调用)"""
        self._is_quiet_mode_enabled = True
        self._last_user_activity = datetime.now() - self.quiet_mode_timeout
        logger.info(f"[ReviewTrigger] 安静模式已启用")
    
    def disable_quiet_mode(self):
        """禁用安静模式"""
        self._is_quiet_mode_enabled = False
        logger.info(f"[ReviewTrigger] 安静模式已禁用")
    
    async def trigger_user_active(self,
                                   context: Optional[Dict] = None) -> bool:
        """用户主动触发复盘
        
        Args:
            context: 上下文信息
            
        Returns:
            bool: 是否成功触发
        """
        logger.info(f"[ReviewTrigger] 用户主动触发复盘")
        
        return await self._queue_trigger(
            trigger_type=TriggerType.USER_ACTIVE,
            priority=TriggerPriority.HIGH,
            context=context
        )
    
    async def _queue_trigger(self,
                             trigger_type: TriggerType,
                             priority: TriggerPriority,
                             context: Optional[Dict] = None):
        """将触发请求加入队列
        
        Args:
            trigger_type: 触发器类型
            priority: 优先级
            context: 上下文信息
        """
        trigger_request = {
            'type': trigger_type,
            'priority': priority,
            'context': context or {},
            'timestamp': datetime.now(),
            'id': f"{trigger_type.value}_{datetime.now().timestamp()}"
        }
        
        # 高优先级插入队首,低优先级加入队尾
        if priority == TriggerPriority.HIGH:
            # 创建临时队列,将高优先级插入
            temp_queue = asyncio.Queue()
            temp_queue.put_nowait(trigger_request)
            
            # 转移原有队列
            while not self._trigger_queue.empty():
                temp_queue.put_nowait(await self._trigger_queue.get())
            
            self._trigger_queue = temp_queue
        else:
            await self._trigger_queue.put(trigger_request)
        
        self.stats['total_triggers'] += 1
        
        logger.debug(f"[ReviewTrigger] 触发请求已排队:{trigger_request['id']}")
        
        return True
    
    async def _trigger_processor(self):
        """触发器处理器(从队列处理触发请求)"""
        try:
            while self._running:
                try:
                    # 获取触发请求
                    trigger_request = await asyncio.wait_for(
                        self._trigger_queue.get(),
                        timeout=1.0
                    )
                    
                    # 检查并发限制
                    if self._active_triggers >= self.max_concurrent_triggers:
                        # 等待有空闲
                        await asyncio.sleep(0.5)
                        await self._trigger_queue.put(trigger_request)
                        continue
                    
                    # 处理触发
                    await self._execute_trigger(trigger_request)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"[ReviewTrigger] 处理触发失败:{e}")
                    self.stats['failed_triggers'] += 1
        
        except asyncio.CancelledError:
            logger.info(f"[ReviewTrigger] 触发器处理器已取消")
    
    async def _execute_trigger(self, trigger_request: Dict):
        """执行触发
        
        Args:
            trigger_request: 触发请求
        """
        trigger_type = trigger_request['type']
        
        try:
            # 增加活跃触发数
            self._active_triggers += 1
            
            # 获取回调
            callback = self._trigger_callbacks.get(trigger_type)
            
            if not callback:
                logger.warning(f"[ReviewTrigger] 未找到 {trigger_type.value} 的回调")
                # 🔥 修复：即使没有回调，也发布事件
                await self._publish_replay_event(trigger_request)
                return
            
            # 调用回调
            if asyncio.iscoroutinefunction(callback):
                await callback(trigger_request)
            else:
                callback(trigger_request)
            
            # 🔥 修复：发布事件
            await self._publish_replay_event(trigger_request)
            
            # 更新统计
            self.stats['last_trigger_time'] = datetime.now().isoformat()
            
            if trigger_type == TriggerType.USER_ACTIVE:
                self.stats['user_active_count'] += 1
            elif trigger_type == TriggerType.QUIET_MODE:
                self.stats['quiet_mode_count'] += 1
            elif trigger_type == TriggerType.NIGHT_SCHEDULE:
                self.stats['night_schedule_count'] += 1
            
            logger.info(f"[ReviewTrigger] 触发执行成功:{trigger_request['id']}")
            
        except Exception as e:
            logger.error(f"[ReviewTrigger] 执行触发失败 {trigger_request['id']}: {e}")
            self.stats['failed_triggers'] += 1
            raise
        
        finally:
            self._active_triggers -= 1
    
    async def _publish_replay_event(self, trigger_request: Dict):
        """发布复盘触发事件到事件总线
        
        Args:
            trigger_request: 触发请求
        """
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, EventPriority, ZulongEvent
            
            event = ZulongEvent(
                type=EventType.REPLAY_TRIGGERED,
                source="ReviewTrigger",
                payload={
                    'trigger_type': trigger_request['type'].value,
                    'priority': trigger_request['priority'].name,
                    'context': trigger_request.get('context', {}),
                    'timestamp': trigger_request.get('timestamp', datetime.now()).isoformat()
                },
                priority=EventPriority.HIGH
            )
            
            # 异步发布事件（在线程池中调用）
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                event_bus.publish,
                event
            )
            
            logger.info(f"[ReviewTrigger] 📡 已发布 REPLAY_TRIGGERED 事件")
            
        except Exception as e:
            logger.error(f"[ReviewTrigger] 发布事件失败：{e}")
    
    async def _quiet_mode_monitor(self):
        """安静模式监控循环"""
        try:
            while self._running:
                await asyncio.sleep(10)  # 每 10 秒检查一次
                
                if not self._is_quiet_mode_enabled:
                    continue
                
                # 检查是否超时
                if self._last_user_activity is None:
                    continue
                
                time_since_activity = datetime.now() - self._last_user_activity
                
                if time_since_activity >= self.quiet_mode_timeout:
                    # 触发安静模式复盘
                    logger.info(f"[ReviewTrigger] 安静模式超时,触发复盘")
                    
                    await self._queue_trigger(
                        trigger_type=TriggerType.QUIET_MODE,
                        priority=TriggerPriority.MEDIUM,
                        context={
                            'quiet_duration_minutes': time_since_activity.total_seconds() / 60
                        }
                    )
                    
                    # 重置计时器(避免重复触发)
                    self._last_user_activity = datetime.now()
        
        except asyncio.CancelledError:
            logger.info(f"[ReviewTrigger] 安静模式监控已取消")
        except Exception as e:
            logger.error(f"[ReviewTrigger] 安静模式监控出错:{e}")
    
    async def _night_scheduler(self):
        """夜间定时调度器"""
        try:
            while self._running:
                now = datetime.now()
                
                # 计算下次触发时间
                next_trigger = now.replace(
                    hour=self.night_trigger_hour,
                    minute=self.night_trigger_minute,
                    second=0,
                    microsecond=0
                )
                
                # 如果已经过了今天的触发时间,设置为明天
                if now >= next_trigger:
                    next_trigger += timedelta(days=1)
                
                # 等待到触发时间
                sleep_seconds = (next_trigger - now).total_seconds()
                logger.debug(f"[ReviewTrigger] 距离夜间触发还有 {sleep_seconds:.0f} 秒")
                
                await asyncio.sleep(sleep_seconds)
                
                # 触发夜间复盘
                logger.info(f"[ReviewTrigger] 夜间定时触发复盘")
                
                await self._queue_trigger(
                    trigger_type=TriggerType.NIGHT_SCHEDULE,
                    priority=TriggerPriority.LOW,
                    context={
                        'scheduled_time': next_trigger.isoformat()
                    }
                )
        
        except asyncio.CancelledError:
            logger.info(f"[ReviewTrigger] 夜间调度器已取消")
        except Exception as e:
            logger.error(f"[ReviewTrigger] 夜间调度器出错:{e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            **self.stats,
            'is_running': self._running,
            'active_triggers': self._active_triggers,
            'queue_size': self._trigger_queue.qsize(),
            'last_user_activity': self._last_user_activity.isoformat() if self._last_user_activity else None,
            'quiet_mode_enabled': self._is_quiet_mode_enabled
        }
    
    def is_healthy(self) -> bool:
        """检查健康状态
        
        Returns:
            bool: 是否健康
        """
        # 正在运行
        if not self._running:
            return False
        
        # 队列未溢出(< 100)
        if self._trigger_queue.qsize() > 100:
            return False
        
        # 活跃触发数未超限
        if self._active_triggers > self.max_concurrent_triggers:
            return False
        
        return True


# 全局单例
_review_trigger_instance = None


def get_review_trigger(
    quiet_mode_timeout_minutes: int = 30,
    night_trigger_hour: int = 2,
    night_trigger_minute: int = 0
) -> ReviewTrigger:
    """获取复盘触发器单例
    
    Args:
        quiet_mode_timeout_minutes: 安静模式超时时间
        night_trigger_hour: 夜间触发小时
        night_trigger_minute: 夜间触发分钟
        
    Returns:
        ReviewTrigger: 单例实例
    """
    global _review_trigger_instance
    
    if _review_trigger_instance is None:
        _review_trigger_instance = ReviewTrigger(
            quiet_mode_timeout_minutes=quiet_mode_timeout_minutes,
            night_trigger_hour=night_trigger_hour,
            night_trigger_minute=night_trigger_minute
        )
    
    return _review_trigger_instance
