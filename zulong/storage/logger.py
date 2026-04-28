# File: zulong/storage/logger.py
# 祖龙 (ZULONG) 异步日志收集器

"""
异步日志收集器 - 高性能日志处理

功能:
1. 异步日志队列
2. 批量刷新
3. L1-B 核心循环集成
4. 日志分级

对应 TSD v2.3 第 9.2 节：日志收集器
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import deque
import json

logger = logging.getLogger(__name__)


class AsyncLogCollector:
    """异步日志收集器"""
    
    def __init__(
        self,
        queue_size: int = 1000,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        storage_backend=None
    ):
        """
        初始化日志收集器
        
        Args:
            queue_size: 队列最大长度
            batch_size: 批量刷新大小
            flush_interval: 刷新间隔（秒）
            storage_backend: 存储后端（如 MongoDB）
        """
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.storage_backend = storage_backend
        
        # 日志队列
        self.log_queue: deque = deque(maxlen=queue_size)
        
        # 刷新任务
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 统计信息
        self.stats = {
            'total_logs': 0,
            'flushed_logs': 0,
            'dropped_logs': 0
        }
        
        logger.info(f"异步日志收集器已初始化 (queue_size={queue_size}, batch_size={batch_size})")
    
    async def start(self):
        """启动日志收集器"""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("异步日志收集器已启动")
    
    async def stop(self):
        """停止日志收集器"""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # 刷新剩余日志
        await self._flush_logs()
        
        logger.info("异步日志收集器已停止")
    
    def log(
        self,
        level: str,
        message: str,
        module: str = "",
        user_input: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        记录日志（同步接口）
        
        Args:
            level: 日志级别 (INFO/WARNING/ERROR)
            message: 日志消息
            module: 模块名称
            user_input: 用户输入
            system_state: 系统状态
            metadata: 元数据
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'module': module,
            'user_input': user_input,
            'system_state': system_state,
            'metadata': metadata or {}
        }
        
        # 检查队列是否已满
        if len(self.log_queue) >= self.queue_size:
            self.stats['dropped_logs'] += 1
            logger.warning(f"日志队列已满，丢弃日志：{message}")
        
        self.log_queue.append(log_entry)
        self.stats['total_logs'] += 1
    
    async def log_async(
        self,
        level: str,
        message: str,
        **kwargs
    ):
        """
        异步记录日志
        
        Args:
            level: 日志级别
            message: 日志消息
            **kwargs: 其他参数
        """
        self.log(level, message, **kwargs)
    
    async def _flush_loop(self):
        """定期刷新循环"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                
                # 检查是否需要刷新
                if len(self.log_queue) >= self.batch_size:
                    await self._flush_logs()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"日志刷新循环错误：{e}")
    
    async def _flush_logs(self):
        """刷新日志到存储后端"""
        if not self.log_queue:
            return
        
        # 批量获取日志
        logs_to_flush = []
        while self.log_queue and len(logs_to_flush) < self.batch_size:
            logs_to_flush.append(self.log_queue.popleft())
        
        try:
            # 存储到后端
            if self.storage_backend:
                await self.storage_backend.batch_insert(logs_to_flush)
            
            self.stats['flushed_logs'] += len(logs_to_flush)
            
            logger.debug(f"已刷新 {len(logs_to_flush)} 条日志")
            
        except Exception as e:
            logger.error(f"刷新日志失败：{e}")
            # 重新加入队列（避免丢失）
            for log in reversed(logs_to_flush):
                self.log_queue.appendleft(log)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息
        """
        return {
            **self.stats,
            'queue_size': len(self.log_queue),
            'queue_max_size': self.queue_size
        }
    
    def get_recent_logs(
        self,
        count: int = 100,
        level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取最近的日志
        
        Args:
            count: 获取数量
            level: 日志级别过滤
        
        Returns:
            日志列表
        """
        logs = list(self.log_queue)[-count:]
        
        if level:
            logs = [log for log in logs if log['level'] == level]
        
        return logs
    
    def export_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json"
    ) -> str:
        """
        导出日志
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            format: 导出格式 (json/csv)
        
        Returns:
            导出的日志字符串
        """
        logs = list(self.log_queue)
        
        # 时间过滤
        if start_time:
            logs = [
                log for log in logs
                if datetime.fromisoformat(log['timestamp']) >= start_time
            ]
        
        if end_time:
            logs = [
                log for log in logs
                if datetime.fromisoformat(log['timestamp']) <= end_time
            ]
        
        # 格式化
        if format == "json":
            return json.dumps(logs, ensure_ascii=False, indent=2)
        elif format == "csv":
            # CSV 格式
            if not logs:
                return ""
            
            headers = list(logs[0].keys())
            lines = [",".join(headers)]
            
            for log in logs:
                values = [str(log.get(h, "")) for h in headers]
                lines.append(",".join(values))
            
            return "\n".join(lines)
        else:
            raise ValueError(f"不支持的格式：{format}")


# 单例模式
_log_collector_instance: Optional[AsyncLogCollector] = None


def get_log_collector(
    queue_size: int = 1000,
    batch_size: int = 50,
    flush_interval: float = 5.0,
    **kwargs
) -> AsyncLogCollector:
    """
    获取日志收集器单例
    
    Args:
        queue_size: 队列大小
        batch_size: 批量大小
        flush_interval: 刷新间隔
        **kwargs: 其他参数
    
    Returns:
        AsyncLogCollector 实例
    """
    global _log_collector_instance
    
    if _log_collector_instance is None:
        _log_collector_instance = AsyncLogCollector(
            queue_size=queue_size,
            batch_size=batch_size,
            flush_interval=flush_interval,
            **kwargs
        )
    
    return _log_collector_instance


# 便捷函数
def log_info(message: str, **kwargs):
    """记录 INFO 级别日志"""
    collector = get_log_collector()
    collector.log("INFO", message, **kwargs)


def log_warning(message: str, **kwargs):
    """记录 WARNING 级别日志"""
    collector = get_log_collector()
    collector.log("WARNING", message, **kwargs)


def log_error(message: str, **kwargs):
    """记录 ERROR 级别日志"""
    collector = get_log_collector()
    collector.log("ERROR", message, **kwargs)
