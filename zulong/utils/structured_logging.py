# File: zulong/utils/structured_logging.py
# 结构化日志模块

"""
祖龙 (ZULONG) 结构化日志模块

对应 TSD v1.7:
- Phase 6 任务 6.4: 系统稳定性增强
- 结构化日志、性能追踪、错误审计

功能:
- JSON 格式日志
- 模块分类
- 性能指标记录
- 错误追踪
- 日志级别控制
"""

import logging
import json
import time
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import traceback


class JSONFormatter(logging.Formatter):
    """JSON 格式日志格式化器
    
    TSD v1.7 对应规则:
    - Phase 6: 结构化日志
    - 机器可读格式
    - 便于日志分析系统处理
    """
    
    def __init__(self, include_extra: bool = True):
        """初始化 JSON 格式化器
        
        Args:
            include_extra: 是否包含额外字段
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为 JSON
        
        Args:
            record: 日志记录
            
        Returns:
            str: JSON 格式日志
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # 添加额外字段
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                              'levelname', 'levelno', 'lineno', 'module', 'msecs',
                              'pathname', 'process', 'processName', 'relativeCreated',
                              'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName']:
                    try:
                        json.dumps(value)  # 验证可序列化
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[f'{key}_str'] = str(value)
        
        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """结构化日志器
    
    TSD v1.7 对应规则:
    - Phase 6: 系统稳定性增强
    - 统一日志接口
    - 性能追踪
    
    功能:
    - JSON 格式日志
    - 性能指标记录
    - 错误追踪
    - 上下文管理
    """
    
    def __init__(self, 
                 name: str,
                 log_file: Optional[str] = None,
                 level: int = logging.INFO,
                 enable_json: bool = True):
        """初始化结构化日志器
        
        Args:
            name: 日志器名称
            log_file: 日志文件路径（可选）
            level: 日志级别
            enable_json: 是否启用 JSON 格式
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # 清除现有处理器
        
        # 创建处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if enable_json:
            # JSON 格式
            console_handler.setFormatter(JSONFormatter())
        else:
            # 传统格式
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        self.logger.addHandler(console_handler)
        
        # 文件处理器（可选）
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)
        
        # 性能追踪
        self._perf_metrics: Dict[str, list] = {}
    
    def debug(self, message: str, **kwargs):
        """调试日志
        
        Args:
            message: 日志消息
            **kwargs: 额外字段
        """
        extra = {'extra_fields': kwargs}
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """信息日志
        
        Args:
            message: 日志消息
            **kwargs: 额外字段
        """
        extra = {'extra_fields': kwargs}
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """警告日志
        
        Args:
            message: 日志消息
            **kwargs: 额外字段
        """
        extra = {'extra_fields': kwargs}
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """错误日志
        
        Args:
            message: 日志消息
            exc_info: 是否包含异常信息
            **kwargs: 额外字段
        """
        extra = {'extra_fields': kwargs}
        self.logger.error(message, exc_info=exc_info, extra=extra)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """严重错误日志
        
        Args:
            message: 日志消息
            exc_info: 是否包含异常信息
            **kwargs: 额外字段
        """
        extra = {'extra_fields': kwargs}
        self.logger.critical(message, exc_info=exc_info, extra=extra)
    
    def performance(self, operation: str, duration_ms: float, **kwargs):
        """性能指标日志
        
        Args:
            operation: 操作名称
            duration_ms: 耗时（毫秒）
            **kwargs: 额外字段
        """
        log_data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': time.time()
        }
        log_data.update(kwargs)
        
        self.info(f"[PERF] {operation} completed in {duration_ms:.2f}ms", **log_data)
        
        # 记录性能指标
        if operation not in self._perf_metrics:
            self._perf_metrics[operation] = []
        self._perf_metrics[operation].append(duration_ms)
        
        # 保留最近 1000 条
        if len(self._perf_metrics[operation]) > 1000:
            self._perf_metrics[operation].pop(0)
    
    def get_performance_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """获取性能统计
        
        Args:
            operation: 操作名称（可选，返回所有）
            
        Returns:
            Dict: 性能统计信息
        """
        if operation:
            if operation not in self._perf_metrics:
                return {}
            
            durations = self._perf_metrics[operation]
            return {
                'operation': operation,
                'count': len(durations),
                'avg_ms': sum(durations) / len(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'p95_ms': sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 20 else max(durations)
            }
        else:
            stats = {}
            for op, durations in self._perf_metrics.items():
                stats[op] = {
                    'count': len(durations),
                    'avg_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations)
                }
            return stats
    
    def context(self, context_data: Dict[str, Any]):
        """创建上下文日志
        
        Args:
            context_data: 上下文数据
            
        Returns:
            StructuredLogger: self
        """
        self.info("Context", **context_data)
        return self


class PerformanceTracker:
    """性能追踪器
    
    TSD v1.7 对应规则:
    - Phase 6: 性能监控
    - 自动记录操作耗时
    
    用法:
        with PerformanceTracker(logger, "operation_name"):
            # 执行操作
            pass
    """
    
    def __init__(self, logger: StructuredLogger, operation: str, **kwargs):
        """初始化性能追踪器
        
        Args:
            logger: 结构化日志器
            operation: 操作名称
            **kwargs: 额外字段
        """
        self.logger = logger
        self.operation = operation
        self.extra_kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        """进入上下文"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        duration_ms = (time.time() - self.start_time) * 1000
        self.logger.performance(
            self.operation,
            duration_ms,
            success=exc_type is None,
            **self.extra_kwargs
        )
        
        # 记录异常
        if exc_type:
            self.logger.error(
                f"[PERF] {self.operation} failed",
                exc_info=True,
                operation=self.operation,
                duration_ms=duration_ms
            )


def get_structured_logger(name: str,
                         log_file: Optional[str] = None,
                         level: int = logging.INFO,
                         enable_json: bool = True) -> StructuredLogger:
    """获取结构化日志器
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径
        level: 日志级别
        enable_json: 是否启用 JSON 格式
        
    Returns:
        StructuredLogger: 结构化日志器
    """
    return StructuredLogger(name, log_file, level, enable_json)


# 全局日志器实例
_global_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str) -> Optional[StructuredLogger]:
    """获取全局日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        StructuredLogger: 日志器实例
    """
    return _global_loggers.get(name)


def init_logger(name: str, **kwargs) -> StructuredLogger:
    """初始化全局日志器
    
    Args:
        name: 日志器名称
        **kwargs: 初始化参数
        
    Returns:
        StructuredLogger: 日志器实例
    """
    logger = get_structured_logger(name, **kwargs)
    _global_loggers[name] = logger
    return logger
