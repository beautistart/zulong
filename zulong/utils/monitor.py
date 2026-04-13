# File: zulong/utils/monitor.py
# 全局监控与日志 - 第五阶段总装
# 对应 TSD v1.7: 全链路追踪和性能监控

import logging
import time
import threading
import uuid
import psutil
from datetime import datetime

# 颜色定义
class Colors:
    L1A = '\033[92m'  # 绿色
    L1B = '\033[94m'  # 蓝色
    L2 = '\033[95m'  # 紫色
    L0 = '\033[91m'  # 红色
    RESET = '\033[0m'  # 重置

# 模块颜色映射
MODULE_COLORS = {
    'L1-A': Colors.L1A,
    'L1B': Colors.L1B,
    'L2': Colors.L2,
    'L0': Colors.L0,
    'EventBus': Colors.L1B,
    'StateManager': Colors.L1B,
    'ReflexController': Colors.L1A,
    'Gatekeeper': Colors.L1B,
    'InterruptController': Colors.L2,
    'InferenceEngine': Colors.L2,
    'TaskStateManager': Colors.L2,
    'SensorSimulator': Colors.L1A,
    'UserSimulator': Colors.L1B,
    'Bootstrap': Colors.L1B,
    'PerformanceTracker': Colors.L2
}

# 自定义日志格式化器
class ZulongFormatter(logging.Formatter):
    """祖龙系统日志格式化器"""
    
    def format(self, record):
        # 获取模块颜色
        module = record.name.split('.')[-1]
        color = MODULE_COLORS.get(module, Colors.RESET)
        
        # 生成或获取 TraceID
        if not hasattr(record, 'trace_id'):
            record.trace_id = str(uuid.uuid4())[:8]
        
        # 格式化时间
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # 构建日志消息
        message = f"[{timestamp}] {color}[{module}] {Colors.RESET}[{record.trace_id}] {record.getMessage()}"
        
        # 添加异常信息
        if record.exc_info:
            message += '\n' + self.formatException(record.exc_info)
        
        return message

# 全局 TraceID 管理
class TraceManager:
    """TraceID 管理器"""
    _local = threading.local()
    
    @classmethod
    def get_trace_id(cls):
        """获取当前线程的 TraceID"""
        if not hasattr(cls._local, 'trace_id'):
            cls._local.trace_id = str(uuid.uuid4())[:8]
        return cls._local.trace_id
    
    @classmethod
    def set_trace_id(cls, trace_id):
        """设置当前线程的 TraceID"""
        cls._local.trace_id = trace_id
    
    @classmethod
    def new_trace_id(cls):
        """生成新的 TraceID"""
        trace_id = str(uuid.uuid4())[:8]
        cls._local.trace_id = trace_id
        return trace_id

# 性能跟踪器
class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self._running = False
        self._latencies = {}
        self._start_time = None
        self._cpu_usage = []
        self._lock = threading.Lock()
    
    def start(self):
        """开始性能监控"""
        self._running = True
        self._start_time = time.time()
        # 启动 CPU 监控线程
        self._cpu_thread = threading.Thread(target=self._monitor_cpu, daemon=True)
        self._cpu_thread.start()
    
    def stop(self):
        """停止性能监控"""
        self._running = False
        if hasattr(self, '_cpu_thread'):
            self._cpu_thread.join(timeout=2)
    
    def record_latency(self, event_type, duration):
        """记录事件处理延迟
        
        Args:
            event_type: 事件类型
            duration: 处理时间（毫秒）
        """
        with self._lock:
            if event_type not in self._latencies:
                self._latencies[event_type] = []
            self._latencies[event_type].append(duration)
    
    def _monitor_cpu(self):
        """监控 CPU 使用率"""
        while self._running:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                with self._lock:
                    self._cpu_usage.append(cpu_percent)
            except Exception:
                pass
    
    def print_summary(self):
        """打印性能摘要"""
        print("\n" + "="*60)
        print(" 📊 ZULONG System Performance Summary")
        print("="*60)
        
        # 计算运行时间
        if self._start_time:
            runtime = time.time() - self._start_time
            print(f"Runtime: {runtime:.2f} seconds")
        
        # 打印 CPU 使用率
        if self._cpu_usage:
            avg_cpu = sum(self._cpu_usage) / len(self._cpu_usage)
            max_cpu = max(self._cpu_usage)
            print(f"CPU Usage: Avg {avg_cpu:.2f}%, Max {max_cpu:.2f}%")
        
        # 打印事件延迟
        print("\nEvent Latency (ms):")
        print("-" * 40)
        for event_type, latencies in self._latencies.items():
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                print(f"{event_type}: Avg {avg_latency:.2f}, Max {max_latency:.2f}")
        
        print("="*60 + "\n")

# 全局性能跟踪器实例
performance_tracker = PerformanceTracker()

# 全局日志配置
def setup_logging():
    """设置日志配置"""
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 移除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 使用自定义格式化器
    formatter = ZulongFormatter()
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根日志器
    root_logger.addHandler(console_handler)

# 增强的日志记录函数
def log_with_trace(logger, level, message, trace_id=None, **kwargs):
    """带 TraceID 的日志记录
    
    Args:
        logger: 日志器实例
        level: 日志级别
        message: 日志消息
        trace_id: 可选的 TraceID
        **kwargs: 其他参数
    """
    if trace_id is None:
        trace_id = TraceManager.get_trace_id()
    
    # 创建日志记录
    extra = {'trace_id': trace_id}
    extra.update(kwargs)
    
    # 记录日志
    if level == logging.DEBUG:
        logger.debug(message, extra=extra)
    elif level == logging.INFO:
        logger.info(message, extra=extra)
    elif level == logging.WARNING:
        logger.warning(message, extra=extra)
    elif level == logging.ERROR:
        logger.error(message, extra=extra)
    elif level == logging.CRITICAL:
        logger.critical(message, extra=extra)
