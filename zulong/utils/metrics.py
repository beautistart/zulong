# File: zulong/utils/metrics.py
# Prometheus 监控指标模块

"""
祖龙 (ZULONG) Prometheus 监控指标模块

对应 TSD v1.7:
- Phase 6 任务 6.4: 系统稳定性增强
- Prometheus 指标导出、性能仪表板

功能:
- Counter 指标（计数器）
- Gauge 指标（仪表）
- Histogram 指标（直方图）
- Summary 指标（摘要）
- Prometheus 格式导出
"""

import time
import threading
from typing import Dict, Any, Optional, List
from collections import defaultdict
import json


class MetricType:
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class Metric:
    """指标基类"""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """初始化指标
        
        Args:
            name: 指标名称
            description: 指标描述
            labels: 标签列表
        """
        self.name = name
        self.description = description
        self.labels = labels or []
        self._lock = threading.Lock()
    
    def get_prometheus_format(self) -> str:
        """获取 Prometheus 格式
        
        Returns:
            str: Prometheus 格式指标
        """
        raise NotImplementedError


class Counter(Metric):
    """计数器指标
    
    特性:
    - 只增不减
    - 用于计数操作次数
    """
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self._values: Dict[str, float] = defaultdict(float)
    
    def inc(self, value: float = 1.0, **label_values):
        """增加计数
        
        Args:
            value: 增加的值
            **label_values: 标签值
        """
        key = self._make_key(**label_values)
        with self._lock:
            self._values[key] += value
    
    def get(self, **label_values) -> float:
        """获取计数值
        
        Args:
            **label_values: 标签值
            
        Returns:
            float: 计数值
        """
        key = self._make_key(**label_values)
        return self._values.get(key, 0.0)
    
    def _make_key(self, **label_values) -> str:
        """生成标签键
        
        Returns:
            str: 标签键
        """
        pairs = []
        for label in self.labels:
            value = label_values.get(label, "")
            pairs.append(f'{label}="{value}"')
        return ",".join(pairs)
    
    def get_prometheus_format(self) -> str:
        """获取 Prometheus 格式
        
        Returns:
            str: Prometheus 格式
        """
        lines = []
        lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} {MetricType.COUNTER}")
        
        for key, value in self._values.items():
            if key:
                lines.append(f"{self.name}{{{key}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        
        return "\n".join(lines)


class Gauge(Metric):
    """仪表指标
    
    特性:
    - 可增可减
    - 用于表示当前状态
    """
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self._values: Dict[str, float] = defaultdict(float)
    
    def set(self, value: float, **label_values):
        """设置值
        
        Args:
            value: 值
            **label_values: 标签值
        """
        key = self._make_key(**label_values)
        with self._lock:
            self._values[key] = value
    
    def inc(self, value: float = 1.0, **label_values):
        """增加"""
        key = self._make_key(**label_values)
        with self._lock:
            self._values[key] += value
    
    def dec(self, value: float = 1.0, **label_values):
        """减少"""
        key = self._make_key(**label_values)
        with self._lock:
            self._values[key] -= value
    
    def get(self, **label_values) -> float:
        """获取值"""
        key = self._make_key(**label_values)
        return self._values.get(key, 0.0)
    
    def _make_key(self, **label_values) -> str:
        """生成标签键"""
        pairs = []
        for label in self.labels:
            value = label_values.get(label, "")
            pairs.append(f'{label}="{value}"')
        return ",".join(pairs)
    
    def get_prometheus_format(self) -> str:
        """获取 Prometheus 格式"""
        lines = []
        lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} {MetricType.GAUGE}")
        
        for key, value in self._values.items():
            if key:
                lines.append(f"{self.name}{{{key}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        
        return "\n".join(lines)


class Histogram(Metric):
    """直方图指标
    
    特性:
    - 记录分布
    - 支持分位数查询
    """
    
    def __init__(self, name: str, description: str, 
                 buckets: Optional[List[float]] = None,
                 labels: Optional[List[str]] = None):
        """初始化直方图
        
        Args:
            name: 指标名称
            description: 指标描述
            buckets: 桶边界（默认：[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]）
            labels: 标签列表
        """
        super().__init__(name, description, labels)
        self.buckets = sorted(buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
        self.buckets.append(float('inf'))
        
        # 每个桶的计数
        self._bucket_counts: Dict[str, Dict[float, int]] = defaultdict(lambda: defaultdict(int))
        # 总和
        self._sums: Dict[str, float] = defaultdict(float)
        # 总计数
        self._counts: Dict[str, int] = defaultdict(int)
    
    def observe(self, value: float, **label_values):
        """观察值
        
        Args:
            value: 观察值
            **label_values: 标签值
        """
        key = self._make_key(**label_values)
        with self._lock:
            # 更新桶计数
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[key][bucket] += 1
            
            # 更新总和和总计数
            self._sums[key] += value
            self._counts[key] += 1
    
    def _make_key(self, **label_values) -> str:
        """生成标签键"""
        pairs = []
        for label in self.labels:
            value = label_values.get(label, "")
            pairs.append(f'{label}="{value}"')
        return ",".join(pairs)
    
    def get_prometheus_format(self) -> str:
        """获取 Prometheus 格式"""
        lines = []
        lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} {MetricType.HISTOGRAM}")
        
        for key in self._counts.keys():
            prefix = f"{self.name}{{{key}}}" if key else f"{self.name}"
            
            # 桶
            for bucket in self.buckets[:-1]:  # 不包括 +Inf
                count = self._bucket_counts[key][bucket]
                lines.append(f"{prefix}_bucket{{le=\"{bucket}\"}} {count}")
            
            # +Inf
            inf_count = self._counts[key]
            lines.append(f"{prefix}_bucket{{le=\"+Inf\"}} {inf_count}")
            
            # 总和和计数
            lines.append(f"{prefix}_sum {self._sums[key]}")
            lines.append(f"{prefix}_count {self._counts[key]}")
        
        return "\n".join(lines)


class MetricsRegistry:
    """指标注册表
    
    TSD v1.7 对应规则:
    - Phase 6: 监控指标
    - 统一管理所有指标
    - Prometheus 格式导出
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._metrics: Dict[str, Metric] = {}
        self._initialized = True
    
    def register_counter(self, name: str, description: str, 
                        labels: Optional[List[str]] = None) -> Counter:
        """注册计数器
        
        Args:
            name: 指标名称
            description: 指标描述
            labels: 标签列表
            
        Returns:
            Counter: 计数器指标
        """
        if name in self._metrics:
            return self._metrics[name]
        
        counter = Counter(name, description, labels)
        self._metrics[name] = counter
        return counter
    
    def register_gauge(self, name: str, description: str,
                      labels: Optional[List[str]] = None) -> Gauge:
        """注册仪表指标"""
        if name in self._metrics:
            return self._metrics[name]
        
        gauge = Gauge(name, description, labels)
        self._metrics[name] = gauge
        return gauge
    
    def register_histogram(self, name: str, description: str,
                          buckets: Optional[List[float]] = None,
                          labels: Optional[List[str]] = None) -> Histogram:
        """注册直方图"""
        if name in self._metrics:
            return self._metrics[name]
        
        histogram = Histogram(name, description, buckets, labels)
        self._metrics[name] = histogram
        return histogram
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """获取指标"""
        return self._metrics.get(name)
    
    def get_prometheus_format(self) -> str:
        """获取 Prometheus 格式所有指标
        
        Returns:
            str: Prometheus 格式
        """
        lines = []
        for metric in self._metrics.values():
            lines.append(metric.get_prometheus_format())
            lines.append("")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for name, metric in self._metrics.items():
            if isinstance(metric, Counter):
                result[name] = {
                    'type': 'counter',
                    'value': {k: v for k, v in metric._values.items()}
                }
            elif isinstance(metric, Gauge):
                result[name] = {
                    'type': 'gauge',
                    'value': {k: v for k, v in metric._values.items()}
                }
        return result


# 全局注册表
_global_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """获取全局指标注册表
    
    Returns:
        MetricsRegistry: 注册表实例
    """
    return _global_registry


# 预定义指标
_zulong_counters = {}
_zulong_gauges = {}
_zulong_histograms = {}


def init_zulong_metrics():
    """初始化祖龙系统指标"""
    global _zulong_counters, _zulong_gauges, _zulong_histograms
    
    registry = get_metrics_registry()
    
    # 计数器
    _zulong_counters['dwa_planning_total'] = registry.register_counter(
        'zulong_dwa_planning_total',
        'Total number of DWA planning operations'
    )
    
    _zulong_counters['navigation_success_total'] = registry.register_counter(
        'zulong_navigation_success_total',
        'Total number of successful navigations'
    )
    
    _zulong_counters['vision_detection_total'] = registry.register_counter(
        'zulong_vision_detection_total',
        'Total number of object detections'
    )
    
    _zulong_counters['skill_invocation_total'] = registry.register_counter(
        'zulong_skill_invocation_total',
        'Total number of skill invocations',
        labels=['skill_type']
    )
    
    # 仪表
    _zulong_gauges['active_skills'] = registry.register_gauge(
        'zulong_active_skills',
        'Number of currently loaded skills'
    )
    
    _zulong_gauges['cpu_memory_usage'] = registry.register_gauge(
        'zulong_cpu_memory_usage_bytes',
        'CPU memory usage in bytes'
    )
    
    _zulong_gauges['gpu_memory_usage'] = registry.register_gauge(
        'zulong_gpu_memory_usage_bytes',
        'GPU memory usage in bytes'
    )
    
    # 直方图
    _zulong_histograms['dwa_planning_duration'] = registry.register_histogram(
        'zulong_dwa_planning_duration_seconds',
        'DWA planning duration in seconds',
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    
    _zulong_histograms['navigation_duration'] = registry.register_histogram(
        'zulong_navigation_duration_seconds',
        'Navigation duration in seconds',
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    
    _zulong_histograms['vision_inference_duration'] = registry.register_histogram(
        'zulong_vision_inference_duration_seconds',
        'Vision inference duration in seconds',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )


def get_zulong_metrics() -> Dict[str, Any]:
    """获取祖龙系统指标
    
    Returns:
        Dict: 指标字典
    """
    return {
        'counters': _zulong_counters,
        'gauges': _zulong_gauges,
        'histograms': _zulong_histograms
    }
