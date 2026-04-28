# File: zulong/utils/memory_manager.py
# 内存管理优化模块（Phase 7 任务 7.2）

"""
祖龙 (ZULONG) 内存管理优化模块

对应 TSD v1.7:
- Phase 7 任务 7.2: 性能优化与调优
- 内存优化
- 懒加载增强
- 智能预加载
- LRU 策略调优

功能:
1. 统一内存监控
2. 懒加载容器
3. 智能预加载
4. LRU 驱逐策略
5. 显存/内存协同管理
"""

import logging
import time
import psutil
import torch
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryStats:
    """内存统计信息"""
    # CPU 内存
    cpu_total_gb: float = 0.0
    cpu_used_gb: float = 0.0
    cpu_available_gb: float = 0.0
    cpu_percent: float = 0.0
    
    # GPU 显存
    gpu_total_gb: float = 0.0
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_percent: float = 0.0
    
    # 模型内存
    model_memory_gb: Dict[str, float] = field(default_factory=dict)
    
    # 时间戳
    timestamp: float = field(default_factory=time.time)


class LazyLoader(Generic[T]):
    """懒加载容器（优化版）
    
    优化点:
    1. 线程安全
    2. 自动卸载（weakref）
    3. 使用统计
    4. 预加载支持
    """
    
    def __init__(
        self,
        name: str,
        factory: Callable[[], T],
        preload: bool = False,
        auto_unload_timeout: Optional[float] = None
    ):
        self.name = name
        self._factory = factory
        self._preload = preload
        self._auto_unload_timeout = auto_unload_timeout
        
        self._instance: Optional[T] = None
        self._lock = threading.Lock()
        self._last_used = time.time()
        self._use_count = 0
        self._load_time: Optional[float] = None
        
        # 预加载
        if preload:
            self._preload_instance()
    
    def _preload_instance(self):
        """后台预加载"""
        logger.info(f"[LazyLoader:{self.name}] 后台预加载...")
        threading.Thread(target=self._load, daemon=True).start()
    
    def _load(self) -> T:
        """加载实例（线程安全）"""
        with self._lock:
            if self._instance is not None:
                return self._instance
            
            start_time = time.time()
            logger.info(f"[LazyLoader:{self.name}] 开始加载...")
            
            try:
                self._instance = self._factory()
                self._load_time = time.time() - start_time
                
                # 创建弱引用（自动卸载）
                if self._auto_unload_timeout:
                    self._weak_ref = weakref.ref(self._instance, self._on_unload)
                
                logger.info(f"[LazyLoader:{self.name}] 加载完成：{self._load_time:.2f}s")
                
                return self._instance
                
            except Exception as e:
                logger.error(f"[LazyLoader:{self.name}] 加载失败：{e}", exc_info=True)
                raise
    
    def _on_unload(self, ref):
        """实例被垃圾回收时的回调"""
        logger.info(f"[LazyLoader:{self.name}] 实例已被垃圾回收")
        self._instance = None
    
    def __getattr__(self, name: str):
        """自动加载并代理属性"""
        if self._instance is None:
            self._load()
        
        self._last_used = time.time()
        self._use_count += 1
        
        return getattr(self._instance, name)
    
    def unload(self):
        """手动卸载"""
        with self._lock:
            if self._instance is not None:
                logger.info(f"[LazyLoader:{self.name}] 手动卸载")
                del self._instance
                self._instance = None
    
    def is_loaded(self) -> bool:
        """检查是否已加载"""
        return self._instance is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'name': self.name,
            'loaded': self._instance is not None,
            'use_count': self._use_count,
            'last_used': time.time() - self._last_used,
            'load_time_s': self._load_time,
        }


class LRUCacheManager:
    """LRU 缓存管理器（优化版）
    
    优化点:
    1. 容量自适应
    2. 优先级驱逐
    3. 内存压力检测
    """
    
    def __init__(self, max_capacity: int = 100):
        self.max_capacity = max_capacity
        self.cache = OrderedDict()
        self.priority = {}  # 优先级映射
        self._lock = threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return default
    
    def put(self, key: str, value: Any, priority: int = 0):
        """添加缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
            priority: 优先级（0-10，越高越重要）
        """
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            self.priority[key] = priority
            
            # 检查容量
            while len(self.cache) > self.max_capacity:
                self._evict()
    
    def _evict(self):
        """驱逐最低优先级的缓存项"""
        if not self.cache:
            return
        
        # 找到最低优先级的项
        min_priority_key = min(self.priority, key=lambda k: self.priority.get(k, 0))
        
        logger.debug(f"[LRUCacheManager] 驱逐：{min_priority_key}")
        
        del self.cache[min_priority_key]
        del self.priority[min_priority_key]
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.priority.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'size': len(self.cache),
            'capacity': self.max_capacity,
            'utilization': len(self.cache) / self.max_capacity if self.max_capacity > 0 else 0,
        }


class MemoryManager:
    """统一内存管理器（单例模式）
    
    TSD v1.7 对应规则:
    - Phase 7 任务 7.2: 内存优化
    - 显存/内存协同管理
    - 智能驱逐策略
    
    功能:
    1. 实时监控
    2. 压力检测
    3. 自动优化
    4. 统一接口
    """
    
    _instance: Optional['MemoryManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'MemoryManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 内存池
        self.lazy_loaders: Dict[str, LazyLoader] = {}
        self.caches: Dict[str, LRUCacheManager] = {}
        
        # 监控
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._memory_pressure = 0.0  # 0.0-1.0
        
        # 统计
        self.stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'peak_memory_gb': 0.0,
            'current_memory_gb': 0.0,
        }
        
        self._initialized = True
        
        logger.info("[MemoryManager] 初始化完成")
    
    @classmethod
    def get_instance(cls) -> 'MemoryManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register_lazy_loader(
        self,
        name: str,
        factory: Callable,
        preload: bool = False,
        auto_unload_timeout: Optional[float] = None
    ):
        """注册懒加载器"""
        loader = LazyLoader(name, factory, preload, auto_unload_timeout)
        self.lazy_loaders[name] = loader
        logger.info(f"[MemoryManager] 注册懒加载器：{name}")
    
    def register_cache(self, name: str, max_capacity: int = 100):
        """注册缓存管理器"""
        cache = LRUCacheManager(max_capacity)
        self.caches[name] = cache
        logger.info(f"[MemoryManager] 注册缓存：{name} (capacity={max_capacity})")
    
    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计信息"""
        stats = MemoryStats()
        
        # CPU 内存
        mem = psutil.virtual_memory()
        stats.cpu_total_gb = mem.total / (1024**3)
        stats.cpu_used_gb = mem.used / (1024**3)
        stats.cpu_available_gb = mem.available / (1024**3)
        stats.cpu_percent = mem.percent
        
        # GPU 显存
        if torch.cuda.is_available():
            stats.gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats.gpu_allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
            stats.gpu_reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
            stats.gpu_percent = stats.gpu_allocated_gb / stats.gpu_total_gb * 100
        
        # 模型内存
        for name, loader in self.lazy_loaders.items():
            if loader.is_loaded():
                # 估算模型内存占用
                stats.model_memory_gb[name] = 2.0  # 预估值
        
        stats.timestamp = time.time()
        
        # 更新统计
        self.stats['current_memory_gb'] = stats.cpu_used_gb
        self.stats['peak_memory_gb'] = max(self.stats['peak_memory_gb'], stats.cpu_used_gb)
        
        return stats
    
    def get_memory_pressure(self) -> float:
        """获取内存压力（0.0-1.0）"""
        stats = self.get_memory_stats()
        
        # CPU 内存压力
        cpu_pressure = stats.cpu_percent / 100.0
        
        # GPU 显存压力
        gpu_pressure = stats.gpu_percent / 100.0 if stats.gpu_total_gb > 0 else 0.0
        
        # 综合压力（加权平均）
        self._memory_pressure = 0.7 * cpu_pressure + 0.3 * gpu_pressure
        
        return self._memory_pressure
    
    def optimize(self):
        """执行内存优化
        
        优化策略:
        1. 卸载未使用的懒加载器
        2. 清空低优先级缓存
        3. 垃圾回收
        """
        pressure = self.get_memory_pressure()
        
        logger.info(f"[MemoryManager] 执行优化，当前压力：{pressure:.2f}")
        
        if pressure < 0.5:
            # 压力低，无需优化
            return
        
        # 卸载长时间未使用的懒加载器
        current_time = time.time()
        for name, loader in list(self.lazy_loaders.items()):
            if loader.is_loaded():
                idle_time = current_time - loader._last_used
                if idle_time > 300:  # 5 分钟未使用
                    logger.info(f"[MemoryManager] 卸载未使用：{name} (idle={idle_time:.1f}s)")
                    loader.unload()
                    self.stats['total_deallocations'] += 1
        
        # 清空低优先级缓存
        for name, cache in self.caches.items():
            if pressure > 0.8:
                # 高压：清空所有缓存
                cache.clear()
                logger.info(f"[MemoryManager] 清空缓存：{name}")
            else:
                # 中压：减少容量
                cache.max_capacity = max(10, cache.max_capacity // 2)
                while len(cache.cache) > cache.max_capacity:
                    cache._evict()
                logger.info(f"[MemoryManager] 缩减缓存：{name} -> {cache.max_capacity}")
        
        # 垃圾回收
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("[MemoryManager] 优化完成")
    
    def start_monitoring(self, interval: float = 5.0):
        """启动内存监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                pressure = self.get_memory_pressure()
                
                if pressure > 0.7:
                    logger.warning(f"[MemoryManager] 内存压力高：{pressure:.2f}")
                    self.optimize()
                
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"[MemoryManager] 监控已启动 (interval={interval}s)")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("[MemoryManager] 监控已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.get_memory_stats()
        
        return {
            'cpu': {
                'total_gb': stats.cpu_total_gb,
                'used_gb': stats.cpu_used_gb,
                'available_gb': stats.cpu_available_gb,
                'percent': stats.cpu_percent,
            },
            'gpu': {
                'total_gb': stats.gpu_total_gb,
                'allocated_gb': stats.gpu_allocated_gb,
                'reserved_gb': stats.gpu_reserved_gb,
                'percent': stats.gpu_percent,
            },
            'lazy_loaders': {
                name: loader.get_stats()
                for name, loader in self.lazy_loaders.items()
            },
            'caches': {
                name: cache.get_stats()
                for name, cache in self.caches.items()
            },
            'manager': self.stats,
            'pressure': self._memory_pressure,
        }


# 全局内存管理器实例
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """获取全局内存管理器"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager.get_instance()
    return _memory_manager


def optimize_memory():
    """快捷优化内存"""
    manager = get_memory_manager()
    manager.optimize()


def monitor_memory(interval: float = 5.0):
    """快捷启动监控"""
    manager = get_memory_manager()
    manager.start_monitoring(interval)
