# File: zulong/core/power_manager.py
# 电源管理器 - 第五阶段总装
# 对应 TSD v1.7: 电源管理

import time
import threading

from zulong.core.state_manager import state_manager
from zulong.core.types import L2Status

import logging
logger = logging.getLogger(__name__)


class PowerManager:
    """电源管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化电源管理器"""
        if not hasattr(self, '_initialized'):
            self._lock = threading.Lock()
            self._initialized = True
            logger.info("PowerManager initialized")
    
    def unload_to_cpu(self):
        """将 L2 卸载到 CPU
        
        对应 TSD v1.7: 安静模式下卸载 L2 以节能
        """
        with self._lock:
            logger.info("Unloading L2 to CPU")
            # 模拟卸载过程
            time.sleep(0.5)  # 模拟卸载时间
            state_manager.set_l2_status(L2Status.UNLOADED)
            logger.info("L2 unloaded to CPU successfully")
    
    def load_to_gpu(self):
        """将 L2 加载到 GPU
        
        对应 TSD v1.7: 唤醒时热加载 L2
        """
        with self._lock:
            logger.info("Loading L2 to GPU")
            # 模拟加载过程
            time.sleep(0.8)  # 模拟加载时间
            state_manager.set_l2_status(L2Status.IDLE)
            logger.info("L2 loaded to GPU successfully")


# 全局电源管理器实例
power_manager = PowerManager()
