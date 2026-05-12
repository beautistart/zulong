# File: zulong/l1b/memory_config_initializer.py
"""
记忆配置初始化器 - TSD v2.4

功能：
1. 在系统启动时初始化动态阈值管理器
2. 从模型配置读取参数并计算阈值
3. 启动显存监控线程
4. 注册回调函数

对应 TSD v2.4: 资源自适应、动态容量管理
"""

import logging
import asyncio
import threading
import time
from typing import Dict, Any

from zulong.l1b.dynamic_threshold_manager import get_dynamic_threshold_manager
from zulong.models.config import ModelID, get_model_config

logger = logging.getLogger(__name__)


class MemoryConfigInitializer:
    """记忆配置初始化器"""
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.threshold_manager = get_dynamic_threshold_manager()
        self.vram_monitor_thread = None
        self.vram_monitor_running = False
        
        logger.info("[MemoryConfigInitializer] 初始化完成")
        self._initialized = True
    
    def initialize_from_model_config(self, model_id: ModelID = ModelID.L2_PRIME):
        """
        从模型配置初始化动态阈值
        
        Args:
            model_id: 模型 ID，默认使用 L2_PRIME
        """
        try:
            # 获取模型配置
            model_config = get_model_config(model_id)
            
            logger.info(f"[MemoryConfigInitializer] 从模型配置初始化：{model_config.model_id.value}")
            logger.info(f"  - 模型名称：{model_config.name}")
            logger.info(f"  - 模型大小：{model_config.size_billions}B")
            logger.info(f"  - 量化级别：{model_config.quantization}")
            logger.info(f"  - 最大上下文：{model_config.max_context_length}")
            
            # 构建配置字典
            config_dict = {
                'name': model_config.name,
                'size_in_billions': model_config.size_billions,
                'max_context_window': model_config.max_context_length,
                'quantization': model_config.quantization,
                'vram_limit_gb': self._estimate_vram_limit(model_config)
            }
            
            # 初始化动态阈值管理器
            self.threshold_manager.initialize_with_model_config(config_dict)
            
            # 启动显存监控线程
            self._start_vram_monitor()
            
            logger.info(f"[MemoryConfigInitializer] ✅ 初始化完成")
            logger.info(f"  - 硬上限：{self.threshold_manager.hard_token_limit} tokens")
            logger.info(f"  - 软上限：{self.threshold_manager.soft_turn_limit} 轮")
            
        except Exception as e:
            logger.error(f"[MemoryConfigInitializer] 初始化失败：{e}", exc_info=True)
    
    def _estimate_vram_limit(self, model_config) -> float:
        """
        估算模型显存限制
        
        Args:
            model_config: 模型配置
            
        Returns:
            float: 估算的显存限制（GB）
        """
        # 简化估算：模型大小 × 2 + 2GB（缓存）
        size_gb = model_config.size_billions * 2 + 2
        
        # 根据量化调整
        if 'int4' in model_config.quantization.lower():
            size_gb *= 0.5
        elif 'int8' in model_config.quantization.lower():
            size_gb *= 0.75
        
        return size_gb
    
    def _start_vram_monitor(self):
        """启动显存监控线程"""
        self.vram_monitor_running = True
        self.vram_monitor_thread = threading.Thread(
            target=self._vram_monitor_loop,
            daemon=True,
            name="VRAM_Monitor"
        )
        self.vram_monitor_thread.start()
        logger.info("[MemoryConfigInitializer] ✅ 显存监控线程已启动")
    
    def _vram_monitor_loop(self):
        """显存监控线程循环"""
        logger.info("[VRAM_Monitor] 显存监控线程运行中...")
        
        while self.vram_monitor_running:
            try:
                # 🔥 获取当前显存使用率
                # 注意：这里需要使用 pynvml 或其他 GPU 监控库
                # 为了简化，这里使用模拟值
                vram_usage = self._get_gpu_vram_usage()
                
                # 更新到阈值管理器
                self.threshold_manager.update_vram_usage(vram_usage)
                
                # 每秒检查一次
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"[VRAM_Monitor] 监控失败：{e}")
                time.sleep(5.0)  # 失败后等待 5 秒再试
    
    def _get_gpu_vram_usage(self) -> float:
        """
        获取 GPU 显存使用率
        
        Returns:
            float: 显存使用率（0.0-1.0）
        """
        try:
            # 🔥 方案 1: 使用 pynvml（需要安装）
            # import pynvml
            # pynvml.nvmlInit()
            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # vram_usage = info.used / info.total
            
            # 🔥 方案 2: 使用 torch（如果已安装）
            # import torch
            # if torch.cuda.is_available():
            #     allocated = torch.cuda.memory_allocated()
            #     total = torch.cuda.get_device_properties(0).total_memory
            #     vram_usage = allocated / total
            # else:
            #     vram_usage = 0.0
            
            # 🔥 方案 3: 模拟值（用于测试）
            import random
            vram_usage = random.uniform(0.5, 0.9)  # 模拟 50%-90% 使用率
            
            return vram_usage
            
        except Exception as e:
            logger.debug(f"[VRAM_Monitor] 获取显存失败：{e}")
            return 0.5  # 默认 50%
    
    def stop_vram_monitor(self):
        """停止显存监控线程"""
        self.vram_monitor_running = False
        if self.vram_monitor_thread:
            self.vram_monitor_thread.join(timeout=2.0)
            logger.info("[MemoryConfigInitializer] 显存监控线程已停止")


# 全局单例
memory_config_initializer = MemoryConfigInitializer()


def initialize_memory_config(model_id: ModelID = ModelID.L2_PRIME):
    """
    初始化记忆配置的便捷函数
    
    Args:
        model_id: 模型 ID，默认使用 L2_PRIME
    """
    initializer = memory_config_initializer
    initializer.initialize_from_model_config(model_id)
    return initializer


async def async_initialize_memory_config(model_id: ModelID = ModelID.L2_PRIME):
    """
    异步初始化记忆配置
    
    Args:
        model_id: 模型 ID
    """
    # 在线程池中执行同步初始化
    loop = asyncio.get_event_loop()
    initializer = await loop.run_in_executor(
        None,
        lambda: initialize_memory_config(model_id)
    )
    return initializer
