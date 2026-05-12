# File: zulong/l1b/dynamic_threshold_manager.py
"""
动态阈值管理器 - TSD v2.4 资源自适应核心组件

功能：
1. 基于模型配置动态计算复盘触发阈值
2. 实时监控显存水位并动态调整
3. 支持用户输入长度预测触发
4. 提供硬上限和软上限双重保护

对应 TSD v2.4: 资源自适应、动态容量管理
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import re

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置数据结构"""
    name: str
    size_in_billions: float  # 模型大小（十亿参数）
    max_context_window: int  # 最大上下文窗口（tokens）
    quantization: str  # 量化级别（fp16, int4, int8）
    vram_limit_gb: float  # 显存限制（GB）
    

@dataclass
class DynamicThresholds:
    """动态阈值数据结构"""
    hard_token_limit: int  # 硬上限（Token 数）
    soft_turn_limit: int  # 软上限（轮次数）
    current_model: str  # 当前模型名称
    safety_factor: float  # 安全系数
    speed_factor: float  # 速度因子
    vram_usage: float  # 当前显存使用率
    is_emergency_mode: bool  # 是否处于紧急模式
    

class DynamicThresholdManager:
    """
    动态阈值管理器
    
    核心职责：
    1. 启动时根据模型配置计算初始阈值
    2. 运行时根据显存水位动态调整
    3. 检测长文本输入时立即触发复盘
    4. 提供线程安全的阈值查询接口
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化动态阈值管理器"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.model_config: Optional[ModelConfig] = None
        self.hard_token_limit = 0
        self.soft_turn_limit = 0
        self.base_hard_limit = 0  # 基础硬上限（未调整前）
        self.base_soft_limit = 0  # 基础软上限（未调整前）
        
        # 显存监控
        self.vram_usage = 0.0
        self.last_vram_check_time = 0
        self.vram_check_interval = 1.0  # 1 秒检查一次
        
        # 紧急模式
        self.is_emergency_mode = False
        self.emergency_mode_start_time = 0
        
        # 长文本检测
        self.long_text_threshold = 1000  # Token 数超过此值判定为长文本
        
        # 回调函数（用于通知外部触发复盘）
        self._on_threshold_updated_callbacks = []
        self._on_emergency_trigger_callbacks = []
        
        logger.info("[DynamicThresholdManager] 初始化完成")
        self._initialized = True
        
        # 尝试从配置文件自动初始化模型参数
        self._auto_initialize_from_config()
    
    def _auto_initialize_from_config(self):
        """从 zulong_config.yaml 自动读取模型信息并初始化阈值
        
        解决 MemoryConfigInitializer 依赖链断裂的问题：
        直接从配置管理器读取模型名称和上下文窗口大小
        """
        try:
            from zulong.config.config_manager import get_config_manager
            cm = get_config_manager()
            
            model_name = cm.get('l2_inference.core_model', 'qwen3.5:4b')
            context_window = cm.get_int('l2_inference.circuit_breaker.context_window_size', 32768)
            
            # 从模型名称解析参数大小，例如 "qwen3.5:4b" -> 4.0
            size_billions = self._parse_model_size(model_name)
            
            config = {
                'name': model_name,
                'size_in_billions': size_billions,
                'max_context_window': context_window,
                'quantization': 'int4',  # Ollama 默认量化
                'vram_limit_gb': 6.0,
            }
            self.initialize_with_model_config(config)
            logger.info(f"[DynamicThresholdManager] 从配置文件自动初始化成功：{model_name}")
        except Exception as e:
            logger.warning(f"[DynamicThresholdManager] 自动初始化失败，使用保底默认值：{e}")
            # 保底默认值：确保记忆预算不为 0
            self.hard_token_limit = 6000
            self.soft_turn_limit = 10
            self.base_hard_limit = 6000
            self.base_soft_limit = 10
    
    @staticmethod
    def _parse_model_size(model_name: str) -> float:
        """从模型名称解析参数量（十亿），例如 'qwen3.5:4b' -> 4.0"""
        match = re.search(r'(\d+(?:\.\d+)?)\s*[bB]', model_name)
        if match:
            return float(match.group(1))
        return 4.0  # 默认假设 4B
    
    def initialize_with_model_config(self, config: Dict[str, Any]):
        """
        使用模型配置初始化阈值计算器
        
        Args:
            config: 模型配置字典，包含：
                - name: 模型名称
                - size_in_billions: 模型大小（B）
                - max_context_window: 最大上下文窗口
                - quantization: 量化级别
                - vram_limit_gb: 显存限制
        """
        self.model_config = ModelConfig(
            name=config.get('name', 'Unknown'),
            size_in_billions=config.get('size_in_billions', 8),
            max_context_window=config.get('max_context_window', 8192),
            quantization=config.get('quantization', 'fp16'),
            vram_limit_gb=config.get('vram_limit_gb', 16)
        )
        
        logger.info(f"[DynamicThresholdManager] 使用模型配置初始化：{self.model_config.name}")
        logger.info(f"  - 模型大小：{self.model_config.size_in_billions}B")
        logger.info(f"  - 最大上下文：{self.model_config.max_context_window} tokens")
        logger.info(f"  - 量化级别：{self.model_config.quantization}")
        logger.info(f"  - 显存限制：{self.model_config.vram_limit_gb}GB")
        
        # 计算初始阈值
        self._calculate_thresholds()
    
    def _calculate_thresholds(self):
        """
        核心计算逻辑：基于模型配置计算硬上限和软上限
        
        公式：
        - 硬上限 = ContextWindow_max × SafetyFactor
        - 软上限 = BaseTurns × SpeedFactor × SizePenalty
        """
        if not self.model_config:
            logger.warning("[DynamicThresholdManager] 模型配置未初始化，使用默认值")
            self.hard_token_limit = 6000
            self.soft_turn_limit = 10
            return
        
        # ========== 1. 计算硬上限（基于上下文窗口） ==========
        max_ctx = self.model_config.max_context_window
        
        # 安全系数：量化模型显存更宽裕，可以使用更高的安全系数
        if 'int4' in self.model_config.quantization.lower():
            self.safety_factor = 0.85  # INT4 量化
        elif 'int8' in self.model_config.quantization.lower():
            self.safety_factor = 0.80  # INT8 量化
        else:
            self.safety_factor = 0.75  # FP16/FP32
        
        self.base_hard_limit = int(max_ctx * self.safety_factor)
        self.hard_token_limit = self.base_hard_limit
        
        # ========== 2. 计算软上限（基于模型大小和推理速度） ==========
        base_turns = 10  # 基础轮次
        
        # 速度因子：模型越大，推理越慢，应该更早复盘
        model_size = self.model_config.size_in_billions
        
        if model_size > 70:
            self.speed_factor = 0.5  # 超大模型（72B+）
        elif model_size > 30:
            self.speed_factor = 0.7  # 大模型（30B-70B）
        elif model_size > 14:
            self.speed_factor = 0.9  # 中型模型（14B-30B）
        else:
            self.speed_factor = 1.0  # 小模型（<14B）
        
        self.base_soft_limit = int(base_turns * self.speed_factor)
        self.soft_turn_limit = max(5, self.base_soft_limit)  # 最少 5 轮
        
        logger.info(f"[DynamicThresholdManager] 阈值计算完成:")
        logger.info(f"  - 硬上限：{self.hard_token_limit} tokens (安全系数 {self.safety_factor})")
        logger.info(f"  - 软上限：{self.soft_turn_limit} 轮 (速度因子 {self.speed_factor})")
    
    def update_vram_usage(self, vram_usage_percent: float):
        """
        更新显存使用率并动态调整阈值
        
        Args:
            vram_usage_percent: 显存使用率（0.0-1.0）
        """
        self.vram_usage = vram_usage_percent
        self.last_vram_check_time = time.time()
        
        # 检查是否进入紧急模式
        if vram_usage_percent > 0.95:
            self._enter_emergency_mode()
        elif vram_usage_percent < 0.85 and self.is_emergency_mode:
            self._exit_emergency_mode()
    
    def _enter_emergency_mode(self):
        """进入紧急模式：显存>95%，强制降低阈值"""
        if not self.is_emergency_mode:
            self.is_emergency_mode = True
            self.emergency_mode_start_time = time.time()
            
            # 临时下调硬上限 20%
            adjusted_hard_limit = int(self.base_hard_limit * 0.8)
            self.hard_token_limit = adjusted_hard_limit
            
            # 强制降低软上限
            self.soft_turn_limit = max(5, int(self.base_soft_limit * 0.6))
            
            logger.warning(f"🚨 [DynamicThresholdManager] 进入紧急模式！显存使用率：{self.vram_usage*100:.1f}%")
            logger.warning(f"  - 硬上限已下调：{self.base_hard_limit} → {self.hard_token_limit}")
            logger.warning(f"  - 软上限已下调：{self.base_soft_limit} → {self.soft_turn_limit}")
            
            # 触发紧急复盘回调
            self._trigger_emergency_callbacks()
    
    def _exit_emergency_mode(self):
        """退出紧急模式：显存恢复正常"""
        if self.is_emergency_mode:
            self.is_emergency_mode = False
            duration = time.time() - self.emergency_mode_start_time
            
            # 恢复原始阈值
            self.hard_token_limit = self.base_hard_limit
            self.soft_turn_limit = self.base_soft_limit
            
            logger.info(f"✅ [DynamicThresholdManager] 退出紧急模式（持续{duration:.1f}秒）")
            logger.info(f"  - 硬上限已恢复：{self.hard_token_limit}")
            logger.info(f"  - 软上限已恢复：{self.soft_turn_limit}")
    
    def check_long_text_input(self, input_text: str) -> bool:
        """
        检测用户输入是否为长文本
        
        Args:
            input_text: 用户输入文本
            
        Returns:
            bool: True 表示需要立即触发复盘
        """
        # 简单估算：中文字符数 × 1.5 + 英文字符数
        # 更精确的方法是使用 TikToken，但为了性能使用简化估算
        chinese_chars = sum(1 for c in input_text if '\u4e00' <= c <= '\u9fff')
        english_chars = sum(1 for c in input_text if c.isascii())
        
        estimated_tokens = int(chinese_chars * 1.5 + english_chars * 0.75)
        
        if estimated_tokens > self.long_text_threshold:
            logger.info(f"📝 [DynamicThresholdManager] 检测到长文本输入：{estimated_tokens} tokens")
            return True
        
        return False
    
    def get_thresholds(self) -> DynamicThresholds:
        """
        获取当前阈值（线程安全）
        
        Returns:
            DynamicThresholds: 包含所有阈值信息的数据对象
        """
        return DynamicThresholds(
            hard_token_limit=self.hard_token_limit,
            soft_turn_limit=self.soft_turn_limit,
            current_model=self.model_config.name if self.model_config else "Unknown",
            safety_factor=self.safety_factor if hasattr(self, 'safety_factor') else 0.75,
            speed_factor=self.speed_factor if hasattr(self, 'speed_factor') else 1.0,
            vram_usage=self.vram_usage,
            is_emergency_mode=self.is_emergency_mode
        )
    
    def get_token_budget(self, current_token_count: int) -> int:
        """
        计算当前可用的 Token 预算
        
        Args:
            current_token_count: 当前已使用的 Token 数
            
        Returns:
            int: 剩余可用的 Token 数
        """
        remaining = self.hard_token_limit - current_token_count
        return max(0, remaining)
    
    def should_trigger_summarization(self, current_tokens: int, current_turns: int) -> Tuple[bool, str]:
        """
        判断是否应该触发复盘
        
        Args:
            current_tokens: 当前 Token 数
            current_turns: 当前轮次数
            
        Returns:
            Tuple[bool, str]: (是否触发，触发原因)
        """
        # 1. 紧急模式：强制触发
        if self.is_emergency_mode:
            return True, "emergency_mode"
        
        # 2. 硬上限检查（Token 数超限）
        if current_tokens >= self.hard_token_limit:
            return True, "token_limit_exceeded"
        
        # 3. 软上限检查（轮次超限）
        if current_turns >= self.soft_turn_limit:
            return True, "turn_limit_exceeded"
        
        # 4. 接近硬上限（90% 水位）
        if current_tokens >= int(self.hard_token_limit * 0.9):
            return True, "token_limit_warning"
        
        return False, "no_trigger"
    
    def register_threshold_updated_callback(self, callback):
        """
        注册阈值更新回调函数
        
        Args:
            callback: 回调函数，签名：callback(old_thresholds, new_thresholds)
        """
        self._on_threshold_updated_callbacks.append(callback)
        logger.debug(f"[DynamicThresholdManager] 注册阈值更新回调：{callback.__name__}")
    
    def register_emergency_trigger_callback(self, callback):
        """
        注册紧急触发回调函数
        
        Args:
            callback: 回调函数，签名：callback(reason)
        """
        self._on_emergency_trigger_callbacks.append(callback)
        logger.debug(f"[DynamicThresholdManager] 注册紧急触发回调：{callback.__name__}")
    
    def _trigger_emergency_callbacks(self):
        """触发所有紧急复盘回调"""
        for callback in self._on_emergency_trigger_callbacks:
            try:
                callback("vram_emergency")
            except Exception as e:
                logger.error(f"[DynamicThresholdManager] 紧急回调执行失败：{e}")
    
    def _notify_threshold_updated(self, old_thresholds: DynamicThresholds, new_thresholds: DynamicThresholds):
        """通知阈值已更新"""
        for callback in self._on_threshold_updated_callbacks:
            try:
                callback(old_thresholds, new_thresholds)
            except Exception as e:
                logger.error(f"[DynamicThresholdManager] 阈值更新回调执行失败：{e}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        获取状态报告
        
        Returns:
            Dict: 包含所有状态信息
        """
        return {
            "model_name": self.model_config.name if self.model_config else "Unknown",
            "hard_token_limit": self.hard_token_limit,
            "base_hard_limit": self.base_hard_limit,
            "soft_turn_limit": self.soft_turn_limit,
            "base_soft_limit": self.base_soft_limit,
            "safety_factor": getattr(self, 'safety_factor', 0.75),
            "speed_factor": getattr(self, 'speed_factor', 1.0),
            "vram_usage": self.vram_usage,
            "is_emergency_mode": self.is_emergency_mode,
            "long_text_threshold": self.long_text_threshold
        }


# 全局单例
dynamic_threshold_manager = DynamicThresholdManager()


def get_dynamic_threshold_manager() -> DynamicThresholdManager:
    """获取动态阈值管理器单例"""
    return dynamic_threshold_manager
