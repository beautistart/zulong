# File: zulong/l1b/l1b_config.py
# 祖龙 (ZULONG) 系统 - L1-B (Gatekeeper) 层配置
# TSD v1.7 规范：L1-B 运行在 GPU，使用无量化模式加载

from pathlib import Path
from typing import Dict, Any
import torch

# 模型路径 (无量化模式)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L1B_AUDIO_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3.5-0.8B-int4-L1B"

# 设备配置 (L1-B 运行在 GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型参数
MAX_LENGTH = 512  # 最大生成长度
TEMPERATURE = 0.7  # 生成温度
TOP_P = 0.9  # Top-p 采样

# 调度器参数
INTERRUPT_COOLDOWN_SEC = 2.0  # 中断冷却时间
CONTEXT_WINDOW_SEC = 30  # 上下文窗口 (秒)


class AudioUnderstandingNodeConfig:
    """音频理解节点配置"""
    
    def __init__(self):
        self.model_path = str(L1B_AUDIO_MODEL_PATH)
        self.device = DEVICE
        self.max_length = MAX_LENGTH
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        
        # 无量化配置
        self.quantization = None
        
        # 音频处理参数
        self.audio_params = {
            "sample_rate": 16000,
            "chunk_size": 0.1,  # 100ms
            "vad_threshold": 0.5,
        }


class L1BSchedulerConfig:
    """L1-B 调度器配置"""
    
    def __init__(self):
        self.model_path = str(L1B_AUDIO_MODEL_PATH)
        self.device = DEVICE
        self.interrupt_cooldown = INTERRUPT_COOLDOWN_SEC
        self.context_window = CONTEXT_WINDOW_SEC
        
        # 紧急关键词列表 (穿透冷却)
        self.urgent_keywords = [
            "停下", "停止", "救命", "紧急", "危险",
            "stop", "help", "emergency", "danger"
        ]
        
        # 去抖动参数
        self.debounce_window = 0.5  # 500ms 去抖动窗口


def get_audio_understanding_config() -> Dict[str, Any]:
    """获取音频理解节点配置"""
    return {
        "model_path": str(L1B_AUDIO_MODEL_PATH),
        "device": str(DEVICE),
        "max_length": MAX_LENGTH,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }


def get_scheduler_config() -> Dict[str, Any]:
    """获取调度器配置"""
    return {
        "model_path": str(L1B_AUDIO_MODEL_PATH),
        "device": str(DEVICE),
        "interrupt_cooldown": INTERRUPT_COOLDOWN_SEC,
        "context_window": CONTEXT_WINDOW_SEC,
        "urgent_keywords": ["停下", "停止", "救命", "紧急", "危险"],
    }
