# File: zulong/l1a/l1a_config.py
# 祖龙 (ZULONG) 系统 - L1-A (Reflex) 层配置
# TSD v1.7 规范：L1-A 运行在 GPU，使用 Intel AutoRound 4bit 量化模型

import os
from pathlib import Path
from typing import Dict, Any
import torch

# 🎯 动态获取项目根目录 (基于当前文件位置)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # zulong/l1a -> zulong -> project_root

# 模型路径 (Intel AutoRound 4bit 量化)
MODEL_ROOT = PROJECT_ROOT / "models"
L1A_VISION_MODEL_PATH = MODEL_ROOT / "Intel_Qwen3.5-0.8B-int4-AutoRead"
L1A_AUDIO_MODEL_PATH = MODEL_ROOT / "Intel_Qwen3.5-0.8B-int4-AutoRead"

# 🎯 缓存与数据目录 (动态路径)
MODEL_CACHE_DIR = PROJECT_ROOT / "data" / "model_cache"
VIDEO_BACKTRACK_DIR = PROJECT_ROOT / "data" / "video_backtrack"
SHARED_VISION_DIR = PROJECT_ROOT / "data" / "shared_vision"

# 确保目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(VIDEO_BACKTRACK_DIR, exist_ok=True)
os.makedirs(SHARED_VISION_DIR, exist_ok=True)

# 设备配置 (L1-A 运行在 GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型参数
MAX_LENGTH = 512  # 最大生成长度
TEMPERATURE = 0.7  # 生成温度
TOP_P = 0.9  # Top-p 采样


class VisionNodeConfig:
    """视觉节点配置"""
    
    def __init__(self):
        self.model_path = str(L1A_VISION_MODEL_PATH)
        self.device = DEVICE
        self.max_length = MAX_LENGTH
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        
        # AutoRound 4bit 量化配置
        self.quantization = {
            "load_in_4bit": False,  # AutoRound 模型已预量化，不需要 bitsandbytes
            "quant_method": "auto-round",
        }
        
        # 视觉处理参数
        self.vision_params = {
            "frame_width": 640,
            "frame_height": 480,
            "fps": 30,
            "motion_threshold": 500,
        }


class VLAudioNodeConfig:
    """VL 音频节点配置"""
    
    def __init__(self):
        self.model_path = str(L1A_AUDIO_MODEL_PATH)
        self.device = DEVICE
        self.max_length = MAX_LENGTH
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        
        # AutoRound 4bit 量化配置
        self.quantization = {
            "load_in_4bit": False,  # AutoRound 模型已预量化，不需要 bitsandbytes
            "quant_method": "auto-round",
        }
        
        # 音频处理参数
        self.audio_params = {
            "sample_rate": 16000,
            "chunk_size": 0.1,  # 100ms
            "vad_threshold": 0.5,
        }


def get_vision_config() -> Dict[str, Any]:
    """获取视觉节点配置"""
    return {
        "model_path": str(L1A_VISION_MODEL_PATH),
        "device": str(DEVICE),
        "max_length": MAX_LENGTH,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }


def get_audio_config() -> Dict[str, Any]:
    """获取 VL 音频节点配置"""
    return {
        "model_path": str(L1A_AUDIO_MODEL_PATH),
        "device": str(DEVICE),
        "max_length": MAX_LENGTH,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
