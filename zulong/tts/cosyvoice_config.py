# File: zulong/tts/cosyvoice_config.py
# CosyVoice2-0.5B TTS 配置
# TSD v1.7 规范：TTS 运行在 CPU 上，使用 safetensors 格式

from pathlib import Path
from typing import Dict, Any

# 模型路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_BASE_DIR = PROJECT_ROOT / "models"
COSYVOICE3_PATH = MODEL_BASE_DIR / "CosyVoice3-0.5B" / "FunAudioLLM" / "Fun-CosyVoice3-0___5B-2512"
TTSFRD_PATH = MODEL_BASE_DIR / "iic" / "CosyVoice-ttsfrd"


class CosyVoiceConfig:
    """
    CosyVoice3-0.5B TTS 配置 (TSD v1.7)
    
    核心规范:
    1. TTS 运行在 CPU 上 (不占用 GPU 显存)
    2. 使用 safetensors 格式模型
    3. 支持零样本语音克隆
    4. 支持多语言 (中文、英文、方言)
    """
    
    def __init__(self):
        # 模型路径
        self.model_path = str(COSYVOICE3_PATH)
        self.ttsfrd_path = str(TTSFRD_PATH)
        
        # 设备配置 (TSD v1.7: TTS 强制 CPU)
        self.device = "cpu"
        self.use_gpu = False
        
        # 语音合成参数
        self.inference_config = {
            "mode": "sft",  # 推理模式：sft (监督微调)
            "text_frontend": True,  # 使用文本前端
            "use_ttsfrd": False,  # 暂时不使用 ttsfrd (需要安装)
            "stream": False,  # 非流式推理
            "cross_lingual": True,  # 支持跨语言
        }
        
        # 音频参数
        self.audio_config = {
            "sample_rate": 22050,
            "volume": 1.0,
            "speed": 1.0,
            "pitch": 1.0,
        }
        
        # 零样本语音克隆配置
        self.zero_shot_config = {
            "prompt_window": 3,  # 提示音频窗口 (秒)
            "prompt_sample_rate": 16000,
        }


class TTSContainerConfig:
    """
    TTS 容器配置 (全局单例模式)
    
    TSD v1.7: TTS 必须使用全局单例，严禁重复加载
    """
    
    # 模型实例缓存
    _instances = {}
    
    # TTS 模型配置
    MODELS = {
        "cosyvoice3": {
            "path": str(COSYVOICE3_PATH),
            "type": "tts",
            "device": "cpu",
            "max_memory": "2GB",  # CPU 内存限制
        }
    }
    
    # 加载策略
    LOAD_STRATEGY = {
        "preload": False,  # 不预加载，按需加载
        "lazy_load": True,  # 懒加载
    }


def get_model_path() -> Path:
    """获取 CosyVoice 模型路径"""
    return COSYVOICE3_PATH


def verify_model_files() -> bool:
    """验证模型文件完整性"""
    print("🔍 验证 CosyVoice3 模型文件...")
    
    # 关键文件
    required_files = [
        COSYVOICE3_PATH / "CosyVoice-BlankEN" / "model.safetensors",
        COSYVOICE3_PATH / "flow.pt",
        COSYVOICE3_PATH / "llm.pt",
        COSYVOICE3_PATH / "hift.pt",
        COSYVOICE3_PATH / "config.json",
    ]
    
    missing_files = []
    for file in required_files:
        if not file.exists():
            missing_files.append(str(file))
            print(f"  ❌ 缺失：{file}")
        else:
            size_mb = file.stat().st_size / (1024**2)
            print(f"  ✅ 存在：{file.name} ({size_mb:.2f} MB)")
    
    if missing_files:
        print(f"\n⚠️ 警告：缺少 {len(missing_files)} 个文件")
        return False
    
    print("\n✅ CosyVoice2 模型验证通过")
    return True


def check_cpu_memory() -> Dict[str, Any]:
    """检查 CPU 内存状态"""
    import psutil
    
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    
    return {
        "total_gb": round(total_gb, 2),
        "available_gb": round(available_gb, 2),
        "percent_used": memory.percent,
        "warning": available_gb < 4.0,  # 如果可用内存小于 4GB，发出警告
    }


# 导出配置
__all__ = [
    "CosyVoiceConfig",
    "TTSContainerConfig",
    "get_model_path",
    "verify_model_files",
    "check_cpu_memory",
]
