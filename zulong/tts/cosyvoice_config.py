# File: zulong/tts/cosyvoice_config.py
# CosyVoice2-0.5B TTS 配置
# TSD v1.7 规范：TTS 运行在 CPU 上，使用 safetensors 格式

import os
import sys
from pathlib import Path
from typing import Dict, Any

# 项目根目录（优先从环境变量 ZULONG_HOME 获取，否则自动检测）
_PROJECT_ROOT = Path(os.environ.get("ZULONG_HOME", Path(__file__).resolve().parent.parent.parent))
MODEL_BASE_DIR = Path(os.environ.get("ZULONG_MODEL_BASE_DIR", str(_PROJECT_ROOT / "models")))
COSYVOICE3_PATH = MODEL_BASE_DIR / "CosyVoice3-0.5B" / "FunAudioLLM" / "Fun-CosyVoice3-0___5B-2512"
TTSFRD_PATH = MODEL_BASE_DIR / "iic" / "CosyVoice-ttsfrd"
COSYVOICE2_PATH = MODEL_BASE_DIR / "iic" / "CosyVoice2-0.5B"
COSYVOICE_CODE_PATH = MODEL_BASE_DIR / "CosyVoice"
COSYVOICE_PROMPT_AUDIO = COSYVOICE_CODE_PATH / "asset" / "zero_shot_prompt.wav"


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


def _config_value(key: str, default: str | None = None) -> str | None:
    try:
        from zulong.config.config_manager import ConfigManager

        value = ConfigManager().get(key, default)
        return str(value) if value not in (None, "") else default
    except Exception:
        return default


def _path_from_env_or_config(env_key: str, config_key: str, default: Path | str | None) -> str:
    value = os.environ.get(env_key)
    if value:
        return str(Path(os.path.expandvars(os.path.expanduser(value))))
    configured = _config_value(config_key)
    if configured:
        return str(Path(os.path.expandvars(os.path.expanduser(configured))))
    if default is None:
        return ""
    return str(Path(os.path.expandvars(os.path.expanduser(str(default)))))


def get_external_runtime_config() -> Dict[str, str]:
    """Return CosyVoice external runtime paths without host-specific defaults."""
    return {
        "integrated_python_path": _path_from_env_or_config(
            "ZULONG_COSYVOICE_PYTHON",
            "audio.tts.cosyvoice.integrated_python_path",
            sys.executable,
        ),
        "code_path": _path_from_env_or_config(
            "ZULONG_COSYVOICE_CODE_PATH",
            "audio.tts.cosyvoice.code_path",
            COSYVOICE_CODE_PATH,
        ),
        "model_dir": _path_from_env_or_config(
            "ZULONG_COSYVOICE_MODEL_DIR",
            "audio.tts.cosyvoice.model_dir",
            COSYVOICE2_PATH,
        ),
        "prompt_audio": _path_from_env_or_config(
            "ZULONG_COSYVOICE_PROMPT_AUDIO",
            "audio.tts.cosyvoice.prompt_audio",
            COSYVOICE_PROMPT_AUDIO,
        ),
        "server_url": os.environ.get(
            "ZULONG_COSYVOICE_GRADIO_URL",
            _config_value("audio.tts.cosyvoice.server_url", "http://localhost:50000") or "http://localhost:50000",
        ),
    }


def get_cosyvoice3_model_path() -> Path:
    """Get the in-process CosyVoice3 model path."""
    configured = _config_value("audio.tts.cosyvoice.model_path")
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured)))
    return COSYVOICE3_PATH


def get_cosyvoice_ttsfrd_path() -> Path:
    """Get the optional CosyVoice ttsfrd path."""
    configured = _config_value("audio.tts.cosyvoice.ttsfrd_path")
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured)))
    return TTSFRD_PATH


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
    "get_cosyvoice3_model_path",
    "get_cosyvoice_ttsfrd_path",
    "get_external_runtime_config",
    "verify_model_files",
    "check_cpu_memory",
]
