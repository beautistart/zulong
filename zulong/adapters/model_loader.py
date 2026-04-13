# File: zulong/adapters/model_loader.py
"""
原子任务 3: 模型自动选择器
目标: 根据硬件环境自动选择合适的模型规格
TSD v1.9: KV Cache 热切换机制 - 模型加载器
"""

import logging
import os
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    NVIDIA_LOW_END = "nvidia_low_end"
    NVIDIA_MID_RANGE = "nvidia_mid_range"
    NVIDIA_HIGH_END = "nvidia_high_end"
    APU = "apu"
    CPU_ONLY = "cpu_only"


@dataclass
class ModelConfig:
    model_id: str
    model_name: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    block_size: int = 16
    max_model_len: int = 4096
    trust_remote_code: bool = True
    dtype: str = "auto"


MODEL_REGISTRY: Dict[HardwareType, ModelConfig] = {
    HardwareType.NVIDIA_LOW_END: ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_name="TinyLlama-1.1B-Chat",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        block_size=16,
        max_model_len=2048,
    ),
    HardwareType.NVIDIA_MID_RANGE: ModelConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        model_name="Qwen2.5-3B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        block_size=16,
        max_model_len=4096,
    ),
    HardwareType.NVIDIA_HIGH_END: ModelConfig(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_name="Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        block_size=16,
        max_model_len=8192,
    ),
    HardwareType.APU: ModelConfig(
        model_id="Qwen/Qwen2.5-32B-Instruct",
        model_name="Qwen2.5-32B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        block_size=16,
        max_model_len=8192,
    ),
    HardwareType.CPU_ONLY: ModelConfig(
        model_id="facebook/opt-1.3b",
        model_name="OPT-1.3B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        block_size=16,
        max_model_len=1024,
    ),
}


def detect_hardware() -> HardwareType:
    """
    检测硬件类型
    
    Returns:
        HardwareType 枚举值
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.info("[ModelLoader] CUDA 不可用，使用 CPU 模式")
            return HardwareType.CPU_ONLY
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_gb = total_mem / (1024 ** 3)
        
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"[ModelLoader] 检测到 GPU: {device_name}, 显存: {total_gb:.1f}GB")
        
        if total_gb > 64:
            logger.info("[ModelLoader] 判定为 APU 统一内存架构")
            return HardwareType.APU
        elif total_gb >= 20:
            logger.info("[ModelLoader] 判定为高端 NVIDIA 显卡")
            return HardwareType.NVIDIA_HIGH_END
        elif total_gb >= 10:
            logger.info("[ModelLoader] 判定为中端 NVIDIA 显卡")
            return HardwareType.NVIDIA_MID_RANGE
        else:
            logger.info("[ModelLoader] 判定为低端 NVIDIA 显卡")
            return HardwareType.NVIDIA_LOW_END
            
    except Exception as e:
        logger.warning(f"[ModelLoader] 硬件检测失败: {e}，默认使用 CPU 模式")
        return HardwareType.CPU_ONLY


def auto_select_model() -> ModelConfig:
    """
    根据硬件环境自动选择模型
    
    Returns:
        ModelConfig 配置对象
    """
    hardware_type = detect_hardware()
    config = MODEL_REGISTRY[hardware_type]
    
    logger.info(
        f"[ModelLoader] 自动选择模型: {config.model_name} "
        f"(硬件类型: {hardware_type.value})"
    )
    
    return config


def get_local_model_path(model_id: str) -> Optional[str]:
    """
    获取本地模型路径
    
    Args:
        model_id: HuggingFace 模型 ID
        
    Returns:
        本地路径，如果不存在则返回 None
    """
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    models_dir = os.path.abspath(models_dir)
    
    model_dir_name = model_id.replace("/", "_").replace("-", "_")
    local_path = os.path.join(models_dir, model_dir_name)
    
    if os.path.exists(local_path):
        logger.info(f"[ModelLoader] 找到本地模型: {local_path}")
        return local_path
    
    return None


def init_l2_engines(
    use_vllm: bool = False,
    custom_model_id: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    初始化 L2 引擎 (PRIME 和 BACKUP)
    
    Args:
        use_vllm: 是否使用 vLLM 引擎
        custom_model_id: 自定义模型 ID
        
    Returns:
        (L2_PRIME 引擎, L2_BACKUP 引擎)
    """
    config = auto_select_model()
    
    if custom_model_id:
        config.model_id = custom_model_id
        logger.info(f"[ModelLoader] 使用自定义模型: {custom_model_id}")
    
    local_path = get_local_model_path(config.model_id)
    model_path = local_path if local_path else config.model_id
    
    if use_vllm:
        try:
            return _init_vllm_engines(model_path, config)
        except ImportError:
            logger.warning("[ModelLoader] vLLM 未安装，使用 Mock 引擎")
            return _init_mock_engines(config)
    else:
        return _init_mock_engines(config)


def _init_vllm_engines(model_path: str, config: ModelConfig) -> Tuple[Any, Any]:
    """
    初始化 vLLM 引擎
    
    Args:
        model_path: 模型路径
        config: 模型配置
        
    Returns:
        (L2_PRIME, L2_BACKUP) vLLM 引擎实例
    """
    from vllm import LLM
    
    logger.info(f"[ModelLoader] 初始化 vLLM 引擎: {model_path}")
    
    common_kwargs = {
        "model": model_path,
        "tensor_parallel_size": config.tensor_parallel_size,
        "block_size": config.block_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "trust_remote_code": config.trust_remote_code,
        "dtype": config.dtype,
        "max_model_len": config.max_model_len,
    }
    
    l2_prime = LLM(**common_kwargs)
    logger.info("[ModelLoader] L2_PRIME vLLM 引擎初始化完成")
    
    l2_backup = LLM(**common_kwargs)
    logger.info("[ModelLoader] L2_BACKUP vLLM 引擎初始化完成")
    
    return l2_prime, l2_backup


def _init_mock_engines(config: ModelConfig) -> Tuple[Any, Any]:
    """
    初始化 Mock 引擎 (用于测试)
    
    Args:
        config: 模型配置
        
    Returns:
        (L2_PRIME, L2_BACKUP) Mock 引擎实例
    """
    from zulong.l1b.hotswap_scheduler import MiniL2Engine
    
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except:
        device = "cpu"
    
    l2_prime = MiniL2Engine(device=device, engine_id="PRIME")
    l2_backup = MiniL2Engine(device=device, engine_id="BACKUP")
    
    logger.info(
        f"[ModelLoader] Mock 引擎初始化完成 "
        f"(模型: {config.model_name}, 设备: {device})"
    )
    
    return l2_prime, l2_backup


def get_model_info() -> Dict:
    """
    获取当前模型信息
    
    Returns:
        模型信息字典
    """
    hardware_type = detect_hardware()
    config = MODEL_REGISTRY[hardware_type]
    
    info = {
        "hardware_type": hardware_type.value,
        "model_id": config.model_id,
        "model_name": config.model_name,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "block_size": config.block_size,
        "max_model_len": config.max_model_len,
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_gb"] = total_mem / (1024 ** 3)
            info["gpu_free_gb"] = free_mem / (1024 ** 3)
    except:
        pass
    
    return info


def list_available_models() -> Dict[str, ModelConfig]:
    """
    列出所有可用的模型配置
    
    Returns:
        模型配置字典
    """
    return {
        hardware.value: config 
        for hardware, config in MODEL_REGISTRY.items()
    }
