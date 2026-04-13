# File: zulong/l2/l2_config.py
# 祖龙 (ZULONG) 系统 - L2 (Cortex) 层配置
# TSD v1.7 规范：L2 运行在 GPU，使用无量化模式加载

from pathlib import Path
from typing import Dict, Any
import torch

# 模型路径 (INT4 量化模式)
# ✅ 已更换为 Qwen3.5-2B 模型，平衡性能和显存占用
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L2_CORE_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen" / "Qwen3___5-2B"

# 设备配置 (L2 运行在 GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型参数 (Qwen3.5-2B)
MAX_LENGTH = 2048  # 最大生成长度 (2B 模型支持足够上下文)
TEMPERATURE = 0.7  # 生成温度 (降低以提高准确性)
TOP_P = 0.9  # Top-p 采样

# 任务管理参数
MAX_TASK_QUEUE_SIZE = 10  # 最大任务队列大小
TASK_TIMEOUT_SEC = 300  # 任务超时时间 (5 分钟)
FREEZE_CHECKPOINT_SEC = 60  # 冻结检查点间隔 (秒)


class L2InferenceNodeConfig:
    """L2 推理节点配置"""
    
    def __init__(self):
        self.model_path = str(L2_CORE_MODEL_PATH)
        self.device = DEVICE
        self.max_length = MAX_LENGTH
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        
        # 无量化配置
        self.quantization = None
        
        # 推理参数
        self.inference_params = {
            "do_sample": True,
            "repetition_penalty": 1.2,
            "presence_penalty": 0.1,
        }


class L2TaskManagerConfig:
    """L2 任务管理器配置"""
    
    def __init__(self):
        self.max_queue_size = MAX_TASK_QUEUE_SIZE
        self.task_timeout = TASK_TIMEOUT_SEC
        self.freeze_checkpoint_interval = FREEZE_CHECKPOINT_SEC
        
        # 任务优先级
        self.priority_levels = {
            "critical": 0,  # 紧急任务
            "high": 1,      # 高优先级
            "normal": 2,    # 普通
            "low": 3,       # 低优先级
        }


class InterruptControllerConfig:
    """中断控制器配置"""
    
    def __init__(self):
        self.enable_auto_resume = True  # 自动恢复
        self.max_freeze_stack_size = 5  # 最大冻结堆栈
        self.re_eval_before_resume = True  # 恢复前重评估
        self.pre_execution_check = True  # 执行前确认


def get_inference_config() -> Dict[str, Any]:
    """获取推理节点配置"""
    return {
        "model_path": str(L2_CORE_MODEL_PATH),
        "device": str(DEVICE),
        "max_length": MAX_LENGTH,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }


def get_task_manager_config() -> Dict[str, Any]:
    """获取任务管理器配置"""
    return {
        "max_queue_size": MAX_TASK_QUEUE_SIZE,
        "task_timeout": TASK_TIMEOUT_SEC,
        "freeze_checkpoint_interval": FREEZE_CHECKPOINT_SEC,
    }


def get_interrupt_config() -> Dict[str, Any]:
    """获取中断控制器配置"""
    return {
        "enable_auto_resume": True,
        "max_freeze_stack_size": 5,
        "re_eval_before_resume": True,
        "pre_execution_check": True,
    }
