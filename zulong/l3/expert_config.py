# File: zulong/l3/expert_config.py
# L3 专家技能池配置 - 通用专家模型管理
# TSD v1.7 规范：L3 层提供专用领域能力，支持动态加载/卸载

from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum
import torch


# ============================================================================
# 模型路径管理（不硬编码具体模型名称）
# ============================================================================

MODEL_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class ExpertModelType(Enum):
    """专家模型类型枚举"""
    GENERAL = "general"  # 通用推理
    LOGIC = "logic"  # 逻辑推理
    CREATIVE = "creative"  # 创意写作
    NAVIGATION = "navigation"  # 导航专家
    MANIPULATION = "manipulation"  # 操作专家
    VISION = "vision"  # 视觉专家
    TTS = "tts"  # 语音合成


class ExpertRole(Enum):
    """专家角色枚举（TSD v1.7 左右脑概念）"""
    LEFT = "left"  # 左脑 - 逻辑推理
    RIGHT = "right"  # 右脑 - 创意情感
    GENERAL = "general"  # 通用


# ============================================================================
# 量化配置（TSD v1.7 强制 4bit Q4_K_M）
# ============================================================================

class QuantizationPreset(Enum):
    """量化预置配置"""
    Q4_K_M = "Q4_K_M"  # TSD v1.7 指定格式
    Q4_K_S = "Q4_K_S"
    Q5_K_M = "Q5_K_M"
    Q8_0 = "Q8_0"


class ExpertQuantizationConfig:
    """
    专家模型量化配置（TSD v1.7 强制规范）
    
    核心要求:
    1. 所有模型必须使用 4bit 量化
    2. 量化格式：Q4_K_M
    3. RTX 3060 6GB 必须支持三模型常驻
    """
    
    def __init__(
        self,
        quantization_type: str = "bnb",  # bitsandbytes
        bits: int = 4,
        preset: QuantizationPreset = QuantizationPreset.Q4_K_M,
        compute_dtype: torch.dtype = torch.float16,
        use_double_quant: bool = True,
    ):
        self.quantization_type = quantization_type
        self.bits = bits
        self.preset = preset
        self.compute_dtype = compute_dtype
        self.use_double_quant = use_double_quant
        
        # TSD v1.7: 显存优化配置
        self.max_memory_per_model = "2GB"  # 每个模型最多 2GB
        self.offload_to_cpu = True  # 支持 CPU 卸载
    
    def to_bnb_config(self) -> Dict[str, Any]:
        """转换为 BitsAndBytes 配置"""
        return {
            "load_in_4bit": (self.bits == 4),
            "bnb_4bit_quant_type": "nf4",  # Normal Float 4-bit
            "bnb_4bit_compute_dtype": self.compute_dtype,
            "bnb_4bit_use_double_quant": self.use_double_quant,
            "llm_int8_threshold": 6.0,
        }


# ============================================================================
# 专家配置（通用化，不绑定具体模型）
# ============================================================================

class ExpertConfig:
    """
    专家配置类（TSD v1.7 L3 专家技能池）
    
    设计原则:
    1. 不硬编码具体模型名称（如 Qwen2.5）
    2. 通过配置文件或注册表动态加载专家
    3. 支持多种专家类型（左右脑、导航、操作等）
    """
    
    def __init__(self):
        # 量化配置
        self.quantization = ExpertQuantizationConfig()
        
        # 专家注册表（动态注册）
        self.expert_registry: Dict[str, Dict[str, Any]] = {}
        
        # 默认专家配置
        self._register_default_experts()
    
    def _register_default_experts(self):
        """注册默认专家（左右脑）"""
        
        # 左脑专家 - 逻辑推理
        self.register_expert(
            expert_id="left_brain",
            expert_type=ExpertModelType.LOGIC,
            role=ExpertRole.LEFT,
            system_prompt=(
                "你是一个逻辑严谨的 AI 助手，擅长数学计算、代码生成和理性分析。"
                "请用清晰、准确、结构化的方式回答问题。"
            ),
            generation_config={
                "max_new_tokens": 512,
                "temperature": 0.3,  # 低温，更确定性
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }
        )
        
        # 右脑专家 - 创意情感
        self.register_expert(
            expert_id="right_brain",
            expert_type=ExpertModelType.CREATIVE,
            role=ExpertRole.RIGHT,
            system_prompt=(
                "你是一个富有创造力和同理心的 AI 伙伴，擅长情感交流、故事创作和艺术表达。"
                "请用温暖、生动、富有想象力的方式回应。"
            ),
            generation_config={
                "max_new_tokens": 1024,
                "temperature": 0.8,  # 高温，更有创造力
                "top_p": 0.95,
                "repetition_penalty": 1.05,
            }
        )
        
        # 通用推理专家（L2 中枢）
        self.register_expert(
            expert_id="l2_inference",
            expert_type=ExpertModelType.GENERAL,
            role=ExpertRole.GENERAL,
            system_prompt="你是一个有帮助的 AI 助手。",
            generation_config={
                "max_new_tokens": 768,
                "temperature": 0.7,
                "top_p": 0.92,
            }
        )
    
    def register_expert(
        self,
        expert_id: str,
        expert_type: ExpertModelType,
        role: ExpertRole,
        system_prompt: str,
        generation_config: Dict[str, Any],
        model_path: Optional[Path] = None,
        vram_limit_gb: float = 2.0,
    ):
        """
        注册专家模型（通用接口）
        
        Args:
            expert_id: 专家唯一标识（如 "left_brain", "nav_expert"）
            expert_type: 专家类型（LOGIC, CREATIVE, NAVIGATION 等）
            role: 专家角色（LEFT, RIGHT, GENERAL）
            system_prompt: 系统提示词
            generation_config: 生成参数配置
            model_path: 模型路径（可选，默认从注册表查找）
            vram_limit_gb: 显存限制（GB）
        """
        self.expert_registry[expert_id] = {
            "expert_id": expert_id,
            "expert_type": expert_type.value,
            "role": role.value,
            "system_prompt": system_prompt,
            "generation_config": generation_config.copy(),
            "model_path": str(model_path) if model_path else None,
            "vram_limit_gb": vram_limit_gb,
            "quantization": self.quantization.to_bnb_config(),
        }
    
    def get_expert_config(self, expert_id: str) -> Dict[str, Any]:
        """获取专家配置"""
        if expert_id not in self.expert_registry:
            raise ValueError(f"未找到专家：{expert_id}")
        return self.expert_registry[expert_id].copy()
    
    def get_expert_ids_by_type(self, expert_type: ExpertModelType) -> List[str]:
        """根据类型获取专家 ID 列表"""
        return [
            expert_id for expert_id, config in self.expert_registry.items()
            if config["expert_type"] == expert_type.value
        ]
    
    def get_expert_ids_by_role(self, role: ExpertRole) -> List[str]:
        """根据角色获取专家 ID 列表"""
        return [
            expert_id for expert_id, config in self.expert_registry.items()
            if config["role"] == role.value
        ]


# ============================================================================
# 模型路径注册表（可配置化）
# ============================================================================

class ModelPathRegistry:
    """
    模型路径注册表（可配置化，不硬编码）
    
    使用方式:
    1. 通过配置文件加载模型路径映射
    2. 支持热更新模型路径
    3. 支持多个模型版本并存
    """
    
    def __init__(self):
        self._paths: Dict[str, Path] = {}
        self._load_default_paths()
    
    def _load_default_paths(self):
        """加载默认模型路径（可从配置文件读取）"""
        # Qwen3.5-0.8B - 适合 RTX 3060 6GB，作为左右脑
        self.register("default_logic_model", MODEL_BASE_DIR / "Qwen" / "Qwen3___5-0___8B")
        self.register("default_creative_model", MODEL_BASE_DIR / "Qwen" / "Qwen3___5-0___8B")
        self.register("default_general_model", MODEL_BASE_DIR / "Qwen" / "Qwen3___5-0___8B")
        
        # 已量化版本 (int4) - 更省显存
        self.register("quantized_logic_model", MODEL_BASE_DIR / "Intel" / "Qwen3___5-0___8B-int4-AutoRound")
        
        # 视觉语言模型
        self.register("vision_model", MODEL_BASE_DIR / "OpenGVLab" / "InternVL2_5-1B")
        self.register("vl_model", MODEL_BASE_DIR / "Qwen" / "Qwen" / "Qwen2___5-VL-3B-Instruct")
        
        # TTS 模型
        self.register("tts_model", MODEL_BASE_DIR / "CosyVoice3-0.5B" / "FunAudioLLM" / "Fun-CosyVoice3-0___5B-2512")
        
        # 嵌入模型 (用于 RAG)
        self.register("embedding_model", MODEL_BASE_DIR / "BAAI" / "bge-small-zh-v1.5")
    
    def register(self, model_id: str, model_path: Path):
        """注册模型路径"""
        if not model_path.exists():
            print(f"⚠️ 警告：模型路径不存在：{model_path}")
        self._paths[model_id] = model_path
    
    def get(self, model_id: str) -> Optional[Path]:
        """获取模型路径"""
        return self._paths.get(model_id)
    
    def list_models(self) -> Dict[str, str]:
        """列出所有注册的模型"""
        return {model_id: str(path) for model_id, path in self._paths.items()}


# ============================================================================
# 全局单例配置
# ============================================================================

class ExpertContainerConfig:
    """
    专家容器全局配置（TSD v1.7 单例模式）
    
    核心规范:
    1. 严禁在不同节点重复加载模型
    2. 必须实现全局单例模式
    3. 支持三模型常驻（RTX 3060 6GB）
    """
    
    # 全局模型实例缓存
    _loaded_models: Dict[str, Any] = {}
    
    # 显存管理
    MAX_VRAM_GB = 5.8  # RTX 3060 实际可用显存
    RESERVED_VRAM_GB = 0.5  # 预留显存给系统
    
    # 加载策略
    PRELOAD_ALL = False  # 不预加载所有模型
    LAZY_LOAD = True  # 懒加载
    AUTO_UNLOAD = True  # 自动卸载不活跃的模型
    
    # 同步策略（TSD v1.7 左右脑同步）
    ENABLE_SYNC = True
    SYNC_CONTEXT = True
    SYNC_KV_CACHE = False  # 可选，耗资源


# ============================================================================
# 导出配置
# ============================================================================

__all__ = [
    # 枚举
    "ExpertModelType",
    "ExpertRole",
    "QuantizationPreset",
    # 配置类
    "ExpertQuantizationConfig",
    "ExpertConfig",
    "ModelPathRegistry",
    "ExpertContainerConfig",
    # 常量
    "MODEL_BASE_DIR",
]
