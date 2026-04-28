# File: zulong/l3/expert_loader.py
# L3 专家技能池加载器 - 通用专家模型加载
# TSD v1.7 规范：全局单例模式，4bit 量化加载，支持动态加载/卸载

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .expert_config import (
    ExpertConfig,
    ExpertModelType,
    ExpertRole,
    ExpertQuantizationConfig,
    ModelPathRegistry,
    ExpertContainerConfig,
)

logger = logging.getLogger(__name__)


class ExpertLoader:
    """
    专家模型加载器（通用化，不绑定具体模型）
    
    TSD v1.7 核心要求:
    1. 4bit 量化加载（Q4_K_M）
    2. 全局单例模式（严禁重复加载）
    3. 支持动态加载/卸载（节能）
    4. 显存优化（RTX 3060 6GB 三模型常驻）
    """
    
    def __init__(self, config: Optional[ExpertConfig] = None):
        """
        初始化专家加载器
        
        Args:
            config: 专家配置（可选，默认使用 ExpertConfig）
        """
        self.config = config or ExpertConfig()
        self.path_registry = ModelPathRegistry()
        
        # 已加载的模型实例（单例缓存）
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        
        # 显存管理
        self.max_vram_gb = ExpertContainerConfig.MAX_VRAM_GB
        self.reserved_vram_gb = ExpertContainerConfig.RESERVED_VRAM_GB
        
        # 检查 GPU
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {gpu_name} ({total_vram:.2f}GB)")
            if total_vram < 6.0:
                logger.warning(f"⚠️ 警告：显存小于 6GB，可能无法同时加载三个模型")
        else:
            logger.warning("⚠️ 未检测到 GPU，将使用 CPU 模式")
    
    def _create_quantization_config(self, quant_config: ExpertQuantizationConfig) -> BitsAndBytesConfig:
        """
        创建 4bit 量化配置
        
        Args:
            quant_config: 量化配置对象
        
        Returns:
            BitsAndBytesConfig 实例
        """
        return BitsAndBytesConfig(
            load_in_4bit=(quant_config.bits == 4),
            bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
            bnb_4bit_compute_dtype=quant_config.compute_dtype,
            bnb_4bit_use_double_quant=quant_config.use_double_quant,
            llm_int8_threshold=6.0,
        )
    
    def load_expert(
        self,
        expert_id: str,
        model_path: Optional[Path] = None,
        role: str = "general",
        system_prompt: Optional[str] = None,
        use_gpu: bool = True,
        quantization: Optional[ExpertQuantizationConfig] = None,
    ) -> Dict[str, Any]:
        """
        加载专家模型（全局单例，避免重复加载）
        
        Args:
            expert_id: 专家唯一标识（如 "left_brain", "nav_expert"）
            model_path: 模型路径（可选，从注册表查找）
            role: 专家角色（"general", "logic", "creative"）
            system_prompt: 系统提示词
            use_gpu: 是否使用 GPU
            quantization: 量化配置（可选）
        
        Returns:
            包含 model, tokenizer, config 的字典
        
        TSD v1.7 规范:
        - 严禁重复加载（检查缓存）
        - 4bit 量化（Q4_K_M）
        - 全局单例模式
        """
        # TSD v1.7: 检查是否已加载（单例模式）
        if expert_id in self.loaded_models:
            logger.info(f"ℹ️ 专家 {expert_id} 已加载，返回缓存实例")
            return self.loaded_models[expert_id]
        
        # 获取模型路径
        if model_path is None:
            # 从注册表查找
            model_id = f"default_{role}_model"
            model_path = self.path_registry.get(model_id)
            if model_path is None:
                raise ValueError(f"未找到模型路径：{model_id}，请提供 model_path 参数")
        
        # 验证模型文件
        if not self._verify_model_files(model_path):
            raise FileNotFoundError(f"模型文件不完整：{model_path}")
        
        logger.info(f"\n🚀 开始加载专家：{expert_id} ({role})")
        logger.info(f"📂 模型路径：{model_path}")
        
        # 创建设备映射
        if use_gpu and self.gpu_available:
            device_map = "auto"
            logger.info(f"💻 设备：GPU (auto)")
        else:
            device_map = "cpu"
            logger.info(f"💻 设备：CPU")
        
        # 创建量化配置
        if quantization is None:
            quantization = self.config.quantization
        
        quant_config = self._create_quantization_config(quantization)
        logger.info(f"🔢 量化：{quantization.bits}bit ({quantization.preset.value})")
        
        try:
            # 加载 tokenizer
            logger.info("📝 加载 Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                padding_side="left",
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            # 加载模型
            logger.info("🧠 加载模型权重...")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                quantization_config=quant_config,
                device_map=device_map,
                torch_dtype=quantization.compute_dtype,
                trust_remote_code=True,
                attn_implementation="sdpa" if self.gpu_available else None,
            )
            
            # 评估模型大小
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"📊 模型参数：{num_params / 1e6:.2f}M")
            
            # 估算显存占用
            vram_usage = self._estimate_vram_usage(model)
            logger.info(f"💾 预估显存占用：{vram_usage:.2f}GB")
            
            # 存储到缓存（单例模式）
            self.loaded_models[expert_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": {
                    "expert_id": expert_id,
                    "model_path": str(model_path),
                    "role": role,
                    "system_prompt": system_prompt or "You are a helpful assistant.",
                    "quantization": f"{quantization.bits}bit_{quantization.preset.value}",
                    "device": device_map,
                    "vram_usage_gb": vram_usage,
                }
            }
            
            logger.info(f"✅ 专家 {expert_id} 加载完成")
            return self.loaded_models[expert_id]
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败：{e}")
            raise
    
    def get_expert(self, expert_id: str) -> Optional[Dict[str, Any]]:
        """获取已加载的专家实例"""
        return self.loaded_models.get(expert_id)
    
    def unload_expert(self, expert_id: str) -> bool:
        """
        卸载专家模型（释放显存）
        
        TSD v1.7: 支持动态卸载以节能（安静模式）
        """
        if expert_id not in self.loaded_models:
            logger.info(f"ℹ️ 专家 {expert_id} 未加载")
            return False
        
        logger.info(f"🗑️ 卸载专家：{expert_id}")
        
        # 删除模型和 tokenizer
        model_data = self.loaded_models[expert_id]
        
        # 清理 GPU 内存
        if hasattr(model_data["model"], 'to'):
            model_data["model"].to('cpu')
        
        del model_data["model"]
        del model_data["tokenizer"]
        del self.loaded_models[expert_id]
        
        # 清理 CUDA 缓存
        if self.gpu_available:
            torch.cuda.empty_cache()
            logger.info(f"🧹 已清理 CUDA 缓存")
        
        logger.info(f"✅ 专家 {expert_id} 已卸载")
        return True
    
    def unload_all_experts(self) -> int:
        """卸载所有专家模型"""
        count = 0
        expert_ids = list(self.loaded_models.keys())
        for expert_id in expert_ids:
            if self.unload_expert(expert_id):
                count += 1
        return count
    
    def get_loaded_experts(self) -> List[str]:
        """获取已加载的专家 ID 列表"""
        return list(self.loaded_models.keys())
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取当前显存/内存使用情况"""
        memory_info = {
            "loaded_experts": self.get_loaded_experts(),
            "num_loaded_experts": len(self.loaded_models),
        }
        
        if self.gpu_available:
            memory_info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
            memory_info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1e9:.2f}GB"
            memory_info["gpu_memory_total"] = f"{self.max_vram_gb:.2f}GB"
        
        return memory_info
    
    def _verify_model_files(self, model_path: Path) -> bool:
        """验证模型文件完整性"""
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors"
        ]
        
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                logger.error(f"❌ 缺失模型文件：{file_path}")
                return False
        
        return True
    
    def _estimate_vram_usage(self, model: torch.nn.Module) -> float:
        """
        估算模型显存占用
        
        公式：参数量 × 精度字节数 + 激活值开销
        4bit 量化：参数量 × 0.5 字节
        """
        num_params = sum(p.numel() for p in model.parameters())
        
        # 4bit 量化参数占用
        param_memory_gb = (num_params * 0.5) / (1024**3)
        
        # 激活值开销（估算为参数的 20%）
        activation_memory_gb = param_memory_gb * 0.2
        
        # 总占用
        total_vram_gb = param_memory_gb + activation_memory_gb
        
        return total_vram_gb


class ExpertContainer:
    """
    全局专家容器单例（TSD v1.7 强制要求）
    
    所有模块必须通过此容器访问专家模型，严禁各自加载
    """
    
    _instance: Optional['ExpertContainer'] = None
    _loader: Optional[ExpertLoader] = None
    
    def __new__(cls, config: Optional[ExpertConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._loader = ExpertLoader(config)
        return cls._instance
    
    def get_loader(self) -> ExpertLoader:
        """获取加载器实例"""
        return self._loader
    
    def get_expert(self, expert_id: str) -> Optional[Dict[str, Any]]:
        """获取专家实例"""
        return self._loader.get_expert(expert_id)
    
    def load_expert(self, **kwargs) -> Dict[str, Any]:
        """加载专家"""
        return self._loader.load_expert(**kwargs)
    
    def unload_expert(self, expert_id: str) -> bool:
        """卸载专家"""
        return self._loader.unload_expert(expert_id)
    
    def unload_all(self) -> int:
        """卸载所有专家"""
        return self._loader.unload_all_experts()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        return self._loader.get_memory_usage()


# 便捷函数
def get_expert_container() -> ExpertContainer:
    """获取全局专家容器单例"""
    return ExpertContainer()


__all__ = [
    "ExpertLoader",
    "ExpertContainer",
    "get_expert_container",
]
