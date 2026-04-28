# File: zulong/l3/expert_container.py
# L3 专家技能池容器 - 通用专家管理
# TSD v1.7 规范：L3 层提供专用领域能力，支持动态加载/卸载、热切换

import torch
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import time

from .expert_config import (
    ExpertConfig,
    ExpertModelType,
    ExpertRole,
    ModelPathRegistry,
)
from .expert_loader import ExpertLoader

logger = logging.getLogger(__name__)


@dataclass
class ExpertContext:
    """
    专家上下文数据
    
    TSD v1.7 对应规则:
    - 专家状态同步策略
    - 同步内容：上下文状态、KV Cache、生成状态
    """
    # 1. 上下文状态（必须同步）
    current_task: str = ""  # 当前任务描述
    conversation_history: List[str] = field(default_factory=list)  # 对话历史
    working_memory: Dict[str, Any] = field(default_factory=dict)  # 工作记忆
    attention_state: str = ""  # 注意力焦点
    
    # 2. KV Cache（可选同步，用于加速切换）
    kv_cache: Optional[Dict[str, torch.Tensor]] = None
    
    # 3. 生成状态（如果正在生成文本）
    current_tokens: List[int] = field(default_factory=list)  # 已生成的 token
    next_token_logits: Optional[torch.Tensor] = None  # 下一个 token 的概率分布
    
    # 元数据
    last_sync_time: float = field(default_factory=time.time)  # 最后同步时间
    version: int = 0  # 版本号（用于冲突检测）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于同步）"""
        return {
            "current_task": self.current_task,
            "conversation_history": self.conversation_history.copy(),
            "working_memory": self.working_memory.copy(),
            "attention_state": self.attention_state,
            "current_tokens": self.current_tokens.copy(),
            "version": self.version
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """从字典加载（用于同步）"""
        self.current_task = data.get("current_task", "")
        self.conversation_history = data.get("conversation_history", []).copy()
        self.working_memory = data.get("working_memory", {}).copy()
        self.attention_state = data.get("attention_state", "")
        self.current_tokens = data.get("current_tokens", []).copy()
        self.version = data.get("version", 0)
        self.last_sync_time = time.time()


@dataclass
class ExpertInstance:
    """
    专家实例 - 封装单个专家模型及其状态
    
    类比：就像大脑的一个半球，有自己的模型权重和上下文
    """
    expert_id: str  # 专家 ID
    expert_type: ExpertModelType  # 专家类型
    role: ExpertRole  # 专家角色
    model: Optional[Any] = None  # 模型实例
    model_id: str = ""  # 模型 ID（具体模型名称）
    is_loaded: bool = False  # 是否已加载到 GPU
    is_active: bool = False  # 是否当前活跃（正在使用）
    context: ExpertContext = field(default_factory=ExpertContext)  # 上下文
    vram_usage_gb: float = 0.0  # 显存占用
    load_time: float = 0.0  # 加载时间
    
    def load_model(self, model_instance: Any, model_id: str, vram_gb: float):
        """加载模型到 GPU
        
        Args:
            model_instance: 模型实例
            model_id: 模型 ID（具体模型名称，如 "Qwen2.5-0.5B-Instruct"）
            vram_gb: 预估显存占用
        """
        self.model = model_instance
        self.model_id = model_id
        self.is_loaded = True
        self.vram_usage_gb = vram_gb
        self.load_time = time.time()
        logger.info(f"[Expert {self.expert_id}] Loaded model: {model_id}, VRAM: {vram_gb:.2f}GB")
    
    def unload_model(self):
        """从 GPU 卸载模型"""
        if self.model is not None:
            # 清理 GPU 内存
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            del self.model
            self.model = None
        
        self.is_loaded = False
        self.model_id = ""
        self.vram_usage_gb = 0.0
        
        # 保留上下文（以便下次快速恢复）
        logger.info(f"[Expert {self.expert_id}] Unloaded model")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            "expert_id": self.expert_id,
            "expert_type": self.expert_type.value,
            "role": self.role.value,
            "model_id": self.model_id,
            "is_loaded": self.is_loaded,
            "is_active": self.is_active,
            "vram_usage_gb": self.vram_usage_gb,
            "context_version": self.context.version,
            "last_sync_time": self.context.last_sync_time
        }


class ExpertPoolContainer:
    """
    L3 专家技能池容器（TSD v1.7 通用实现）
    
    TSD v1.7 对应规则:
    - 2.2.4 L3: 专家技能池 - 提供专用领域能力
    - 管理多个专家模型（左右脑、导航、操作等）
    - 实现热备架构和状态同步
    
    功能:
    - 专家模型管理（通用化，不绑定具体模型）
    - 实时状态同步
    - 快速角色切换
    - 显存优化
    """
    
    def __init__(
        self,
        config: Optional[ExpertConfig] = None,
        max_vram_gb: float = 5.8
    ):
        """
        初始化专家池容器
        
        Args:
            config: 专家配置（可选）
            max_vram_gb: 最大可用显存（默认 5.8GB，RTX 3060）
        """
        self.config = config or ExpertConfig()
        self.max_vram_gb = max_vram_gb
        
        # 专家实例池（动态管理）
        self.expert_pool: Dict[str, ExpertInstance] = {}
        
        # 加载器
        self.loader = ExpertLoader(self.config)
        
        # 同步配置
        self.sync_enabled = True  # 是否启用同步
        self.sync_context = True  # 同步上下文
        self.sync_kv_cache = False  # 同步 KV Cache（可选，耗资源）
        self.sync_generation = True  # 同步生成状态
        
        # 统计信息
        self.total_switches = 0  # 总切换次数
        self.total_syncs = 0  # 总同步次数
        self.last_switch_time = 0.0  # 最后切换时间
        
        logger.info(f"[ExpertPool] Initialized with max VRAM: {max_vram_gb:.2f}GB")
    
    def register_expert(
        self,
        expert_id: str,
        expert_type: ExpertModelType,
        role: ExpertRole,
        system_prompt: str,
    ) -> ExpertInstance:
        """
        注册专家到池中（不立即加载）
        
        Args:
            expert_id: 专家唯一标识
            expert_type: 专家类型
            role: 专家角色
            system_prompt: 系统提示词
        
        Returns:
            ExpertInstance 实例
        """
        # 创建专家实例
        expert = ExpertInstance(
            expert_id=expert_id,
            expert_type=expert_type,
            role=role,
        )
        
        # 注册到池中
        self.expert_pool[expert_id] = expert
        
        # 注册到配置
        self.config.register_expert(
            expert_id=expert_id,
            expert_type=expert_type,
            role=role,
            system_prompt=system_prompt,
            generation_config=self._get_default_generation_config(role),
        )
        
        logger.info(f"[ExpertPool] Registered expert: {expert_id} ({role.value})")
        return expert
    
    def load_expert(self, expert_id: str, model_path: Optional[str] = None) -> bool:
        """
        加载专家模型到 GPU
        
        Args:
            expert_id: 专家 ID
            model_path: 模型路径（可选）
        
        Returns:
            是否加载成功
        """
        if expert_id not in self.expert_pool:
            logger.error(f"❌ 专家 {expert_id} 未注册")
            return False
        
        expert = self.expert_pool[expert_id]
        
        # 检查显存是否足够
        if self._get_total_vram_usage() + 2.0 > self.max_vram_gb:
            logger.warning(f"⚠️ 显存不足，尝试卸载不活跃的专家")
            self._unload_inactive_experts()
        
        # 检查是否仍然不足
        if self._get_total_vram_usage() + 2.0 > self.max_vram_gb:
            logger.error(f"❌ 显存不足，无法加载专家 {expert_id}")
            return False
        
        # 获取专家配置
        expert_config = self.config.get_expert_config(expert_id)
        
        # 使用加载器加载模型
        try:
            model_data = self.loader.load_expert(
                expert_id=expert_id,
                model_path=model_path,
                role=expert.role.value,
                system_prompt=expert_config["system_prompt"],
            )
            
            # 更新专家实例
            expert.load_model(
                model_instance=model_data["model"],
                model_id=expert_config.get("model_path", "unknown"),
                vram_gb=model_data["config"]["vram_usage_gb"],
            )
            
            logger.info(f"[ExpertPool] Loaded expert: {expert_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载专家失败：{e}")
            return False
    
    def unload_expert(self, expert_id: str) -> bool:
        """卸载专家模型"""
        if expert_id not in self.expert_pool:
            return False
        
        expert = self.expert_pool[expert_id]
        
        # 从加载器卸载
        self.loader.unload_expert(expert_id)
        
        # 更新专家实例
        expert.unload_model()
        
        logger.info(f"[ExpertPool] Unloaded expert: {expert_id}")
        return True
    
    def get_active_expert(self) -> Optional[ExpertInstance]:
        """获取当前活跃的专家实例"""
        for expert in self.expert_pool.values():
            if expert.is_active:
                return expert
        return None
    
    def activate_expert(self, expert_id: str) -> bool:
        """
        激活指定专家（切换到该专家）
        
        Args:
            expert_id: 专家 ID
        
        Returns:
            是否激活成功
        """
        if expert_id not in self.expert_pool:
            logger.error(f"❌ 专家 {expert_id} 不存在")
            return False
        
        expert = self.expert_pool[expert_id]
        
        # 如果专家未加载，先加载
        if not expert.is_loaded:
            logger.info(f"🔄 专家 {expert_id} 未加载，开始加载...")
            if not self.load_expert(expert_id):
                return False
        
        # 获取当前活跃专家
        current_active = self.get_active_expert()
        
        # 如果已经是目标专家，直接返回
        if current_active and current_active.expert_id == expert_id:
            return True
        
        # 切换活跃专家
        if current_active:
            current_active.is_active = False
            logger.info(f"🔄 停用专家：{current_active.expert_id}")
        
        expert.is_active = True
        self.total_switches += 1
        self.last_switch_time = time.time()
        
        logger.info(f"🔄 激活专家：{expert_id} ({expert.role.value})")
        return True
    
    def sync_experts(self, from_expert_id: str, to_expert_id: str) -> bool:
        """
        同步两个专家的上下文状态
        
        TSD v1.7: 左右脑完全同步策略
        
        Args:
            from_expert_id: 源专家 ID
            to_expert_id: 目标专家 ID
        
        Returns:
            是否同步成功
        """
        if from_expert_id not in self.expert_pool or to_expert_id not in self.expert_pool:
            logger.error("❌ 专家不存在")
            return False
        
        from_expert = self.expert_pool[from_expert_id]
        to_expert = self.expert_pool[to_expert_id]
        
        if not self.sync_enabled:
            return True
        
        # 同步上下文
        if self.sync_context:
            to_expert.context.from_dict(from_expert.context.to_dict())
        
        # 同步 KV Cache（可选）
        if self.sync_kv_cache and from_expert.context.kv_cache is not None:
            to_expert.context.kv_cache = from_expert.context.kv_cache.clone()
        
        # 同步生成状态
        if self.sync_generation:
            to_expert.context.current_tokens = from_expert.context.current_tokens.copy()
            if from_expert.context.next_token_logits is not None:
                to_expert.context.next_token_logits = from_expert.context.next_token_logits.clone()
        
        # 更新同步时间
        to_expert.context.last_sync_time = time.time()
        self.total_syncs += 1
        
        logger.info(f"🔄 同步专家：{from_expert_id} -> {to_expert_id}")
        return True
    
    def get_expert_status(self, expert_id: str) -> Dict[str, Any]:
        """获取专家状态"""
        if expert_id not in self.expert_pool:
            return {"error": "专家不存在"}
        
        expert = self.expert_pool[expert_id]
        return expert.get_status()
    
    def get_all_experts_status(self) -> List[Dict[str, Any]]:
        """获取所有专家状态"""
        return [expert.get_status() for expert in self.expert_pool.values()]
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取显存使用情况"""
        return self.loader.get_memory_usage()
    
    def _get_total_vram_usage(self) -> float:
        """获取总显存使用量"""
        return sum(
            expert.vram_usage_gb for expert in self.expert_pool.values()
            if expert.is_loaded
        )
    
    def _unload_inactive_experts(self) -> int:
        """卸载不活跃的专家（LRU 策略）"""
        count = 0
        for expert in self.expert_pool.values():
            if expert.is_loaded and not expert.is_active:
                self.unload_expert(expert.expert_id)
                count += 1
        return count
    
    def _get_default_generation_config(self, role: ExpertRole) -> Dict[str, Any]:
        """获取默认生成配置"""
        base_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }
        
        if role == ExpertRole.LEFT:
            base_config["temperature"] = 0.3
            base_config["top_p"] = 0.9
        elif role == ExpertRole.RIGHT:
            base_config["temperature"] = 0.8
            base_config["top_p"] = 0.95
        
        return base_config


# 便捷函数
def get_expert_pool() -> ExpertPoolContainer:
    """获取全局专家池单例"""
    # 使用单例模式
    if not hasattr(get_expert_pool, '_instance'):
        get_expert_pool._instance = ExpertPoolContainer()
    return get_expert_pool._instance


__all__ = [
    "ExpertPoolContainer",
    "ExpertInstance",
    "ExpertContext",
    "get_expert_pool",
]
