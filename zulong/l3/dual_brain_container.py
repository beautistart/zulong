# File: zulong/l3/dual_brain_container.py
# 左右脑模型容器 - 管理两个热备模型及其同步

import torch
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class BrainRole(Enum):
    """脑角色枚举"""
    LEFT = "left"  # 左脑
    RIGHT = "right"  # 右脑


@dataclass
class BrainContext:
    """脑上下文数据
    
    TSD v1.7 对应规则:
    - 左右脑完全同步策略
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
class BrainInstance:
    """脑实例 - 封装单个模型及其状态
    
    类比：就像大脑的一个半球，有自己的模型权重和上下文
    """
    role: BrainRole  # 左脑/右脑
    model: Optional[Any] = None  # 模型实例
    model_id: str = ""  # 模型 ID
    is_loaded: bool = False  # 是否已加载到 GPU
    is_active: bool = False  # 是否当前活跃（正在使用）
    context: BrainContext = field(default_factory=BrainContext)  # 上下文
    vram_usage_gb: float = 0.0  # 显存占用
    load_time: float = 0.0  # 加载时间
    
    def load_model(self, model_instance: Any, model_id: str, vram_gb: float):
        """加载模型到 GPU
        
        Args:
            model_instance: 模型实例
            model_id: 模型 ID
            vram_gb: 预估显存占用
        """
        self.model = model_instance
        self.model_id = model_id
        self.is_loaded = True
        self.vram_usage_gb = vram_gb
        self.load_time = time.time()
        logger.info(f"[Brain {self.role.value}] Loaded model: {model_id}, VRAM: {vram_gb:.2f}GB")
    
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
        logger.info(f"[Brain {self.role.value}] Unloaded model")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            "role": self.role.value,
            "model_id": self.model_id,
            "is_loaded": self.is_loaded,
            "is_active": self.is_active,
            "vram_usage_gb": self.vram_usage_gb,
            "context_version": self.context.version,
            "last_sync_time": self.context.last_sync_time
        }


class DualBrainContainer:
    """左右脑模型容器
    
    TSD v1.7 对应规则:
    - 2.2.4 L3: 专家技能池 - 左右脑热备架构
    - 管理两个热备模型（左脑 + 右脑）
    - 实现完全同步策略
    
    功能:
    - 左右脑模型管理
    - 实时状态同步
    - 快速角色切换
    - 显存优化
    """
    
    def __init__(self, max_vram_gb: float = 5.8):
        """初始化左右脑容器
        
        Args:
            max_vram_gb: 最大可用显存（默认 5.8GB，RTX 3060）
        """
        self.max_vram_gb = max_vram_gb
        
        # 左右脑实例
        self.left_brain = BrainInstance(role=BrainRole.LEFT)
        self.right_brain = BrainInstance(role=BrainRole.RIGHT)
        
        # 同步配置
        self.sync_enabled = True  # 是否启用同步
        self.sync_context = True  # 同步上下文
        self.sync_kv_cache = False  # 同步 KV Cache（可选，耗资源）
        self.sync_generation = True  # 同步生成状态
        
        # 统计信息
        self.total_switches = 0  # 总切换次数
        self.total_syncs = 0  # 总同步次数
        self.last_switch_time = 0.0  # 最后切换时间
        
        logger.info(f"[DualBrain] Initialized with max VRAM: {max_vram_gb:.2f}GB")
    
    def load_brains(self, left_model: Any, right_model: Any, 
                    left_id: str, right_id: str,
                    vram_per_model: float = 1.5):
        """加载左右脑模型到 GPU
        
        Args:
            left_model: 左脑模型实例
            right_model: 右脑模型实例
            left_id: 左脑模型 ID
            right_id: 右脑模型 ID
            vram_per_model: 每个模型的显存占用
        """
        # 检查显存是否足够
        total_required = vram_per_model * 2
        if total_required > self.max_vram_gb:
            logger.warning(f"[DualBrain] Not enough VRAM: need {total_required:.2f}GB, have {self.max_vram_gb:.2f}GB")
            # 尝试卸载一个模型
            if self.right_brain.is_loaded:
                self.right_brain.unload_model()
        
        # 加载左脑
        self.left_brain.load_model(left_model, left_id, vram_per_model)
        
        # 加载右脑
        self.right_brain.load_model(right_model, right_id, vram_per_model)
        
        # 设置活跃脑（默认左脑）
        self.left_brain.is_active = True
        self.right_brain.is_active = False
        
        logger.info(f"[DualBrain] Loaded: Left={left_id}, Right={right_id}")
    
    def get_active_brain(self) -> BrainInstance:
        """获取当前活跃的脑实例"""
        if self.left_brain.is_active:
            return self.left_brain
        elif self.right_brain.is_active:
            return self.right_brain
        else:
            # 默认返回左脑
            return self.left_brain
    
    def switch_active_brain(self, new_active_role: BrainRole) -> bool:
        """切换活跃脑
        
        类比：就像切换使用左脑或右脑思考
        
        Args:
            new_active_role: 新的活跃脑角色
            
        Returns:
            bool: 切换是否成功
        """
        start_time = time.time()
        
        # 确定目标脑和源脑
        if new_active_role == BrainRole.LEFT:
            new_active = self.left_brain
            old_active = self.right_brain
        else:
            new_active = self.right_brain
            old_active = self.left_brain
        
        # 如果已经是活跃脑，无需切换
        if new_active.is_active:
            logger.debug(f"[DualBrain] Brain {new_active_role.value} already active")
            return True
        
        # 切换前同步（确保状态一致）
        if self.sync_enabled:
            self._sync_brains(source=old_active, target=new_active)
        
        # 执行切换
        old_active.is_active = False
        new_active.is_active = True
        
        # 更新统计
        self.total_switches += 1
        switch_duration = time.time() - start_time
        self.last_switch_time = switch_duration  # 存储切换耗时（秒）
        
        logger.info(f"[DualBrain] Switched to {new_active_role.value} brain "
                   f"(time: {switch_duration*1000:.2f}ms)")
        
        return True
    
    def _sync_brains(self, source: BrainInstance, target: BrainInstance):
        """同步两个脑的状态
        
        TSD v1.7 对应规则:
        - 完全同步策略：源脑状态复制到目标脑
        
        Args:
            source: 源脑（状态来源）
            target: 目标脑（同步目标）
        """
        if not self.sync_enabled:
            return
        
        start_time = time.time()
        
        # 1. 同步上下文状态（必须）
        if self.sync_context:
            target.context.from_dict(source.context.to_dict())
        
        # 2. 同步 KV Cache（可选）
        if self.sync_kv_cache and source.context.kv_cache is not None:
            # 深度复制 KV Cache
            target.context.kv_cache = {
                k: v.clone() for k, v in source.context.kv_cache.items()
            }
        
        # 3. 同步生成状态（可选）
        if self.sync_generation:
            target.context.current_tokens = source.context.current_tokens.copy()
            if source.context.next_token_logits is not None:
                target.context.next_token_logits = source.context.next_token_logits.clone()
        
        # 更新同步时间
        target.context.last_sync_time = time.time()
        target.context.version = source.context.version + 1
        
        # 更新统计
        self.total_syncs += 1
        sync_time = time.time() - start_time
        
        logger.debug(f"[DualBrain] Synced {source.role.value} -> {target.role.value} "
                    f"(time: {sync_time*1000:.2f}ms, version: {target.context.version})")
    
    def update_context(self, **kwargs):
        """更新当前活跃脑的上下文
        
        并自动同步到另一个脑
        
        Args:
            **kwargs: 上下文参数（current_task, conversation_history 等）
        """
        active_brain = self.get_active_brain()
        
        # 更新参数
        for key, value in kwargs.items():
            if hasattr(active_brain.context, key):
                setattr(active_brain.context, key, value)
            else:
                logger.warning(f"[DualBrain] Unknown context key: {key}")
        
        # 更新版本号
        active_brain.context.version += 1
        active_brain.context.last_sync_time = time.time()
        
        # 同步到另一个脑
        if self.sync_enabled:
            other_brain = self.right_brain if active_brain == self.left_brain else self.right_brain
            self._sync_brains(source=active_brain, target=other_brain)
    
    def get_context(self) -> BrainContext:
        """获取当前活跃脑的上下文"""
        return self.get_active_brain().context
    
    def get_vram_usage(self) -> float:
        """获取总显存占用"""
        total = 0.0
        if self.left_brain.is_loaded:
            total += self.left_brain.vram_usage_gb
        if self.right_brain.is_loaded:
            total += self.right_brain.vram_usage_gb
        return total
    
    def get_status(self) -> Dict[str, Any]:
        """获取容器状态"""
        return {
            "left_brain": self.left_brain.get_status(),
            "right_brain": self.right_brain.get_status(),
            "total_vram_usage_gb": self.get_vram_usage(),
            "max_vram_gb": self.max_vram_gb,
            "sync_enabled": self.sync_enabled,
            "total_switches": self.total_switches,
            "total_syncs": self.total_syncs,
            "last_switch_time_ms": self.last_switch_time * 1000
        }
    
    def print_status(self):
        """打印状态信息"""
        status = self.get_status()
        print("\n" + "=" * 60)
        print("左右脑模型容器状态")
        print("=" * 60)
        print(f"左脑：{status['left_brain']['model_id']} "
              f"(loaded={status['left_brain']['is_loaded']}, "
              f"active={status['left_brain']['is_active']})")
        print(f"右脑：{status['right_brain']['model_id']} "
              f"(loaded={status['right_brain']['is_loaded']}, "
              f"active={status['right_brain']['is_active']})")
        print(f"总显存占用：{status['total_vram_usage_gb']:.2f}GB / {status['max_vram_gb']:.2f}GB")
        print(f"切换次数：{status['total_switches']}")
        print(f"同步次数：{status['total_syncs']}")
        print(f"最后切换时间：{status['last_switch_time_ms']:.2f}ms")
        print("=" * 60 + "\n")
