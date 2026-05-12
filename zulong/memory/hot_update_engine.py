# File: zulong/memory/hot_update_engine.py
# 祖龙 (ZULONG) 热更新引擎 - 实现动态经验注入

"""
热更新引擎 - 让系统真正"从经验中学习变聪明"

功能:
1. 监听经验库变化
2. 将经验转换为参数调整（热补丁）
3. 实时应用到 L0/L1 执行层
4. 版本管理和回滚

对应 TSD v2.3 第 10.3 节：动态经验注入
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class PatchType(Enum):
    """补丁类型"""
    PARAMETER = "parameter"      # 参数调整
    RULE = "rule"               # 规则更新
    STRATEGY = "strategy"       # 策略调整
    THRESHOLD = "threshold"     # 阈值调整


class PatchStatus(Enum):
    """补丁状态"""
    PENDING = "pending"         # 待处理
    APPLYING = "applying"       # 应用中
    APPLIED = "applied"         # 已应用
    FAILED = "failed"           # 失败
    ROLLED_BACK = "rolled_back" # 已回滚


@dataclass
class SystemPatch:
    """系统热补丁"""
    patch_id: str
    patch_type: PatchType
    target_layer: str  # "l0", "l1a", "l1b", "l2"
    condition: str     # 触发条件（如"抓取杯子"）
    adjustment: Dict[str, Any]  # 调整内容
    priority: int = 1  # 优先级（1-10）
    status: PatchStatus = PatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None  # 过期时间
    version: int = 1
    parent_patch_id: Optional[str] = None  # 父补丁 ID（用于版本链）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'patch_id': self.patch_id,
            'patch_type': self.patch_type.value,
            'target_layer': self.target_layer,
            'condition': self.condition,
            'adjustment': self.adjustment,
            'priority': self.priority,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'version': self.version,
            'parent_patch_id': self.parent_patch_id
        }


@dataclass
class ExperienceChange:
    """经验变更事件"""
    experience_id: str
    change_type: str  # "added", "updated", "accessed"
    experience_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HotUpdateEngine:
    """热更新引擎（事件驱动版）
    
    采用回调机制，在经验库写入时立即触发补丁生成
    优势：
    1. 毫秒级响应（无需等待轮询）
    2. 零空闲开销（无数据时不消耗算力）
    3. 事件驱动（符合系统架构原则）
    """
    
    def __init__(self, experience_store=None):
        """
        初始化热更新引擎
        
        Args:
            experience_store: 经验库实例
        """
        self.experience_store = experience_store
        
        # 补丁存储
        self._patches: Dict[str, SystemPatch] = {}
        self._active_patches: Dict[str, List[SystemPatch]] = {}  # layer -> patches
        
        # 回调函数
        self._patch_appliers: Dict[str, Callable] = {}  # layer -> applier
        
        # 【废弃】监控任务 - 改为事件驱动
        # self._monitoring = False
        # self._monitor_task: Optional[asyncio.Task] = None
        
        # 版本历史
        self._patch_history: List[SystemPatch] = []
        
        # 统计
        self.stats = {
            'total_patches': 0,
            'applied_patches': 0,
            'failed_patches': 0,
            'rolled_back_patches': 0
        }
        
        logger.info("[HotUpdateEngine] 初始化完成（事件驱动版）")
    
    def register_applier(self, layer: str, applier: Callable):
        """
        注册补丁应用器
        
        Args:
            layer: 层级名称（"l0", "l1a", "l1b", "l2"）
            applier: 应用函数 (async def apply(patch: SystemPatch))
        """
        self._patch_appliers[layer] = applier
        logger.info(f"[HotUpdateEngine] 已注册 {layer} 层应用器")
    
    async def on_experience_added(self, experience: Any) -> bool:
        """
        【事件驱动核心】当经验添加时立即触发
        
        调用时机：经验库 add_experience() 成功后立即调用
        优势：毫秒级响应，零空闲开销
        
        Args:
            experience: 经验对象
        
        Returns:
            bool: 是否成功生成并应用补丁
        """
        try:
            logger.info(f"[HotUpdateEngine] 检测到新经验：{experience.id[:8] if hasattr(experience, 'id') else 'unknown'}")
            
            # 1. 生成补丁
            patch = await self._generate_patch_from_experience(experience)
            
            if patch:
                logger.info(f"[HotUpdateEngine] 生成补丁：{patch.patch_id}")
                
                # 2. 应用补丁
                success = await self.apply_patch(patch)
                
                if success:
                    logger.info(f"[HotUpdateEngine] ✅ 补丁已应用：{patch.patch_id}")
                    return True
                else:
                    logger.warning(f"[HotUpdateEngine] ⚠️ 补丁应用失败：{patch.patch_id}")
                    return False
            else:
                logger.debug(f"[HotUpdateEngine] 未生成补丁（经验类型不适合）")
                return False
        
        except Exception as e:
            logger.error(f"[HotUpdateEngine] 处理经验失败：{e}")
            return False
    
    def _get_recent_experiences(self) -> List[Dict[str, Any]]:
        """【已废弃】获取最近的经验
        
        保留此方法仅用于向后兼容
        新方法：on_experience_added()
        """
        logger.warning("[HotUpdateEngine] _get_recent_experiences 已废弃，请使用 on_experience_added()")
        return []
    
    async def _generate_patch_from_experience(
        self,
        experience: Any
    ) -> Optional[SystemPatch]:
        """
        从经验生成补丁
        
        Args:
            experience: 经验对象
        
        Returns:
            Optional[SystemPatch]: 生成的补丁
        """
        try:
            # 1. 分析经验类型
            exp_type = getattr(experience, 'experience_type', 'unknown')
            content = getattr(experience, 'content', '')
            tags = getattr(experience, 'tags', [])
            metadata = getattr(experience, 'metadata', {})
            
            # 2. 根据经验类型生成补丁
            if exp_type == 'failure':
                # 失败案例：生成参数调整补丁
                patch = await self._generate_failure_patch(experience)
            elif exp_type == 'success':
                # 成功案例：生成策略优化补丁
                patch = await self._generate_success_patch(experience)
            elif exp_type == 'preference':
                # 用户偏好：生成规则更新补丁
                patch = await self._generate_preference_patch(experience)
            else:
                return None
            
            if patch:
                logger.info(f"[HotUpdateEngine] 生成补丁：{patch.patch_id}")
                return patch
            
        except Exception as e:
            logger.error(f"[HotUpdateEngine] 生成补丁失败：{e}")
        
        return None
    
    async def _generate_failure_patch(self, experience: Any) -> Optional[SystemPatch]:
        """从失败案例生成补丁"""
        # 简单示例：从元数据中提取参数调整
        metadata = getattr(experience, 'metadata', {})
        
        if 'parameter_adjustment' in metadata:
            adjustment = metadata['parameter_adjustment']
            
            patch_id = f"patch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(adjustment).encode()).hexdigest()[:8]}"
            
            return SystemPatch(
                patch_id=patch_id,
                patch_type=PatchType.PARAMETER,
                target_layer="l0",  # 执行层
                condition=getattr(experience, 'content', ''),
                adjustment=adjustment,
                priority=8  # 高优先级
            )
        
        return None
    
    async def _generate_success_patch(self, experience: Any) -> Optional[SystemPatch]:
        """从成功案例生成补丁"""
        # 简单示例：从元数据中提取策略优化
        metadata = getattr(experience, 'metadata', {})
        
        if 'strategy_optimization' in metadata:
            optimization = metadata['strategy_optimization']
            
            patch_id = f"patch_success_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            return SystemPatch(
                patch_id=patch_id,
                patch_type=PatchType.STRATEGY,
                target_layer="l1b",  # 调度层
                condition=getattr(experience, 'content', ''),
                adjustment=optimization,
                priority=5  # 中优先级
            )
        
        return None
    
    async def _generate_preference_patch(self, experience: Any) -> Optional[SystemPatch]:
        """从用户偏好生成补丁"""
        # 简单示例：从元数据中提取规则更新
        metadata = getattr(experience, 'metadata', {})
        
        if 'rule_update' in metadata:
            update = metadata['rule_update']
            
            patch_id = f"patch_pref_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            return SystemPatch(
                patch_id=patch_id,
                patch_type=PatchType.RULE,
                target_layer="l1a",  # 反射层
                condition=getattr(experience, 'content', ''),
                adjustment=update,
                priority=6
            )
        
        return None
    
    async def apply_patch(self, patch: SystemPatch) -> bool:
        """
        应用补丁
        
        Args:
            patch: 补丁对象
        
        Returns:
            bool: 是否成功应用
        """
        try:
            logger.info(f"[HotUpdateEngine] 开始应用补丁：{patch.patch_id}")
            
            # 1. 更新状态
            patch.status = PatchStatus.APPLYING
            self._patches[patch.patch_id] = patch
            
            # 2. 查找应用器
            if patch.target_layer not in self._patch_appliers:
                raise ValueError(f"未找到 {patch.target_layer} 层的应用器")
            
            applier = self._patch_appliers[patch.target_layer]
            
            # 3. 应用补丁
            success = await applier(patch)
            
            if success:
                # 4. 更新状态
                patch.status = PatchStatus.APPLIED
                patch.applied_at = datetime.utcnow()
                
                # 5. 添加到活跃补丁
                if patch.target_layer not in self._active_patches:
                    self._active_patches[patch.target_layer] = []
                self._active_patches[patch.target_layer].append(patch)
                
                # 6. 更新统计
                self.stats['applied_patches'] += 1
                
                logger.info(f"[HotUpdateEngine] 补丁已应用：{patch.patch_id}")
                return True
            else:
                raise Exception("应用器返回失败")
        
        except Exception as e:
            logger.error(f"[HotUpdateEngine] 应用补丁失败：{e}")
            patch.status = PatchStatus.FAILED
            self.stats['failed_patches'] += 1
            return False
    
    async def rollback_patch(self, patch_id: str) -> bool:
        """
        回滚补丁
        
        Args:
            patch_id: 补丁 ID
        
        Returns:
            bool: 是否成功回滚
        """
        if patch_id not in self._patches:
            logger.error(f"[HotUpdateEngine] 补丁不存在：{patch_id}")
            return False
        
        patch = self._patches[patch_id]
        
        try:
            logger.info(f"[HotUpdateEngine] 开始回滚补丁：{patch.patch_id}")
            
            # 1. 从活跃补丁移除
            if patch.target_layer in self._active_patches:
                self._active_patches[patch.target_layer] = [
                    p for p in self._active_patches[patch.target_layer]
                    if p.patch_id != patch_id
                ]
            
            # 2. 更新状态
            patch.status = PatchStatus.ROLLED_BACK
            self.stats['rolled_back_patches'] += 1
            
            # 3. 添加到历史
            self._patch_history.append(patch)
            
            logger.info(f"[HotUpdateEngine] 补丁已回滚：{patch.patch_id}")
            return True
        
        except Exception as e:
            logger.error(f"[HotUpdateEngine] 回滚失败：{e}")
            return False
    
    def get_active_patches(
        self,
        layer: Optional[str] = None,
        patch_type: Optional[PatchType] = None
    ) -> List[SystemPatch]:
        """
        获取活跃补丁
        
        Args:
            layer: 层级过滤
            patch_type: 类型过滤
        
        Returns:
            List[SystemPatch]: 补丁列表
        """
        patches = []
        
        if layer:
            patches = self._active_patches.get(layer, [])
        else:
            for layer_patches in self._active_patches.values():
                patches.extend(layer_patches)
        
        if patch_type:
            patches = [p for p in patches if p.patch_type == patch_type]
        
        return patches
    
    def get_patch_stats(self) -> Dict[str, Any]:
        """获取补丁统计"""
        return {
            **self.stats,
            'total_patches': len(self._patches),
            'active_patches': sum(len(p) for p in self._active_patches.values()),
            'history_size': len(self._patch_history)
        }


# 单例模式
_hot_update_engine: Optional[HotUpdateEngine] = None


def get_hot_update_engine(experience_store=None) -> HotUpdateEngine:
    """
    获取热更新引擎单例
    
    Args:
        experience_store: 经验库实例
    
    Returns:
        HotUpdateEngine 实例
    """
    global _hot_update_engine
    
    if _hot_update_engine is None:
        _hot_update_engine = HotUpdateEngine(experience_store)
    
    return _hot_update_engine
