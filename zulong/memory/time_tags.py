# 记忆管理：时间标签体系

"""
功能:
- 三维时间标签 (创建/最后使用/最后验证)
- 时间衰减策略
- 经验新鲜度评估
- 使用频率统计

对应 TSD v2.3 第 12.1 节
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ExperienceStatus(Enum):
    """经验状态枚举"""
    ACTIVE = "active"           # 活跃 (7 天内使用)
    VALIDATED = "validated"     # 已验证 (30 天内验证)
    STALE = "stale"            # 过期 (30-90 天未使用)
    ARCHIVED = "archived"      # 归档 (90 天以上未使用)


class TimeTags:
    """时间标签数据类"""
    
    def __init__(self,
                 created_at: Optional[datetime] = None,
                 last_used_at: Optional[datetime] = None,
                 last_validated_at: Optional[datetime] = None):
        """初始化时间标签
        
        Args:
            created_at: 创建时间
            last_used_at: 最后使用时间
            last_validated_at: 最后验证时间
        """
        self.created_at = created_at or datetime.utcnow()
        self.last_used_at = last_used_at
        self.last_validated_at = last_validated_at
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'created_at': self.created_at.isoformat(),
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'last_validated_at': self.last_validated_at.isoformat() if self.last_validated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeTags':
        """从字典创建"""
        return cls(
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.utcnow(),
            last_used_at=datetime.fromisoformat(data['last_used_at']) if data.get('last_used_at') else None,
            last_validated_at=datetime.fromisoformat(data['last_validated_at']) if data.get('last_validated_at') else None
        )
    
    def get_status(self) -> ExperienceStatus:
        """获取经验状态
        
        Returns:
            ExperienceStatus: 当前状态
        """
        now = datetime.utcnow()
        
        # 优先检查最后使用时间
        if self.last_used_at:
            days_since_use = (now - self.last_used_at).days
            
            if days_since_use <= 7:
                return ExperienceStatus.ACTIVE
            elif days_since_use <= 30:
                return ExperienceStatus.VALIDATED
            elif days_since_use <= 90:
                return ExperienceStatus.STALE
            else:
                return ExperienceStatus.ARCHIVED
        
        # 如果没有使用记录，检查验证时间
        if self.last_validated_at:
            days_since_validate = (now - self.last_validated_at).days
            
            if days_since_validate <= 30:
                return ExperienceStatus.VALIDATED
            elif days_since_validate <= 90:
                return ExperienceStatus.STALE
            else:
                return ExperienceStatus.ARCHIVED
        
        # 默认活跃 (新创建)
        return ExperienceStatus.ACTIVE
    
    def get_age_days(self) -> int:
        """获取经验年龄 (天)
        
        Returns:
            int: 年龄天数
        """
        return (datetime.utcnow() - self.created_at).days
    
    def get_days_since_last_use(self) -> Optional[int]:
        """获取距最后使用天数
        
        Returns:
            Optional[int]: 天数，如果没有使用记录返回 None
        """
        if self.last_used_at:
            return (datetime.utcnow() - self.last_used_at).days
        return None
    
    def get_days_since_validation(self) -> Optional[int]:
        """获取距最后验证天数
        
        Returns:
            Optional[int]: 天数，如果没有验证记录返回 None
        """
        if self.last_validated_at:
            return (datetime.utcnow() - self.last_validated_at).days
        return None


class TimeDecayStrategy:
    """时间衰减策略"""
    
    def __init__(self,
                 decay_rate: float = 0.1,
                 half_life_days: int = 30,
                 min_weight: float = 0.1):
        """初始化衰减策略
        
        Args:
            decay_rate: 衰减率 (每天)
            half_life_days: 半衰期 (天)
            min_weight: 最小权重
        """
        self.decay_rate = decay_rate
        self.half_life_days = half_life_days
        self.min_weight = min_weight
        
        logger.info(f"[TimeDecayStrategy] 初始化完成："
                   f"decay_rate={decay_rate}, half_life={half_life_days}d")
    
    def calculate_weight(self,
                         time_tags: TimeTags,
                         base_weight: float = 1.0) -> float:
        """计算时间权重
        
        Args:
            time_tags: 时间标签
            base_weight: 基础权重
            
        Returns:
            float: 衰减后的权重
        """
        # 获取经验年龄
        age_days = time_tags.get_age_days()
        
        # 使用指数衰减
        weight = base_weight * (0.5 ** (age_days / self.half_life_days))
        
        # 应用最小权重
        weight = max(weight, self.min_weight)
        
        return weight
    
    def calculate_recency_score(self,
                                 time_tags: TimeTags) -> float:
        """计算新鲜度分数
        
        Args:
            time_tags: 时间标签
            
        Returns:
            float: 新鲜度分数 (0-1)
        """
        # 优先使用最后使用时间
        reference_date = time_tags.last_used_at or time_tags.last_validated_at
        
        if not reference_date:
            # 没有使用或验证记录，使用创建时间
            reference_date = time_tags.created_at
        
        age_days = (datetime.utcnow() - reference_date).days
        
        # 使用半衰期计算新鲜度
        score = 0.5 ** (age_days / self.half_life_days)
        
        return max(score, 0.0)
    
    def calculate_usage_frequency(self,
                                   time_tags: TimeTags,
                                   usage_count: int) -> float:
        """计算使用频率分数
        
        Args:
            time_tags: 时间标签
            usage_count: 使用次数
            
        Returns:
            float: 使用频率分数 (0-1)
        """
        age_days = time_tags.get_age_days()
        
        if age_days == 0:
            return min(usage_count / 10, 1.0)
        
        # 计算每天平均使用次数
        daily_usage = usage_count / age_days
        
        # 归一化 (假设每天使用 1 次为满分)
        frequency_score = min(daily_usage, 1.0)
        
        return frequency_score


class TimeTagManager:
    """时间标签管理器"""
    
    def __init__(self,
                 decay_strategy: Optional[TimeDecayStrategy] = None):
        """初始化管理器
        
        Args:
            decay_strategy: 时间衰减策略
        """
        self.decay_strategy = decay_strategy or TimeDecayStrategy()
        
        # 状态阈值 (天)
        self.thresholds = {
            'active': 7,
            'validated': 30,
            'stale': 90
        }
        
        logger.info("[TimeTagManager] 初始化完成")
    
    def create_time_tags(self) -> TimeTags:
        """创建新的时间标签
        
        Returns:
            TimeTags: 时间标签实例
        """
        tags = TimeTags()
        logger.debug(f"[TimeTagManager] 创建时间标签：{tags.created_at}")
        return tags
    
    def update_usage(self,
                     time_tags: TimeTags,
                     usage_count: int = 1) -> TimeTags:
        """更新使用时间
        
        Args:
            time_tags: 时间标签
            usage_count: 使用次数增量
            
        Returns:
            TimeTags: 更新后的时间标签
        """
        time_tags.last_used_at = datetime.utcnow()
        
        logger.debug(f"[TimeTagManager] 更新使用时间：{time_tags.last_used_at}")
        
        return time_tags
    
    def update_validation(self,
                          time_tags: TimeTags) -> TimeTags:
        """更新验证时间
        
        Args:
            time_tags: 时间标签
            
        Returns:
            TimeTags: 更新后的时间标签
        """
        time_tags.last_validated_at = datetime.utcnow()
        
        logger.debug(f"[TimeTagManager] 更新验证时间：{time_tags.last_validated_at}")
        
        return time_tags
    
    def evaluate_experience(self,
                            time_tags: TimeTags,
                            usage_count: int = 0) -> Dict[str, Any]:
        """评估经验
        
        Args:
            time_tags: 时间标签
            usage_count: 使用次数
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 计算各项指标
        status = time_tags.get_status()
        age_days = time_tags.get_age_days()
        recency_score = self.decay_strategy.calculate_recency_score(time_tags)
        frequency_score = self.decay_strategy.calculate_usage_frequency(
            time_tags, usage_count
        )
        time_weight = self.decay_strategy.calculate_weight(time_tags)
        
        # 综合评分
        overall_score = (
            recency_score * 0.5 +      # 新鲜度占 50%
            frequency_score * 0.3 +    # 使用频率占 30%
            time_weight * 0.2          # 时间权重占 20%
        )
        
        evaluation = {
            'status': status.value,
            'age_days': age_days,
            'days_since_use': time_tags.get_days_since_last_use(),
            'days_since_validation': time_tags.get_days_since_validation(),
            'recency_score': recency_score,
            'frequency_score': frequency_score,
            'time_weight': time_weight,
            'overall_score': overall_score,
            'recommendation': self._get_recommendation(status, overall_score)
        }
        
        logger.debug(f"[TimeTagManager] 经验评估完成："
                    f"status={status.value}, score={overall_score:.3f}")
        
        return evaluation
    
    def _get_recommendation(self,
                            status: ExperienceStatus,
                            overall_score: float) -> str:
        """获取建议
        
        Args:
            status: 经验状态
            overall_score: 综合评分
            
        Returns:
            str: 建议
        """
        if status == ExperienceStatus.ACTIVE:
            return "保持使用"
        elif status == ExperienceStatus.VALIDATED:
            return "建议验证"
        elif status == ExperienceStatus.STALE:
            if overall_score < 0.3:
                return "建议归档"
            else:
                return "需要重新验证"
        else:  # ARCHIVED
            return "已归档，建议删除或重新激活"
    
    def get_stale_experiences(self,
                               experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取过期经验列表
        
        Args:
            experiences: 经验列表 (每个包含 time_tags 字段)
            
        Returns:
            List[Dict]: 过期经验列表
        """
        stale = []
        
        for exp in experiences:
            time_tags_data = exp.get('time_tags')
            
            if not time_tags_data:
                continue
            
            time_tags = TimeTags.from_dict(time_tags_data)
            status = time_tags.get_status()
            
            if status in [ExperienceStatus.STALE, ExperienceStatus.ARCHIVED]:
                stale.append({
                    **exp,
                    'status': status.value,
                    'evaluation': self.evaluate_experience(time_tags)
                })
        
        return stale
    
    def cleanup_recommendations(self,
                                 experiences: List[Dict[str, Any]]) -> Dict[str, List]:
        """生成清理建议
        
        Args:
            experiences: 经验列表
            
        Returns:
            Dict[str, List]: 分类建议
        """
        recommendations = {
            'to_archive': [],    # 需要归档
            'to_validate': [],   # 需要验证
            'to_delete': []      # 建议删除
        }
        
        for exp in experiences:
            time_tags_data = exp.get('time_tags')
            
            if not time_tags_data:
                continue
            
            time_tags = TimeTags.from_dict(time_tags_data)
            evaluation = self.evaluate_experience(time_tags)
            
            status = time_tags.get_status()
            
            if status == ExperienceStatus.ARCHIVED:
                if evaluation['overall_score'] < 0.2:
                    recommendations['to_delete'].append({
                        **exp,
                        'reason': '长期未使用且评分过低'
                    })
                else:
                    recommendations['to_archive'].append({
                        **exp,
                        'reason': '90 天以上未使用'
                    })
            elif status == ExperienceStatus.STALE:
                recommendations['to_validate'].append({
                    **exp,
                    'reason': '30-90 天未验证'
                })
        
        logger.info(f"[TimeTagManager] 清理建议："
                   f"归档={len(recommendations['to_archive'])}, "
                   f"验证={len(recommendations['to_validate'])}, "
                   f"删除={len(recommendations['to_delete'])}")
        
        return recommendations


# 全局单例
_time_tag_manager_instance = None


def get_time_tag_manager(
    decay_strategy: Optional[TimeDecayStrategy] = None
) -> TimeTagManager:
    """获取时间标签管理器单例
    
    Args:
        decay_strategy: 时间衰减策略
        
    Returns:
        TimeTagManager: 单例实例
    """
    global _time_tag_manager_instance
    
    if _time_tag_manager_instance is None:
        _time_tag_manager_instance = TimeTagManager(
            decay_strategy=decay_strategy
        )
    
    return _time_tag_manager_instance
