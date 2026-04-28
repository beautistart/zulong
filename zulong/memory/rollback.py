# 记忆管理：降智回滚机制

"""
功能:
- 30 天未验证降级
- 90 天未使用归档
- 经验降级策略
- 自动归档清理

对应 TSD v2.3 第 12.2 节
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from .time_tags import (
    TimeTags,
    TimeTagManager,
    TimeDecayStrategy,
    ExperienceStatus
)

logger = logging.getLogger(__name__)


class RollbackAction(Enum):
    """回滚动作枚举"""
    NONE = "none"               # 无需操作
    DOWNGRADE = "downgrade"     # 降级
    ARCHIVE = "archive"         # 归档
    DELETE = "delete"           # 删除
    REACTIVATE = "reactivate"   # 重新激活


class ExperienceLevel(Enum):
    """经验等级"""
    LEVEL_1 = 1     # 新经验 (< 7 天)
    LEVEL_2 = 2     # 成熟经验 (7-30 天)
    LEVEL_3 = 3     # 稳定经验 (30-90 天)
    LEVEL_4 = 4     # 过期经验 (> 90 天)


class RollbackResult:
    """回滚结果数据类"""
    
    def __init__(self,
                 action: RollbackAction,
                 experience_id: str,
                 reason: str,
                 old_level: Optional[ExperienceLevel] = None,
                 new_level: Optional[ExperienceLevel] = None,
                 metadata: Optional[Dict] = None):
        self.action = action
        self.experience_id = experience_id
        self.reason = reason
        self.old_level = old_level
        self.new_level = new_level
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'action': self.action.value,
            'experience_id': self.experience_id,
            'reason': self.reason,
            'old_level': self.old_level.value if self.old_level else None,
            'new_level': self.new_level.value if self.new_level else None,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class RollbackStrategy:
    """回滚策略"""
    
    def __init__(self,
                 downgrade_days: int = 30,
                 archive_days: int = 90,
                 delete_days: int = 180,
                 reactivation_threshold: float = 0.7):
        """初始化回滚策略
        
        Args:
            downgrade_days: 降级天数阈值
            archive_days: 归档天数阈值
            delete_days: 删除天数阈值
            reactivation_threshold: 重新激活阈值
        """
        self.downgrade_days = downgrade_days
        self.archive_days = archive_days
        self.delete_days = delete_days
        self.reactivation_threshold = reactivation_threshold
        
        logger.info(f"[RollbackStrategy] 初始化完成："
                   f"downgrade={downgrade_days}d, archive={archive_days}d, "
                   f"delete={delete_days}d")
    
    def evaluate_rollback(self,
                          time_tags: TimeTags,
                          overall_score: float) -> RollbackAction:
        """评估回滚动作
        
        Args:
            time_tags: 时间标签
            overall_score: 综合评分
            
        Returns:
            RollbackAction: 建议的回滚动作
        """
        # 获取使用年龄
        days_since_use = time_tags.get_days_since_last_use()
        days_since_validation = time_tags.get_days_since_validation()
        
        # 检查是否需要删除
        if days_since_use and days_since_use >= self.delete_days:
            return RollbackAction.DELETE
        
        # 检查是否需要归档
        if days_since_use and days_since_use >= self.archive_days:
            return RollbackAction.ARCHIVE
        
        # 检查是否需要降级
        if days_since_validation and days_since_validation >= self.downgrade_days:
            return RollbackAction.DOWNGRADE
        
        # 检查是否可以重新激活
        if overall_score >= self.reactivation_threshold:
            return RollbackAction.REACTIVATE
        
        # 默认无需操作
        return RollbackAction.NONE
    
    def get_experience_level(self,
                              time_tags: TimeTags) -> ExperienceLevel:
        """获取经验等级
        
        Args:
            time_tags: 时间标签
            
        Returns:
            ExperienceLevel: 经验等级
        """
        age_days = time_tags.get_age_days()
        
        if age_days < 7:
            return ExperienceLevel.LEVEL_1
        elif age_days < 30:
            return ExperienceLevel.LEVEL_2
        elif age_days < 90:
            return ExperienceLevel.LEVEL_3
        else:
            return ExperienceLevel.LEVEL_4


class RollbackManager:
    """回滚管理器"""
    
    def __init__(self,
                 time_tag_manager: Optional[TimeTagManager] = None,
                 rollback_strategy: Optional[RollbackStrategy] = None):
        """初始化管理器
        
        Args:
            time_tag_manager: 时间标签管理器
            rollback_strategy: 回滚策略
        """
        self.time_tag_manager = time_tag_manager or TimeTagManager()
        self.rollback_strategy = rollback_strategy or RollbackStrategy()
        
        # 回调函数
        self._rollback_callbacks: List[Callable] = []
        
        # 统计信息
        self.stats = {
            'total_evaluations': 0,
            'downgrades': 0,
            'archives': 0,
            'deletes': 0,
            'reactivations': 0,
            'no_action': 0
        }
        
        logger.info("[RollbackManager] 初始化完成")
    
    def register_callback(self, callback: Callable):
        """注册回滚回调
        
        Args:
            callback: 回调函数 (接收 RollbackResult 参数)
        """
        self._rollback_callbacks.append(callback)
        logger.info(f"[RollbackManager] 注册回调：{callback.__name__}")
    
    async def evaluate_experience(self,
                                   experience: Dict[str, Any]) -> RollbackResult:
        """评估单个经验
        
        Args:
            experience: 经验数据 (包含 time_tags)
            
        Returns:
            RollbackResult: 回滚结果
        """
        self.stats['total_evaluations'] += 1
        
        # 提取时间标签
        time_tags_data = experience.get('time_tags')
        
        if not time_tags_data:
            logger.warning(f"[RollbackManager] 经验 {experience.get('id')} 缺少时间标签")
            return RollbackResult(
                action=RollbackAction.NONE,
                experience_id=experience.get('id', 'unknown'),
                reason="缺少时间标签"
            )
        
        time_tags = TimeTags.from_dict(time_tags_data)
        
        # 评估经验
        evaluation = self.time_tag_manager.evaluate_experience(
            time_tags,
            experience.get('usage_count', 0)
        )
        
        # 评估回滚动作
        action = self.rollback_strategy.evaluate_rollback(
            time_tags,
            evaluation['overall_score']
        )
        
        # 获取经验等级
        old_level = self.rollback_strategy.get_experience_level(time_tags)
        
        # 生成结果
        result = RollbackResult(
            action=action,
            experience_id=experience.get('id', 'unknown'),
            reason=evaluation['recommendation'],
            old_level=old_level,
            metadata={
                'evaluation': evaluation,
                'days_since_use': time_tags.get_days_since_last_use(),
                'days_since_validation': time_tags.get_days_since_validation()
            }
        )
        
        # 更新统计
        self._update_stats(action)
        
        # 触发回调
        await self._trigger_callbacks(result)
        
        logger.info(f"[RollbackManager] 经验评估完成："
                   f"{experience.get('id')} -> {action.value}")
        
        return result
    
    async def evaluate_batch(self,
                              experiences: List[Dict[str, Any]],
                              batch_size: int = 100) -> List[RollbackResult]:
        """批量评估经验
        
        Args:
            experiences: 经验列表
            batch_size: 批次大小
            
        Returns:
            List[RollbackResult]: 回滚结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(experiences), batch_size):
            batch = experiences[i:i + batch_size]
            
            # 并发评估
            tasks = [self.evaluate_experience(exp) for exp in batch]
            batch_results = await asyncio.gather(*tasks)
            
            results.extend(batch_results)
            
            logger.info(f"[RollbackManager] 完成批次 {i//batch_size + 1}/"
                       f"{(len(experiences) + batch_size - 1)//batch_size}")
        
        return results
    
    def execute_rollback(self,
                         result: RollbackResult,
                         experience_store) -> bool:
        """执行回滚操作
        
        Args:
            result: 回滚结果
            experience_store: 经验库实例
            
        Returns:
            bool: 是否成功执行
        """
        try:
            if result.action == RollbackAction.NONE:
                logger.debug(f"[RollbackManager] 无需操作：{result.experience_id}")
                return True
            
            elif result.action == RollbackAction.DOWNGRADE:
                # 降级：降低权重
                self._execute_downgrade(result, experience_store)
                
            elif result.action == RollbackAction.ARCHIVE:
                # 归档：移动到归档库
                self._execute_archive(result, experience_store)
                
            elif result.action == RollbackAction.DELETE:
                # 删除：从库中移除
                self._execute_delete(result, experience_store)
                
            elif result.action == RollbackAction.REACTIVATE:
                # 重新激活：提升权重
                self._execute_reactivate(result, experience_store)
            
            logger.info(f"[RollbackManager] 回滚执行成功：{result.experience_id}")
            return True
            
        except Exception as e:
            logger.error(f"[RollbackManager] 回滚执行失败：{e}")
            return False
    
    def _execute_downgrade(self,
                           result: RollbackResult,
                           experience_store):
        """执行降级
        
        Args:
            result: 回滚结果
            experience_store: 经验库
        """
        # 降低检索权重
        if experience_store:
            experience_store.update_experience_weight(
                result.experience_id,
                weight_multiplier=0.5  # 降低 50%
            )
        
        logger.info(f"[RollbackManager] 经验已降级：{result.experience_id}")
        self.stats['downgrades'] += 1
    
    def _execute_archive(self,
                         result: RollbackResult,
                         experience_store):
        """执行归档
        
        Args:
            result: 回滚结果
            experience_store: 经验库
        """
        # 移动到归档库
        if experience_store:
            experience_store.archive_experience(result.experience_id)
        
        logger.info(f"[RollbackManager] 经验已归档：{result.experience_id}")
        self.stats['archives'] += 1
    
    def _execute_delete(self,
                        result: RollbackResult,
                        experience_store):
        """执行删除
        
        Args:
            result: 回滚结果
            experience_store: 经验库
        """
        # 从库中删除
        if experience_store:
            experience_store.delete_experience(result.experience_id)
        
        logger.info(f"[RollbackManager] 经验已删除：{result.experience_id}")
        self.stats['deletes'] += 1
    
    def _execute_reactivate(self,
                            result: RollbackResult,
                            experience_store):
        """执行重新激活
        
        Args:
            result: 回滚结果
            experience_store: 经验库
        """
        # 提升权重
        if experience_store:
            experience_store.update_experience_weight(
                result.experience_id,
                weight_multiplier=1.5  # 提升 50%
            )
        
        logger.info(f"[RollbackManager] 经验已重新激活：{result.experience_id}")
        self.stats['reactivations'] += 1
    
    async def _trigger_callbacks(self, result: RollbackResult):
        """触发回调
        
        Args:
            result: 回滚结果
        """
        for callback in self._rollback_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"[RollbackManager] 回调执行失败：{e}")
    
    def _update_stats(self, action: RollbackAction):
        """更新统计
        
        Args:
            action: 回滚动作
        """
        if action == RollbackAction.DOWNGRADE:
            self.stats['downgrades'] += 1
        elif action == RollbackAction.ARCHIVE:
            self.stats['archives'] += 1
        elif action == RollbackAction.DELETE:
            self.stats['deletes'] += 1
        elif action == RollbackAction.REACTIVATE:
            self.stats['reactivations'] += 1
        else:
            self.stats['no_action'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            **self.stats,
            'total_evaluations': self.stats['total_evaluations']
        }
    
    def get_rollback_summary(self,
                              results: List[RollbackResult]) -> Dict[str, Any]:
        """获取回滚摘要
        
        Args:
            results: 回滚结果列表
            
        Returns:
            Dict: 摘要信息
        """
        summary = {
            'total': len(results),
            'by_action': {},
            'by_level': {},
            'recommendations': []
        }
        
        # 按动作分类
        for result in results:
            action = result.action.value
            summary['by_action'][action] = summary['by_action'].get(action, 0) + 1
            
            if result.old_level:
                level = f"LEVEL_{result.old_level.value}"
                summary['by_level'][level] = summary['by_level'].get(level, 0) + 1
            
            if result.action != RollbackAction.NONE:
                summary['recommendations'].append({
                    'experience_id': result.experience_id,
                    'action': result.action.value,
                    'reason': result.reason
                })
        
        return summary


# 全局单例
_rollback_manager_instance = None


def get_rollback_manager(
    time_tag_manager: Optional[TimeTagManager] = None,
    rollback_strategy: Optional[RollbackStrategy] = None
) -> RollbackManager:
    """获取回滚管理器单例
    
    Args:
        time_tag_manager: 时间标签管理器
        rollback_strategy: 回滚策略
        
    Returns:
        RollbackManager: 单例实例
    """
    global _rollback_manager_instance
    
    if _rollback_manager_instance is None:
        _rollback_manager_instance = RollbackManager(
            time_tag_manager=time_tag_manager,
            rollback_strategy=rollback_strategy
        )
    
    return _rollback_manager_instance
