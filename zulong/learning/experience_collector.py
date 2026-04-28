# File: zulong/learning/experience_collector.py
# 经验收集器 (Phase 9.3)
# 收集和过滤经验数据

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from zulong.learning.online_learner import Experience

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """收集配置"""
    min_reward: float = -1.0        # 最小奖励阈值
    max_reward: float = 1.0         # 最大奖励阈值
    filter_fn: Optional[Callable] = None  # 自定义过滤函数
    batch_size: int = 100           # 批量大小


class ExperienceCollector:
    """
    经验收集器
    
    功能:
    - 经验数据收集
    - 质量过滤
    - 批量刷新
    - 统计报告
    
    使用示例:
    ```python
    collector = ExperienceCollector(min_reward=0.0)
    
    # 收集经验
    collector.collect(state, action, reward, next_state)
    
    # 获取批量经验
    batch = collector.get_batch()
    ```
    """
    
    def __init__(self, config: Optional[CollectionConfig] = None):
        """
        初始化经验收集器
        
        Args:
            config: 收集配置
        """
        self._config = config or CollectionConfig()
        self._buffer: List[Experience] = []
        self._collected_count = 0
        self._filtered_count = 0
        
        logger.info("[ExperienceCollector] 初始化完成")
    
    def collect(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any],
        episode_id: Optional[str] = None
    ) -> bool:
        """
        收集经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 奖励
            next_state: 下一状态
            episode_id: 剧集 ID
            
        Returns:
            bool: 是否成功收集
        """
        # 奖励过滤
        if reward < self._config.min_reward or reward > self._config.max_reward:
            self._filtered_count += 1
            logger.debug(f"[ExperienceCollector] 过滤低质量经验: reward={reward}")
            return False
        
        # 创建经验
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            episode_id=episode_id
        )
        
        # 自定义过滤
        if self._config.filter_fn and not self._config.filter_fn(experience):
            self._filtered_count += 1
            return False
        
        # 添加到缓冲区
        self._buffer.append(experience)
        self._collected_count += 1
        
        return True
    
    def get_batch(self, batch_size: Optional[int] = None) -> List[Experience]:
        """
        获取批量经验
        
        Args:
            batch_size: 批量大小 (默认使用配置)
            
        Returns:
            List[Experience]: 经验列表
        """
        size = batch_size or self._config.batch_size
        
        if len(self._buffer) < size:
            return []
        
        # 取出批量经验
        batch = self._buffer[:size]
        self._buffer = self._buffer[size:]
        
        logger.debug(f"[ExperienceCollector] 获取批量: {len(batch)} 条经验")
        return batch
    
    def get_all(self) -> List[Experience]:
        """
        获取所有经验
        
        Returns:
            List[Experience]: 经验列表
        """
        all_experiences = self._buffer.copy()
        self._buffer.clear()
        return all_experiences
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取收集统计"""
        return {
            "collected_count": self._collected_count,
            "filtered_count": self._filtered_count,
            "buffer_size": len(self._buffer),
            "filter_rate": (
                self._filtered_count / self._collected_count
                if self._collected_count > 0
                else 0.0
            )
        }
    
    def clear(self):
        """清空缓冲区"""
        self._buffer.clear()
        logger.info("[ExperienceCollector] 缓冲区已清空")
