# File: zulong/learning/online_learner.py
# 在线学习框架 (Phase 9.3)
# 支持增量学习和模型微调

import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class LearningStrategy(str, Enum):
    """学习策略"""
    REPLAY = "replay"                   # 经验回放
    FINE_TUNE = "fine_tune"             # 微调
    CONTINUOUS = "continuous"           # 持续学习


@dataclass
class Experience:
    """经验数据"""
    state: Dict[str, Any]               # 状态
    action: str                         # 动作
    reward: float                       # 奖励
    next_state: Dict[str, Any]          # 下一状态
    timestamp: float = field(default_factory=time.time)
    episode_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        return cls(**data)


class OnlineLearner:
    """
    在线学习器
    
    功能:
    - 经验收集与存储
    - 在线学习 (批量/增量)
    - 灾难性遗忘防护
    - 学习效果评估
    
    使用示例:
    ```python
    learner = OnlineLearner(learning_rate=0.01)
    
    # 添加经验
    learner.add_experience(Experience(
        state={"task": "clean"},
        action="start_cleaning",
        reward=0.8,
        next_state={"task": "clean", "progress": 0.5}
    ))
    
    # 在线学习
    metrics = learner.learn(strategy=LearningStrategy.REPLAY)
    print(metrics)
    ```
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        replay_buffer_size: int = 1000,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        初始化在线学习器
        
        Args:
            learning_rate: 学习率
            batch_size: 批量大小
            replay_buffer_size: 经验回放缓冲区大小
            checkpoint_dir: 检查点目录
        """
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._replay_buffer_size = replay_buffer_size
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 经验存储
        self._experiences: List[Experience] = []
        self._old_experiences: List[Experience] = []  # 用于防遗忘
        
        # 学习统计
        self._total_experiences = 0
        self._learning_steps = 0
        self._cumulative_reward = 0.0
        
        logger.info(f"[OnlineLearner] 初始化完成 (lr={learning_rate}, batch={batch_size})")
    
    def add_experience(self, experience: Experience):
        """
        添加经验
        
        Args:
            experience: 经验数据
        """
        self._experiences.append(experience)
        self._total_experiences += 1
        self._cumulative_reward += experience.reward
        
        # 限制缓冲区大小
        if len(self._experiences) > self._replay_buffer_size:
            # 移动旧经验到长期存储 (防遗忘)
            old_exp = self._experiences.pop(0)
            self._old_experiences.append(old_exp)
            
            # 限制旧经验大小
            if len(self._old_experiences) > self._replay_buffer_size // 2:
                self._old_experiences.pop(0)
        
        logger.debug(f"[OnlineLearner] 添加经验，总计: {len(self._experiences)}")
    
    def add_batch_experiences(self, experiences: List[Experience]):
        """
        批量添加经验
        
        Args:
            experiences: 经验列表
        """
        for exp in experiences:
            self.add_experience(exp)
    
    def learn(
        self,
        strategy: LearningStrategy = LearningStrategy.REPLAY
    ) -> Dict[str, Any]:
        """
        执行学习
        
        Args:
            strategy: 学习策略
            
        Returns:
            Dict: 学习指标
        """
        if len(self._experiences) < self._batch_size:
            logger.warning(f"[OnlineLearner] 经验不足，需要至少 {self._batch_size} 条")
            return {"status": "insufficient_data"}
        
        self._learning_steps += 1
        
        if strategy == LearningStrategy.REPLAY:
            metrics = self._replay_learning()
        elif strategy == LearningStrategy.FINE_TUNE:
            metrics = self._fine_tuning()
        elif strategy == LearningStrategy.CONTINUOUS:
            metrics = self._continuous_learning()
        else:
            raise ValueError(f"不支持的学习策略: {strategy}")
        
        logger.info(f"[OnlineLearner] 学习完成，步骤: {self._learning_steps}")
        
        return metrics
    
    def _replay_learning(self) -> Dict[str, Any]:
        """
        经验回放学习
        
        从缓冲区随机采样批量学习
        
        Returns:
            Dict: 学习指标
        """
        import random
        
        # 采样当前经验
        batch = random.sample(self._experiences, min(self._batch_size, len(self._experiences)))
        
        # 采样旧经验 (防遗忘)
        if self._old_experiences:
            old_batch_size = min(self._batch_size // 4, len(self._old_experiences))
            old_batch = random.sample(self._old_experiences, old_batch_size)
            batch.extend(old_batch)
        
        # 计算平均奖励
        avg_reward = sum(exp.reward for exp in batch) / len(batch)
        
        # 模拟学习更新 (实际应用中应更新模型权重)
        loss = self._compute_loss(batch)
        
        metrics = {
            "strategy": "replay",
            "batch_size": len(batch),
            "avg_reward": avg_reward,
            "loss": loss,
            "learning_rate": self._learning_rate,
            "step": self._learning_steps
        }
        
        return metrics
    
    def _fine_tuning(self) -> Dict[str, Any]:
        """
        微调学习
        
        使用所有可用经验进行微调
        
        Returns:
            Dict: 学习指标
        """
        # 使用所有经验
        all_experiences = self._experiences + self._old_experiences
        
        avg_reward = sum(exp.reward for exp in all_experiences) / len(all_experiences)
        loss = self._compute_loss(all_experiences)
        
        # 微调使用较小的学习率
        fine_tune_lr = self._learning_rate * 0.1
        
        metrics = {
            "strategy": "fine_tune",
            "total_experiences": len(all_experiences),
            "avg_reward": avg_reward,
            "loss": loss,
            "learning_rate": fine_tune_lr,
            "step": self._learning_steps
        }
        
        return metrics
    
    def _continuous_learning(self) -> Dict[str, Any]:
        """
        持续学习
        
        只使用最新经验，快速适应
        
        Returns:
            Dict: 学习指标
        """
        # 使用最新经验
        recent_size = min(self._batch_size, len(self._experiences))
        recent_experiences = self._experiences[-recent_size:]
        
        avg_reward = sum(exp.reward for exp in recent_experiences) / len(recent_experiences)
        loss = self._compute_loss(recent_experiences)
        
        # 持续学习使用较大的学习率
        continuous_lr = self._learning_rate * 2.0
        
        metrics = {
            "strategy": "continuous",
            "recent_experiences": len(recent_experiences),
            "avg_reward": avg_reward,
            "loss": loss,
            "learning_rate": continuous_lr,
            "step": self._learning_steps
        }
        
        return metrics
    
    def _compute_loss(self, experiences: List[Experience]) -> float:
        """
        计算损失 (简化版)
        
        Args:
            experiences: 经验列表
            
        Returns:
            float: 损失值
        """
        if not experiences:
            return 0.0
        
        # 简化：使用负奖励作为损失
        avg_reward = sum(exp.reward for exp in experiences) / len(experiences)
        loss = -avg_reward + 1.0  # 归一化到 0-2 范围
        
        return max(loss, 0.0)
    
    def save_checkpoint(self, checkpoint_name: str = "latest"):
        """
        保存检查点
        
        Args:
            checkpoint_name: 检查点名称
        """
        checkpoint_path = self._checkpoint_dir / f"{checkpoint_name}.json"
        
        data = {
            "experiences": [exp.to_dict() for exp in self._experiences],
            "old_experiences": [exp.to_dict() for exp in self._old_experiences],
            "statistics": {
                "total_experiences": self._total_experiences,
                "learning_steps": self._learning_steps,
                "cumulative_reward": self._cumulative_reward
            }
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[OnlineLearner] 保存检查点: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_name: str = "latest") -> bool:
        """
        加载检查点
        
        Args:
            checkpoint_name: 检查点名称
            
        Returns:
            bool: 是否成功
        """
        checkpoint_path = self._checkpoint_dir / f"{checkpoint_name}.json"
        
        if not checkpoint_path.exists():
            logger.warning(f"[OnlineLearner] 检查点不存在: {checkpoint_path}")
            return False
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._experiences = [Experience.from_dict(exp) for exp in data["experiences"]]
        self._old_experiences = [Experience.from_dict(exp) for exp in data["old_experiences"]]
        self._total_experiences = data["statistics"]["total_experiences"]
        self._learning_steps = data["statistics"]["learning_steps"]
        self._cumulative_reward = data["statistics"]["cumulative_reward"]
        
        logger.info(f"[OnlineLearner] 加载检查点: {checkpoint_path}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取学习统计"""
        avg_reward = (
            self._cumulative_reward / self._total_experiences
            if self._total_experiences > 0
            else 0.0
        )
        
        return {
            "total_experiences": self._total_experiences,
            "current_buffer_size": len(self._experiences),
            "old_buffer_size": len(self._old_experiences),
            "learning_steps": self._learning_steps,
            "cumulative_reward": self._cumulative_reward,
            "avg_reward": avg_reward
        }
    
    def clear(self):
        """清空所有经验"""
        self._experiences.clear()
        self._old_experiences.clear()
        self._total_experiences = 0
        self._learning_steps = 0
        self._cumulative_reward = 0.0
        logger.info("[OnlineLearner] 已清空")
