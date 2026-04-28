# 混合检索配置管理

"""
功能:
- 混合检索权重配置
- 自适应权重调整
- 检索性能监控
- 参数优化建议

对应 TSD v2.3 第 14.2 节
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """混合检索配置"""
    
    # 向量检索权重 (0-1)
    vector_weight: float = 0.7
    
    # BM25 权重 (0-1)
    bm25_weight: float = 0.3
    
    # 时间衰减因子 (每天衰减比例)
    time_decay_factor: float = 0.1
    
    # 最大保留天数
    max_age_days: int = 30
    
    # 半衰期天数
    half_life_days: int = 7
    
    # 最小权重阈值
    min_weight_threshold: float = 0.1
    
    def __post_init__(self):
        """验证配置"""
        # 确保权重和为 1
        total = self.vector_weight + self.bm25_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"[HybridSearchConfig] 权重和不为 1: {total}，自动调整")
            self.vector_weight = self.vector_weight / total
            self.bm25_weight = self.bm25_weight / total
        
        logger.info(f"[HybridSearchConfig] 初始化完成："
                   f"vector={self.vector_weight}, bm25={self.bm25_weight}, "
                   f"decay={self.time_decay_factor}, half_life={self.half_life_days}")


class HybridSearchOptimizer:
    """混合检索优化器
    
    功能:
    1. 自适应权重调整
    2. 检索性能监控
    3. 参数优化建议
    """
    
    def __init__(self, config: Optional[HybridSearchConfig] = None):
        """初始化优化器
        
        Args:
            config: 混合检索配置
        """
        self.config = config or HybridSearchConfig()
        
        # 性能监控
        self.search_history: List[Dict] = []
        self.feedback_history: List[Dict] = []
        
        # 统计信息
        self.stats = {
            'total_searches': 0,
            'avg_latency_ms': 0.0,
            'user_satisfaction': 0.0
        }
        
        logger.info("[HybridSearchOptimizer] 初始化完成")
    
    def record_search(self,
                      query: str,
                      results: List[Dict],
                      latency_ms: float,
                      user_clicked: Optional[int] = None):
        """记录搜索历史
        
        Args:
            query: 查询文本
            results: 搜索结果
            latency_ms: 延迟（毫秒）
            user_clicked: 用户点击的结果索引（如果有）
        """
        record = {
            'query': query,
            'num_results': len(results),
            'latency_ms': latency_ms,
            'user_clicked': user_clicked,
            'config': {
                'vector_weight': self.config.vector_weight,
                'bm25_weight': self.config.bm25_weight
            }
        }
        
        self.search_history.append(record)
        
        # 更新统计
        self.stats['total_searches'] += 1
        
        # 移动平均延迟
        n = self.stats['total_searches']
        self.stats['avg_latency_ms'] = (
            (self.stats['avg_latency_ms'] * (n - 1) + latency_ms) / n
        )
        
        logger.debug(f"[HybridSearchOptimizer] 搜索记录：{query[:30]}... "
                    f"延迟={latency_ms:.2f}ms")
    
    def record_feedback(self,
                        query: str,
                        result_id: str,
                        relevance_score: float):
        """记录用户反馈
        
        Args:
            query: 查询文本
            result_id: 结果 ID
            relevance_score: 相关性评分 (0-1)
        """
        feedback = {
            'query': query,
            'result_id': result_id,
            'relevance_score': relevance_score,
            'config': {
                'vector_weight': self.config.vector_weight,
                'bm25_weight': self.config.bm25_weight
            }
        }
        
        self.feedback_history.append(feedback)
        
        # 更新用户满意度
        n = len(self.feedback_history)
        self.stats['user_satisfaction'] = (
            (self.stats['user_satisfaction'] * (n - 1) + relevance_score) / n
        )
        
        logger.info(f"[HybridSearchOptimizer] 反馈记录：{query[:30]}... "
                   f"评分={relevance_score:.2f}")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析检索性能
        
        Returns:
            Dict: 性能分析报告
        """
        if not self.search_history:
            return {'error': '没有搜索历史'}
        
        # 计算平均延迟
        avg_latency = np.mean([r['latency_ms'] for r in self.search_history])
        
        # 计算点击率（如果有点击数据）
        clicked_searches = [r for r in self.search_history if r.get('user_clicked') is not None]
        click_through_rate = len(clicked_searches) / len(self.search_history) if self.search_history else 0
        
        # 分析不同配置的性能
        config_performance = self._analyze_config_performance()
        
        report = {
            'total_searches': self.stats['total_searches'],
            'avg_latency_ms': avg_latency,
            'click_through_rate': click_through_rate,
            'user_satisfaction': self.stats['user_satisfaction'],
            'current_config': {
                'vector_weight': self.config.vector_weight,
                'bm25_weight': self.config.bm25_weight,
                'time_decay_factor': self.config.time_decay_factor
            },
            'config_performance': config_performance
        }
        
        return report
    
    def _analyze_config_performance(self) -> Dict:
        """分析不同配置的性能
        
        Returns:
            Dict: 配置性能分析
        """
        if len(self.feedback_history) < 10:
            return {'info': '反馈数据不足'}
        
        # 按配置分组
        config_groups = {}
        
        for feedback in self.feedback_history:
            key = f"v{feedback['config']['vector_weight']:.1f}_b{feedback['config']['bm25_weight']:.1f}"
            
            if key not in config_groups:
                config_groups[key] = []
            
            config_groups[key].append(feedback['relevance_score'])
        
        # 计算平均评分
        performance = {}
        for key, scores in config_groups.items():
            performance[key] = {
                'avg_score': np.mean(scores),
                'std_score': np.std(scores),
                'count': len(scores)
            }
        
        return performance
    
    def suggest_optimization(self) -> Dict[str, float]:
        """建议优化参数
        
        Returns:
            Dict: 建议的新参数
        """
        if len(self.feedback_history) < 20:
            logger.info("[HybridSearchOptimizer] 反馈数据不足，使用默认参数")
            return {
                'vector_weight': self.config.vector_weight,
                'bm25_weight': self.config.bm25_weight
            }
        
        # 分析不同权重的表现
        high_vector_scores = []
        high_bm25_scores = []
        
        for feedback in self.feedback_history:
            vector_weight = feedback['config']['vector_weight']
            score = feedback['relevance_score']
            
            if vector_weight > 0.6:
                high_vector_scores.append(score)
            elif vector_weight < 0.4:
                high_bm25_scores.append(score)
        
        # 比较哪种权重更好
        if high_vector_scores and high_bm25_scores:
            avg_vector = np.mean(high_vector_scores)
            avg_bm25 = np.mean(high_bm25_scores)
            
            if avg_vector > avg_bm25:
                logger.info(f"[HybridSearchOptimizer] 建议：增加向量权重 "
                           f"(向量={avg_vector:.2f} vs BM25={avg_bm25:.2f})")
                return {
                    'vector_weight': 0.8,
                    'bm25_weight': 0.2
                }
            else:
                logger.info(f"[HybridSearchOptimizer] 建议：增加 BM25 权重 "
                           f"(向量={avg_vector:.2f} vs BM25={avg_bm25:.2f})")
                return {
                    'vector_weight': 0.5,
                    'bm25_weight': 0.5
                }
        
        # 默认建议
        return {
            'vector_weight': self.config.vector_weight,
            'bm25_weight': self.config.bm25_weight
        }
    
    def apply_optimization(self, suggestions: Dict[str, float]):
        """应用优化建议
        
        Args:
            suggestions: 优化建议
        """
        if 'vector_weight' in suggestions:
            old_weight = self.config.vector_weight
            self.config.vector_weight = suggestions['vector_weight']
            logger.info(f"[HybridSearchOptimizer] 向量权重：{old_weight} -> "
                       f"{self.config.vector_weight}")
        
        if 'bm25_weight' in suggestions:
            old_weight = self.config.bm25_weight
            self.config.bm25_weight = suggestions['bm25_weight']
            logger.info(f"[HybridSearchOptimizer] BM25 权重：{old_weight} -> "
                       f"{self.config.bm25_weight}")
        
        # 重新归一化
        total = self.config.vector_weight + self.config.bm25_weight
        self.config.vector_weight /= total
        self.config.bm25_weight /= total
        
        logger.info(f"[HybridSearchOptimizer] 优化已应用")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            **self.stats,
            'config': {
                'vector_weight': self.config.vector_weight,
                'bm25_weight': self.config.bm25_weight,
                'time_decay_factor': self.config.time_decay_factor,
                'max_age_days': self.config.max_age_days
            },
            'history_size': len(self.search_history),
            'feedback_size': len(self.feedback_history)
        }


# 预定义配置模板

def get_balanced_config() -> HybridSearchConfig:
    """平衡配置（默认）"""
    return HybridSearchConfig(
        vector_weight=0.7,
        bm25_weight=0.3,
        time_decay_factor=0.1,
        half_life_days=7
    )


def get_vector_focused_config() -> HybridSearchConfig:
    """向量优先配置（适合语义搜索）"""
    return HybridSearchConfig(
        vector_weight=0.9,
        bm25_weight=0.1,
        time_decay_factor=0.05,
        half_life_days=14
    )


def get_keyword_focused_config() -> HybridSearchConfig:
    """关键词优先配置（适合精准匹配）"""
    return HybridSearchConfig(
        vector_weight=0.5,
        bm25_weight=0.5,
        time_decay_factor=0.15,
        half_life_days=5
    )


def get_fresh_content_config() -> HybridSearchConfig:
    """新鲜内容优先配置（适合实时数据）"""
    return HybridSearchConfig(
        vector_weight=0.7,
        bm25_weight=0.3,
        time_decay_factor=0.2,
        half_life_days=3
    )
