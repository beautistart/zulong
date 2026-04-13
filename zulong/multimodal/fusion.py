# File: zulong/multimodal/fusion.py
# 多模态融合器 (Phase 9.1)
# 实现视觉/音频/文本的联合推理

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """融合策略"""
    EARLY = "early"           # 早期融合 (特征级)
    LATE = "late"             # 晚期融合 (决策级)
    HYBRID = "hybrid"         # 混合融合


@dataclass
class ModalityFeatures:
    """单模态特征"""
    modality: str             # "vision", "audio", "text"
    features: np.ndarray      # 特征向量
    confidence: float         # 置信度
    timestamp: float          # 时间戳


@dataclass
class FusionResult:
    """融合结果"""
    fused_features: np.ndarray    # 融合后的特征
    confidence: float             # 融合置信度
    modality_weights: Dict[str, float]  # 各模态权重
    interpretation: str           # 融合解释


class MultimodalFusion:
    """
    多模态融合器
    
    功能:
    - 早期融合 (特征级连接)
    - 晚期融合 (加权决策)
    - 混合融合 (Transformer 风格注意力)
    - 时间对齐
    - 模态缺失处理
    
    使用示例:
    ```python
    fusion = MultimodalFusion()
    
    # 准备多模态特征
    vision = ModalityFeatures(
        modality="vision",
        features=np.random.randn(128),
        confidence=0.9,
        timestamp=0.0
    )
    
    text = ModalityFeatures(
        modality="text",
        features=np.random.randn(64),
        confidence=0.8,
        timestamp=0.0
    )
    
    # 融合
    result = fusion.fuse([vision, text], strategy=FusionStrategy.LATE)
    print(result.confidence)
    print(result.modality_weights)
    ```
    """
    
    def __init__(self):
        """初始化多模态融合器"""
        self._fusion_weights = {
            "vision": 0.4,
            "audio": 0.3,
            "text": 0.3
        }
        
        logger.info("[MultimodalFusion] 初始化完成")
    
    def fuse(
        self,
        modalities: List[ModalityFeatures],
        strategy: FusionStrategy = FusionStrategy.LATE
    ) -> FusionResult:
        """
        融合多模态特征
        
        Args:
            modalities: 多模态特征列表
            strategy: 融合策略
            
        Returns:
            FusionResult: 融合结果
        """
        if not modalities:
            raise ValueError("模态列表不能为空")
        
        if strategy == FusionStrategy.EARLY:
            return self._early_fusion(modalities)
        elif strategy == FusionStrategy.LATE:
            return self._late_fusion(modalities)
        elif strategy == FusionStrategy.HYBRID:
            return self._hybrid_fusion(modalities)
        else:
            raise ValueError(f"不支持的融合策略: {strategy}")
    
    def _early_fusion(self, modalities: List[ModalityFeatures]) -> FusionResult:
        """
        早期融合 (特征级连接)
        
        将所有模态的特征向量连接起来
        
        Args:
            modalities: 模态特征列表
            
        Returns:
            FusionResult: 融合结果
        """
        # 按时间戳排序
        modalities.sort(key=lambda m: m.timestamp)
        
        # 连接特征
        fused = np.concatenate([m.features for m in modalities])
        
        # 计算加权置信度
        total_weight = sum(self._fusion_weights.get(m.modality, 0.3) * m.confidence 
                          for m in modalities)
        weight_sum = sum(self._fusion_weights.get(m.modality, 0.3) 
                        for m in modalities)
        confidence = total_weight / weight_sum if weight_sum > 0 else 0.0
        
        # 模态权重
        modality_weights = {
            m.modality: self._fusion_weights.get(m.modality, 0.3)
            for m in modalities
        }
        
        interpretation = f"早期融合: 连接 {len(modalities)} 个模态的特征"
        
        return FusionResult(
            fused_features=fused,
            confidence=confidence,
            modality_weights=modality_weights,
            interpretation=interpretation
        )
    
    def _late_fusion(self, modalities: List[ModalityFeatures]) -> FusionResult:
        """
        晚期融合 (加权平均)
        
        各模态独立推理后加权平均
        
        Args:
            modalities: 模态特征列表
            
        Returns:
            FusionResult: 融合结果
        """
        # 归一化特征维度
        normalized_features = []
        for m in modalities:
            # L2 归一化
            norm = np.linalg.norm(m.features)
            if norm > 0:
                normalized_features.append(m.features / norm)
            else:
                normalized_features.append(m.features)
        
        # 加权融合
        fused = np.zeros_like(normalized_features[0])
        total_weight = 0.0
        
        for m, norm_feat in zip(modalities, normalized_features):
            weight = self._fusion_weights.get(m.modality, 0.3) * m.confidence
            
            # 如果维度不匹配，调整
            if norm_feat.shape != fused.shape:
                # 简单填充或截断
                if norm_feat.shape[0] < fused.shape[0]:
                    norm_feat = np.pad(norm_feat, (0, fused.shape[0] - norm_feat.shape[0]))
                else:
                    norm_feat = norm_feat[:fused.shape[0]]
            
            fused += weight * norm_feat
            total_weight += weight
        
        if total_weight > 0:
            fused /= total_weight
        
        # 置信度
        confidence = total_weight / len(modalities) if modalities else 0.0
        
        modality_weights = {
            m.modality: self._fusion_weights.get(m.modality, 0.3) * m.confidence
            for m in modalities
        }
        
        interpretation = f"晚期融合: 加权平均 {len(modalities)} 个模态的决策"
        
        return FusionResult(
            fused_features=fused,
            confidence=min(confidence, 1.0),
            modality_weights=modality_weights,
            interpretation=interpretation
        )
    
    def _hybrid_fusion(self, modalities: List[ModalityFeatures]) -> FusionResult:
        """
        混合融合 (简化版注意力机制)
        
        使用注意力权重融合多模态特征
        
        Args:
            modalities: 模态特征列表
            
        Returns:
            FusionResult: 融合结果
        """
        # 计算注意力分数 (基于置信度和模态权重)
        attention_scores = []
        for m in modalities:
            score = self._fusion_weights.get(m.modality, 0.3) * m.confidence
            attention_scores.append(score)
        
        # Softmax 归一化
        attention_scores = np.array(attention_scores)
        exp_scores = np.exp(attention_scores - np.max(attention_scores))
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # 加权融合 (使用注意力权重)
        # 将所有特征投影到相同维度 (取最大维度)
        max_dim = max(m.features.shape[0] for m in modalities)
        
        projected_features = []
        for m in modalities:
            feat = m.features
            if feat.shape[0] < max_dim:
                # 零填充
                feat = np.pad(feat, (0, max_dim - feat.shape[0]))
            else:
                feat = feat[:max_dim]
            projected_features.append(feat)
        
        # 注意力加权求和
        fused = np.zeros(max_dim)
        for i, (m, proj_feat) in enumerate(zip(modalities, projected_features)):
            fused += attention_weights[i] * proj_feat
        
        confidence = np.max(attention_weights)
        
        modality_weights = {
            m.modality: float(attention_weights[i])
            for i, m in enumerate(modalities)
        }
        
        interpretation = f"混合融合: 注意力机制融合 {len(modalities)} 个模态"
        
        return FusionResult(
            fused_features=fused,
            confidence=float(confidence),
            modality_weights=modality_weights,
            interpretation=interpretation
        )
    
    def set_weights(self, weights: Dict[str, float]):
        """
        设置模态权重
        
        Args:
            weights: 模态权重字典 {"vision": 0.5, "audio": 0.3, "text": 0.2}
        """
        self._fusion_weights.update(weights)
        logger.info(f"[MultimodalFusion] 更新模态权重: {weights}")
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前模态权重"""
        return self._fusion_weights.copy()
