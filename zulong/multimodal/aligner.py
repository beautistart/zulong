# File: zulong/multimodal/aligner.py
# 模态时间对齐器 (Phase 9.1)
# 对齐不同模态的时间戳

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from zulong.multimodal.fusion import ModalityFeatures

logger = logging.getLogger(__name__)


@dataclass
class AlignedWindow:
    """对齐的时间窗口"""
    start_time: float
    end_time: float
    modalities: Dict[str, ModalityFeatures]  # modality_name -> feature


class ModalityAligner:
    """
    模态时间对齐器
    
    功能:
    - 时间窗口对齐
    - 缺失模态插值
    - 时间戳同步
    
    使用示例:
    ```python
    aligner = ModalityAligner(window_size=1.0, step_size=0.5)
    
    # 添加不同时间戳的模态数据
    aligner.add_modality("vision", vision_features)
    aligner.add_modality("audio", audio_features)
    aligner.add_modality("text", text_features)
    
    # 获取对齐窗口
    windows = aligner.get_aligned_windows()
    ```
    """
    
    def __init__(self, window_size: float = 1.0, step_size: float = 0.5):
        """
        初始化对齐器
        
        Args:
            window_size: 窗口大小 (秒)
            step_size: 步长 (秒)
        """
        self._window_size = window_size
        self._step_size = step_size
        self._modalities: Dict[str, List[ModalityFeatures]] = {}
        
        logger.info(f"[ModalityAligner] 初始化完成 (window={window_size}s, step={step_size}s)")
    
    def add_modality(self, name: str, features: ModalityFeatures):
        """
        添加模态数据
        
        Args:
            name: 模态名称
            features: 模态特征
        """
        if name not in self._modalities:
            self._modalities[name] = []
        
        self._modalities[name].append(features)
        # 按时间戳排序
        self._modalities[name].sort(key=lambda m: m.timestamp)
    
    def get_aligned_windows(self) -> List[AlignedWindow]:
        """
        获取对齐的时间窗口
        
        Returns:
            List[AlignedWindow]: 对齐窗口列表
        """
        if not self._modalities:
            return []
        
        # 找到全局时间范围
        all_timestamps = []
        for modality_features in self._modalities.values():
            for m in modality_features:
                all_timestamps.append(m.timestamp)
        
        if not all_timestamps:
            return []
        
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # 生成时间窗口
        windows = []
        current_start = min_time
        
        while current_start + self._window_size <= max_time + self._window_size:
            current_end = current_start + self._window_size
            
            # 收集当前窗口内的所有模态
            window_modalities = {}
            
            for modality_name, modality_features in self._modalities.items():
                # 找到窗口内最近的特征
                closest = self._find_closest_in_window(
                    modality_features, current_start, current_end
                )
                
                if closest:
                    window_modalities[modality_name] = closest
            
            if window_modalities:
                windows.append(AlignedWindow(
                    start_time=current_start,
                    end_time=current_end,
                    modalities=window_modalities
                ))
            
            current_start += self._step_size
        
        return windows
    
    def _find_closest_in_window(
        self,
        features: List[ModalityFeatures],
        start: float,
        end: float
    ) -> Optional[ModalityFeatures]:
        """
        找到窗口内最近的模态特征
        
        Args:
            features: 模态特征列表
            start: 窗口开始时间
            end: 窗口结束时间
            
        Returns:
            Optional[ModalityFeatures]: 最近的特征，如果没有则返回 None
        """
        # 找到窗口内的特征
        in_window = [f for f in features if start <= f.timestamp <= end]
        
        if in_window:
            # 返回窗口中心的最近特征
            center = (start + end) / 2
            return min(in_window, key=lambda f: abs(f.timestamp - center))
        
        # 如果窗口内没有，找最近的
        if features:
            center = (start + end) / 2
            return min(features, key=lambda f: abs(f.timestamp - center))
        
        return None
    
    def interpolate_missing(
        self,
        windows: List[AlignedWindow]
    ) -> List[AlignedWindow]:
        """
        插值缺失的模态
        
        Args:
            windows: 对齐窗口列表
            
        Returns:
            List[AlignedWindow]: 插值后的窗口
        """
        if not windows:
            return windows
        
        # 找到所有模态名称
        all_modalities = set()
        for window in windows:
            all_modalities.update(window.modalities.keys())
        
        # 对每个窗口插值
        for i, window in enumerate(windows):
            for modality in all_modalities:
                if modality not in window.modalities:
                    # 尝试从相邻窗口插值
                    interpolated = self._interpolate_modality(
                        windows, i, modality
                    )
                    if interpolated:
                        window.modalities[modality] = interpolated
        
        return windows
    
    def _interpolate_modality(
        self,
        windows: List[AlignedWindow],
        current_idx: int,
        modality: str
    ) -> Optional[ModalityFeatures]:
        """
        插值特定模态
        
        Args:
            windows: 窗口列表
            current_idx: 当前窗口索引
            modality: 模态名称
            
        Returns:
            Optional[ModalityFeatures]: 插值后的特征
        """
        # 找前一个和后一个有这个模态的窗口
        prev_feat = None
        next_feat = None
        
        for i in range(current_idx - 1, -1, -1):
            if modality in windows[i].modalities:
                prev_feat = windows[i].modalities[modality]
                break
        
        for i in range(current_idx + 1, len(windows)):
            if modality in windows[i].modalities:
                next_feat = windows[i].modalities[modality]
                break
        
        # 线性插值
        if prev_feat and next_feat:
            # 简单平均
            avg_features = (prev_feat.features + next_feat.features) / 2
            avg_confidence = (prev_feat.confidence + next_feat.confidence) / 2
            
            return ModalityFeatures(
                modality=modality,
                features=avg_features,
                confidence=avg_confidence * 0.8,  # 插值降低置信度
                timestamp=(prev_feat.timestamp + next_feat.timestamp) / 2
            )
        elif prev_feat:
            return prev_feat
        elif next_feat:
            return next_feat
        
        return None
    
    def clear(self):
        """清空所有模态数据"""
        self._modalities.clear()
        logger.info("[ModalityAligner] 已清空")
