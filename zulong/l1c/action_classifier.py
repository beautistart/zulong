# File: zulong/l1c/action_classifier.py
"""
L1-C 动作分类器 (MobileNetV4-TSM)

TSD v1.7 对应:
- 4.4 感知预处理
- 5.2 显存约束 (4bit 量化)

优化方案核心:
- 替代 ST-GCN (无需骨骼数据)
- SlowFast 双流架构
- TSM (Temporal Shift Module) 引入时序信息

架构优势:
- 无需骨骼关键点 (解决 ST-GCN 训练数据难的问题)
- 推理速度快 (适合 L1 层实时处理)
- 参数量小 (<5MB)
"""

import numpy as np
import cv2
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
import logging
import time
import os
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger("ActionClassifier")


class MobileNetV4_TSM:
    """
    MobileNetV4 + Temporal Shift Module 动作分类器
    
    核心逻辑 (SlowFast 双流架构):
    1. **Slow Pathway**: 低帧率 (8 FPS), 捕捉空间语义
    2. **Fast Pathway**: 高帧率 (30 FPS), 捕捉微小运动
    3. **TSM**: 通道位移引入时序信息
    4. **融合**: 双路特征融合，输出意图分数
    
    TSD v1.7 对应:
    - 4.2.1 L1-B 注意力控制器
    - 5.2 显存约束
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        logger.info("🧠 [MobileNetV4_TSM.__init__] Creating action classifier...")
        
        self._config = config or {
            'slow_fps': 8,  # Slow 流帧率
            'fast_fps': 30,  # Fast 流帧率
            'slow_frame_interval': 4,  # Slow 流采样间隔 (30/8≈4)
            'num_frames_slow': 8,  # Slow 流输入帧数
            'num_frames_fast': 16,  # Fast 流输入帧数
            'intent_threshold': 0.6,  # 意图阈值
            'interact_threshold': 0.8,  # 交互阈值
        }
        
        # 帧缓冲区
        self.slow_buffer = deque(maxlen=self._config.get('num_frames_slow', 8))
        self.fast_buffer = deque(maxlen=self._config.get('num_frames_fast', 16))
        
        # 时间戳追踪
        self.last_slow_frame_time = 0.0
        self.frame_count = 0
        
        # 模型占位符
        self._model = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 意图类型
        self.intent_types = [
            "WAVING",      # 挥手
            "APPROACHING", # 靠近
            "GAZING",      # 注视
            "STILL",       # 静止
            "UNKNOWN",     # 未知
        ]
        
        # 自定义意图分类头（全连接层）
        self._intent_classifier = None
        
        logger.info("✅ [MobileNetV4_TSM] Instance created")
        logger.info(f"   - Slow FPS: {self._config['slow_fps']}")
        logger.info(f"   - Fast FPS: {self._config['fast_fps']}")
        logger.info(f"   - Intent Threshold: {self._config['intent_threshold']}")
        logger.info(f"   - Device: {self._device}")
    
    def load_model(self, model_path: Optional[str] = None):
        """
        加载 MobileNetV3 模型（替代 MobileNetV4-TSM）
        
        Args:
            model_path: 模型路径 (可选，默认使用 PyTorch 预训练模型)
        """
        try:
            # ========== 1. 加载 MobileNetV3  backbone ==========
            logger.info("📦 [MobileNetV4_TSM] 加载 MobileNetV3-Large...")
            
            # 使用 PyTorch 官方预训练模型
            self._model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            self._model.to(self._device)
            self._model.eval()
            
            # 移除原始分类头，只保留特征提取
            self._model.classifier = nn.Identity()
            
            logger.info(f"✅ [MobileNetV4_TSM] MobileNetV3-Large 加载成功 (预训练)")
            
            # ========== 2. 自定义意图分类头（可选） ==========
            # MobileNetV3-Large 输出 1280 维特征
            # 我们不需要训练分类头，使用规则分类
            num_features = 1280
            num_intents = 5
            
            self._intent_classifier = None  # 不使用神经网络分类头，使用规则分类
            
            logger.info("✅ [MobileNetV4_TSM] 模型加载完成 (使用规则分类)")
            
        except Exception as e:
            logger.error(f"❌ [MobileNetV4_TSM] 模型加载失败：{e}")
            logger.warning("⚠️ [MobileNetV4_TSM] 降级到模拟模式")
            self._model = None
            self._intent_classifier = None
    
    def add_frame(self, frame: np.ndarray, timestamp: float) -> bool:
        """
        添加帧到缓冲区
        
        Args:
            frame: BGR 格式帧
            timestamp: 时间戳
        
        Returns:
            bool: 是否可以进行动作分类 (Slow 缓冲区已满)
        """
        try:
            # 预处理帧 (resize + normalize)
            processed_frame = self._preprocess_frame(frame)
            
            # 添加到 Fast 缓冲区 (每帧都加)
            self.fast_buffer.append(processed_frame.copy())
            
            # 添加到 Slow 缓冲区 (按间隔采样)
            if self.frame_count % self._config['slow_frame_interval'] == 0:
                self.slow_buffer.append(processed_frame.copy())
                self.last_slow_frame_time = timestamp
            
            self.frame_count += 1
            
            # 检查 Slow 缓冲区是否已满
            return len(self.slow_buffer) >= self._config['num_frames_slow']
            
        except Exception as e:
            logger.error(f"❌ [add_frame] 错误：{e}")
            return False
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理帧 (适配 MobileNetV4 输入)
        
        Args:
            frame: BGR 格式帧
        
        Returns:
            预处理后的帧 (224x224, normalized)
        """
        try:
            # Resize to 224x224
            resized = cv2.resize(frame, (224, 224))
            
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
            
            return rgb
            
        except Exception as e:
            logger.error(f"❌ [_preprocess_frame] 错误：{e}")
            return np.zeros((224, 224, 3), dtype=np.float32)
    
    def classify_action(self) -> Tuple[float, str, Dict[str, Any]]:
        """
        动作分类 (核心逻辑)
        
        流程:
        1. Slow 流：提取空间语义特征
        2. Fast 流：提取时序运动特征
        3. TSM：通道位移融合时序信息
        4. 输出：意图分数 + 意图类型 + 详细特征
        
        Returns:
            (意图分数，意图类型，详细特征字典)
        """
        try:
            # 检查缓冲区
            if len(self.slow_buffer) < self._config['num_frames_slow']:
                logger.warning("⚠️ [classify_action] Slow 缓冲区未满")
                return 0.0, "UNKNOWN", {}
            
            if len(self.fast_buffer) < self._config['num_frames_fast']:
                logger.warning("⚠️ [classify_action] Fast 缓冲区未满")
                return 0.0, "UNKNOWN", {}
            
            # ========== 1. Slow 流：空间语义特征 ==========
            slow_features = self._slow_pathway(list(self.slow_buffer))
            
            # ========== 2. Fast 流：时序运动特征 ==========
            fast_features = self._fast_pathway(list(self.fast_buffer))
            
            # ========== 3. TSM：通道位移融合 ==========
            fused_features = self._temporal_shift_fusion(slow_features, fast_features)
            
            # ========== 4. 意图分类 ==========
            intent_score, intent_type, intent_details = self._classify_intent(fused_features)
            
            logger.debug(f"🧠 [Action] 意图分类：{intent_type} ({intent_score:.2f})")
            
            return intent_score, intent_type, intent_details
            
        except Exception as e:
            logger.error(f"❌ [classify_action] 错误：{e}")
            return 0.0, "ERROR", {}
    
    def _slow_pathway(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Slow Pathway: 低帧率空间语义特征提取
        
        Args:
            frames: 预处理后的帧列表 (8 帧)
        
        Returns:
            空间语义特征向量
        """
        try:
            if self._model is None:
                # 模拟模式降级
                logger.warning("⚠️ [_slow_pathway] 使用模拟模式")
                avg_frame = np.mean(frames, axis=0)
                return self._extract_simple_features(avg_frame)
            
            # ========== 真实推理 ==========
            # 计算平均帧（空间语义）
            avg_frame = np.mean(frames, axis=0)
            
            # 转为 Tensor
            input_tensor = torch.from_numpy(avg_frame).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
            
            # MobileNetV3 特征提取
            with torch.no_grad():
                features = self._model(input_tensor)
            
            # 转为 numpy
            features_np = features.cpu().numpy()[0]
            
            logger.debug(f"🧠 [Slow] 特征维度：{features_np.shape}")
            
            return features_np
            
        except Exception as e:
            logger.error(f"❌ [_slow_pathway] 错误：{e}")
            return np.zeros(1280, dtype=np.float32)
    
    def _fast_pathway(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Fast Pathway: 高帧率时序运动特征提取
        
        Args:
            frames: 预处理后的帧列表 (16 帧)
        
        Returns:
            时序运动特征向量
        """
        try:
            if self._model is None:
                # 模拟模式降级
                logger.warning("⚠️ [_fast_pathway] 使用模拟模式")
                motion_magnitude = self._compute_motion_magnitude(frames)
                motion_consistency = self._compute_motion_consistency(frames)
                features = np.zeros(1280, dtype=np.float32)
                features[0] = motion_magnitude
                features[1] = motion_consistency
                return features
            
            # ========== 真实推理：使用时序特征 ==========
            # 策略：对 Fast 流 16 帧进行特征提取，然后聚合
            
            all_features = []
            for frame in frames[-8:]:  # 使用最近 8 帧
                input_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
                
                with torch.no_grad():
                    feat = self._model(input_tensor)
                
                all_features.append(feat.cpu().numpy()[0])
            
            # 计算时序统计特征
            features_stack = np.stack(all_features, axis=0)  # (8, 1280)
            
            # 聚合：只使用均值（保持 1280 维）
            mean_features = np.mean(features_stack, axis=0)
            
            logger.debug(f"🧠 [Fast] 特征维度：{mean_features.shape}")
            
            return mean_features
            
        except Exception as e:
            logger.error(f"❌ [_fast_pathway] 错误：{e}")
            return np.zeros(1280, dtype=np.float32)
    
    def _temporal_shift_fusion(
        self, 
        slow_features: np.ndarray, 
        fast_features: np.ndarray
    ) -> np.ndarray:
        """
        TSM: 时序位移融合
        
        核心逻辑:
        1. 将 Slow 和 Fast 特征按通道拼接
        2. 对部分通道进行时序位移 (shift)
        3. 输出融合特征
        
        Args:
            slow_features: Slow 流特征 (1280 维)
            fast_features: Fast 流特征 (1280 维)
        
        Returns:
            融合特征 (2560 维)
        """
        try:
            # 拼接特征
            fused = np.concatenate([slow_features, fast_features], axis=0)
            
            # TSM: 对部分通道进行时序位移 (模拟)
            # 真实实现会使用 PyTorch 的 channel_shift 操作
            shift_ratio = 0.125  # 12.5% 通道位移
            num_shift_channels = int(len(fused) * shift_ratio)
            
            # 位移操作 (简化版)
            fused[:num_shift_channels] = np.roll(fused[:num_shift_channels], shift=1)
            
            logger.debug(f"🧠 [TSM] 融合特征维度：{fused.shape}")
            
            return fused
            
        except Exception as e:
            logger.error(f"❌ [_temporal_shift_fusion] 错误：{e}")
            return np.zeros(2560, dtype=np.float32)
    
    def _classify_intent(
        self, 
        features: np.ndarray
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        意图分类（基于规则的简化版本）
        
        Args:
            features: 融合特征 (2560 维)
        
        Returns:
            (意图分数，意图类型，详细特征字典)
        """
        try:
            # 检查特征维度
            if len(features) != 2560:
                logger.warning(f"⚠️ [_classify_intent] 特征维度异常：{len(features)}, 期望 2560")
                # 尝试修复维度
                if len(features) < 2560:
                    features = np.pad(features, (0, 2560 - len(features)))
                else:
                    features = features[:2560]
            
            # ========== 基于规则的意图分类 ==========
            # 不使用未训练的权重，而是使用特征统计
            
            # 1. 计算特征能量（运动强度）
            feature_energy = np.mean(np.abs(features))
            
            # 2. 计算特征变化率（Fast 流前 1280 维 vs 后 1280 维）
            slow_feat = features[:1280]
            fast_feat = features[1280:]
            feature_diff = np.mean(np.abs(fast_feat - slow_feat))
            
            # 3. 规则分类（优化版）
            # 挥手：高能量 + 高变化（手臂大幅度摆动）
            # 靠近：中等能量 + 低变化（缓慢靠近，特征稳定）
            # 注视：低能量 + 中等变化（微小头部动作）
            # 静止：低能量 + 低变化
            
            # 动态阈值（基于特征统计）
            high_energy_threshold = 0.05  # 挥手需要较高能量
            low_energy_threshold = 0.02   # 靠近/注视的中等能量
            high_diff_threshold = 0.015   # 挥手需要高变化
            low_diff_threshold = 0.008    # 注视的中等变化
            
            # 分类逻辑
            if feature_energy > high_energy_threshold and feature_diff > high_diff_threshold:
                intent_type = "WAVING"  # 挥手
                intent_score = min(1.0, 0.75 + (feature_energy + feature_diff) * 0.25)
            elif feature_energy > low_energy_threshold and feature_diff <= high_diff_threshold:
                intent_type = "APPROACHING"  # 靠近
                intent_score = min(1.0, 0.7 + feature_energy * 0.3)
            elif feature_energy <= low_energy_threshold and feature_diff > low_diff_threshold:
                intent_type = "GAZING"  # 注视
                intent_score = min(1.0, 0.7 + feature_diff * 0.3)
            else:
                intent_type = "STILL"  # 静止
                intent_score = max(0.7, 1.0 - (feature_energy + feature_diff) * 0.3)
            
            intent_details = {
                'feature_energy': float(feature_energy),
                'feature_diff': float(feature_diff),
                'method': 'rule_based'
            }
            
            # 每次推理都打印详细日志（调试用）
            logger.info(f"🧠 [Action] 意图分类：{intent_type} (score={intent_score:.2f}, energy={feature_energy:.4f}, diff={feature_diff:.4f}, frames={self.frame_count})")
            
            return intent_score, intent_type, intent_details
            
        except Exception as e:
            logger.error(f"❌ [_classify_intent] 错误：{e}", exc_info=True)
            return 0.0, "ERROR", {}
            
            # 详细特征
            intent_details = {
                "all_scores": dict(zip(self.intent_types, probs_np)),
                "predicted_idx": predicted_idx,
            }
            
            logger.info(f"🧠 [Action] 意图分类：{intent_type} (置信度：{intent_score:.2f})")
            
            return intent_score, intent_type, intent_details
            
        except Exception as e:
            logger.error(f"❌ [_classify_intent] 错误：{e}")
            return 0.0, "ERROR", {}
    
    def _simulate_intent_classification(self, features: np.ndarray) -> Tuple[float, str, Dict[str, Any]]:
        """
        模拟意图分类（降级方案）
        
        Args:
            features: 融合特征
        
        Returns:
            (意图分数，意图类型，详细特征字典)
        """
        # 提取关键特征
        motion_magnitude = features[0] if len(features) > 0 else 0.0
        motion_consistency = features[1] if len(features) > 1 else 0.0
        
        # 简单规则分类
        intent_scores = {}
        
        # 挥手检测
        if motion_magnitude > 5.0:
            intent_scores["WAVING"] = 0.85
        else:
            intent_scores["WAVING"] = 0.2
        
        # 靠近检测
        if motion_magnitude > 2.0 and motion_consistency > 0.7:
            intent_scores["APPROACHING"] = 0.65
        else:
            intent_scores["APPROACHING"] = 0.3
        
        # 注视检测
        if motion_magnitude < 2.0 and motion_consistency > 0.5:
            intent_scores["GAZING"] = 0.55
        else:
            intent_scores["GAZING"] = 0.2
        
        # 静止检测
        if motion_magnitude < 1.0:
            intent_scores["STILL"] = 0.9
        else:
            intent_scores["STILL"] = 0.1
        
        # 选择最高分
        intent_type = max(intent_scores, key=intent_scores.get)
        intent_score = intent_scores[intent_type]
        
        intent_details = {
            "motion_magnitude": float(motion_magnitude),
            "motion_consistency": float(motion_consistency),
            "all_scores": intent_scores,
        }
        
        return intent_score, intent_type, intent_details
    
    def _extract_simple_features(self, frame: np.ndarray) -> np.ndarray:
        """提取简单特征 (颜色直方图 + 边缘)"""
        try:
            # 转为灰度图
            gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Canny 边缘检测
            edges = cv2.Canny(gray, 100, 200)
            
            # 特征向量
            features = np.zeros(512, dtype=np.float32)
            features[0] = np.mean(edges) / 255.0  # 边缘密度
            features[1] = np.std(edges) / 255.0  # 边缘变化
            
            return features
            
        except Exception as e:
            logger.error(f"❌ [_extract_simple_features] 错误：{e}")
            return np.zeros(512, dtype=np.float32)
    
    def _compute_motion_magnitude(self, frames: List[np.ndarray]) -> float:
        """计算运动幅值"""
        try:
            if len(frames) < 2:
                return 0.0
            
            # 计算帧间差分
            motion_sum = 0.0
            for i in range(1, len(frames)):
                diff = np.abs(frames[i] - frames[i-1])
                motion_sum += np.mean(diff)
            
            return motion_sum / (len(frames) - 1)
            
        except Exception as e:
            logger.error(f"❌ [_compute_motion_magnitude] 错误：{e}")
            return 0.0
    
    def _compute_motion_consistency(self, frames: List[np.ndarray]) -> float:
        """计算运动一致性"""
        try:
            if len(frames) < 3:
                return 0.0
            
            # 计算光流方向一致性
            consistency_scores = []
            
            for i in range(2, len(frames)):
                # 简化：使用帧差方向
                diff1 = frames[i-1] - frames[i-2]
                diff2 = frames[i] - frames[i-1]
                
                # 计算相关性
                corr = np.corrcoef(diff1.flatten(), diff2.flatten())[0, 1]
                consistency_scores.append(corr)
            
            return float(np.mean(consistency_scores))
            
        except Exception as e:
            logger.error(f"❌ [_compute_motion_consistency] 错误：{e}")
            return 0.0
    
    def _extract_motion_patterns(self, frames: List[np.ndarray]) -> np.ndarray:
        """提取运动模式特征"""
        try:
            patterns = np.zeros(8, dtype=np.float32)
            
            # 计算运动轨迹
            motion_trajectory = []
            for i in range(1, len(frames)):
                diff = np.mean(frames[i] - frames[i-1], axis=(0, 1))
                motion_trajectory.append(diff)
            
            # 特征 1-3: X/Y/Z 方向运动
            if motion_trajectory:
                trajectory = np.array(motion_trajectory)
                patterns[0] = np.mean(np.abs(trajectory[:, 0]))  # X 方向
                patterns[1] = np.mean(np.abs(trajectory[:, 1]))  # Y 方向
                patterns[2] = np.mean(np.abs(trajectory[:, 2]))  # Z 方向 (亮度变化)
                
                # 特征 4-8: 运动模式 (挥手/靠近/注视)
                patterns[3] = np.std(trajectory[:, 1])  # Y 方向变化 (挥手特征)
                patterns[4] = np.mean(trajectory[:, 0])  # X 方向趋势 (靠近特征)
                patterns[5] = np.max(np.abs(trajectory))  # 最大运动
                patterns[6] = np.mean(np.abs(trajectory))  # 平均运动
                patterns[7] = len([t for t in trajectory if np.abs(t[1]) > 0.01])  # 垂直运动次数
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ [_extract_motion_patterns] 错误：{e}")
            return np.zeros(8, dtype=np.float32)
    
    def reset_buffers(self):
        """重置缓冲区"""
        self.slow_buffer.clear()
        self.fast_buffer.clear()
        self.frame_count = 0
        logger.debug("🧠 [Action] 缓冲区已重置")
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """获取缓冲区状态"""
        return {
            'slow_buffer_size': len(self.slow_buffer),
            'fast_buffer_size': len(self.fast_buffer),
            'frame_count': self.frame_count,
            'ready_for_classification': len(self.slow_buffer) >= self._config.get('num_frames_slow', 8),
        }
