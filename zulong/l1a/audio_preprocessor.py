#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙 (ZULONG) 系统 - 音频预处理模块

文件：zulong/l1a/audio_preprocessor.py

功能:
- 环境噪音过滤（librosa）
- 声音特征提取（MFCC、频谱、过零率）
- VAD（语音活动检测）增强
- 输出结构化音频特征给 VL 模型

TSD v1.7 对应:
- 4.4 感知预处理 - VAD (语音活动检测)
- 2.2.2 L1-A - SENSOR_* 事件处理
- 4.3 RAG 与专家模块 - 特征提取
"""

import asyncio
import logging
import numpy as np
import librosa
from typing import Dict, Any, Optional
from dataclasses import dataclass

from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """音频特征数据结构"""
    
    # 原始音频
    audio_data: np.ndarray  # 归一化音频数据（-1.0 ~ 1.0）
    sample_rate: int  # 采样率
    
    # 时域特征
    rms_energy: float  # 均方根能量
    zero_crossing_rate: float  # 过零率
    
    # 频域特征
    mfcc: np.ndarray  # MFCC 特征（13 维）
    spectral_centroid: float  # 频谱质心
    spectral_rolloff: float  # 频谱滚降点
    
    # VAD 特征
    is_speech: bool  # 是否语音
    speech_probability: float  # 语音概率
    
    # 元数据
    timestamp: float  # 时间戳
    duration: float  # 时长（秒）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "audio_data": self.audio_data,
            "sample_rate": self.sample_rate,
            "rms_energy": self.rms_energy,
            "zero_crossing_rate": self.zero_crossing_rate,
            "mfcc": self.mfcc,
            "spectral_centroid": self.spectral_centroid,
            "spectral_rolloff": self.spectral_rolloff,
            "is_speech": self.is_speech,
            "speech_probability": self.speech_probability,
            "timestamp": self.timestamp,
            "duration": self.duration
        }


class AudioPreprocessor:
    """
    音频预处理器
    
    功能:
    - 环境噪音过滤（谱减法）
    - 声音特征提取（MFCC、频谱、过零率）
    - VAD（语音活动检测）
    - 输出结构化音频特征
    
    使用示例:
    ```python
    preprocessor = AudioPreprocessor()
    
    # 处理音频数据
    features = await preprocessor.process(audio_data, sample_rate=16000)
    
    # 批量处理
    features_batch = await preprocessor.process_batch(audio_chunks)
    ```
    """
    
    # VAD 参数配置
    VAD_ENERGY_THRESHOLD = 0.02  # 能量阈值
    VAD_ZCR_THRESHOLD = 0.1      # 过零率阈值
    VAD_SPECTRAL_THRESHOLD = 1000  # 频谱质心阈值
    
    # MFCC 参数
    N_MFCC = 13  # MFCC 维度
    N_FFT = 2048  # FFT 窗口大小
    HOP_LENGTH = 512  # 跳跃长度
    
    def __init__(self):
        """初始化音频预处理器"""
        self.noise_profile: Optional[np.ndarray] = None
        self.is_calibrated = False
        
        logger.info("🎵 音频预处理器初始化完成")
    
    async def calibrate_noise_profile(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        校准环境噪音轮廓（从静音片段学习）
        
        Args:
            audio_data: 静音片段音频数据
            sample_rate: 采样率
        """
        logger.info("🔇 正在校准环境噪音轮廓...")
        
        # 计算频谱
        D = librosa.stft(audio_data, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        self.noise_profile = np.mean(np.abs(D), axis=1)
        
        self.is_calibrated = True
        logger.info(f"✅ 噪音轮廓校准完成（谱线数：{len(self.noise_profile)}）")
    
    def denoise(self, audio_data: np.ndarray, alpha: float = 2.0) -> np.ndarray:
        """
        环境噪音过滤（谱减法）
        
        算法:
        1. 短时傅里叶变换（STFT）
        2. 谱减：|X|² - α|N|²
        3. 逆 STFT 还原
        
        Args:
            audio_data: 原始音频数据
            alpha: 降噪强度（默认 2.0）
        
        Returns:
            np.ndarray: 降噪后的音频
        """
        if not self.is_calibrated:
            logger.warning("⚠️ 噪音轮廓未校准，跳过降噪")
            return audio_data
        
        # STFT
        D = librosa.stft(audio_data, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # 谱减
        magnitude_denoised = np.maximum(magnitude**2 - alpha * self.noise_profile[:, None]**2, 0)
        magnitude_denoised = np.sqrt(magnitude_denoised)
        
        # 逆 STFT
        D_reconstructed = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(D_reconstructed, hop_length=self.HOP_LENGTH, length=len(audio_data))
        
        logger.debug(f"🔇 降噪完成，信噪比提升约 {alpha:.1f}dB")
        return audio_denoised
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        提取音频特征
        
        提取特征:
        - 时域：RMS 能量、过零率
        - 频域：MFCC、频谱质心、频谱滚降点
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
        
        Returns:
            dict: 特征字典
        """
        # 处理静音情况
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < 1e-10:  # 静音
            return {
                "rms_energy": 0.0,
                "zero_crossing_rate": 0.0,
                "mfcc": np.zeros(self.N_MFCC),
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0
            }
        
        # 归一化
        audio_normalized = audio_data / max_amplitude
        
        # 时域特征
        rms_energy = np.sqrt(np.mean(audio_normalized**2))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_normalized))
        
        # 频域特征
        mfcc = librosa.feature.mfcc(y=audio_normalized, sr=sample_rate, n_mfcc=self.N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)  # 平均 MFCC
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_normalized, sr=sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_normalized, sr=sample_rate))
        
        return {
            "rms_energy": float(rms_energy),
            "zero_crossing_rate": float(zero_crossing_rate),
            "mfcc": mfcc_mean,
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(spectral_rolloff)
        }
    
    def vad(self, features: Dict[str, Any]) -> tuple:
        """
        语音活动检测（VAD）
        
        算法:
        1. 能量阈值：RMS > threshold
        2. 过零率：ZCR 在人声范围（0.1-0.5）
        3. 频谱质心：人声频谱质心 < 1000Hz
        
        Args:
            features: 音频特征字典
        
        Returns:
            tuple: (is_speech, probability)
        """
        rms = features["rms_energy"]
        zcr = features["zero_crossing_rate"]
        spectral_centroid = features["spectral_centroid"]
        
        # 多特征融合决策
        score = 0.0
        
        # 能量检测
        if rms > self.VAD_ENERGY_THRESHOLD:
            score += 0.4
        
        # 过零率检测（人声范围 0.1-0.5）
        if 0.1 < zcr < 0.5:
            score += 0.3
        
        # 频谱质心检测（人声 < 1000Hz）
        if spectral_centroid < self.VAD_SPECTRAL_THRESHOLD:
            score += 0.3
        
        is_speech = score > 0.5
        probability = score
        
        return is_speech, probability
    
    async def process(self, audio_data: np.ndarray, sample_rate: int = 16000) -> AudioFeatures:
        """
        处理音频数据（完整流程）
        
        流程:
        1. 降噪
        2. 特征提取
        3. VAD 检测
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
        
        Returns:
            AudioFeatures: 音频特征对象
        """
        # 降噪
        audio_denoised = self.denoise(audio_data)
        
        # 特征提取
        features = self.extract_features(audio_denoised, sample_rate)
        
        # VAD 检测
        is_speech, speech_probability = self.vad(features)
        
        # 构建特征对象
        audio_features = AudioFeatures(
            audio_data=audio_denoised,
            sample_rate=sample_rate,
            rms_energy=features["rms_energy"],
            zero_crossing_rate=features["zero_crossing_rate"],
            mfcc=features["mfcc"],
            spectral_centroid=features["spectral_centroid"],
            spectral_rolloff=features["spectral_rolloff"],
            is_speech=is_speech,
            speech_probability=speech_probability,
            timestamp=asyncio.get_event_loop().time(),
            duration=len(audio_data) / sample_rate
        )
        
        logger.debug(f"🎵 音频特征提取完成：能量={features['rms_energy']:.4f}, "
                    f"VAD={is_speech} ({speech_probability:.2f})")
        
        return audio_features
    
    async def process_batch(self, audio_chunks: list, sample_rate: int = 16000) -> list:
        """
        批量处理音频块
        
        Args:
            audio_chunks: 音频块列表
            sample_rate: 采样率
        
        Returns:
            list: 特征列表
        """
        features_list = []
        
        for chunk in audio_chunks:
            features = await self.process(chunk, sample_rate)
            features_list.append(features)
        
        return features_list
    
    async def publish_features(self, features: AudioFeatures):
        """
        发布音频特征到 EventBus
        
        Args:
            features: 音频特征对象
        """
        event = ZulongEvent(
            type=EventType.SENSOR_SOUND,
            source="audio_preprocessor",
            payload=features.to_dict(),
            priority=EventPriority.NORMAL
        )
        
        await event_bus.publish(event)
        logger.debug(f"📨 发布音频特征事件：{event.id}")


# 全局预处理器实例
audio_preprocessor = AudioPreprocessor()


# 测试函数
async def test_audio_preprocessor():
    """测试音频预处理器"""
    print("="*60)
    print("🎵 祖龙系统 - 音频预处理器测试")
    print("="*60)
    
    preprocessor = AudioPreprocessor()
    
    # 生成测试音频（正弦波 + 噪音）
    print("\n🎵 生成测试音频（440Hz 正弦波 + 白噪音）...")
    sample_rate = 16000
    duration = 1.0
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 440Hz 正弦波
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    # 白噪音
    noise = 0.1 * np.random.randn(len(t))
    # 混合
    audio_data = signal + noise
    
    print(f"   音频长度：{len(audio_data)} 采样点")
    print(f"   时长：{duration}秒")
    print(f"   信噪比：约 14dB")
    
    # 校准噪音轮廓
    print("\n🔇 校准噪音轮廓（使用纯噪音片段）...")
    noise_only = 0.1 * np.random.randn(int(sample_rate * 0.5))
    await preprocessor.calibrate_noise_profile(noise_only, sample_rate)
    
    # 处理音频
    print("\n🎵 处理音频（降噪 + 特征提取 + VAD）...")
    features = await preprocessor.process(audio_data, sample_rate)
    
    # 打印特征
    print("\n📊 音频特征:")
    print(f"   - RMS 能量：{features.rms_energy:.4f}")
    print(f"   - 过零率：{features.zero_crossing_rate:.4f}")
    print(f"   - 频谱质心：{features.spectral_centroid:.1f} Hz")
    print(f"   - MFCC 维度：{features.mfcc.shape}")
    print(f"   - VAD 结果：{'语音' if features.is_speech else '非语音'} ({features.speech_probability:.2f})")
    print(f"   - 时长：{features.duration:.2f}秒")
    
    # 测试批量处理
    print("\n📦 批量处理测试（10 个音频块）...")
    audio_chunks = [audio_data[:1600] for _ in range(10)]  # 10 个 100ms 块
    features_list = await preprocessor.process_batch(audio_chunks, sample_rate)
    print(f"   处理完成：{len(features_list)} 个特征")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    asyncio.run(test_audio_preprocessor())
