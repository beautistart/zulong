# File: zulong/l1a/audio_processor_adapter.py
# 听觉处理器适配器 (TSD v2.5 共享池架构)
# 功能：桥接 AudioPreprocessor 与 SharedMemoryPool

import asyncio
import time
import numpy as np
from typing import Optional, Dict, Any
import logging

from zulong.infrastructure.shared_memory_pool import shared_memory_pool, ZoneType, DataType
from zulong.infrastructure.data_ingestion import data_ingestion
from zulong.l1a.audio_preprocessor import AudioPreprocessor, AudioFeatures
from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

logger = logging.getLogger(__name__)


class AudioProcessorAdapter:
    """
    听觉处理器适配器 (TSD v2.5)
    
    功能:
    - 监听 DataIngestion 的音频入池事件
    - 从共享池读取原始音频数据
    - 调用 AudioPreprocessor 进行处理 (降噪、特征提取、VAD)
    - 将处理结果写回共享池 (Feature Zone)
    
    数据流向:
    MicrophoneDevice → DataIngestion → Raw Zone
    → (事件触发) → AudioProcessorAdapter
    → AudioPreprocessor.process()
    → Feature Zone (audio_features, vad_result)
    → L1-B 读取
    """
    
    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.event_bus = event_bus
        self.pool = shared_memory_pool
        
        # 订阅音频入池事件
        self._setup_event_handlers()
        
        logger.info("✅ [AudioProcessorAdapter] 初始化完成")
    
    def _setup_event_handlers(self):
        """设置事件监听"""
        # 监听 SENSOR_AUDIO 事件 (由 DataIngestion 发布)
        self.event_bus.subscribe(
            EventType.SENSOR_AUDIO,
            self._on_audio_ingested,
            subscriber="AudioProcessorAdapter"
        )
        logger.debug("📡 [AudioProcessorAdapter] 已订阅 SENSOR_AUDIO 事件")
    
    async def _on_audio_ingested(self, event: ZulongEvent):
        """
        音频入池事件处理
        
        Args:
            event: SENSOR_AUDIO 事件
        """
        try:
            trace_id = event.payload.get("trace_id")
            if not trace_id:
                logger.warning(f"⚠️ [AudioProcessorAdapter] 事件缺少 trace_id: {event.payload}")
                return
            
            # 从 Raw Zone 读取原始音频
            envelope = self.pool.read_raw(trace_id)
            if not envelope:
                logger.warning(f"⚠️ [AudioProcessorAdapter] 未找到原始音频：{trace_id[:15]}")
                return
            
            audio_data = envelope.payload
            sample_rate = envelope.metadata.get("sample_rate", 16000)
            timestamp = envelope.timestamp
            
            logger.debug(f"📖 [AudioProcessorAdapter] 读取原始音频：{trace_id[:15]} ({sample_rate}Hz, {len(audio_data)}样本)")
            
            # 调用 AudioPreprocessor 进行处理
            features = await self._process_audio(audio_data, sample_rate)
            
            # 将处理结果写回 Feature Zone
            feature_trace_id = f"feature_{trace_id[6:]}"  # trace_xxx → feature_xxx
            self.pool.write_feature(
                key=feature_trace_id,
                data=features.to_dict(),
                data_type=DataType.AUDIO_FEATURE,
                parent_trace_id=trace_id
            )
            
            logger.debug(f"💾 [AudioProcessorAdapter] 处理结果写入 Feature Zone: {feature_trace_id[:15]}")
            
            # 如果检测到语音，发布语音事件
            if features.is_speech:
                speech_event = ZulongEvent(
                    type=EventType.SENSOR_AUDIO,
                    priority=EventPriority.NORMAL,
                    payload={
                        "trace_id": trace_id,
                        "feature_trace_id": feature_trace_id,
                        "is_speech": True,
                        "speech_probability": features.speech_probability
                    },
                    source="AudioProcessorAdapter"
                )
                self.event_bus.publish(speech_event)
                logger.info(f"🎤 [AudioProcessorAdapter] 检测到语音：{features.speech_probability:.2%}")
        
        except Exception as e:
            logger.error(f"❌ [AudioProcessorAdapter] 处理音频失败：{e}", exc_info=True)
    
    async def _process_audio(self, audio_data: np.ndarray, sample_rate: int) -> AudioFeatures:
        """
        处理音频数据
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
        
        Returns:
            AudioFeatures: 音频特征
        """
        # 1. 降噪
        if self.preprocessor.is_calibrated:
            audio_denoised = self.preprocessor.denoise(audio_data)
        else:
            audio_denoised = audio_data
        
        # 2. 提取特征
        features_dict = self.preprocessor.extract_features(audio_denoised, sample_rate)
        
        # 3. VAD 检测
        is_speech, speech_prob = self.preprocessor.vad(features_dict)
        
        # 4. 构建 AudioFeatures 对象
        features = AudioFeatures(
            audio_data=audio_denoised,
            sample_rate=sample_rate,
            rms_energy=features_dict["rms_energy"],
            zero_crossing_rate=features_dict["zero_crossing_rate"],
            mfcc=features_dict["mfcc"],
            spectral_centroid=features_dict["spectral_centroid"],
            spectral_rolloff=features_dict["spectral_rolloff"],
            is_speech=is_speech,
            speech_probability=speech_prob,
            timestamp=time.time(),
            duration=len(audio_data) / sample_rate
        )
        
        logger.debug(
            f"🎵 [AudioProcessorAdapter] 特征提取完成："
            f"RMS={features.rms_energy:.3f}, ZCR={features.zero_crossing_rate:.3f}, "
            f"语音={is_speech} ({speech_prob:.2%})"
        )
        
        return features
    
    async def calibrate(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        校准环境噪音轮廓
        
        Args:
            audio_data: 静音片段音频数据
            sample_rate: 采样率
        """
        await self.preprocessor.calibrate_noise_profile(audio_data, sample_rate)
        logger.info("✅ [AudioProcessorAdapter] 噪音轮廓校准完成")
    
    def shutdown(self):
        """关闭适配器"""
        logger.info("🛑 [AudioProcessorAdapter] 已关闭")


# 全局单例
audio_processor_adapter = AudioProcessorAdapter()
