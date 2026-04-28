#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙 (ZULONG) 系统 - 音频理解节点（基于规则）

文件：zulong/l1a/reflex/audio_understanding_node.py

功能:
- 接收音频预处理模块的特征
- 使用规则模板生成结构化文本
- 输出：{"type": "speech/non_speech", "text": "...", "timestamp": ...}

TSD v1.7 对应:
- 2.2.2 L1-A - 受控反射
- 4.4 感知预处理 - 结构化输出

优势:
- 无需加载大模型，延迟极低（<10ms）
- 适合实时音频处理
- 可解释性强
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import time

from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

logger = logging.getLogger(__name__)


class AudioUnderstandingNode:
    """
    音频理解节点（基于规则）
    
    功能:
    - 根据音频特征生成结构化文本
    - 使用规则模板，无需大模型
    - 超低延迟（<10ms）
    """
    
    def __init__(self):
        """初始化音频理解节点"""
        self.is_initialized = False
        
        # 规则模板
        self.templates = {
            "speech": [
                "检测到语音，能量 {rms:.2f}，音调 {pitch}Hz",
                "用户正在说话，置信度 {confidence:.0%}",
                "语音内容：能量 {rms:.2f}，频谱质心 {centroid:.0f}Hz"
            ],
            "non_speech": [
                "检测到非语音声音：{sound_type}，能量 {rms:.2f}",
                "环境声音：{sound_type}，频谱质心 {centroid:.0f}Hz",
                "背景噪音：能量 {rms:.2f}，过零率 {zcr:.2f}"
            ]
        }
        
        logger.info("🎵 音频理解节点初始化完成（基于规则）")
    
    async def initialize(self) -> bool:
        """初始化节点"""
        self.is_initialized = True
        logger.info("✅ 音频理解节点已初始化")
        return True
    
    def classify_sound(self, features: Dict[str, Any]) -> str:
        """
        分类声音类型
        
        Args:
            features: 音频特征
        
        Returns:
            str: 声音类型
        """
        if features.get("is_speech", False):
            return "speech"
        
        # 根据特征判断声音类型
        zcr = features.get("zero_crossing_rate", 0)
        centroid = features.get("spectral_centroid", 0)
        
        if zcr > 0.3:
            return "high_frequency_noise"  # 高频噪音
        elif centroid > 2000:
            return "background_noise"  # 背景噪音
        else:
            return "low_frequency_sound"  # 低频声音
    
    def generate_description(self, features: Dict[str, Any]) -> str:
        """
        生成声音描述
        
        Args:
            features: 音频特征
        
        Returns:
            str: 描述文本
        """
        sound_type = self.classify_sound(features)
        
        # 提取特征
        rms = features.get("rms_energy", 0)
        zcr = features.get("zero_crossing_rate", 0)
        centroid = features.get("spectral_centroid", 0)
        confidence = features.get("speech_probability", 0)
        
        # 估算音调（基于频谱质心）
        pitch = int(centroid / 2)
        
        if sound_type == "speech":
            template = self.templates["speech"][0]
            return template.format(rms=rms, pitch=pitch, confidence=confidence)
        
        elif sound_type == "high_frequency_noise":
            template = self.templates["non_speech"][0]
            return template.format(rms=rms, sound_type="高频噪音", centroid=centroid)
        
        elif sound_type == "background_noise":
            template = self.templates["non_speech"][1]
            return template.format(rms=rms, sound_type="背景噪音", centroid=centroid)
        
        else:
            template = self.templates["non_speech"][2]
            return template.format(rms=rms, zcr=zcr)
    
    async def process(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理音频特征，生成结构化文本
        
        Args:
            audio_features: 音频特征字典
        
        Returns:
            dict: 结构化文本结果
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # 分类声音类型
            sound_type = self.classify_sound(audio_features)
            
            # 生成描述
            description = self.generate_description(audio_features)
            
            # 构建结果
            result = {
                "success": True,
                "type": sound_type,
                "text": description,
                "timestamp": asyncio.get_event_loop().time(),
                "confidence": audio_features.get("speech_probability", 0),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            logger.debug(f"📝 音频理解完成：{result['type']} - {result['text']} "
                        f"({result['processing_time_ms']:.1f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 音频理解失败：{e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def process_batch(self, audio_features_list: list) -> list:
        """
        批量处理音频特征
        
        Args:
            audio_features_list: 音频特征列表
        
        Returns:
            list: 结构化文本结果列表
        """
        results = []
        
        for features in audio_features_list:
            result = await self.process(features)
            results.append(result)
        
        return results
    
    async def publish_result(self, result: Dict[str, Any]):
        """
        发布处理结果到 EventBus
        
        Args:
            result: 处理结果
        """
        event = ZulongEvent(
            type=EventType.L2_OUTPUT,
            source="audio_understanding_node",
            payload=result,
            priority=EventPriority.NORMAL
        )
        
        await event_bus.publish(event)
        logger.debug(f"📨 发布音频理解结果：{event.id}")


# 全局音频理解节点实例
audio_understanding_node = AudioUnderstandingNode()


# 测试函数
async def test_audio_understanding_node():
    """测试音频理解节点"""
    print("="*60)
    print("🧠 祖龙系统 - 音频理解节点测试（基于规则）")
    print("="*60)
    
    node = AudioUnderstandingNode()
    
    # 初始化
    print("\n📥 初始化节点...")
    await node.initialize()
    
    # 测试 1: 语音特征
    print("\n🎤 测试 1: 语音特征处理...")
    speech_features = {
        "rms_energy": 0.7,
        "zero_crossing_rate": 0.05,
        "spectral_centroid": 450.0,
        "is_speech": True,
        "speech_probability": 0.9
    }
    
    result = await node.process(speech_features)
    print(f"   类型：{result['type']}")
    print(f"   文本：{result['text']}")
    print(f"   置信度：{result['confidence']:.2f}")
    print(f"   处理时间：{result['processing_time_ms']:.1f}ms")
    
    # 测试 2: 非语音特征（高频噪音）
    print("\n🔊 测试 2: 高频噪音处理...")
    noise_features = {
        "rms_energy": 0.3,
        "zero_crossing_rate": 0.4,
        "spectral_centroid": 3000.0,
        "is_speech": False,
        "speech_probability": 0.1
    }
    
    result = await node.process(noise_features)
    print(f"   类型：{result['type']}")
    print(f"   文本：{result['text']}")
    print(f"   置信度：{result['confidence']:.2f}")
    print(f"   处理时间：{result['processing_time_ms']:.1f}ms")
    
    # 测试 3: 非语音特征（背景噪音）
    print("\n🔊 测试 3: 背景噪音处理...")
    bg_noise = {
        "rms_energy": 0.2,
        "zero_crossing_rate": 0.2,
        "spectral_centroid": 2500.0,
        "is_speech": False,
        "speech_probability": 0.15
    }
    
    result = await node.process(bg_noise)
    print(f"   类型：{result['type']}")
    print(f"   文本：{result['text']}")
    print(f"   置信度：{result['confidence']:.2f}")
    print(f"   处理时间：{result['processing_time_ms']:.1f}ms")
    
    # 测试 4: 批量处理
    print("\n📦 测试 4: 批量处理（10 个样本）...")
    features_list = [speech_features, noise_features, bg_noise] * 3 + [speech_features]
    
    results = await node.process_batch(features_list)
    avg_time = sum(r['processing_time_ms'] for r in results) / len(results)
    print(f"   处理完成：{len(results)} 个结果")
    print(f"   平均处理时间：{avg_time:.1f}ms")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    asyncio.run(test_audio_understanding_node())
