#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙 (ZULONG) 系统 - VL 模型音频处理节点

文件：zulong/l1a/reflex/vl_audio_node.py

功能:
- 使用 Qwen2.5-0.5B 语言模型处理音频特征
- 音频特征 → 结构化文本
- 输出：{"type": "speech", "text": "...", "timestamp": ...}

TSD v1.7 对应:
- 2.2.2 L1-A - 受控反射
- 4.4 感知预处理 - VL 模型处理
- 5.2 显存约束 - 4bit 量化加载

注意：
- 音频特征已经是结构化数据，不需要真正的 VL 模型
- 使用 Qwen2.5-0.5B-Instruct 语言模型即可
"""

import asyncio
import logging
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

logger = logging.getLogger(__name__)


class VLAudioNode:
    """
    VL 模型音频处理节点
    
    功能:
    - 加载 Qwen3.5-0.8B-Base 模型（4bit 量化）
    - 将音频特征转换为结构化文本
    - 支持动态卸载/加载（显存管理）
    
    使用示例:
    ```python
    node = VLAudioNode()
    await node.initialize()
    
    # 处理音频特征
    result = await node.process(audio_features)
    
    # 输出结构化文本
    print(result["text"])
    ```
    """
    
    # 模型配置
    MODEL_PATH = r"d:\AI\project\zulong_beta4\models\Qwen\Qwen3___5-0___8B-Base"  # 使用 Qwen3.5-0.8B-Base 模型 (4bit 量化加载)
    MAX_LENGTH = 512  # 最大生成长度
    
    def __init__(self):
        """初始化 VL 音频节点"""
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.is_loaded = False
        # L1-A 运行在 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🧠 VL 音频节点初始化完成（设备：{self.device}）")
    
    async def initialize(self) -> bool:
        """
        加载 Intel_Qwen3.5-0.8B-int4-AutoRound 模型（AutoRound 4bit 量化）
        
        Returns:
            bool: 加载是否成功
        """
        if self.is_loaded:
            logger.info("✅ 模型已加载，跳过初始化")
            return True
        
        try:
            logger.info(f"📥 正在加载 Intel_Qwen3.5-0.8B-int4-AutoRound 模型：{self.MODEL_PATH}")
            logger.info(f"   - 量化：Intel AutoRound 4bit")
            logger.info(f"   - 设备：{self.device}")
            
            # 加载 tokenizer
            logger.info("📥 加载 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_PATH,
                trust_remote_code=True
            )
            
            # 加载模型（AutoRound 4bit 量化）
            logger.info("📥 加载模型（AutoRound 4bit 量化）...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_PATH,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16  # AutoRound 量化模型使用 float16 加载
            )
            
            self.is_loaded = True
            logger.info("✅ Intel_Qwen3.5-0.8B-int4-AutoRound 模型加载完成（AutoRound 4bit 量化）")
            logger.info("⚠️ 注意：AutoRound 4bit 量化模型占用约 0.8-1.0GB 显存")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败：{e}")
            return False
    
    async def unload(self):
        """卸载模型（释放显存）"""
        if self.model:
            logger.info("📤 正在卸载 VL 模型...")
            del self.model
            self.model = None
            self.is_loaded = False
            
            # 清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ VL 模型已卸载")
    
    def build_prompt(self, audio_features: Dict[str, Any]) -> str:
        """
        构建音频理解提示词
        
        Args:
            audio_features: 音频特征字典
        
        Returns:
            str: 提示词
        """
        # 提取特征
        rms = audio_features.get("rms_energy", 0)
        zcr = audio_features.get("zero_crossing_rate", 0)
        spectral_centroid = audio_features.get("spectral_centroid", 0)
        is_speech = audio_features.get("is_speech", False)
        
        # 构建描述
        prompt = f"""你是一个音频分析专家。根据以下音频特征，描述你听到的声音：

音频特征:
- 能量 (RMS): {rms:.4f}
- 过零率：{zcr:.4f}
- 频谱质心：{spectral_centroid:.1f} Hz
- 语音检测：{'是' if is_speech else '否'}

请用简洁的语言描述这个声音，包括:
1. 是否是语音
2. 如果是语音，推测内容
3. 如果是非语音，描述声音类型

回答格式:
类型：[语音/非语音]
描述：[具体描述]
"""
        return prompt
    
    async def process(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理音频特征，生成结构化文本
        
        Args:
            audio_features: 音频特征字典
        
        Returns:
            dict: 结构化文本结果
        """
        if not self.is_loaded:
            if not await self.initialize():
                return {
                    "success": False,
                    "error": "模型未加载",
                    "text": "",
                    "timestamp": asyncio.get_event_loop().time()
                }
        
        try:
            # 构建提示词
            prompt = self.build_prompt(audio_features)
            
            # 分词
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.MAX_LENGTH,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # 解码
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回答部分
            response = generated_text[len(prompt):].strip()
            
            # 结构化输出
            result = {
                "success": True,
                "type": "speech" if audio_features.get("is_speech", False) else "non_speech",
                "text": response,
                "timestamp": asyncio.get_event_loop().time(),
                "confidence": audio_features.get("speech_probability", 0)
            }
            
            logger.info(f"📝 VL 音频理解完成：{result['type']} - {result['text'][:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ VL 音频处理失败：{e}")
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
            source="vl_audio_node",
            payload=result,
            priority=EventPriority.NORMAL
        )
        
        await event_bus.publish(event)
        logger.debug(f"📨 发布 VL 音频处理结果：{event.id}")


# 全局 VL 音频节点实例
vl_audio_node = VLAudioNode()


# 测试函数
async def test_vl_audio_node():
    """测试 VL 音频节点"""
    print("="*60)
    print("🧠 祖龙系统 - VL 音频处理节点测试")
    print("="*60)
    
    node = VLAudioNode()
    
    # 初始化
    print("\n📥 加载 VL 模型（4bit 量化）...")
    success = await node.initialize()
    
    if not success:
        print("❌ 模型加载失败")
        return
    
    print("✅ 模型加载成功")
    
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
    print(f"   文本：{result['text'][:100]}")
    print(f"   置信度：{result['confidence']:.2f}")
    
    # 测试 2: 非语音特征
    print("\n🔊 测试 2: 非语音特征处理...")
    noise_features = {
        "rms_energy": 0.3,
        "zero_crossing_rate": 0.3,
        "spectral_centroid": 2000.0,
        "is_speech": False,
        "speech_probability": 0.2
    }
    
    result = await node.process(noise_features)
    print(f"   类型：{result['type']}")
    print(f"   文本：{result['text'][:100]}")
    print(f"   置信度：{result['confidence']:.2f}")
    
    # 卸载模型
    print("\n📤 卸载模型...")
    await node.unload()
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    asyncio.run(test_vl_audio_node())
