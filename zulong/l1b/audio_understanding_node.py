#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙 (ZULONG) 系统 - L1-B 音频理解模块

文件：zulong/l1b/audio_understanding_node.py

功能:
- 接收 L1-A 的音频预处理特征
- 使用 L2 模型（Qwen2.5-0.5B-Instruct）理解音频
- 输出结构化音频理解结果
- 支持上下文打包和历史回溯

TSD v1.7 对应:
- 2.2.2 L1-B - 调度与意图守门层
- 4.2 上下文打包 - 视听信息流回溯
- 4.4 感知预处理 - 语言模型处理

架构说明:
- L1-A: 音频预处理（降噪、特征提取、VAD）+ VL 模型视觉处理
- L1-B: 音频理解（L2 模型、结构化输出）+ 上下文打包
- L2: 对话管理（决策、响应生成）

模型使用:
- L1-A: Qwen2.5-0.5B-VL (视觉理解)
- L1-B: Qwen2.5-0.5B-Instruct (音频理解)
- L2: Qwen2.5-0.5B-Instruct (推理决策)
- L3-TTS: CosyVoice3-0.5B (语音合成)
"""

import asyncio
import logging
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

logger = logging.getLogger(__name__)


class L1BAudioUnderstandingNode:
    """
    L1-B 音频理解节点
    
    功能:
    - 使用 L2 模型（Qwen2.5-0.5B-Instruct）理解音频特征
    - 生成结构化音频理解结果
    - 支持上下文打包
    """
    
    # 模型配置（使用 L2 模型）
    MODEL_PATH = r"d:\AI\project\zulong_beta4\models\Intel\Qwen3___5-0___8B-int4-AutoRound"  # int4 量化模型
    MAX_LENGTH = 200  # 最大生成长度
    
    def __init__(self):
        """初始化 L1-B 音频理解节点"""
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.is_loaded = False
        # L1-B 运行在 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 上下文缓冲区（过去 30 秒）
        self.context_buffer: List[Dict[str, Any]] = []
        self.buffer_max_size = 30  # 最多保留 30 个样本（约 30 秒）
        
        logger.info(f"🧠 L1-B 音频理解节点初始化完成（设备：{self.device}）")
    
    async def initialize(self) -> bool:
        """
        加载 Qwen3.5-0.8B-int4-L1B 模型（无量化模式）
        
        Returns:
            bool: 加载是否成功
        """
        if self.is_loaded:
            logger.info("✅ 模型已加载，跳过初始化")
            return True
        
        try:
            model_path = Path(self.MODEL_PATH)
            
            logger.info(f"📥 正在加载 Qwen3.5-0.8B-int4-L1B 模型：{model_path}")
            logger.info(f"   - 量化：无量化模式")
            logger.info(f"   - 设备：{self.device}")
            
            # 加载 tokenizer
            logger.info("📥 加载 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            # 加载模型（int4 量化）
            logger.info("📥 加载模型（int4 量化）...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                trust_remote_code=True
            )
            
            self.is_loaded = True
            logger.info("✅ Qwen3.5-0.8B-int4 模型加载完成")
            logger.info("⚠️ 注意：int4 量化模型占用约 500MB 显存")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Qwen3.5-0.8B-int4-L1B 模型加载失败：{e}")
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
    
    def build_prompt(self, audio_features: Dict[str, Any], context: Optional[str] = None) -> str:
        """
        构建音频理解提示词
        
        Args:
            audio_features: 音频特征字典
            context: 上下文信息（可选）
        
        Returns:
            str: 提示词
        """
        # 提取特征
        rms = audio_features.get("rms_energy", 0)
        zcr = audio_features.get("zero_crossing_rate", 0)
        centroid = audio_features.get("spectral_centroid", 0)
        is_speech = audio_features.get("is_speech", False)
        confidence = audio_features.get("speech_probability", 0)
        
        # 构建提示词
        prompt = f"""你是一个音频分析专家。根据以下音频特征，理解并描述这个声音：

音频特征:
- 能量 (RMS): {rms:.4f}
- 过零率：{zcr:.4f}
- 频谱质心：{centroid:.1f} Hz
- 语音检测：{'是' if is_speech else '否'}
- 置信度：{confidence:.0%}

"""
        
        if context:
            prompt += f"上下文信息：{context}\n\n"
        
        prompt += """请用简洁的语言描述：
1. 声音类型（语音/非语音/环境音）
2. 如果是语音，推测可能的内容或意图
3. 如果是非语音，描述声音特征

回答格式：
类型：[声音类型]
描述：[具体描述]
"""
        
        return prompt
    
    async def process(self, audio_features: Dict[str, Any], include_context: bool = True) -> Dict[str, Any]:
        """
        处理音频特征，生成结构化理解结果
        
        Args:
            audio_features: 音频特征字典
            include_context: 是否包含上下文
        
        Returns:
            dict: 结构化理解结果
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
            # 获取上下文（如果需要）
            context = None
            if include_context and self.context_buffer:
                # 取最近 3 个样本作为上下文
                recent_context = self.context_buffer[-3:]
                context = f"过去 {len(recent_context)} 秒检测到 {sum(1 for c in recent_context if c.get('is_speech', False))} 次语音"
            
            # 构建提示词
            prompt = self.build_prompt(audio_features, context)
            
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
            
            # 更新上下文缓冲区
            self._update_context(audio_features)
            
            # 结构化输出
            result = {
                "success": True,
                "type": "speech" if audio_features.get("is_speech", False) else "non_speech",
                "text": response,
                "timestamp": asyncio.get_event_loop().time(),
                "confidence": audio_features.get("speech_probability", 0),
                "context_included": include_context
            }
            
            logger.info(f"📝 L1-B 音频理解完成：{result['type']} - {result['text'][:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ L1-B 音频理解失败：{e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "timestamp": asyncio.get_event_loop().time()
            }
    
    def _update_context(self, audio_features: Dict[str, Any]):
        """
        更新上下文缓冲区
        
        Args:
            audio_features: 音频特征
        """
        # 添加到缓冲区
        self.context_buffer.append(audio_features.copy())
        
        # 限制缓冲区大小
        if len(self.context_buffer) > self.buffer_max_size:
            self.context_buffer.pop(0)
    
    def clear_context(self):
        """清空上下文缓冲区"""
        self.context_buffer.clear()
        logger.info("🗑️ 上下文缓冲区已清空")
    
    async def process_batch(self, audio_features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理音频特征
        
        Args:
            audio_features_list: 音频特征列表
        
        Returns:
            list: 结构化理解结果列表
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
            source="l1b_audio_understanding",
            payload=result,
            priority=EventPriority.NORMAL
        )
        
        await event_bus.publish(event)
        logger.debug(f"📨 发布 L1-B 音频理解结果：{event.id}")


# 全局 L1-B 音频理解节点实例
l1b_audio_understanding = L1BAudioUnderstandingNode()


# 测试函数
async def test_l1b_audio():
    """测试 L1-B 音频理解节点"""
    print("="*60)
    print("🧠 祖龙系统 - L1-B 音频理解节点测试")
    print("="*60)
    
    node = L1BAudioUnderstandingNode()
    
    # 加载模型
    print("\n📥 加载 VL 模型（FP16 无量化）...")
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
    
    result = await node.process(speech_features, include_context=True)
    print(f"   类型：{result['type']}")
    print(f"   文本：{result['text'][:100]}...")
    print(f"   置信度：{result['confidence']:.2f}")
    print(f"   包含上下文：{result['context_included']}")
    
    # 测试 2: 非语音特征
    print("\n🔊 测试 2: 非语音特征处理...")
    noise_features = {
        "rms_energy": 0.3,
        "zero_crossing_rate": 0.3,
        "spectral_centroid": 2000.0,
        "is_speech": False,
        "speech_probability": 0.2
    }
    
    result = await node.process(noise_features, include_context=True)
    print(f"   类型：{result['type']}")
    print(f"   文本：{result['text'][:100]}...")
    print(f"   置信度：{result['confidence']:.2f}")
    
    # 卸载模型
    print("\n📤 卸载模型...")
    await node.unload()
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    asyncio.run(test_l1b_audio())
