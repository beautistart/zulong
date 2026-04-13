#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙 (ZULONG) 系统 - 扬声器驱动与音频播放模块

文件：zulong/l0/devices/speaker_device.py

功能:
- 扬声器设备管理（初始化、启动、停止）
- 音频流播放（支持 16kHz, 24kHz, 48kHz）
- TTS 音频输出（流式播放）
- 播放状态回调
- 发布 SPEAKER_PLAYING/SPEAKER_STOPPED 事件

TSD v1.7 对应:
- 2.2.4 L3: 专家技能池 - TTS 专家输出
- 4.3 RAG 与专家模块 - 音频输出接口
"""

import asyncio
import logging
from typing import Optional, Callable, Any
import pyaudio
import numpy as np

from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

logger = logging.getLogger(__name__)


class SpeakerDevice:
    """
    扬声器设备驱动类
    
    功能:
    - 封装 PyAudio，调用 Windows 音频驱动
    - 实现音频流播放
    - 支持 TTS 流式输出
    - 发布播放状态事件
    
    硬件要求:
    - Windows 系统已安装扬声器/耳机驱动
    - 音频输出设备可用
    
    使用示例:
    ```python
    speaker = SpeakerDevice()
    await speaker.start()
    
    # 播放音频数据（numpy 数组）
    await speaker.play_audio(audio_data, sample_rate=24000)
    
    # 流式播放（TTS）
    async for chunk in tts_stream:
        await speaker.play_chunk(chunk)
    
    await speaker.stop()
    ```
    """
    
    # 音频参数配置
    DEFAULT_SAMPLE_RATE = 24000    # 默认采样率 24kHz（TTS 常用）
    CHUNK_SIZE = 1024              # 每次播放的帧数
    SAMPLE_WIDTH = 2               # 16bit = 2 bytes
    CHANNELS = 1                   # 单声道
    
    def __init__(self, device_index: Optional[int] = None):
        """
        初始化扬声器设备
        
        Args:
            device_index: 扬声器设备索引，None 表示使用默认设备
        """
        self.device_index = device_index
        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.is_playing = False
        self.is_streaming = False
        
        # 播放队列
        self._audio_queue = asyncio.Queue()
        
        # 回调函数
        self._on_play_start: Optional[Callable] = None
        self._on_play_end: Optional[Callable] = None
        
        # 🎯 事件循环用于处理异步任务
        self._event_loop = asyncio.new_event_loop()
        
        # 🎯 订阅 ACTION_SPEAK 事件 (TSD v1.7 规范)
        self._register_event_handlers()
        
        logger.info("🔊 扬声器设备初始化完成")
        logger.info("🎙️ 已订阅 ACTION_SPEAK 事件")
        
        # 🎯 自动启动扬声器
        try:
            self._event_loop.run_until_complete(self.start())
            logger.info("✅ 扬声器自动启动成功")
        except Exception as e:
            logger.warning(f"⚠️ 扬声器自动启动失败：{e}，将在首次播放时启动")
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        # 订阅 L2 发出的语音合成指令
        event_bus.subscribe(
            EventType.ACTION_SPEAK,
            self._on_speak_command_sync,  # 🎯 使用同步包装器
            "SpeakerDevice"
        )
        logger.debug("✅ SpeakerDevice subscribed to ACTION_SPEAK events")
    
    def _on_speak_command_sync(self, event: ZulongEvent):
        """
        同步包装器 - 处理语音合成指令
        
        Args:
            event: ACTION_SPEAK 事件
        """
        # 🎯 在事件循环中运行异步方法
        try:
            self._event_loop.run_until_complete(self._on_speak_command(event))
        except Exception as e:
            logger.error(f"❌ _on_speak_command_sync 执行失败：{e}", exc_info=True)
    
    async def _on_speak_command(self, event: ZulongEvent):
        """
        处理语音合成指令 (TSD v1.7 规范)
        
        Args:
            event: ACTION_SPEAK 事件
        """
        logger.info(f"🎙️ [Speaker] 收到 ACTION_SPEAK 事件")
        
        try:
            text = event.payload.get("text", "")
            style = event.payload.get("style", "normal")
            voice_mode = event.payload.get("voice_mode", "TEXT_ONLY")
            
            # 🔥 关键修复：TTS 前文本清洗 (双重保险，TSD v1.7 第 4.2 节)
            from zulong.utils.text_cleaner import clean_text_for_tts
            original_text = text
            text = clean_text_for_tts(text)
            
            # 对比日志
            if text != original_text:
                logger.info(f"✨ [Speaker] TTS 文本二次清洗：{len(original_text)} → {len(text)} 字符")
                logger.debug(f"   原始：'{original_text[:100]}'")
                logger.debug(f"   清洗：'{text[:100]}'")
            else:
                logger.debug("✅ [Speaker] 文本已足够纯净，无需二次清洗")
            
            logger.info(f"   - 文本：'{text[:50]}...'")
            logger.info(f"   - 风格：{style}")
            logger.info(f"   - 语音模式：{voice_mode}")
            
            # 🎯 调用 TTS 专家生成音频并播放
            from zulong.l3.tts_expert_node import TTSExpertNode
            
            tts_expert = TTSExpertNode()
            
            logger.info("📥 开始 TTS 合成...")
            
            # 使用清洗后的文本生成音频
            result = tts_expert.execute({
                "task_description": "将文本转为语音",
                "text": text,  # 🔥 使用清洗后的纯净文本
                "sample_rate": 24000
            })
            
            audio_data = result.get("audio_data")
            
            if audio_data is not None:
                logger.info(f"✅ TTS 合成完成，播放音频...")
                await self.play_audio(audio_data, sample_rate=24000)
                logger.info("✅ 语音播放完成")
            else:
                logger.warning("⚠️ TTS 合成失败，无音频数据")
            
        except Exception as e:
            logger.error(f"❌ ACTION_SPEAK 处理失败：{e}", exc_info=True)
    
    def list_devices(self) -> list:
        """
        列出所有可用的音频输出设备
        
        Returns:
            list: 设备信息列表
        """
        audio = pyaudio.PyAudio()
        devices = []
        
        for i in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:  # 输出设备
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxOutputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })
            except Exception as e:
                logger.warning(f"获取设备 {i} 信息失败：{e}")
        
        audio.terminate()
        return devices
    
    async def initialize(self) -> bool:
        """
        初始化 PyAudio 和扬声器设备
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.audio = pyaudio.PyAudio()
            
            # 确定设备索引
            if self.device_index is None:
                self.device_index = self.audio.get_default_output_device_info()['index']
                logger.info(f"📍 使用默认扬声器设备：{self.device_index}")
            
            # 获取设备信息
            device_info = self.audio.get_device_info_by_index(self.device_index)
            logger.info(f"🔊 扬声器设备：{device_info['name']}")
            logger.info(f"   - 通道数：{device_info['maxOutputChannels']}")
            logger.info(f"   - 采样率：{int(device_info['defaultSampleRate'])} Hz")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 扬声器初始化失败：{e}")
            return False
    
    async def start(self) -> bool:
        """
        启动音频流播放
        
        Returns:
            bool: 启动是否成功
        """
        if not self.audio:
            if not await self.initialize():
                return False
        
        try:
            # 打开音频流
            # 🎯 关键修复：使用设备的实际配置
            device_info = self.audio.get_device_info_by_index(self.device_index)
            device_rate = int(device_info['defaultSampleRate'])
            device_channels = device_info['maxOutputChannels']
            
            logger.info(f"🔊 启动音频流...")
            logger.info(f"   - 设备采样率：{device_rate} Hz")
            logger.info(f"   - 设备声道数：{device_channels}")
            
            # 🎯 关键修复：音频流声道数必须与设备一致
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=device_channels,  # 使用设备实际声道数
                rate=device_rate,  # 使用设备默认采样率
                output=True,
                output_device_index=self.device_index,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            self.is_playing = True
            
            # 启动播放处理循环
            asyncio.create_task(self._playback_loop())
            
            logger.info("✅ 扬声器音频流已启动")
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动音频流失败：{e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio 回调函数（底层驱动调用）
        
        Args:
            in_data: 输入数据（未使用）
            frame_count: 帧数
            time_info: 时间信息
            status: 状态标志
        
        Returns:
            tuple: (data, flag)
        """
        # 流式播放模式下从队列读取数据
        if self.is_streaming and not self._audio_queue.empty():
            try:
                data = self._audio_queue.get_nowait()
                return (data, pyaudio.paContinue)
            except asyncio.QueueEmpty:
                pass
        
        # 非流式模式返回静音
        return (b'\x00' * (frame_count * self.CHANNELS * self.SAMPLE_WIDTH), pyaudio.paContinue)
    
    async def _playback_loop(self):
        """
        音频播放主循环
        
        功能:
        - 从队列读取音频数据
        - 写入音频流播放
        - 发布播放状态事件
        """
        logger.info("🎵 音频播放循环已启动")
        
        while self.is_playing:
            try:
                if not self._audio_queue.empty():
                    audio_data = await self._audio_queue.get()
                    
                    if self.stream and self.stream.is_active():
                        # 播放音频
                        self.stream.write(audio_data, exception_on_underflow=False)
                        logger.debug(f"🔊 播放音频块：{len(audio_data)} bytes")
                    else:
                        # Stream 未激活，等待
                        await asyncio.sleep(0.01)
                
                await asyncio.sleep(0)  # 让出事件循环
                
            except Exception as e:
                logger.error(f"音频播放循环错误：{e}")
                await asyncio.sleep(0.1)
    
    async def play_audio(self, audio_data: np.ndarray, sample_rate: int = 24000):
        """
        播放音频数据
        
        Args:
            audio_data: 音频数据（numpy 数组，归一化到 -1.0 ~ 1.0）
            sample_rate: 音频原始采样率
        
        Returns:
            bool: 播放是否成功
        """
        if not self.stream:
            logger.error("❌ 扬声器未启动，请先调用 start()")
            return False
        
        try:
            # 获取设备采样率
            device_info = self.audio.get_device_info_by_index(self.device_index)
            device_rate = int(device_info['defaultSampleRate'])
            device_channels = device_info['maxOutputChannels']
            
            logger.info(f"🔊 设备信息：采样率={device_rate}Hz, 声道数={device_channels}")
            
            # 🎯 关键修复 1: 重采样到设备采样率
            if sample_rate != device_rate:
                logger.info(f"🔄 重采样：{sample_rate}Hz -> {device_rate}Hz")
                import scipy.signal as signal
                num_samples = int(len(audio_data) * device_rate / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                sample_rate = device_rate
            
            # 🎯 关键修复 2: 转换为 16-bit PCM
            audio_int16 = (audio_data * 32767).astype('<i2')  # little-endian int16
            
            # 🎯 关键修复 3: 单声道 -> 立体声 (如果设备是立体声)
            if device_channels == 2:
                logger.info("🔄 转换为立体声 (单声道 -> 双声道)")
                # 将单声道数据复制为双声道 (左右声道相同)
                audio_stereo = np.repeat(audio_int16, 2).reshape(-1, 2)
                audio_bytes = audio_stereo.tobytes()
            else:
                audio_bytes = audio_int16.tobytes()
            
            logger.info(f"🔊 播放音频：{len(audio_bytes)} 字节，采样率={sample_rate}Hz, 声道数={device_channels}")
            logger.info(f"   - 数据类型：{audio_int16.dtype}")
            logger.info(f"   - 最小值：{audio_int16.min()}, 最大值：{audio_int16.max()}")
            logger.info(f"   - 时长：{len(audio_data) / sample_rate:.2f}秒")
            
            # 发布播放开始事件
            self._publish_event(EventType.SPEAKER_PLAYING, {
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate
            })
            
            # 🎯 关键修复 4: 直接播放完整音频块，不分块
            logger.info("🔊 开始播放...")
            self.stream.write(audio_bytes, exception_on_underflow=False)
            
            # 🎯 关键修复 5: 正确计算播放时长
            # 立体声 16-bit: 每帧 4 字节 (2 声道 × 2 字节)
            device_channels = device_info['maxOutputChannels']
            bytes_per_sample = 2 * device_channels  # 16-bit = 2 bytes × 声道数
            duration = len(audio_bytes) / (sample_rate * bytes_per_sample)
            
            logger.info(f"   等待播放：{duration + 0.1:.2f}秒 (音频时长：{duration:.2f}秒)")
            
            # 等待播放完成 (音频时长 + 缓冲时间)
            import time
            time.sleep(duration + 0.1)
            
            logger.info(f"✅ 音频播放完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 音频播放失败：{e}")
            return False
    
    async def play_chunk(self, audio_chunk: bytes):
        """
        播放音频块（流式播放）
        
        Args:
            audio_chunk: 音频数据（bytes）
        """
        if not self.is_streaming:
            self.is_streaming = True
            logger.debug("🔊 开始流式播放")
        
        await self._audio_queue.put(audio_chunk)
    
    async def stop_stream(self):
        """停止流式播放"""
        if self.is_streaming:
            self.is_streaming = False
            logger.debug("🔇 流式播放已停止")
    
    def set_callbacks(self, on_play_start: Optional[Callable] = None, on_play_end: Optional[Callable] = None):
        """
        设置播放回调函数
        
        Args:
            on_play_start: 播放开始回调
            on_play_end: 播放结束回调
        """
        self._on_play_start = on_play_start
        self._on_play_end = on_play_end
    
    def _publish_event(self, event_type: EventType, payload: dict):
        """
        发布事件到 EventBus
        
        Args:
            event_type: 事件类型
            payload: 事件载荷
        """
        event = ZulongEvent(
            type=event_type,
            source="speaker_device",
            payload=payload,
            priority=EventPriority.NORMAL
        )
        
        event_bus.publish(event)
        logger.debug(f"📨 发布事件：{event.type.name}")
    
    async def stop(self):
        """停止音频流播放"""
        logger.info("🛑 正在停止扬声器...")
        
        self.is_playing = False
        self.is_streaming = False
        
        # 清空播放队列
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                logger.warning(f"关闭音频流失败：{e}")
        
        if self.audio:
            try:
                self.audio.terminate()
                self.audio = None
            except Exception as e:
                logger.warning(f"终止 PyAudio 失败：{e}")
        
        logger.info("✅ 扬声器已停止")
    
    def cleanup(self):
        """清理资源"""
        asyncio.create_task(self.stop())


# TTS 专家接口（L3 专家层使用）
class TTSOutput:
    """
    TTS 输出接口（供 L3 TTS 专家调用）
    
    使用示例:
    ```python
    from zulong.l3.experts.tts_expert import tts_expert
    
    # TTS 专家内部调用
    audio_data = await tts_expert.generate_audio(text)
    await TTSOutput.play(audio_data)
    ```
    """
    
    _speaker: Optional[SpeakerDevice] = None
    
    @classmethod
    async def initialize(cls) -> bool:
        """初始化扬声器"""
        if cls._speaker is None:
            cls._speaker = SpeakerDevice()
            return await cls._speaker.start()
        return True
    
    @classmethod
    async def play(cls, audio_data: np.ndarray, sample_rate: int = 24000) -> bool:
        """
        播放 TTS 音频
        
        Args:
            audio_data: 音频数据（numpy 数组）
            sample_rate: 采样率
        
        Returns:
            bool: 播放是否成功
        """
        if cls._speaker is None:
            if not await cls.initialize():
                return False
        
        return await cls._speaker.play_audio(audio_data, sample_rate)
    
    @classmethod
    async def play_stream(cls, audio_chunks: list) -> bool:
        """
        流式播放 TTS 音频
        
        Args:
            audio_chunks: 音频块列表
        
        Returns:
            bool: 播放是否成功
        """
        if cls._speaker is None:
            if not await cls.initialize():
                return False
        
        for chunk in audio_chunks:
            await cls._speaker.play_chunk(chunk)
        
        await cls._speaker.stop_stream()
        return True
    
    @classmethod
    async def stop(cls):
        """停止播放"""
        if cls._speaker:
            await cls._speaker.stop()


# 测试函数
async def test_speaker():
    """测试扬声器设备"""
    print("="*60)
    print("🔊 祖龙系统 - 扬声器设备测试")
    print("="*60)
    
    speaker = SpeakerDevice()
    
    # 列出设备
    print("\n📋 可用扬声器设备:")
    devices = speaker.list_devices()
    for device in devices:
        print(f"   [{device['index']}] {device['name']}")
        print(f"       通道：{device['channels']}, 采样率：{device['sample_rate']} Hz")
    
    # 启动测试
    print("\n🚀 启动扬声器...")
    if await speaker.start():
        print("✅ 扬声器已启动")
        
        # 生成测试音频（正弦波）
        print("\n🎵 生成测试音频（440Hz 正弦波，1 秒）...")
        sample_rate = 24000
        duration = 1.0
        frequency = 440.0  # A4 音
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        print("🔊 开始播放...")
        await speaker.play_audio(audio_data, sample_rate)
        
        await asyncio.sleep(2)
        await speaker.stop()
        print("✅ 测试完成")
    else:
        print("❌ 扬声器启动失败")


if __name__ == "__main__":
    asyncio.run(test_speaker())
