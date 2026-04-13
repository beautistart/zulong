#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙 (ZULONG) 系统 - 麦克风驱动与音频采集模块

文件：zulong/l0/devices/microphone_device.py

功能:
- 麦克风设备管理（初始化、启动、停止）
- 音频流采集（16kHz, 16bit, 单声道）
- 声音变化检测（能量阈值 + 过零率）
- 环境噪音过滤（librosa 降噪）
- 发布 SENSOR_AUDIO_CHUNK 事件

TSD v1.7 对应:
- 4.4 感知预处理 - VAD (语音活动检测)
- 2.2.2 L1-A - SENSOR_* 事件处理
"""

import asyncio
import logging
from typing import Optional, Callable
import pyaudio
import numpy as np
import librosa

from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import EventBus

logger = logging.getLogger(__name__)


class MicrophoneDevice:
    """
    麦克风设备驱动类
    
    功能:
    - 封装 PyAudio，调用 Windows 音频驱动
    - 实现声音变化检测
    - 发布音频事件到 EventBus
    
    硬件要求:
    - Windows 系统已安装麦克风驱动
    - 麦克风设备可用
    
    使用示例:
    ```python
    mic = MicrophoneDevice()
    await mic.start()
    # 自动采集音频并发布事件
    await mic.stop()
    ```
    """
    
    # 音频参数配置
    SAMPLE_RATE = 16000      # 采样率 16kHz
    CHUNK_SIZE = 1024        # 每次读取的帧数
    SAMPLE_WIDTH = 2         # 16bit = 2 bytes
    CHANNELS = 1             # 单声道
    
    # 声音检测阈值
    ENERGY_THRESHOLD = 0.02  # 能量阈值（归一化）
    ZCR_THRESHOLD = 0.1      # 过零率阈值
    SILENCE_DURATION = 0.5   # 静音判定时长（秒）
    
    def __init__(self, device_index: Optional[int] = None):
        """
        初始化麦克风设备
        
        Args:
            device_index: 麦克风设备索引，None 表示使用默认设备
        """
        self.device_index = device_index
        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.is_running = False
        
        # 噪音基线（用于声音变化检测）
        self.noise_baseline: Optional[float] = None
        self.baseline_samples: list = []
        self.baseline_initialized = False
        
        # 状态跟踪
        self.last_sound_time = 0.0
        self.is_speaking = False
        
        # EventBus（单例）
        from zulong.core.event_bus import event_bus
        self.event_bus = event_bus
        
        logger.info("🎤 麦克风设备初始化完成")
    
    def list_devices(self) -> list:
        """
        列出所有可用的音频输入设备
        
        Returns:
            list: 设备信息列表
        """
        audio = pyaudio.PyAudio()
        devices = []
        
        for i in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:  # 输入设备
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })
            except Exception as e:
                logger.warning(f"获取设备 {i} 信息失败：{e}")
        
        audio.terminate()
        return devices
    
    async def initialize(self) -> bool:
        """
        初始化 PyAudio 和麦克风设备
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.audio = pyaudio.PyAudio()
            
            # 确定设备索引
            if self.device_index is None:
                self.device_index = self.audio.get_default_input_device_info()['index']
                logger.info(f"📍 使用默认麦克风设备：{self.device_index}")
            
            # 获取设备信息
            device_info = self.audio.get_device_info_by_index(self.device_index)
            logger.info(f"🎤 麦克风设备：{device_info['name']}")
            logger.info(f"   - 通道数：{device_info['maxInputChannels']}")
            logger.info(f"   - 采样率：{int(device_info['defaultSampleRate'])} Hz")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 麦克风初始化失败：{e}")
            return False
    
    async def start(self) -> bool:
        """
        启动音频流采集
        
        Returns:
            bool: 启动是否成功
        """
        if not self.audio:
            if not await self.initialize():
                return False
        
        try:
            # 打开音频流
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            
            self.is_running = True
            logger.info("✅ 麦克风音频流已启动")
            
            # 启动噪音基线校准（前 0.5 秒）
            asyncio.create_task(self._calibrate_noise_baseline())
            
            # 启动音频处理循环
            asyncio.create_task(self._process_audio_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动音频流失败：{e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio 回调函数（底层驱动调用）
        
        Args:
            in_data: 音频数据（bytes）
            frame_count: 帧数
            time_info: 时间信息
            status: 状态标志
        
        Returns:
            tuple: (data, flag)
        """
        # 持续读取数据
        return (in_data, pyaudio.paContinue)
    
    async def _calibrate_noise_baseline(self):
        """
        校准环境噪音基线（启动时采集 0.5 秒环境音）
        
        TSD v1.7 对应:
        - 4.4 感知预处理 - 环境噪音过滤
        """
        logger.info("🔇 正在校准环境噪音基线（0.5 秒）...")
        
        # 等待音频流完全启动
        await asyncio.sleep(0.1)
        
        self.baseline_samples = []
        calibration_samples = int(0.5 * self.SAMPLE_RATE / self.CHUNK_SIZE)
        
        for _ in range(calibration_samples):
            if not self.is_running:
                break
            
            try:
                if self.stream:
                    data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    self.baseline_samples.append(data)
            except OSError as e:
                # 流尚未完全打开，跳过本次读取
                logger.debug(f"音频流尚未就绪，跳过校准采样：{e}")
            except Exception as e:
                logger.warning(f"校准期间读取音频失败：{e}")
            
            await asyncio.sleep(0)  # 让出事件循环
        
        # 计算噪音基线（平均能量）
        if self.baseline_samples:
            energies = []
            for sample in self.baseline_samples:
                audio_array = np.frombuffer(sample, dtype=np.int16).astype(np.float32)
                energy = np.sqrt(np.mean(audio_array**2))
                energies.append(energy)
            
            self.noise_baseline = np.mean(energies) * 1.5  # 1.5 倍作为阈值
            self.baseline_initialized = True
            
            logger.info(f"✅ 噪音基线校准完成：{self.noise_baseline:.2f}")
        else:
            logger.warning("⚠️ 噪音基线校准失败，使用默认阈值")
            self.noise_baseline = self.ENERGY_THRESHOLD
            self.baseline_initialized = True
    
    async def _process_audio_loop(self):
        """
        音频处理主循环
        
        功能:
        - 持续读取音频流
        - 检测声音变化
        - 发布音频事件
        """
        logger.info("🎧 音频处理循环已启动")
        
        while self.is_running:
            try:
                if self.stream is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                if not self.stream.is_active():
                    await asyncio.sleep(0.1)
                    continue
                    
                data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                
                if self._detect_sound_change(data):
                    await self._publish_audio_event(data)
                
                await asyncio.sleep(0)
                
            except IOError as e:
                if "Stream not open" in str(e):
                    await asyncio.sleep(0.5)
                    continue
                logger.error(f"音频处理循环错误：{e}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"音频处理循环错误：{e}")
                await asyncio.sleep(0.1)
    
    def _detect_sound_change(self, audio_data: bytes) -> bool:
        """
        检测声音变化（语音活动检测 VAD）
        
        算法:
        1. 计算音频能量（RMS）
        2. 计算过零率（Zero Crossing Rate）
        3. 与噪音基线对比
        
        Args:
            audio_data: 音频数据（bytes）
        
        Returns:
            bool: 是否检测到声音变化
        """
        # 转换为 numpy 数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # 归一化
        audio_normalized = audio_array / 32768.0
        
        # 计算能量（RMS）
        energy = np.sqrt(np.mean(audio_normalized**2))
        
        # 计算过零率
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_normalized)))) / len(audio_normalized)
        
        # 检测逻辑
        if not self.baseline_initialized:
            return False
        
        # 能量超过阈值 且 过零率合理（人声范围）
        if energy > self.noise_baseline and zero_crossings > self.ZCR_THRESHOLD:
            self.last_sound_time = asyncio.get_event_loop().time()
            
            if not self.is_speaking:
                self.is_speaking = True
                logger.debug(f"🔊 检测到声音：能量={energy:.4f}, 过零率={zero_crossings:.4f}")
            
            return True
        else:
            # 静音检测
            if self.is_speaking and (asyncio.get_event_loop().time() - self.last_sound_time) > self.SILENCE_DURATION:
                self.is_speaking = False
                logger.debug("🔇 检测到静音")
            
            return False
    
    async def _publish_audio_event(self, audio_data: bytes):
        """
        发布音频事件到 EventBus
        
        TSD v1.7 对应:
        - 3.1 事件定义 - SENSOR_AUDIO_CHUNK
        
        Args:
            audio_data: 音频数据（bytes）
        """
        event = ZulongEvent(
            type=EventType.SENSOR_SOUND,
            source="microphone_device",
            payload={
                "audio_data": audio_data,
                "sample_rate": self.SAMPLE_RATE,
                "sample_width": self.SAMPLE_WIDTH,
                "channels": self.CHANNELS,
                "chunk_size": self.CHUNK_SIZE,
                "timestamp": asyncio.get_event_loop().time()
            },
            priority=EventPriority.NORMAL
        )
        
        await self.event_bus.publish(event)
        logger.debug(f"📨 发布音频事件：{event.id}")
    
    async def stop(self):
        """停止音频流采集"""
        logger.info("🛑 正在停止麦克风...")
        
        self.is_running = False
        
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
        
        logger.info("✅ 麦克风已停止")
    
    def cleanup(self):
        """清理资源"""
        asyncio.create_task(self.stop())


# 测试函数
async def test_microphone():
    """测试麦克风设备"""
    print("="*60)
    print("🎤 祖龙系统 - 麦克风设备测试")
    print("="*60)
    
    mic = MicrophoneDevice()
    
    # 列出设备
    print("\n📋 可用麦克风设备:")
    devices = mic.list_devices()
    for device in devices:
        print(f"   [{device['index']}] {device['name']}")
        print(f"       通道：{device['channels']}, 采样率：{device['sample_rate']} Hz")
    
    # 启动测试
    print("\n🚀 启动麦克风（测试 5 秒）...")
    if await mic.start():
        print("✅ 麦克风已启动，请说话测试...")
        await asyncio.sleep(5)
        await mic.stop()
        print("✅ 测试完成")
    else:
        print("❌ 麦克风启动失败")


if __name__ == "__main__":
    asyncio.run(test_microphone())
