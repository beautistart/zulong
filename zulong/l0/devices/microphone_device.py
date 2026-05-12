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
import time
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
        
        # 手动录音模式（前端按钮触发）
        self.manual_recording = False
        self.manual_audio_chunks: list = []
        self.manual_start_time: Optional[float] = None
        
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
        
        # SharedMemoryPool（用于写入 audio.raw_frame 供 L1D 插件读取）
        self._shared_memory_pool = None
        
        # 三层注意力状态机（手动录音模式）
        self._attention_layer = "L0_SENSOR"  # L0_SENSOR / L1_SILENT / L2_INTERACTIVE
        self._speech_detected = False
        self._speech_start_time = 0.0
        self._speech_buffer = []
        self._silence_frames = 0
        self._max_silence_frames = int(0.5 * self.SAMPLE_RATE / self.CHUNK_SIZE)  # 0.5秒静音判定
        self._min_speech_frames = int(0.3 * self.SAMPLE_RATE / self.CHUNK_SIZE)   # 0.3秒最小语音长度
        
        # YAMNet 环境音分类器（背景音过滤）
        self._yamnet_classifier = None
        self._yamnet_classify_interval = 1.0  # 每 1 秒分类一次
        self._last_yamnet_time = 0.0
        self._audio_accumulator = []  # 累积音频用于 YAMNet 分类
        self._last_sound_category = "unknown"
        
        # ALBERT 意图分类器（L1-B 阶段使用，此处不调用）
        # L1-D 仅做二分类：有无交互意图
        
        # 交互意图识别关键词（fallback）
        self._interaction_keywords = [
            "你好", "您好", "嘿", "hi", "hello", "hey",
            "帮我", "请帮", "可以帮", "能不能",
            "查", "查一下", "告诉我", "说一说",
            "什么", "怎么", "为什么", "哪里", "谁",
            "吗", "呢", "吧", "啊",
            "今天", "明天", "后天",
            "天气", "时间", "日期",
            "讲个", "说个", "来个",
            "打开", "关闭", "启动", "停止"
        ]
        
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
            # 打开音频流（阻塞读取模式，由 _process_audio_loop 主动读取）
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.CHUNK_SIZE,
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
                audio_normalized = audio_array / 32768.0
                energy = np.sqrt(np.mean(audio_normalized**2))
                energies.append(energy)
            
            self.noise_baseline = np.mean(energies) * 1.5
            self.baseline_initialized = True
            
            logger.info(f"✅ 噪音基线校准完成：{self.noise_baseline:.2f}")
        else:
            logger.warning("⚠️ 噪音基线校准失败，使用默认阈值")
            self.noise_baseline = self.ENERGY_THRESHOLD
            self.baseline_initialized = True
    
    async def _get_shared_memory(self):
        """获取 SharedMemoryPool 单例（异步）"""
        if self._shared_memory_pool is None:
            try:
                from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
                self._shared_memory_pool = await SharedMemoryPool.get_instance()
            except Exception as e:
                logger.warning(f"获取 SharedMemoryPool 失败: {e}")
        return self._shared_memory_pool
    
    def _init_yamnet(self):
        """初始化 YAMNet 环境音分类器"""
        if self._yamnet_classifier is not None:
            return
        
        try:
            from zulong.models.audio_model_container import YAMNetClassifier
            self._yamnet_classifier = YAMNetClassifier()
            if self._yamnet_classifier.initialize(device="cpu"):
                logger.info("✅ [YAMNet] 环境音分类器已启用")
            else:
                self._yamnet_classifier = None
                logger.warning("⚠️ [YAMNet] 初始化失败，禁用背景音过滤")
        except Exception as e:
            self._yamnet_classifier = None
            logger.warning(f"⚠️ [YAMNet] 加载失败: {e}")
    
    def _classify_sound(self, audio_array: np.ndarray) -> str:
        """
        使用 YAMNet 分类音频
        
        Returns:
            str: "speech" / "noise" / "music" / "other"
        """
        if self._yamnet_classifier is None:
            return "unknown"
        
        try:
            results = self._yamnet_classifier.classify(audio_array, self.SAMPLE_RATE)
            if results:
                top_category = results[0].category
                top_label = results[0].label
                top_score = results[0].score
                
                self._last_sound_category = top_category
                logger.debug(f"🔊 [YAMNet] {top_label} ({top_score:.2f}) → {top_category}")
                
                return top_category
        except Exception as e:
            logger.debug(f"[YAMNet] 分类失败: {e}")
        
        return "unknown"
    
    async def _process_audio_loop(self):
        """
        音频处理主循环 - 三层注意力机制
        
        L0_SENSOR:    纯数据采集，无注意力，持续写入 shared_memory
        L1_SILENT:    静默注意，检测语音开始（能量/过零率），不发布事件
        L2_INTERACTIVE: 交互注意，检测语音结束，发布 USER_SPEECH 事件
        
        功能:
        - 持续读取音频流
        - 三层注意力过滤
        - 写入 shared_memory["audio.raw_frame"] 供 L1D 插件读取
        - 检测交互音频，发布事件
        """
        logger.info("🎧 音频处理循环已启动（三层注意力模式）")
        
        while self.is_running:
            try:
                if self.stream is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                if not self.stream.is_active():
                    await asyncio.sleep(0.1)
                    continue
                    
                data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                
                # 手动录音模式：三层注意力机制
                if self.manual_recording:
                    self.manual_audio_chunks.append(data)
                    
                    # ── L0: 持续采集 ──
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    
                    # 写入 shared_memory 供 L1D 插件读取
                    shared_memory = await self._get_shared_memory()
                    if shared_memory and hasattr(shared_memory, 'set'):
                        try:
                            shared_memory.set(
                                "audio.raw_frame",
                                audio_array,
                                zone="raw",
                                metadata={"sample_rate": self.SAMPLE_RATE, "timestamp": time.time()}
                            )
                        except Exception as e:
                            logger.debug(f"写入 shared_memory 失败: {e}")
                    
                    # ── L1: 静默注意 - 检测语音开始 ──
                    if self._attention_layer == "L0_SENSOR":
                        if self._detect_voice_activity(audio_array):
                            self._attention_layer = "L1_SILENT"
                            self._speech_detected = True
                            self._speech_start_time = time.time()
                            self._speech_buffer = [data]
                            self._silence_frames = 0
                            logger.debug(f"🎤 [L1] 检测到语音开始")
                    
                    # ── L1→L2: 语音持续检测 ──
                    elif self._attention_layer == "L1_SILENT":
                        self._speech_buffer.append(data)
                        
                        if self._detect_voice_activity(audio_array):
                            self._silence_frames = 0
                        else:
                            self._silence_frames += 1
                            
                            # 静音超过阈值 → 语音结束
                            if self._silence_frames >= self._max_silence_frames:
                                speech_duration = time.time() - self._speech_start_time
                                speech_frames = len(self._speech_buffer)
                                
                                # 语音长度足够 → 发布 USER_SPEECH 事件
                                if speech_frames >= self._min_speech_frames:
                                    self._attention_layer = "L2_INTERACTIVE"
                                    await self._publish_speech_event(speech_duration)
                                
                                # 重置状态
                                self._attention_layer = "L0_SENSOR"
                                self._speech_detected = False
                                self._speech_buffer = []
                                self._silence_frames = 0
                    
                    logger.debug(f"🎤 录音中：层={self._attention_layer}, 块数={len(self.manual_audio_chunks)}")
                
                # 自动检测模式（保留原有逻辑）
                elif self._detect_sound_change(data):
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
    
    def _detect_voice_activity(self, audio_array: np.ndarray) -> bool:
        """
        检测语音活动（VAD）
        
        Args:
            audio_array: 音频数组 (int16)
        
        Returns:
            bool: 是否检测到语音
        """
        if not self.baseline_initialized:
            return False
        
        # 归一化
        audio_normalized = audio_array.astype(np.float32) / 32768.0
        
        # 计算能量（RMS）
        energy = np.sqrt(np.mean(audio_normalized**2))
        
        # 计算过零率
        if len(audio_normalized) > 1:
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_normalized)))) / len(audio_normalized)
        else:
            zero_crossings = 0
        
        # 检测逻辑：能量超过阈值 且 过零率合理（人声范围）
        return energy > self.noise_baseline and zero_crossings > self.ZCR_THRESHOLD
    
    async def _publish_speech_event(self, duration: float):
        """
        发布 USER_SPEECH 事件（三层注意力触发）
        
        三级过滤流程：
        1. YAMNet 分类 → 过滤非语音（噪音、音乐、车辆等）
        2. ASR 转录 → 获取文本
        3. 语义意图识别 → 区分背景对话 vs 直接交互
        
        Args:
            duration: 语音时长（秒）
        """
        if not self._speech_buffer:
            return
        
        # 合并音频
        combined_audio = b"".join(self._speech_buffer)
        audio_array = np.frombuffer(combined_audio, dtype=np.int16)
        
        # ── Level 1: YAMNet 背景音过滤 ──
        self._init_yamnet()
        sound_category = self._classify_sound(audio_array)
        
        if sound_category not in ["speech", "unknown"]:
            logger.info(f"🔇 [L1-过滤] YAMNet 分类为 {sound_category}，保持静默注意")
            return
        
        # ── Level 2: ASR 转录 + 情感/事件检测 ──
        transcription_text = ""
        emotion = "neutral"
        asr_events = []
        
        try:
            from zulong.models.audio_model_container import AudioModelContainer
            
            container = AudioModelContainer()
            if not hasattr(container, '_initialized') or not container._initialized:
                container.initialize(enable_sensevoice=True, enable_whisper=True)
            
            audio_normalized = audio_array.astype(np.float32) / 32768.0
            result = container.transcribe_speech(audio_normalized, self.SAMPLE_RATE, "zh")
            
            if result and result.text:
                transcription_text = result.text.strip()
                emotion = result.emotion  # angry/happy/sad/neutral/frightened/surprised
                asr_events = result.events  # music/applause/laughter/cry/emphasis
                logger.info(f"🎤 [L2-ASR] 转录：'{transcription_text}' (引擎={result.engine}, 情感={emotion}, 事件={asr_events})")
        except Exception as e:
            logger.error(f"🎤 [ASR] 转录失败: {e}")
        
        if not transcription_text:
            logger.debug("🔇 [L1-过滤] ASR 无转录结果，保持静默注意")
            return
        
        # ── Level 2.5: SenseVoice 事件过滤 ──
        if self._filter_by_asr_events(asr_events, emotion):
            logger.info(f"🔇 [L1-事件过滤] 检测到背景事件：{asr_events}，保持静默注意")
            return
        
        # ── Level 3: 交互意图二分类（SenseVoice 情感/事件）──
        has_interaction_intent = self._detect_interaction_intent(transcription_text, emotion, asr_events)
        
        if not has_interaction_intent:
            logger.info(f"🔇 [L1-静默] 检测到背景对话：'{transcription_text}'，无交互意图，保持静默注意")
            return
        
        # ── 发布 USER_SPEECH 事件 ──
        try:
            from zulong.core.types import ZulongEvent, EventType, EventPriority
            event = ZulongEvent(
                type=EventType.USER_SPEECH,
                source="microphone_attention",
                payload={
                    "text": transcription_text,
                    "audio_data": combined_audio,
                    "sample_rate": self.SAMPLE_RATE,
                    "duration": duration,
                    "confidence": 0.9,
                    "attention_layer": "L2_INTERACTIVE",
                    "sound_category": sound_category,
                    "emotion": emotion,  # SenseVoice 情感
                    "asr_events": asr_events,  # SenseVoice 事件
                    "timestamp": time.time()
                },
                priority=EventPriority.HIGH
            )
            self.event_bus.publish(event)
            logger.info(f"📨 [L2] 发布 USER_SPEECH 事件：'{transcription_text}'（情感={emotion}, 意图识别）")
        except Exception as e:
            logger.error(f"🎤 [事件发布] 失败: {e}", exc_info=True)
    
    def _filter_by_asr_events(self, events: list, emotion: str) -> bool:
        """
        根据 SenseVoice 事件和情感过滤背景音
        
        Args:
            events: 事件列表 ["music", "applause", "laughter", "cry", "emphasis"]
            emotion: 情感 "angry/happy/sad/neutral/frightened/surprised"
        
        Returns:
            bool: True 表示需要过滤（静默），False 表示通过
        """
        # 背景音乐 → 过滤
        if "music" in events:
            return True
        
        # 掌声/笑声（可能是电视/视频） → 过滤
        if "applause" in events or "laughter" in events:
            # 除非同时有明确的语音内容
            if emotion != "neutral":
                return False
            return True
        
        # 哭声（可能是背景婴儿） → 过滤
        if "cry" in events and emotion in ["sad", "frightened"]:
            return True
        
        return False
    
    def _detect_interaction_intent(self, text: str, emotion: str, events: list) -> bool:
        """
        检测交互意图（L1-D 二分类：有/无交互意图）
        
        策略：
        1. SenseVoice 情感增强：强烈情感 → 交互意图
        2. SenseVoice 事件检测：emphasis → 交互意图
        3. 关键词匹配：问候词/请求词/疑问词 → 交互意图
        
        注意：ALBERT 意图分类在 L1-B 阶段进行（15类细分类）
        
        Args:
            text: 转录文本
            emotion: SenseVoice 情感 (angry/happy/sad/neutral/frightened/surprised)
            events: SenseVoice 事件 (music/applause/laughter/cry/emphasis)
        
        Returns:
            bool: 是否有交互意图
        """
        # ── 策略1: SenseVoice 情感增强 ──
        # 强烈情感通常表示直接交互
        if emotion in ["angry", "surprised", "frightened"]:
            logger.info(f"🎯 [情感] 检测到强烈情感：{emotion} → 交互意图")
            return True
        
        # ── 策略2: SenseVoice 事件检测 ──
        # emphasis（强调）通常表示交互意图
        if "emphasis" in events:
            logger.info(f"🎯 [事件] 检测到强调事件：emphasis → 交互意图")
            return True
        
        # ── 策略3: 关键词匹配 ──
        text_lower = text.lower()
        
        # 检查关键词
        for keyword in self._interaction_keywords:
            if keyword in text_lower:
                logger.debug(f"🎯 [关键词] 检测到交互关键词：'{keyword}' → 交互意图")
                return True
        
        # 检查疑问句模式
        if "？" in text or "?" in text_lower:
            logger.debug(f"🎯 [句式] 检测到疑问句 → 交互意图")
            return True
        
        # 检查祈使句模式
        imperative_verbs = ["请", "把", "让", "叫", "给"]
        for verb in imperative_verbs:
            if text.startswith(verb):
                logger.debug(f"🎯 [句式] 检测到祈使句：'{verb}' → 交互意图")
                return True
        
        # ── 无交互意图：背景对话 ──
        logger.debug(f"🔇 [L1-D] 无交互意图标识，判定为背景对话")
        return False
    
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
    
    def start_manual_recording(self):
        """开始手动录音模式（前端按钮触发）- 三层注意力模式"""
        self.manual_recording = True
        self.manual_audio_chunks = []
        self.manual_start_time = asyncio.get_event_loop().time()
        
        # 重置三层注意力状态
        self._attention_layer = "L0_SENSOR"
        self._speech_detected = False
        self._speech_buffer = []
        self._silence_frames = 0
        
        logger.info("🎤 [手动录音] 开始（三层注意力模式）")
    
    async def stop_manual_recording(self) -> bytes:
        """
        停止手动录音并返回音频数据
        
        注意：三层注意力机制已实时处理语音事件，
        此方法仅停止音频流收集，不再重复 ASR
        
        Returns:
            bytes: 合并的音频数据（PCM 16bit, 16kHz, 单声道）
        """
        if not self.manual_recording:
            logger.warning("🎤 [手动录音] 未在录音状态")
            return b""
        
        self.manual_recording = False
        
        duration = asyncio.get_event_loop().time() - self.manual_start_time if self.manual_start_time else 0
        logger.info(f"🎤 [手动录音] 结束：时长={duration:.2f}s, 块数={len(self.manual_audio_chunks)}, 注意力层={self._attention_layer}")
        
        # 合并所有音频块
        combined_audio = b"".join(self.manual_audio_chunks)
        
        # 清空缓冲
        self.manual_audio_chunks = []
        self.manual_start_time = None
        
        # 注意：三层注意力机制已实时发布 USER_SPEECH 事件
        # 此处不再重复 ASR，仅返回音频数据
        
        return combined_audio
    
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
