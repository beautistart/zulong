# File: zulong/plugins/voice/l1d_audio_plugin.py
"""
L1-D 听觉系统 - 三层注意力音频插件 (增强版)

TSD v1.8 对应:
- 2.2.0 三层注意力机制
- 4.1.2 静默注意实现规范
- 4.2.1 L1-B 注意力控制器

架构:
- L0_SENSOR: 音频流采集、预加重、低切滤波
- L1_SILENT: YAMNet 环境音分类、VAD 检测
- L2_INTERACTIVE: Whisper 语音转文本、事件生成

功能:
- 三层注意力过滤（无需注意、静默注意、交互注意）
- YAMNet 环境音分类 (521 类)
- Whisper 语音转文本
- WebRTC VAD 高精度语音活动检测
- 能量和过零率特征提取
"""

import logging
import time
import collections
import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    logging.warning("webrtcvad not available, using fallback VAD")

from zulong.modules.l1.core.interface import (
    IL1Module, L1PluginBase, ZulongEvent, EventPriority, EventType, create_event
)
from zulong.core.attention_atoms import AttentionLayer, AttentionEvent

logger = logging.getLogger(__name__)


class AudioState(Enum):
    """音频状态枚举"""
    SILENT = "SILENT"
    NOISE = "NOISE"
    BACKGROUND_CHAT = "BACKGROUND_CHAT"
    SPEAKING = "SPEAKING"


@dataclass
class AudioFrame:
    """音频帧数据结构"""
    data: np.ndarray
    sample_rate: int
    timestamp: float
    energy: float = 0.0
    zcr: float = 0.0
    is_voice: bool = False
    state: AudioState = AudioState.SILENT
    sound_label: str = ""
    sound_category: str = ""


class L1D_AudioPlugin(L1PluginBase):
    """
    L1-D 听觉系统 - 三层注意力音频插件 (增强版)
    
    职责:
    - L0: 音频采集、预加重、低切滤波
    - L1: YAMNet 分类、VAD 检测、特征提取
    - L2: Whisper 转录、事件生成
    
    输入 (shared_memory):
    - "audio.raw_frame": 原始音频帧 (np.ndarray, 16kHz, int16)
    
    输出 (shared_memory):
    - "audio.status": 当前状态
    - "audio.energy": 当前能量
    - "audio.speech_buffer": 语音缓冲区
    - "audio.env_log": 环境噪音日志
    - "audio.sound_label": YAMNet 分类标签
    - "audio.transcription": Whisper 转录文本
    
    输出 (ZulongEvent):
    - AUDIO_SPEECH_START: 语音开始事件
    - AUDIO_SPEECH_END: 语音结束事件 (包含转录文本)
    """
    
    @property
    def module_id(self) -> str:
        return "L1D/Audio"
    
    @property
    def priority(self) -> EventPriority:
        return EventPriority.CRITICAL
    
    def initialize(self, shared_memory: Dict) -> bool:
        """初始化音频插件"""
        try:
            logger.info("🔌 [L1D/Audio] 正在初始化三层注意力音频插件 (增强版)...")
            
            if not super().initialize(shared_memory):
                return False
            
            self._sample_rate = self.get_config("sample_rate", 16000)
            self._frame_duration_ms = self.get_config("frame_duration_ms", 30)
            self._frame_size = int(self._sample_rate * self._frame_duration_ms / 1000)
            
            self._voice_threshold_energy = self.get_config("voice_threshold_energy", 500)
            self._noise_floor = self.get_config("noise_floor", 100)
            self._min_speech_duration = self.get_config("min_speech_duration", 0.5)
            self._speech_collect_window = self.get_config("speech_collect_window", 1.5)
            
            self._enable_yamnet = self.get_config("enable_yamnet", True)
            self._enable_whisper = self.get_config("enable_whisper", True)
            self._yamnet_classify_interval = self.get_config("yamnet_classify_interval", 1.0)
            
            if WEBRTC_VAD_AVAILABLE:
                vad_mode = self.get_config("vad_mode", 3)
                self._vad = webrtcvad.Vad(vad_mode)
                logger.info(f"   - WebRTC VAD 已初始化 (模式 {vad_mode})")
            else:
                self._vad = None
                logger.warning("   - WebRTC VAD 不可用，使用能量阈值作为备用")
            
            self._audio_model_container = None
            self._yamnet_enabled = False
            self._whisper_enabled = False
            
            if self._enable_yamnet or self._enable_whisper:
                try:
                    from zulong.models.audio_model_container import get_audio_model_container
                    self._audio_model_container = get_audio_model_container()
                    
                    success = self._audio_model_container.initialize(
                        enable_yamnet=self._enable_yamnet,
                        enable_whisper=self._enable_whisper,
                        whisper_device="cuda",
                        yamnet_device="cuda"
                    )
                    
                    self._yamnet_enabled = self._audio_model_container.yamnet_enabled
                    self._whisper_enabled = self._audio_model_container.whisper_enabled
                    
                    if self._yamnet_enabled:
                        logger.info("   - YAMNet 环境音分类已启用 (GPU)")
                    if self._whisper_enabled:
                        logger.info("   - Whisper 语音转文本已启用 (GPU)")
                        
                except Exception as e:
                    logger.warning(f"   - 音频模型加载失败: {e}，将使用备用方案")
            
            self._is_speaking = False
            self._speech_start_time = 0.0
            self._last_voice_time = 0.0
            self._speech_buffer: List[np.ndarray] = []
            self._speech_buffer_max_frames = int(self._speech_collect_window * 1000 / self._frame_duration_ms)
            
            self._state = AudioState.SILENT
            self._frame_count = 0
            self._env_log_counter = 0
            self._last_yamnet_time = 0.0
            self._last_sound_label = ""
            self._last_sound_category = ""
            
            self._audio_accumulator: List[np.ndarray] = []
            self._accumulator_max_frames = int(self._yamnet_classify_interval * 1000 / self._frame_duration_ms)
            
            shared_memory["audio.status"] = "SILENT"
            shared_memory["audio.energy"] = 0.0
            shared_memory["audio.speech_buffer"] = []
            shared_memory["audio.env_log"] = ""
            shared_memory["audio.last_voice_time"] = 0.0
            shared_memory["audio.raw_stream"] = collections.deque(maxlen=333)
            shared_memory["audio.sound_label"] = ""
            shared_memory["audio.sound_category"] = ""
            shared_memory["audio.transcription"] = ""
            
            logger.info(f"   - 采样率: {self._sample_rate} Hz")
            logger.info(f"   - 帧时长: {self._frame_duration_ms} ms")
            logger.info(f"   - 帧大小: {self._frame_size} 采样点")
            logger.info(f"   - 语音能量阈值: {self._voice_threshold_energy}")
            logger.info(f"   - 底噪阈值: {self._noise_floor}")
            logger.info(f"   - 最小语音时长: {self._min_speech_duration} s")
            logger.info(f"   - 语音收集窗口: {self._speech_collect_window} s")
            logger.info("✅ [L1D/Audio] 初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ [L1D/Audio] 初始化失败：{e}", exc_info=True)
            return False
    
    def process_cycle(self, shared_memory: Dict) -> List[ZulongEvent]:
        """单周期处理 - 三层注意力"""
        events: List[ZulongEvent] = []
        current_time = time.time()
        
        try:
            raw_frame = shared_memory.get("audio.raw_frame")
            if raw_frame is None:
                return events
            
            if isinstance(raw_frame, np.ndarray):
                audio_frame = raw_frame
            else:
                return events
            
            self._frame_count += 1
            
            clean_frame = self._low_cut_filter(audio_frame)
            
            if "audio.raw_stream" in shared_memory:
                shared_memory["audio.raw_stream"].append(clean_frame.copy())
            
            frame_data = AudioFrame(
                data=clean_frame,
                sample_rate=self._sample_rate,
                timestamp=current_time
            )
            
            frame_data.energy = self._compute_energy(clean_frame)
            frame_data.zcr = self._compute_zcr(clean_frame)
            
            shared_memory["audio.energy"] = frame_data.energy
            
            if frame_data.energy < self._noise_floor:
                frame_data.state = AudioState.SILENT
                self._update_state(shared_memory, AudioState.SILENT)
                return events
            
            self._audio_accumulator.append(clean_frame.copy())
            if len(self._audio_accumulator) > self._accumulator_max_frames:
                self._audio_accumulator.pop(0)
            
            if self._yamnet_enabled and len(self._audio_accumulator) >= self._accumulator_max_frames:
                if current_time - self._last_yamnet_time >= self._yamnet_classify_interval:
                    self._classify_with_yamnet(frame_data, shared_memory)
                    self._last_yamnet_time = current_time
            
            is_voice = self._detect_voice(clean_frame)
            frame_data.is_voice = is_voice
            
            if self._yamnet_enabled and frame_data.sound_category:
                if frame_data.sound_category not in ["speech", "other"]:
                    frame_data.state = AudioState.NOISE
                    self._update_state(shared_memory, AudioState.NOISE)
                    self._log_environment_sound(frame_data, shared_memory)
                    return events
            
            if not is_voice:
                frame_data.state = AudioState.NOISE
                self._update_state(shared_memory, AudioState.NOISE)
                self._log_environment_sound(frame_data, shared_memory)
                return events
            
            if frame_data.energy > self._voice_threshold_energy:
                if not self._is_speaking:
                    self._is_speaking = True
                    self._speech_start_time = current_time
                    self._speech_buffer = []
                    
                    logger.info(f"🎙️ [L1D/Audio] 检测到语音开始 (能量={frame_data.energy:.1f})")
                    
                    speech_start_event = create_event(
                        event_type=EventType.USER_SPEECH,
                        priority=EventPriority.HIGH,
                        source=self.module_id,
                        event_subtype="AUDIO_SPEECH_START",
                        energy=frame_data.energy,
                        sound_label=frame_data.sound_label,
                        sound_category=frame_data.sound_category,
                        timestamp=current_time
                    )
                    events.append(speech_start_event)
                
                self._speech_buffer.append(clean_frame.copy())
                if len(self._speech_buffer) > self._speech_buffer_max_frames:
                    self._speech_buffer.pop(0)
                
                shared_memory["audio.speech_buffer"] = self._speech_buffer.copy()
                
                frame_data.state = AudioState.SPEAKING
                self._update_state(shared_memory, AudioState.SPEAKING)
                self._last_voice_time = current_time
                shared_memory["audio.last_voice_time"] = current_time
                
            else:
                frame_data.state = AudioState.BACKGROUND_CHAT
                self._update_state(shared_memory, AudioState.BACKGROUND_CHAT)
                
                if self._is_speaking:
                    speech_duration = current_time - self._speech_start_time
                    if speech_duration >= self._min_speech_duration:
                        transcription = ""
                        confidence = 0.0
                        
                        if self._whisper_enabled and len(self._speech_buffer) > 0:
                            transcription, confidence = self._transcribe_speech(shared_memory)
                        
                        logger.info(f"🎙️ [L1D/Audio] 语音结束 (时长={speech_duration:.2f}s, 转录='{transcription}')")
                        
                        speech_end_event = create_event(
                            event_type=EventType.USER_SPEECH,
                            priority=EventPriority.HIGH,
                            source=self.module_id,
                            event_subtype="AUDIO_SPEECH_END",
                            duration=speech_duration,
                            buffer_frames=len(self._speech_buffer),
                            transcription=transcription,
                            confidence=confidence,
                            timestamp=current_time
                        )
                        events.append(speech_end_event)
                    
                    self._is_speaking = False
                    self._speech_buffer = []
            
            if self._frame_count % 30 == 0:
                logger.debug(f"🎵 [L1D/Audio] 状态={frame_data.state.value}, "
                           f"能量={frame_data.energy:.1f}, 人声={is_voice}, "
                           f"标签={frame_data.sound_label}")
            
        except Exception as e:
            logger.error(f"❌ [L1D/Audio] process_cycle 错误：{e}", exc_info=True)
        
        return events
    
    def _classify_with_yamnet(self, frame_data: AudioFrame, shared_memory: Dict):
        """使用 YAMNet 分类声音"""
        try:
            audio_segment = np.concatenate(self._audio_accumulator)
            
            if audio_segment.dtype == np.int16:
                audio_float = audio_segment.astype(np.float32) / 32768.0
            else:
                audio_float = audio_segment.astype(np.float32)
            
            classifications = self._audio_model_container.classify_sound(audio_float, self._sample_rate)
            
            if classifications:
                top_class = classifications[0]
                frame_data.sound_label = top_class.label
                frame_data.sound_category = top_class.category
                
                self._last_sound_label = top_class.label
                self._last_sound_category = top_class.category
                
                shared_memory["audio.sound_label"] = top_class.label
                shared_memory["audio.sound_category"] = top_class.category
                
                logger.debug(f"🔊 [YAMNet] 分类: {top_class.label} ({top_class.category}) "
                           f"置信度={top_class.score:.2f}")
                
        except Exception as e:
            logger.error(f"❌ [YAMNet] 分类失败: {e}")
    
    def _transcribe_speech(self, shared_memory: Dict) -> tuple:
        """使用 Whisper 转录语音"""
        try:
            audio_segment = np.concatenate(self._speech_buffer)
            
            if audio_segment.dtype == np.int16:
                audio_float = audio_segment.astype(np.float32) / 32768.0
            else:
                audio_float = audio_segment.astype(np.float32)
            
            text, confidence = self._audio_model_container.transcribe_speech(
                audio_float, 
                self._sample_rate,
                language="zh"
            )
            
            shared_memory["audio.transcription"] = text
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"❌ [Whisper] 转录失败: {e}")
            return "", 0.0
    
    def _low_cut_filter(self, data: np.ndarray, cutoff_hz: float = 80.0) -> np.ndarray:
        """L0: 低切滤波，去除低频噪音"""
        if len(data) < 10:
            return data
        
        try:
            from scipy import signal
            
            nyquist = self._sample_rate / 2
            normalized_cutoff = cutoff_hz / nyquist
            
            if normalized_cutoff >= 1.0:
                return data
            
            b, a = signal.butter(1, normalized_cutoff, btype='high')
            filtered = signal.filtfilt(b, a, data)
            
            return filtered.astype(data.dtype)
            
        except ImportError:
            return data
        except Exception:
            return data
    
    def _compute_energy(self, data: np.ndarray) -> float:
        """计算音频能量"""
        if data.dtype == np.int16:
            return float(np.sum(np.abs(data.astype(np.float64))))
        else:
            return float(np.sum(np.abs(data)) * 32768)
    
    def _compute_zcr(self, data: np.ndarray) -> float:
        """计算过零率"""
        if len(data) < 2:
            return 0.0
        
        if data.dtype == np.int16:
            data = data.astype(np.float64) / 32768.0
        
        signs = np.sign(data)
        signs[signs == 0] = 1
        crossings = np.sum(np.abs(np.diff(signs)) / 2)
        
        return float(crossings / len(data))
    
    def _detect_voice(self, data: np.ndarray) -> bool:
        """VAD 语音活动检测"""
        if WEBRTC_VAD_AVAILABLE and self._vad is not None:
            try:
                if data.dtype != np.int16:
                    data = (data * 32768).astype(np.int16)
                
                frame_bytes = data.tobytes()
                
                if len(frame_bytes) < 480:
                    padding_needed = 480 - len(frame_bytes)
                    frame_bytes += b'\x00' * padding_needed
                elif len(frame_bytes) > 480:
                    frame_bytes = frame_bytes[:480]
                
                return self._vad.is_speech(frame_bytes, self._sample_rate)
                
            except Exception as e:
                logger.debug(f"WebRTC VAD 错误: {e}")
                return self._fallback_vad(data)
        else:
            return self._fallback_vad(data)
    
    def _fallback_vad(self, data: np.ndarray) -> bool:
        """备用 VAD（基于能量和过零率）"""
        energy = self._compute_energy(data)
        zcr = self._compute_zcr(data)
        
        energy_score = 1.0 if energy > self._noise_floor * 2 else 0.0
        zcr_score = 1.0 if 0.1 < zcr < 0.5 else 0.0
        
        return (energy_score + zcr_score) >= 1.0
    
    def _update_state(self, shared_memory: Dict, new_state: AudioState):
        """更新状态"""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            shared_memory["audio.status"] = new_state.value
            
            if new_state != AudioState.SILENT:
                logger.debug(f"🎵 [L1D/Audio] 状态变更: {old_state.value} -> {new_state.value}")
    
    def _log_environment_sound(self, frame_data: AudioFrame, shared_memory: Dict):
        """记录环境噪音"""
        self._env_log_counter += 1
        
        if self._env_log_counter % 100 == 0:
            sound_info = f"{frame_data.sound_label}" if frame_data.sound_label else "未知"
            noise_log = f"环境音: {sound_info} (能量={frame_data.energy:.1f}) at {time.strftime('%H:%M:%S')}"
            shared_memory["audio.env_log"] = noise_log
            logger.debug(f"🔊 [L1D/Audio] {noise_log}")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health = {
            "status": "OK" if self._initialized else "ERROR",
            "details": {
                "state": self._state.value,
                "is_speaking": self._is_speaking,
                "vad_available": WEBRTC_VAD_AVAILABLE,
                "sample_rate": self._sample_rate,
                "frame_count": self._frame_count,
                "last_voice_time": self._last_voice_time,
                "yamnet_enabled": self._yamnet_enabled,
                "whisper_enabled": self._whisper_enabled
            },
            "last_update": time.time()
        }
        
        if self._audio_model_container:
            health["models"] = self._audio_model_container.health_check()
        
        return health
    
    def shutdown(self):
        """关闭插件"""
        logger.info("🔌 [L1D/Audio] 正在关闭...")
        self._initialized = False
        self._speech_buffer = []
        self._audio_accumulator = []
        logger.info("✅ [L1D/Audio] 已关闭")


def create_plugin(config: Dict = None) -> IL1Module:
    """工厂函数"""
    return L1D_AudioPlugin(config=config)
