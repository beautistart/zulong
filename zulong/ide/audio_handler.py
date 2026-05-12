"""
音频输入处理器

处理前端 WebSocket 发送的实时音频流,调用 ASR 引擎进行转录
"""
import asyncio
import base64
import io
import os
import tempfile
import time
from typing import Dict, Optional, Any

from zulong.ide.audio_logger import logger

_audio_container = None

def _get_audio_container():
    """懒加载音频容器"""
    global _audio_container
    if _audio_container is None:
        try:
            from zulong.models.audio_model_container import AudioModelContainer
            _audio_container = AudioModelContainer()  # 单例模式通过构造函数获取
            
            # 初始化模型
            from zulong.config.config_manager import ConfigManager
            cm = ConfigManager()
            
            sensevoice_model_path = cm.get('audio.asr.model_path', './models/OpenASR/sensevoice-small-onnx')
            asr_device = cm.get('audio.asr.device', 'cuda')
            
            initialized = _audio_container.initialize(
                enable_yamnet=False,
                enable_sensevoice=True,
                enable_whisper=True,
                sensevoice_device=asr_device,
                sensevoice_model_path=sensevoice_model_path,
            )
            
            if initialized:
                logger.info("[AudioHandler] AudioModelContainer 初始化成功")
            else:
                logger.warning("[AudioHandler] AudioModelContainer 初始化失败，ASR 不可用")
                
        except Exception as e:
            logger.error(f"[AudioHandler] AudioModelContainer 加载失败: {e}")
            return None
    return _audio_container


class AudioStreamSession:
    """音频流会话 - 管理单个 WebSocket 连接的音频缓冲"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audio_chunks: list[bytes] = []
        self.start_time: Optional[float] = None
        self.is_streaming: bool = False
        self.last_transcript_time: float = 0
        self.transcript_buffer: str = ""
    
    def add_chunk(self, audio_data: bytes):
        """添加音频块"""
        if not self.is_streaming:
            return
        self.audio_chunks.append(audio_data)
        logger.debug(f"[AudioSession:{self.session_id[:8]}] 添加音频块: {len(audio_data)} bytes, 总块数: {len(self.audio_chunks)}")
    
    def get_combined_audio(self) -> bytes:
        """获取合并的音频数据"""
        return b"".join(self.audio_chunks)
    
    def clear(self):
        """清空缓冲"""
        self.audio_chunks = []
        self.transcript_buffer = ""
        logger.debug(f"[AudioSession:{self.session_id[:8]}] 缓冲已清空")


_audio_sessions: Dict[str, AudioStreamSession] = {}


def get_or_create_session(session_id: str) -> AudioStreamSession:
    """获取或创建音频会话"""
    if session_id not in _audio_sessions:
        _audio_sessions[session_id] = AudioStreamSession(session_id)
        logger.info(f"[AudioHandler] 创建音频会话: {session_id[:8]}")
    return _audio_sessions[session_id]


async def handle_audio_start(session_id: str) -> Dict[str, Any]:
    """
    处理音频流开始
    
    Args:
        session_id: WebSocket 会话 ID
    
    Returns:
        响应消息
    """
    audio_session = get_or_create_session(session_id)
    audio_session.is_streaming = True
    audio_session.start_time = time.time()
    audio_session.clear()
    
    logger.info(f"[AudioHandler] 音频流开始: session={session_id[:8]}")
    
    return {
        "status": "ok",
        "message": "audio_stream_started",
    }


async def handle_audio_chunk(session_id: str, payload: Dict) -> Optional[Dict[str, Any]]:
    """
    处理音频块
    
    Args:
        session_id: WebSocket 会话 ID
        payload: 包含 base64 编码的音频数据
    
    Returns:
        实时转录结果 (如果有)
    """
    audio_session = get_or_create_session(session_id)
    
    if not audio_session.is_streaming:
        logger.warning(f"[AudioHandler] 收到音频块但流未开始: session={session_id[:8]}")
        return None
    
    audio_base64 = payload.get("audio", "")
    if not audio_base64:
        return None
    
    try:
        audio_data = base64.b64decode(audio_base64)
        audio_session.add_chunk(audio_data)
        
        chunk_count = len(audio_session.audio_chunks)
        if chunk_count % 5 == 0:
            logger.debug(f"[AudioHandler] 已接收 {chunk_count} 个音频块")
        
        if chunk_count >= 10 and time.time() - audio_session.last_transcript_time >= 1.0:
            return await _transcribe_buffer(session_id, is_final=False)
        
        return None
        
    except Exception as e:
        logger.error(f"[AudioHandler] 音频块处理失败: {e}")
        return None


async def handle_audio_end(session_id: str) -> Dict[str, Any]:
    """
    处理音频流结束,执行最终转录
    
    Args:
        session_id: WebSocket 会话 ID
    
    Returns:
        最终转录结果
    """
    audio_session = get_or_create_session(session_id)
    audio_session.is_streaming = False
    
    if not audio_session.audio_chunks:
        logger.warning(f"[AudioHandler] 音频流结束但无数据: session={session_id[:8]}")
        return {
            "status": "error",
            "message": "no_audio_data",
            "text": "",
        }
    
    duration = time.time() - audio_session.start_time if audio_session.start_time else 0
    logger.info(f"[AudioHandler] 音频流结束: session={session_id[:8]}, 时长={duration:.2f}s, 块数={len(audio_session.audio_chunks)}")
    
    result = await _transcribe_buffer(session_id, is_final=True)
    
    if session_id in _audio_sessions:
        del _audio_sessions[session_id]
    
    return result or {
        "status": "ok",
        "text": "",
        "is_final": True,
    }


async def _transcribe_buffer(session_id: str, is_final: bool = False) -> Optional[Dict[str, Any]]:
    """
    转录音频缓冲
    
    Args:
        session_id: WebSocket 会话 ID
        is_final: 是否为最终转录
    
    Returns:
        转录结果
    """
    audio_session = get_or_create_session(session_id)
    
    if not audio_session.audio_chunks:
        return None
    
    container = _get_audio_container()
    if container is None:
        logger.error("[AudioHandler] AudioModelContainer 不可用")
        return {
            "status": "error",
            "message": "asr_not_available",
            "text": "",
            "is_final": is_final,
        }
    
    try:
        combined_audio = audio_session.get_combined_audio()
        
        # 写入临时 webm 文件
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
            webm_file.write(combined_audio)
            webm_path = webm_file.name
        
        wav_path = None
        try:
            # 获取 ffmpeg 路径
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_exe = "ffmpeg"  # 回退到系统 PATH
            
            # 使用 subprocess 调用 ffmpeg 转换
            wav_path = webm_path.replace(".webm", ".wav")
            
            import subprocess
            subprocess.run([
                ffmpeg_exe, "-y", "-i", webm_path,
                "-ar", "16000",  # 16kHz
                "-ac", "1",      # 单声道
                "-f", "wav",
                wav_path
            ], check=True, capture_output=True)
            
            logger.debug(f"[AudioHandler] ffmpeg 转换成功: {webm_path} -> {wav_path}")
            
            # 读取 wav
            import soundfile as sf
            audio_data, sample_rate = sf.read(wav_path)
            
            logger.debug(f"[AudioHandler] 开始转录: {len(audio_data)} samples, sr={sample_rate}, is_final={is_final}")
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: container.transcribe_speech(
                    audio_data.astype('float32'),
                    int(sample_rate),
                    "zh"  # 明确指定中文
                )
            )
            
            audio_session.last_transcript_time = time.time()
            
            if result and result.text:
                text = result.text.strip()
                
                if is_final:
                    logger.info(f"[AudioHandler] 最终转录: \"{text}\" (引擎={result.engine}, 情感={result.emotion})")
                else:
                    logger.debug(f"[AudioHandler] 实时转录: \"{text}\"")
                
                return {
                    "status": "ok",
                    "text": text,
                    "is_final": is_final,
                    "emotion": result.emotion,
                    "language": result.language,
                    "engine": result.engine,
                    "confidence": result.confidence,
                }
            else:
                logger.debug(f"[AudioHandler] 转录无结果")
                return {
                    "status": "ok",
                    "text": "",
                    "is_final": is_final,
                }
        finally:
            # 清理临时文件
            if os.path.exists(webm_path):
                os.unlink(webm_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            
    except Exception as e:
        logger.error(f"[AudioHandler] 转录失败: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "text": "",
            "is_final": is_final,
        }
