# File: zulong/l0/audio/__init__.py
# L0 音频处理模块

"""
祖龙系统 L0 音频处理模块

提供:
- 原生音频解码 (跨平台)
- 音频格式转换
- 音频播放支持
"""

from zulong.l0.audio.native_decoder import NativeAudioDecoder, decode_audio

__all__ = [
    "NativeAudioDecoder",
    "decode_audio"
]
