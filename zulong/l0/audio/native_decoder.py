# File: zulong/l0/audio/native_decoder.py
# L0 原生音频解码器 - 跨平台 MP3 解码

"""
祖龙系统原生音频解码器

使用 subprocess 直接调用 ffmpeg.exe 进行 MP3 解码:
- 无需用户安装任何外部依赖
- 无需配置环境变量
- 100% 可控，所有用户使用同一版本
- 高性能，直接管道传输数据
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NativeAudioDecoder:
    """
    跨平台原生音频解码器
    
    使用项目自带的 ffmpeg.exe 进行解码:
    - 自动定位项目根目录的 bin/ffmpeg.exe
    - 如果找不到，尝试使用系统 PATH 中的 ffmpeg
    - 使用 subprocess 直接调用，高性能管道传输
    
    使用示例:
    ```python
    # 解码 MP3 字节流
    pcm_data, sample_rate = NativeAudioDecoder.decode_mp3_bytes(mp3_bytes)
    
    # 解码 MP3 文件
    pcm_data, sample_rate = NativeAudioDecoder.decode_mp3_file("audio.mp3")
    ```
    """
    
    @staticmethod
    def _find_ffmpeg() -> Optional[str]:
        """
        查找 ffmpeg.exe 的路径
        
        优先级:
        1. 项目 bin/ 目录下的 ffmpeg.exe
        2. 系统 PATH 中的 ffmpeg
        
        Returns:
            Optional[str]: ffmpeg.exe 路径，如果找不到则返回 None
        """
        # 1. 尝试在项目 bin/ 目录下查找
        try:
            current_file = os.path.abspath(__file__)
            # zulong/l0/audio/native_decoder.py -> ../../../../bin/ffmpeg.exe
            project_root = os.path.abspath(os.path.join(
                current_file, "..", "..", "..", ".."
            ))
            ffmpeg_path = os.path.join(project_root, "bin", "ffmpeg.exe")
            
            if os.path.exists(ffmpeg_path):
                logger.info(f"✅ 找到内置 ffmpeg: {ffmpeg_path}")
                return ffmpeg_path
        except Exception as e:
            logger.debug(f"查找项目 ffmpeg 失败：{e}")
        
        # 2. 尝试在系统 PATH 中查找
        try:
            import shutil
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                logger.info(f"✅ 找到系统 ffmpeg: {ffmpeg_path}")
                return ffmpeg_path
        except Exception as e:
            logger.debug(f"查找系统 ffmpeg 失败：{e}")
        
        # 3. 找不到
        logger.warning("⚠️ 未找到 ffmpeg.exe")
        return None
    
    @staticmethod
    def decode_mp3_bytes(mp3_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        解码 MP3 字节流为 PCM
        
        使用 ffmpeg 通过管道直接将 MP3 解码为 PCM 数据:
        ffmpeg -i pipe:0 -f s16le -acodec pcm_s16le -ar 24000 -ac 1 pipe:1
        
        Args:
            mp3_bytes: MP3 格式的二进制数据
        
        Returns:
            Tuple[np.ndarray, int]: (PCM 数据 [int16], 采样率)
            
        Raises:
            RuntimeError: 当解码失败时
        """
        # 1. 查找 ffmpeg 路径
        ffmpeg_path = NativeAudioDecoder._find_ffmpeg()
        if not ffmpeg_path:
            raise RuntimeError(
                "未找到 ffmpeg.exe\n"
                "请从 https://www.gyan.dev/ffmpeg/builds/ 下载\n"
                "并将 ffmpeg.exe 放入项目 bin/ 目录"
            )
        
        # 2. 使用临时文件保存 MP3 数据
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(mp3_bytes)
            
            logger.debug(f"临时 MP3 文件：{tmp_path} ({len(mp3_bytes)} 字节)")
            
            # 3. 使用 ffmpeg 解码为 PCM
            # 命令：ffmpeg -i input.mp3 -f s16le -acodec pcm_s16le -ar 24000 -ac 1 pipe:1
            # -i: 输入文件
            # -f s16le: 输出格式为 16-bit little-endian PCM
            # -acodec pcm_s16le: 音频编码器
            # -ar 24000: 采样率 24kHz (edge-tts 原生采样率)
            # -ac 1: 单声道
            # pipe:1: 输出到 stdout
            
            cmd = [
                ffmpeg_path,
                "-i", tmp_path,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", "24000",  # edge-tts 原生采样率
                "-ac", "1",      # 单声道
                "pipe:1"
            ]
            
            logger.debug(f"执行命令：{' '.join(cmd)}")
            
            # 4. 执行 ffmpeg，通过管道获取 PCM 数据
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False  # 不自动抛出异常，我们自己处理
            )
            
            if result.returncode != 0:
                stderr_msg = result.stderr.decode('utf-8', errors='ignore')
                logger.error(f"ffmpeg 错误：{stderr_msg}")
                raise RuntimeError(f"ffmpeg 解码失败：{stderr_msg[:500]}")
            
            # 5. 解析 PCM 数据
            pcm_bytes = result.stdout
            sample_rate = 24000  # 我们指定的采样率
            
            logger.debug(f"PCM 数据：{len(pcm_bytes)} 字节")
            
            if len(pcm_bytes) == 0:
                raise ValueError("解码后的音频数据为空")
            
            # 转换为 numpy array (int16)
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            
            logger.debug(f"最终 PCM: {len(pcm_array)} 采样点，范围 [{pcm_array.min()}, {pcm_array.max()}]")
            
            return pcm_array, sample_rate
        
        except subprocess.SubprocessError as e:
            logger.error(f"ffmpeg 进程错误：{e}")
            raise RuntimeError(f"ffmpeg 解码失败：{str(e)}")
        except Exception as e:
            logger.error(f"音频解码失败：{e}")
            raise RuntimeError(f"音频解码失败：{str(e)}")
        
        finally:
            # 清理临时文件
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    logger.debug(f"已清理临时文件：{tmp_path}")
                except PermissionError:
                    # Windows 下偶尔会出现文件被占用的延迟，忽略
                    logger.warning(f"无法清理临时文件 {tmp_path}")
    
    @staticmethod
    def decode_mp3_file(mp3_path: str) -> Tuple[np.ndarray, int]:
        """
        解码 MP3 文件为 PCM
        
        Args:
            mp3_path: MP3 文件路径
        
        Returns:
            Tuple[np.ndarray, int]: (PCM 数据 [int16], 采样率)
            
        Raises:
            FileNotFoundError: 当文件不存在时
            RuntimeError: 当解码失败时
        """
        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"MP3 文件不存在：{mp3_path}")
        
        # 读取文件
        with open(mp3_path, 'rb') as f:
            mp3_bytes = f.read()
        
        # 调用字节流解码
        return NativeAudioDecoder.decode_mp3_bytes(mp3_bytes)


# 便捷函数
def decode_audio(mp3_data: bytes) -> Tuple[np.ndarray, int]:
    """
    便捷函数：解码 MP3 字节流为 PCM
    
    Args:
        mp3_data: MP3 格式的二进制数据
    
    Returns:
        Tuple[np.ndarray, int]: (PCM 数据 [int16], 采样率)
    """
    return NativeAudioDecoder.decode_mp3_bytes(mp3_data)
