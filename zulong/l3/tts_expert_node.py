# File: zulong/l3/tts_expert_node.py
# L3 TTS 专家节点 - CosyVoice3-0.5B 语音合成
# TSD v1.7 规范：TTS 运行在 CPU 上，使用 safetensors 格式

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import logging

from zulong.l3.base_expert_node import BaseExpertNode, ExpertExecutionError
from zulong.models.container import ModelContainer
from zulong.models.config import ModelID

logger = logging.getLogger(__name__)


class TTSError(ExpertExecutionError):
    """TTS 执行异常"""
    pass


class TTSExpertNode(BaseExpertNode):
    """
    L3 TTS 专家节点
    
    功能:
    - 文本转语音 (TTS)
    - 支持零样本语音克隆
    - 支持多语言 (中文、英文、方言)
    - 运行在 CPU 上 (不占用 GPU 显存)
    
    TSD v1.7 对应:
    - 2.2.4 L3: 专家技能池 - TTS 专家输出
    - 4.3 RAG 与专家模块 - 音频输出接口
    - 5.3 TTS 约束：CPU 运行，safetensors 格式
    
    使用示例:
    ```python
    tts_expert = TTSExpertNode()
    
    # 同步调用
    result = tts_expert.execute({
        "task_description": "将文本转为语音",
        "text": "晚上好，我是祖龙机器人",
        "speaker": "default",  # 或使用参考音频
        "sample_rate": 22050
    })
    
    # 获取音频数据
    audio_data = result["audio_data"]  # numpy array
    ```
    """
    
    def __init__(self):
        """初始化 TTS 专家节点"""
        super().__init__(expert_type="EXPERT_TTS")
        
        # TTS 模型配置
        _project_root = Path(__file__).resolve().parent.parent.parent
        self.model_path = _project_root / "models" / "CosyVoice3-0.5B" / "FunAudioLLM" / "Fun-CosyVoice3-0___5B-2512"
        self.ttsfrd_path = _project_root / "models" / "iic" / "CosyVoice-ttsfrd"
        
        # 模型实例 (懒加载)
        self.tts_model = None
        self.ttsfrd = None
        
        # 音频参数
        self.default_sample_rate = 22050
        self.default_volume = 1.0
        self.default_speed = 1.0
        
        # 设备配置 (TSD v1.7: TTS 强制 CPU)
        self.device = "cpu"
        
        logger.info(f"🎤 TTS 专家节点初始化完成")
        logger.info(f"   - 模型路径：{self.model_path}")
        logger.info(f"   - 设备：{self.device}")
        logger.info(f"   - 默认采样率：{self.default_sample_rate} Hz")
    
    def _load_model(self) -> bool:
        """
        加载 TTS 模型 (懒加载)
        
        Returns:
            bool: 加载是否成功
        """
        if self.tts_model is not None:
            logger.debug("TTS 模型已加载")
            return True
        
        try:
            logger.info("📥 正在加载 CosyVoice3-0.5B TTS 模型...")
            
            # 直接加载 CosyVoice 模型 (不通过 ModelContainer)
            self._load_model_directly()
            
            return self.tts_model is not None
            
        except Exception as e:
            logger.error(f"❌ TTS 模型加载失败：{e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_model_directly(self):
        """
        加载 TTS 模型 (支持 CosyVoice 和 edge-tts)
        
        优先级:
        1. CosyVoice (如果已安装)
        2. edge-tts (如果已安装)
        3. Mock 模式 (降级方案)
        """
        # 1. 尝试 CosyVoice
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
            
            logger.info(f"📥 从 {self.model_path} 加载 CosyVoice...")
            self.tts_model = CosyVoice(str(self.model_path))
            
            logger.info("✅ CosyVoice 模型加载成功")
            return
            
        except ImportError:
            logger.debug("⚠️ cosyvoice 库未安装，尝试 edge-tts")
        except Exception as e:
            logger.warning(f"⚠️ CosyVoice 加载失败：{e}，尝试 edge-tts")
        
        # 2. 尝试 edge-tts
        try:
            import edge_tts
            import asyncio
            
            logger.info("✅ 使用 edge-tts 作为 TTS 引擎")
            logger.info("   - 支持中文、英文等多种语言")
            logger.info("   - 在线合成，需要网络连接")
            
            # 设置 edge-tts 语音 (中文女声)
            self.tts_model = "edge-tts"
            self.edge_voice = "zh-CN-XiaoxiaoNeural"  # 中文女声
            self.edge_sample_rate = 22050
            
            logger.info("✅ edge-tts 配置完成")
            return
            
        except ImportError:
            logger.warning("⚠️ edge-tts 库未安装")
            logger.info("💡 安装方法：pip install edge-tts")
        except Exception as e:
            logger.warning(f"⚠️ edge-tts 配置失败：{e}")
        
        # 3. 降级为 Mock 模式
        logger.warning("⚠️ 降级为 Mock TTS 模式")
        self.tts_model = "MOCK_TTS_MODEL"
        logger.info("✅ Mock TTS 模式已启用 (生成的音频为静音)")
    
    def _load_ttsfrd(self):
        """
        加载 TTSFRD (文本前端，可选)
        
        TTSFRD 提供更好的文本处理能力，但不是必需的
        """
        if self.ttsfrd is not None:
            return
        
        try:
            if not self.ttsfrd_path.exists():
                logger.warning(f"⚠️ TTSFRD 模型不存在：{self.ttsfrd_path}")
                return
            
            # 暂时不加载，需要安装特定库
            logger.info("⚠️ TTSFRD 需要额外依赖，暂时跳过")
            
        except Exception as e:
            logger.warning(f"⚠️ TTSFRD 加载失败：{e}")
    
    def _synthesize(self, text: str, **kwargs) -> np.ndarray:
        """
        语音合成核心方法
        
        Args:
            text: 输入文本
            **kwargs: 其他参数 (speaker, speed, volume 等)
        
        Returns:
            np.ndarray: 音频数据 (归一化到 -1.0 ~ 1.0)
        """
        if self.tts_model is None:
            if not self._load_model():
                raise TTSError("TTS 模型未加载")
        
        try:
            logger.info(f"🎤 开始 TTS 推理：'{text[:50]}...' (长度：{len(text)})")
            
            # 提取参数
            speaker = kwargs.get('speaker', 'default')
            speed = kwargs.get('speed', self.default_speed)
            volume = kwargs.get('volume', self.default_volume)
            
            # 🎯 Mock 模式处理
            if self.tts_model == "MOCK_TTS_MODEL":
                logger.warning("⚠️ Mock TTS 模式：生成 1 秒静音")
                # 生成 1 秒静音 (22050 Hz 采样率)
                duration = 1.0  # 秒
                audio_data = np.zeros(int(self.default_sample_rate * duration), dtype=np.float32)
                return audio_data
            
            # 🎯 edge-tts 处理
            elif self.tts_model == "edge-tts":
                logger.info(f"🎤 使用 edge-tts 合成：voice={self.edge_voice}")
                return self._synthesize_with_edge_tts(text, voice=self.edge_voice, speed=speed, volume=volume)
            
            # 🎯 CosyVoice 处理
            else:
                # 调用模型生成
                if hasattr(self.tts_model, 'inference_sft'):
                    # CosyVoice 的标准调用方式
                    result = self.tts_model.inference_sft(
                        tts_text=text,
                        spk_id=speaker if speaker != 'default' else '中文女',
                        speed=speed
                    )
                    audio_data = result['tts_speech']
                    
                elif hasattr(self.tts_model, 'generate'):
                    # 通用 generate 接口
                    audio_data = self.tts_model.generate(text)
                    
                elif hasattr(self.tts_model, 'synthesize'):
                    # synthesize 接口
                    audio_data = self.tts_model.synthesize(text)
                    
                else:
                    # 尝试直接调用 (兼容模式)
                    logger.warning("⚠️ 未找到标准 TTS 接口，尝试直接调用...")
                    audio_data = self.tts_model(text)
                
                # 处理音频数据
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()
                
                # 应用音量增益
                if volume != 1.0:
                    audio_data = audio_data * volume
                    # 限幅避免削波
                    audio_data = np.clip(audio_data, -1.0, 1.0)
            
            logger.info(f"✅ TTS 推理完成，音频形状：{audio_data.shape}")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ TTS 推理失败：{e}")
            import traceback
            traceback.print_exc()
            raise TTSError(f"TTS 推理失败：{e}")
    
    def _synthesize_with_edge_tts(self, text: str, voice: str = "zh-CN-XiaoxiaoNeural", speed: float = 1.0, volume: float = 1.0) -> np.ndarray:
        """
        使用 edge-tts 合成语音 (同步方法，内部处理异步)
        
        Args:
            text: 输入文本
            voice: 语音 ID (默认：zh-CN-XiaoxiaoNeural)
            speed: 语速 (1.0 为正常)
            volume: 音量 (1.0 为正常)
        
        Returns:
            np.ndarray: 音频数据
        """
        import edge_tts
        import tempfile
        import os
        import asyncio
        
        try:
            logger.info(f"🎤 edge-tts 合成中... voice={voice}, speed={speed}x")
            
            # edge-tts 的 rate 格式需要 +/- 前缀，例如 +0%, -10%
            rate_percent = int((speed - 1.0) * 100)
            if rate_percent >= 0:
                rate_str = f"+{rate_percent}%"
            else:
                rate_str = f"{rate_percent}%"
            
            # 🎯 关键修复：edge-tts stream 返回的是 MP3 格式，不是 PCM!
            # 必须先保存为 MP3 文件，然后用 pydub 解码为 PCM
            
            # 步骤 1: 保存 MP3 到临时文件
            mp3_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            mp3_path = mp3_file.name
            mp3_file.close()
            
            try:
                # 使用 save 方法保存 MP3
                communicate = edge_tts.Communicate(text, voice, rate=rate_str)
                
                # 🎯 关键修复：处理事件循环嵌套问题
                try:
                    # 尝试在运行中的事件循环中执行
                    loop = asyncio.get_running_loop()
                    # 使用线程池运行异步代码
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, communicate.save(mp3_path))
                        future.result()
                except RuntimeError:
                    # 没有运行中的事件循环，直接运行
                    asyncio.run(communicate.save(mp3_path))
                
                if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
                    logger.error("❌ edge-tts 未生成 MP3 文件")
                    return np.zeros(int(self.default_sample_rate * 1), dtype=np.float32)
                
                logger.info(f"✅ MP3 已保存：{mp3_path} ({os.path.getsize(mp3_path)} 字节)")
                
                # 步骤 2: 解码 MP3 为 PCM
                # 🎯 关键修复：edge-tts 返回 MP3 格式，需要使用系统原生解码器
                
                try:
                    # 使用祖龙系统原生音频解码器 (自动适配 Windows Media Foundation / Linux GStreamer / macOS CoreAudio)
                    from zulong.l0.audio.native_decoder import decode_audio
                    
                    logger.info("🔄 使用系统原生解码器解码 MP3...")
                    
                    # 读取 MP3 文件
                    with open(mp3_path, 'rb') as f:
                        mp3_bytes = f.read()
                    
                    # 解码为 PCM
                    audio_int16, sample_rate = decode_audio(mp3_bytes)
                    
                    logger.info(f"   - 采样率：{sample_rate} Hz")
                    logger.info(f"   - 数据点数：{len(audio_int16)}")
                    logger.info(f"   - 范围：[{audio_int16.min()}, {audio_int16.max()}]")
                    
                    # 转换为 float32 (-1.0 ~ 1.0)
                    audio_data = audio_int16.astype(np.float32) / 32768.0
                    
                    logger.info(f"✅ 解码完成：{len(audio_data)} 采样点")
                    
                except Exception as e:
                    logger.error(f"❌ MP3 解码失败：{e}")
                    logger.info("💡 请确认已安装 audioread: pip install audioread")
                    logger.info("⚠️ 降级为静音输出")
                    duration = len(text) * 0.3
                    return np.zeros(int(self.default_sample_rate * duration), dtype=np.float32)
                    
            finally:
                # 清理临时文件
                if os.path.exists(mp3_path):
                    os.unlink(mp3_path)
                    logger.debug(f"已清理临时文件：{mp3_path}")
            
            logger.info(f"   - 原始数据类型：{audio_data.dtype}")
            logger.info(f"   - 数据点数：{len(audio_data)}")
            logger.info(f"   - 最小值：{audio_data.min():.4f}, 最大值：{audio_data.max():.4f}")
            
            # 应用音量
            if volume != 1.0:
                audio_data = audio_data * volume
                audio_data = np.clip(audio_data, -1.0, 1.0)
            
            logger.info(f"✅ edge-tts 音频转换完成，形状：{audio_data.shape}")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ edge-tts 合成失败：{e}", exc_info=True)
            # 返回静音音频作为降级
            duration = len(text) * 0.3  # 估算时长
            return np.zeros(int(self.default_sample_rate * duration), dtype=np.float32)
    
    def execute(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 TTS 任务
        
        Args:
            task_payload: 任务载荷
                - text: str, 必选，输入文本
                - speaker: str, 可选，说话人 ID
                - speed: float, 可选，语速
                - volume: float, 可选，音量
        
        Returns:
            Dict[str, Any]: 执行结果
                - status: str, "success" | "error"
                - audio_data: np.ndarray, 音频数据 (成功时)
                - text: str, 输入文本
                - sample_rate: int, 采样率 (固定 24000Hz, edge-tts 原生采样率)
                - duration: float, 音频时长 (秒)
                - error: str, 错误信息 (失败时)
        """
        start_time = time.time()
        
        try:
            # 验证载荷
            if not self.validate_payload(task_payload):
                raise TTSError("任务载荷验证失败")
            
            # 提取参数
            text = task_payload.get('text', '')
            speaker = task_payload.get('speaker', 'default')
            speed = task_payload.get('speed', self.default_speed)
            volume = task_payload.get('volume', self.default_volume)
            
            if not text or not text.strip():
                raise TTSError("输入文本为空")
            
            logger.info(f"🎤 TTS 专家开始执行：'{text[:50]}...'")
            
            # 执行 TTS 推理
            audio_data = self._synthesize(
                text=text,
                speaker=speaker,
                speed=speed,
                volume=volume
            )
            
            # 🎯 关键修改：使用 edge-tts 原生采样率 24000Hz
            # 重采样工作交给 SpeakerDevice 处理
            edge_sample_rate = 24000  # edge-tts 固定输出 24kHz
            
            # 计算时长 (使用实际采样率)
            duration = len(audio_data) / edge_sample_rate
            
            # 执行时间
            execution_time = time.time() - start_time
            
            logger.info(f"✅ TTS 任务完成：时长={duration:.2f}秒，耗时={execution_time:.2f}秒")
            
            return {
                "status": "success",
                "audio_data": audio_data,
                "text": text,
                "sample_rate": edge_sample_rate,  # 返回 edge-tts 原生采样率
                "duration": duration,
                "execution_time": execution_time,
                "speaker": speaker
            }
            
        except TTSError as e:
            logger.error(f"❌ TTS 任务失败：{e}")
            return {
                "status": "error",
                "error": str(e),
                "text": task_payload.get('text', ''),
                "execution_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"❌ TTS 任务异常：{e}", exc_info=True)
            return {
                "status": "error",
                "error": f"系统异常：{str(e)}",
                "text": task_payload.get('text', ''),
                "execution_time": time.time() - start_time
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        获取 TTS 专家能力描述
        
        Returns:
            Dict[str, Any]: 能力描述
        """
        return {
            "expert_type": "EXPERT_TTS",
            "model": "CosyVoice3-0.5B",
            "device": self.device,
            "sample_rates": [16000, 22050, 24000, 48000],
            "languages": ["zh-CN", "en-US", "zh-TW", "zh-HK", "ja-JP", "ko-KR"],
            "features": [
                "text_to_speech",
                "zero_shot_voice_clone",
                "multi_language",
                "speed_control",
                "volume_control"
            ],
            "max_text_length": 5000,
            "default_sample_rate": self.default_sample_rate,
            "is_loaded": self.tts_model is not None
        }
    
    def validate_payload(self, task_payload: Dict[str, Any]) -> bool:
        """
        验证 TTS 任务载荷
        
        Args:
            task_payload: 任务载荷
            
        Returns:
            bool: 载荷是否有效
        """
        if not isinstance(task_payload, dict):
            logger.error("TTS 任务载荷必须是字典")
            return False
        
        if "text" not in task_payload:
            logger.error("TTS 任务载荷缺少 text 字段")
            return False
        
        text = task_payload["text"]
        if not isinstance(text, str) or not text.strip():
            logger.error("TTS 输入文本必须是非空字符串")
            return False
        
        if len(text) > 5000:
            logger.warning(f"⚠️ 文本过长 ({len(text)} 字符),可能影响质量")
        
        return True


# 导出单例
_tts_expert_instance: Optional[TTSExpertNode] = None


def get_tts_expert() -> TTSExpertNode:
    """
    获取 TTS 专家单例
    
    Returns:
        TTSExpertNode: TTS 专家实例
    """
    global _tts_expert_instance
    if _tts_expert_instance is None:
        _tts_expert_instance = TTSExpertNode()
    return _tts_expert_instance


# 便捷函数
def synthesize_speech(text: str, **kwargs) -> Dict[str, Any]:
    """
    便捷函数：文本转语音
    
    Args:
        text: 输入文本
        **kwargs: 其他参数
    
    Returns:
        Dict[str, Any]: TTS 执行结果
    """
    expert = get_tts_expert()
    return expert.execute({"text": text, **kwargs})
