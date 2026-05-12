# File: zulong/l3/tts_expert_node.py
# L3 TTS 专家节点 - Kokoro-82M 为主 + CosyVoice3-0.5B / edge-tts 备选
# TSD v1.7 规范：TTS 运行在 CPU 上，不占用 GPU 显存

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import logging
import threading

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
    - 中文/英文双语合成
    - 多音色选择
    - 运行在 CPU 上 (不占用 GPU 显存)

    TSD v1.7 对应:
    - 2.2.4 L3: 专家技能池 - TTS 专家输出
    - 4.3 RAG 与专家模块 - 音频输出接口

    引擎优先级:
    1. Kokoro-82M (本地CPU, 82M参数, ~0.3s/次)
    2. CosyVoice3-0.5B (本地CPU, 备选)
    3. edge-tts (微软云端, 备选)
    4. Mock (兜底)

    使用示例:
    ```python
    tts_expert = TTSExpertNode()

    # 同步调用
    result = tts_expert.execute({
        "task_description": "将文本转为语音",
        "text": "晚上好，我是祖龙机器人",
        "voice": "zf_001",           # Kokoro 中文音色
        "speed": 1.0,
        "sample_rate": 24000
    })

    # 获取音频数据
    audio_data = result["audio_data"]  # numpy array
    ```
    """

    # Kokoro 中文音色预设（对应本地 models/hexgrad/Kokoro-82M/voices/ 目录）
    KOKORO_ZH_VOICES = {
        "default": "zf_xiaoyi",  # 默认中文女声（更自然，推荐）
        "zf_xiaobei": "zf_xiaobei",  # 中文女声
        "zf_xiaoni": "zf_xiaoni",    # 中文女声
        "zf_xiaoxiao": "zf_xiaoxiao",  # 中文女声
        "zf_xiaoyi": "zf_xiaoyi",    # 中文女声（推荐，更自然）
        "zm_yunjian": "zm_yunjian",  # 中文男声
        "zm_yunxi": "zm_yunxi",      # 中文男声
        "zm_yunxia": "zm_yunxia",    # 中文男声
        "zm_yunyang": "zm_yunyang",  # 中文男声
    }

    # Kokoro 英文音色
    KOKORO_EN_VOICES = {
        "am_adam": "am_adam",
        "am_bobby": "am_bobby",
        "am_emma": "am_emma",
        "am_fenrir": "am_fenrir",
        "am_liam": "am_liam",
        "am_michael": "am_michael",
        "am_puck": "am_puck",
        "am_santa": "am_santa",
        "bf_emma": "bf_emma",
        "bf_isabella": "bf_isabella",
        "bm_george": "bm_george",
        "bm_lewis": "bm_lewis",
    }

    def __init__(self):
        """初始化 TTS 专家节点"""
        super().__init__(expert_type="EXPERT_TTS")

        # Kokoro 配置 (主引擎)
        self.kokoro_lang = "z"  # 'z'=中文, 'a'=美式英语, 'b'=英式英语
        self.kokoro_voice = "zf_xiaoyi"  # 默认中文女声
        self.kokoro_pipeline = None
        self._kokoro_lock = threading.Lock()

        # CosyVoice 配置 (备选引擎 1)
        self.model_path = Path(r"d:\AI\project\zulong_beta4\models\CosyVoice3-0.5B\FunAudioLLM\Fun-CosyVoice3-0___5B-2512")
        self.ttsfrd_path = Path(r"d:\AI\project\zulong_beta4\models\iic\CosyVoice-ttsfrd")

        # 模型实例 (懒加载)
        self.tts_model = None  # CosyVoice 实例
        self.ttsfrd = None
        self.engine = None  # 当前使用的引擎名称: "kokoro" / "cosyvoice" / "edge-tts" / "mock"

        # 音频参数
        self.default_sample_rate = 24000  # Kokoro 原生 24kHz
        self.default_volume = 1.0
        self.default_speed = 1.0

        # 设备配置 (强制 CPU)
        self.device = "cpu"

        logger.info(f"🎤 TTS 专家节点初始化完成")
        logger.info(f"   - 主引擎：Kokoro-82M (82M参数, CPU <0.3s)")
        logger.info(f"   - 备选：CosyVoice3-0.5B / edge-tts / Mock")
        logger.info(f"   - 设备：{self.device}")
        logger.info(f"   - 默认采样率：{self.default_sample_rate} Hz")
        logger.info(f"   - 默认中文音色：{self.kokoro_voice}")

    # ── 模型加载 ────────────────────────────────────────

    def _load_model(self) -> bool:
        """
        加载 TTS 模型 (懒加载)

        Returns:
            bool: 加载是否成功
        """
        if self.tts_model is not None or self.kokoro_pipeline is not None:
            logger.debug("TTS 模型已加载")
            return True

        try:
            logger.info("📥 正在加载 TTS 模型 (Kokoro-82M 优先)...")
            self._load_model_directly()
            return self.tts_model is not None or self.kokoro_pipeline is not None

        except Exception as e:
            logger.error(f"❌ TTS 模型加载失败：{e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_model_directly(self):
        """
        加载 TTS 模型 (支持 Kokoro / CosyVoice / edge-tts)

        优先级:
        1. Kokoro-82M (超轻量, CPU <0.3s) ← 主推
        2. CosyVoice3-0.5B (备选, 需下载权重)
        3. edge-tts (云端备选, 需网络)
        4. Mock 模式 (兜底)
        """
        # ── 1. 尝试 Kokoro-82M ──────────────────────
        try:
            logger.info("📥 尝试加载 Kokoro-82M...")
            from kokoro import KPipeline

            # 使用线程锁保护, 防止并发加载
            with self._kokoro_lock:
                if self.kokoro_pipeline is None:
                    self.kokoro_pipeline = KPipeline(lang_code=self.kokoro_lang)

            self.engine = "kokoro"
            logger.info("✅ Kokoro-82M 模型加载成功 (82M 参数, CPU 实时推理)")
            return

        except ImportError:
            logger.debug("⚠️ kokoro 库未安装, 尝试 CosyVoice")
            logger.debug("💡 安装方法: pip install kokoro")
        except Exception as e:
            logger.warning(f"⚠️ Kokoro 加载失败: {e}, 尝试 CosyVoice")

        # ── 2. 尝试 CosyVoice ────────────────────────
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice

            logger.info(f"📥 从 {self.model_path} 加载 CosyVoice...")
            self.tts_model = CosyVoice(str(self.model_path))
            self.engine = "cosyvoice"

            logger.info("✅ CosyVoice 模型加载成功")
            return

        except ImportError:
            logger.debug("⚠️ cosyvoice 库未安装, 尝试 edge-tts")
        except Exception as e:
            logger.warning(f"⚠️ CosyVoice 加载失败: {e}, 尝试 edge-tts")

        # ── 3. 尝试 edge-tts ─────────────────────────
        try:
            import edge_tts
            import asyncio

            logger.info("✅ 使用 edge-tts 作为 TTS 引擎")
            logger.info("   - 支持中文、英文等多种语言")
            logger.info("   - 在线合成, 需要网络连接")

            self.tts_model = "edge-tts"
            self.edge_voice = "zh-CN-XiaoxiaoNeural"
            self.edge_sample_rate = 22050
            self.engine = "edge-tts"

            logger.info("✅ edge-tts 配置完成")
            return

        except ImportError:
            logger.warning("⚠️ edge-tts 库未安装")
            logger.info("💡 安装方法: pip install edge-tts")
        except Exception as e:
            logger.warning(f"⚠️ edge-tts 配置失败: {e}")

        # ── 4. 降级为 Mock 模式 ──────────────────────
        logger.warning("⚠️ 降级为 Mock TTS 模式")
        self.tts_model = "MOCK_TTS_MODEL"
        self.engine = "mock"
        logger.info("✅ Mock TTS 模式已启用 (生成的音频为静音)")

    # ── 语音合成 ────────────────────────────────────────

    def _synthesize(self, text: str, **kwargs) -> np.ndarray:
        """
        语音合成核心方法

        Args:
            text: 输入文本
            **kwargs: 其他参数 (voice, speed, volume 等)

        Returns:
            np.ndarray: 音频数据 (归一化到 -1.0 ~ 1.0, 24kHz)
        """
        if self.tts_model is None and self.kokoro_pipeline is None:
            if not self._load_model():
                raise TTSError("TTS 模型未加载")

        try:
            logger.info(f"🎤 开始 TTS 推理: '{text[:50]}...' (长度: {len(text)}, 引擎: {self.engine})")

            voice = kwargs.get('voice', 'default')
            speed = kwargs.get('speed', self.default_speed)
            volume = kwargs.get('volume', self.default_volume)

            # ── Mock 模式 ────────────────────────────
            if self.engine == "mock":
                logger.warning("⚠️ Mock TTS 模式: 生成 1 秒静音")
                duration = 1.0
                audio_data = np.zeros(int(self.default_sample_rate * duration), dtype=np.float32)
                return audio_data

            # ── Kokoro 引擎 ──────────────────────────
            if self.engine == "kokoro":
                return self._synthesize_with_kokoro(text, voice=voice, speed=speed, volume=volume)

            # ── edge-tts 引擎 ────────────────────────
            if self.engine == "edge-tts":
                logger.info(f"🎤 使用 edge-tts 合成: voice={self.edge_voice}")
                return self._synthesize_with_edge_tts(text, voice=self.edge_voice, speed=speed, volume=volume)

            # ── CosyVoice 引擎 ───────────────────────
            if hasattr(self.tts_model, 'inference_sft'):
                result = self.tts_model.inference_sft(
                    tts_text=text,
                    spk_id=voice if voice != 'default' else '中文女',
                    speed=speed
                )
                audio_data = result['tts_speech']
            elif hasattr(self.tts_model, 'generate'):
                audio_data = self.tts_model.generate(text)
            elif hasattr(self.tts_model, 'synthesize'):
                audio_data = self.tts_model.synthesize(text)
            else:
                logger.warning("⚠️ 未找到标准 TTS 接口, 尝试直接调用...")
                audio_data = self.tts_model(text)

            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()

            if volume != 1.0:
                audio_data = audio_data * volume
                audio_data = np.clip(audio_data, -1.0, 1.0)

            logger.info(f"✅ TTS 推理完成, 音频形状: {audio_data.shape}")
            return audio_data

        except Exception as e:
            logger.error(f"❌ TTS 推理失败: {e}")
            import traceback
            traceback.print_exc()
            raise TTSError(f"TTS 推理失败: {e}")

    def _synthesize_with_kokoro(self, text: str, voice: str = "default", speed: float = 1.0, volume: float = 1.0) -> np.ndarray:
        """
        使用 Kokoro-82M 合成语音

        Args:
            text: 输入文本
            voice: 音色 (zf_001~008 中文女声, zm_001~003 中文男声, am_* 英文)
            speed: 语速 (1.0 正常)
            volume: 音量 (1.0 正常)

        Returns:
            np.ndarray: 24kHz float32 音频数据, 归一化到 -1.0~1.0
        """
        try:
            # 映射音色
            if voice == "default" or voice not in {**self.KOKORO_ZH_VOICES, **self.KOKORO_EN_VOICES}:
                voice = self.kokoro_voice

            logger.info(f"🎤 Kokoro 合成中... voice={voice}, speed={speed}x")

            # Kokoro 流式生成, 拼接所有音频块
            audio_chunks = []
            gen = self.kokoro_pipeline(text, voice=voice, speed=speed)

            for result in gen:
                if hasattr(result, 'audio') and result.audio is not None:
                    audio_chunks.append(result.audio)
                elif isinstance(result, dict) and 'audio' in result:
                    audio_chunks.append(result['audio'])

            if not audio_chunks:
                logger.warning("⚠️ Kokoro 未生成音频, 返回静音")
                return np.zeros(int(self.default_sample_rate * max(len(text) * 0.1, 1.0)), dtype=np.float32)

            # 拼接所有音频块
            audio_data = np.concatenate(audio_chunks)

            # 确保是 float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # 应用音量
            if volume != 1.0:
                audio_data = audio_data * volume
                audio_data = np.clip(audio_data, -1.0, 1.0)

            duration = len(audio_data) / self.default_sample_rate
            logger.info(f"✅ Kokoro 合成完成: {duration:.2f}秒, {len(audio_data)} 采样点")

            return audio_data

        except Exception as e:
            logger.error(f"❌ Kokoro 合成失败: {e}", exc_info=True)
            duration = max(len(text) * 0.1, 1.0)
            return np.zeros(int(self.default_sample_rate * duration), dtype=np.float32)

    def _synthesize_with_edge_tts(self, text: str, voice: str = "zh-CN-XiaoxiaoNeural", speed: float = 1.0, volume: float = 1.0) -> np.ndarray:
        """
        使用 edge-tts 合成语音 (同步方法, 内部处理异步)

        Args:
            text: 输入文本
            voice: 语音 ID (默认: zh-CN-XiaoxiaoNeural)
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

            rate_percent = int((speed - 1.0) * 100)
            rate_str = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"

            # 保存 MP3 到临时文件
            mp3_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            mp3_path = mp3_file.name
            mp3_file.close()

            try:
                communicate = edge_tts.Communicate(text, voice, rate=rate_str)

                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, communicate.save(mp3_path))
                        future.result()
                except RuntimeError:
                    asyncio.run(communicate.save(mp3_path))

                if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
                    logger.error("❌ edge-tts 未生成 MP3 文件")
                    return np.zeros(int(self.default_sample_rate * 1), dtype=np.float32)

                logger.info(f"✅ MP3 已保存: {mp3_path} ({os.path.getsize(mp3_path)} 字节)")

                try:
                    from zulong.l0.audio.native_decoder import decode_audio

                    logger.info("🔄 使用系统原生解码器解码 MP3...")
                    with open(mp3_path, 'rb') as f:
                        mp3_bytes = f.read()

                    audio_int16, sample_rate = decode_audio(mp3_bytes)
                    audio_data = audio_int16.astype(np.float32) / 32768.0

                    logger.info(f"✅ 解码完成: {len(audio_data)} 采样点")

                except Exception as e:
                    logger.error(f"❌ MP3 解码失败: {e}")
                    duration = len(text) * 0.3
                    return np.zeros(int(self.default_sample_rate * duration), dtype=np.float32)

            finally:
                if os.path.exists(mp3_path):
                    os.unlink(mp3_path)

            if volume != 1.0:
                audio_data = audio_data * volume
                audio_data = np.clip(audio_data, -1.0, 1.0)

            logger.info(f"✅ edge-tts 合成完成, 形状: {audio_data.shape}")
            return audio_data

        except Exception as e:
            logger.error(f"❌ edge-tts 合成失败: {e}", exc_info=True)
            duration = len(text) * 0.3
            return np.zeros(int(self.default_sample_rate * duration), dtype=np.float32)

    # ── 公开接口 ────────────────────────────────────────

    def execute(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 TTS 任务

        Args:
            task_payload: 任务载荷
                - text: str, 必选, 输入文本
                - voice: str, 可选, 音色 (Kokoro: zf_001~008 / zm_001~003)
                - speed: float, 可选, 语速
                - volume: float, 可选, 音量
                - lang: str, 可选, 语言 ('z'=中文, 'a'=美式英语)

        Returns:
            Dict[str, Any]: 执行结果
                - status: "success" | "error"
                - audio_data: np.ndarray, 24kHz float32
                - text: 输入文本
                - sample_rate: 24000
                - duration: 秒
                - engine: 使用的引擎
                - execution_time: 秒
        """
        start_time = time.time()

        try:
            if not self.validate_payload(task_payload):
                raise TTSError("任务载荷验证失败")

            text = task_payload.get('text', '')
            voice = task_payload.get('voice', 'default')
            speed = task_payload.get('speed', self.default_speed)
            volume = task_payload.get('volume', self.default_volume)

            # 如果指定了语言, 切换 Kokoro pipeline
            lang = task_payload.get('lang', None)
            if lang and lang != self.kokoro_lang and self.engine == "kokoro":
                with self._kokoro_lock:
                    from kokoro import KPipeline
                    self.kokoro_lang = lang
                    self.kokoro_pipeline = KPipeline(lang_code=lang)
                    logger.info(f"🔄 Kokoro 语言已切换为: {lang}")

            if not text or not text.strip():
                raise TTSError("输入文本为空")

            logger.info(f"🎤 TTS 专家开始执行: '{text[:50]}...' (引擎: {self.engine})")

            audio_data = self._synthesize(
                text=text,
                voice=voice,
                speed=speed,
                volume=volume,
            )

            # Kokoro / edge-tts 原生输出 24kHz
            output_sample_rate = self.default_sample_rate

            duration = len(audio_data) / output_sample_rate
            execution_time = time.time() - start_time

            logger.info(f"✅ TTS 任务完成: 时长={duration:.2f}秒, 耗时={execution_time:.2f}秒, 引擎={self.engine}")

            return {
                "status": "success",
                "audio_data": audio_data,
                "text": text,
                "sample_rate": output_sample_rate,
                "duration": duration,
                "engine": self.engine,
                "execution_time": execution_time,
                "voice": voice,
            }

        except TTSError as e:
            logger.error(f"❌ TTS 任务失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "text": task_payload.get('text', ''),
                "execution_time": time.time() - start_time,
            }
        except Exception as e:
            logger.error(f"❌ TTS 任务异常: {e}", exc_info=True)
            return {
                "status": "error",
                "error": f"系统异常: {str(e)}",
                "text": task_payload.get('text', ''),
                "execution_time": time.time() - start_time,
            }

    def execute_streaming(self, task_payload: Dict[str, Any]):
        """
        流式执行 TTS 任务（生成音频块并立即yield）

        Args:
            task_payload: 任务载荷
                - text: str, 必选, 输入文本
                - voice: str, 可选, 音色
                - speed: float, 可选, 语速
                - volume: float, 可选, 音量
                - lang: str, 可选, 语言

        Yields:
            Dict[str, Any]: 音频块或状态
                - type: "audio_chunk" | "complete" | "error"
                - audio_data: np.ndarray (音频块)
                - text: 对应的文本片段
                - error: 错误信息（如果type是error）
        """
        start_time = time.time()

        try:
            if not self.validate_payload(task_payload):
                raise TTSError("任务载荷验证失败")

            text = task_payload.get('text', '')
            voice = task_payload.get('voice', 'default')
            speed = task_payload.get('speed', self.default_speed)
            volume = task_payload.get('volume', self.default_volume)

            # 如果指定了语言, 切换 Kokoro pipeline
            lang = task_payload.get('lang', None)
            if lang and lang != self.kokoro_lang and self.engine == "kokoro":
                with self._kokoro_lock:
                    from kokoro import KPipeline
                    self.kokoro_lang = lang
                    self.kokoro_pipeline = KPipeline(lang_code=lang)
                    logger.info(f"🔄 Kokoro 语言已切换为: {lang}")

            if not text or not text.strip():
                raise TTSError("输入文本为空")

            logger.info(f"🎤 TTS 流式开始: '{text[:50]}...' (引擎: {self.engine})")

            # 根据引擎选择不同的流式策略
            if self.engine == "kokoro":
                yield from self._synthesize_streaming_kokoro(
                    text=text, voice=voice, speed=speed, volume=volume
                )
            else:
                # 其他引擎：一次性生成完整音频后分块yield
                audio_data = self._synthesize(
                    text=text, voice=voice, speed=speed, volume=volume
                )
                # 分块yield（每块约0.5秒音频）
                chunk_size = int(self.default_sample_rate * 0.5)
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    yield {
                        "type": "audio_chunk",
                        "audio_data": chunk,
                        "text": text[max(0, i//self.default_sample_rate*10):min(len(text), (i+len(chunk))//self.default_sample_rate*10)],
                    }

            # 完成信号
            yield {
                "type": "complete",
                "execution_time": time.time() - start_time,
            }

        except TTSError as e:
            logger.error(f"❌ TTS 流式任务失败: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
            }
        except Exception as e:
            logger.error(f"❌ TTS 流式任务异常: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": f"系统异常: {str(e)}",
                "execution_time": time.time() - start_time,
            }

    def _synthesize_streaming_kokoro(self, text: str, voice: str = "default", speed: float = 1.0, volume: float = 1.0):
        """
        Kokoro 流式合成（逐块yield）

        Yields:
            Dict[str, Any]: 音频块
        """
        try:
            # 映射音色
            if voice == "default" or voice not in {**self.KOKORO_ZH_VOICES, **self.KOKORO_EN_VOICES}:
                voice = self.kokoro_voice

            logger.info(f"🎤 Kokoro 流式合成中... voice={voice}, speed={speed}x")

            # Kokoro 流式生成
            gen = self.kokoro_pipeline(text, voice=voice, speed=speed)

            for result in gen:
                audio_chunk = None
                if hasattr(result, 'audio') and result.audio is not None:
                    audio_chunk = result.audio
                elif isinstance(result, dict) and 'audio' in result:
                    audio_chunk = result['audio']

                if audio_chunk is not None:
                    # 确保是 float32
                    if audio_chunk.dtype != np.float32:
                        audio_chunk = audio_chunk.astype(np.float32)

                    # 应用音量
                    if volume != 1.0:
                        audio_chunk = audio_chunk * volume
                        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

                    yield {
                        "type": "audio_chunk",
                        "audio_data": audio_chunk,
                        "text": text,  # 完整文本（Kokoro不返回分段文本）
                    }

        except Exception as e:
            logger.error(f"❌ Kokoro 流式合成失败: {e}", exc_info=True)
            raise TTSError(f"Kokoro 流式合成失败: {e}")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        获取 TTS 专家能力描述

        Returns:
            Dict[str, Any]: 能力描述
        """
        return {
            "expert_type": "EXPERT_TTS",
            "engine": self.engine or "kokoro (优先)",
            "model": "Kokoro-82M",
            "fallback_models": ["CosyVoice3-0.5B", "edge-tts", "Mock"],
            "device": self.device,
            "sample_rate": self.default_sample_rate,
            "languages": ["zh-CN", "en-US"],
            "voices": {
                "zh-CN": list(self.KOKORO_ZH_VOICES.keys()),
                "en-US": list(self.KOKORO_EN_VOICES.keys()),
            },
            "features": [
                "text_to_speech",
                "cpu_inference",
                "multi_voice",
                "speed_control",
                "volume_control",
                "streaming_support",
            ],
            "max_text_length": 5000,
            "is_loaded": self.tts_model is not None or self.kokoro_pipeline is not None,
            "estimated_latency": "~0.3s (Kokoro CPU) / ~3s (CosyVoice CPU) / ~1s (edge-tts API)",
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
            logger.warning(f"⚠️ 文本过长 ({len(text)} 字符), 可能影响质量")

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
    便捷函数: 文本转语音

    Args:
        text: 输入文本
        **kwargs: 其他参数

    Returns:
        Dict[str, Any]: TTS 执行结果
    """
    expert = get_tts_expert()
    return expert.execute({"text": text, **kwargs})
