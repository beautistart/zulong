#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙 (ZULONG) 系统 - VL 模型视觉处理节点

文件：zulong/l1a/reflex/vision_node.py

功能:
- 使用 Qwen3.5-0.8B-Base 模型（带视觉编码器）处理视频帧
- 视频帧 → 结构化文本
- 输出：{"type": "vision", "text": "...", "objects": [...], "timestamp": ...}

TSD v1.7 对应:
- 2.2.2 L1-A - 受控反射
- 4.4 感知预处理 - VL 模型处理
- 5.2 显存约束 - 4bit 量化加载
"""

import asyncio
import logging
import time
import torch
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from transformers import AutoProcessor

from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus
from zulong.l1a.vision_short_term_memory import VisionShortTermMemory
from zulong.l1a.l1a_config import PROJECT_ROOT, MODEL_ROOT, SHARED_VISION_DIR, VIDEO_BACKTRACK_DIR
from zulong.models.container import ModelContainer
from zulong.models.config import ModelID

logger = logging.getLogger(__name__)


class VisionNode:
    """
    VL 模型视觉处理节点
    
    功能:
    - 加载 Qwen3.5-0.8B-Base 模型（使用视觉编码器）
    - 将视频帧转换为结构化文本
    - 支持物体识别、场景理解、人脸检测
    - 支持动态卸载/加载（显存管理）
    
    使用示例:
    ```python
    node = VisionNode()
    await node.initialize()
    
    # 处理视频帧
    result = await node.process(frame)
    
    # 输出结构化文本
    print(result["text"])
    print(result["objects"])
    ```
    """
    
    # 🎯 模型配置 (动态路径)
    MODEL_PATH = MODEL_ROOT / "Qwen" / "Qwen3___5-0___8B-Base"  # Qwen3.5-0.8B-Base (带视觉编码器)
    MAX_LENGTH = 512  # 最大生成长度
    
    def __init__(self):
        """初始化视觉节点"""
        self.model_container = None
        self.vl_model = None  # InternVL2_5-1B 多模态模型
        self.processor: Optional[AutoProcessor] = None
        self.is_loaded = False
        # L1-A 运行在 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 👁️ 视觉短期记忆管理器 (环形缓冲区)
        self.short_term_memory = VisionShortTermMemory(
            duration=5,  # 5 秒
            fps=30,
            cache_dir=str(SHARED_VISION_DIR)  # 🎯 使用动态路径
        )
        
        logger.info(f"👁️ 视觉节点初始化完成（设备：{self.device}）")
        logger.info(f"   - 视觉短期记忆：已启用 (5 秒/30FPS)")
    
    async def initialize(self) -> bool:
        """
        从 ModelContainer 获取已加载的 InternVL2_5-1B 模型
        
        Returns:
            bool: 加载是否成功
        """
        if self.is_loaded:
            logger.info("✅ 模型已加载，跳过初始化")
            return True
        
        try:
            logger.info("📥 从 ModelContainer 获取 InternVL2_5-1B 模型...")
            
            # 获取 ModelContainer 单例
            self.model_container = ModelContainer()
            
            # 从常驻模型中获取 L1_PERCEPTION 模型 (InternVL2_5-1B)
            if ModelID.L1_PERCEPTION not in self.model_container.resident_models:
                logger.error("❌ ModelContainer 中没有 L1_PERCEPTION 模型")
                return False
            
            # 获取 RealModelLoader 实例
            loader = self.model_container.resident_models[ModelID.L1_PERCEPTION]
            
            # 从 loader 获取模型和 processor
            self.vl_model = loader.model
            self.processor = loader.processor
            
            if self.vl_model is None or self.processor is None:
                logger.error("❌ 模型或 processor 为空")
                return False
            
            self.is_loaded = True
            logger.info("✅ InternVL2_5-1B 模型获取成功 (GPU)")
            logger.info("✅ 视觉编码器已启用，支持图像理解")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型获取失败：{e}", exc_info=True)
            return False
    
    async def unload(self):
        """卸载模型（释放显存）"""
        if self.model:
            logger.info("📤 正在卸载视觉模型...")
            del self.model
            self.model = None
            self.is_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ 视觉模型已卸载")
    
    def build_vision_prompt(self, frame: np.ndarray, context: Dict[str, Any] = None) -> str:
        """
        构建视觉理解提示词
        
        Args:
            frame: 视频帧
            context: 上下文信息
        
        Returns:
            str: 提示词
        """
        # 获取帧信息
        height, width = frame.shape[:2]
        
        # 构建描述
        prompt = f"""你是一个视觉分析专家。请分析这张摄像头捕获的图像：

图像信息:
- 分辨率：{width}x{height}
- 时间：{context.get('timestamp', '未知')}

请详细描述:
1. 场景中有哪些物体
2. 是否有人类，如果有，描述其位置和动作
3. 场景的类型（客厅、厨房、办公室等）
4. 任何显著的变化或异常

回答格式:
场景：[场景类型]
物体：[物体列表]
人物：[有/无，位置和动作]
描述：[详细描述]
"""
        return prompt
    
    async def analyze_scene(self, frame: np.ndarray, prompt: str = None) -> Dict[str, Any]:
        """
        分析场景（使用视觉编码器）
        
        Args:
            frame: 视频帧 (BGR 格式)
            prompt: 分析问题
        
        Returns:
            dict: 分析结果
        """
        if not self.is_loaded:
            if not await self.initialize():
                return {
                    "success": False,
                    "error": "模型未加载",
                    "text": "",
                    "objects": [],
                    "timestamp": asyncio.get_event_loop().time()
                }
        
        try:
            # 转换 BGR 到 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 使用默认 prompt
            if prompt is None:
                prompt = "请详细描述这个场景中的内容，包括物体、人物、颜色等。"
            
            # Qwen3.5-VL Base 模型没有 chat_template，需要手动构建 prompt
            # 格式: <|im_start|>user
            # <|image_pad|><|im_end|>
            # <|im_start|>user
            # {prompt}<|im_end|>
            # <|im_start|>assistant
            #
            
            # 检查 processor 是否有 chat_template
            if self.processor.chat_template is None:
                # 手动构建 prompt
                image_token = getattr(self.processor, 'image_token', '<|image_pad|>')
                text = f"<|im_start|>user\n{image_token}\n\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                # 使用 processor 的 chat_template
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame_rgb},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # 处理图像和文本
            inputs = self.processor(
                text=[text],
                images=[frame_rgb],
                return_tensors="pt",
                padding=True
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
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
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # 提取结构化信息
            result = self._parse_result(generated_text)
            result["success"] = True
            result["timestamp"] = asyncio.get_event_loop().time()
            
            logger.debug(f"👁️ 视觉分析完成：{len(result.get('objects', []))} 个物体")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 视觉处理失败：{e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "objects": [],
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理视频帧，生成结构化文本
        
        Args:
            frame: 视频帧
            context: 上下文信息
        
        Returns:
            dict: 结构化文本结果
        """
        # 调用 analyze_scene (使用视觉编码器)
        return await self.analyze_scene(frame, self.build_vision_prompt(frame, context))
    
    def _parse_result(self, text: str) -> Dict[str, Any]:
        """
        解析模型输出，提取结构化信息
        
        Args:
            text: 模型生成的文本
        
        Returns:
            dict: 结构化结果
        """
        result = {
            "text": text,
            "scene": "",
            "objects": [],
            "person": None
        }
        
        # 简单解析（实际应该用更复杂的 NLP）
        lines = text.split('\n')
        for line in lines:
            if line.startswith("场景："):
                result["scene"] = line.replace("场景：", "").strip()
            elif line.startswith("物体："):
                objects_str = line.replace("物体：", "").strip()
                result["objects"] = [obj.strip() for obj in objects_str.split(',')]
            elif line.startswith("人物："):
                result["person"] = line.replace("人物：", "").strip()
        
        return result
    
    async def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测物体（简化版，使用 OpenCV）
        
        Args:
            frame: 视频帧
        
        Returns:
            List[Dict]: 物体列表
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用 Haar 级联检测人脸（示例）
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        objects = []
        for (x, y, w, h) in faces:
            objects.append({
                "type": "face",
                "bbox": [x, y, w, h],
                "confidence": 0.9
            })
        
        return objects
    
    async def push_frame(self, frame: np.ndarray):
        """
        推送视频帧到短期记忆缓冲区
        
        Args:
            frame: 视频帧 (BGR 格式)
        """
        timestamp = time.time()
        self.short_term_memory.add_frame(frame, timestamp)
    
    async def save_motion_backtrack_video(self, trigger_reason: str = "motion_detected"):
        """
        保存变动帧回溯视频 (从 ring buffer 保存最近 N 帧，最长 5 秒)
        
        Args:
            trigger_reason: 触发原因
        
        Returns:
            tuple: (视频路径，元数据), 失败返回 (None, None)
        
        TSD v1.7 对应:
        - 4.4 感知预处理：变动帧回溯保存
        """
        try:
            # 🎯 关键修改：从 ring buffer 获取最近 N 帧，限制最长 5 秒 (150 帧@30FPS)
            frames_to_save = self.short_term_memory.get_frames_list(num_frames=150)  # 5 秒 = 150 帧
            
            if not frames_to_save or len(frames_to_save) < 3:
                logger.warning(f"👁️ Ring buffer 帧数不足 ({len(frames_to_save)}/3)，无法保存回溯视频")
                return None, None
            
            logger.info(f"👁️ 保存 ring buffer 视频：{len(frames_to_save)}帧")
            
            # 🎯 关键修改：从短期记忆中获取带时间戳的帧
            motion_frames = self.short_term_memory.get_frames_list_with_metadata(num_frames=150)
            
            # 如果获取失败，使用简化的方式
            if not motion_frames:
                motion_frames = [
                    (frame, {"timestamp": time.time(), "source": "ring_buffer"})
                    for frame in frames_to_save
                ]
            
            # 使用 VisionShortTermMemory 保存
            video_path, metadata = self.short_term_memory.save_motion_frames_to_video(
                motion_frames=motion_frames,
                trigger_reason=trigger_reason
            )
            
            if video_path:
                logger.info(f"✅ 变动帧视频已保存：{video_path}")
                
                # 🎯 关键修改：不清空 ring buffer，让摄像头持续推送帧
                # 这样下次检测运动时，ring buffer 中已经有足够的帧了
                # self.short_term_memory.clear_buffer()
            else:
                logger.error("❌ 变动帧视频保存失败")
            
            # 发布 VISION_DATA_READY 事件
            vision_event = ZulongEvent(
                type=EventType.VISION_DATA_READY,
                source="vision_node",
                priority=EventPriority.HIGH,
                payload={
                    "video_path": video_path,
                    "duration": metadata["duration_seconds"],
                    "frame_count": metadata["frame_count"],
                    "trigger_reason": trigger_reason,
                    "metadata": metadata,
                    "is_motion_backtrack": True
                }
            )
            
            event_bus.publish(vision_event)
            logger.info(f"📨 发布 VISION_DATA_READY 事件 (变动回溯)")
            
            return video_path, metadata
        
        except Exception as e:
            logger.error(f"❌ 变动帧回溯视频保存失败：{e}", exc_info=True)
            return None, None
    
    async def handle_vision_request(self, event: ZulongEvent):
        """
        处理视觉捕获请求事件
        
        Args:
            event: SENSOR_VISION_REQUEST 事件
        """
        logger.info("👁️ 收到视觉捕获请求")
        
        # 保存视频片段
        trigger_reason = event.payload.get("reason", "user_request")
        video_path, metadata = self.short_term_memory.save_to_file(trigger_reason=trigger_reason)
        
        if video_path:
            # 发布 VISION_DATA_READY 事件
            vision_event = ZulongEvent(
                type=EventType.VISION_DATA_READY,
                source="vision_node",
                priority=EventPriority.HIGH,
                payload={
                    "video_path": video_path,
                    "duration": metadata["duration_seconds"],
                    "frame_count": metadata["frame_count"],
                    "trigger_reason": trigger_reason,
                    "metadata": metadata
                }
            )
            
            logger.info(f"👁️ 发布 VISION_DATA_READY 事件：{video_path}")
            event_bus.publish(vision_event)
        else:
            logger.warning("👁️ 视觉数据保存失败，无法发布 VISION_DATA_READY 事件")
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """
        获取缓冲区信息
        
        Returns:
            Dict: 缓冲区信息
        """
        return self.short_term_memory.get_buffer_info()
    
    def get_keyframes(self, num_keyframes: int = 5) -> List[np.ndarray]:
        """
        提取关键帧序列 (供多模态模型使用)
        
        Args:
            num_keyframes: 关键帧数量 (默认 5 帧)
        
        Returns:
            List[np.ndarray]: 关键帧列表
        """
        return self.short_term_memory.extract_keyframes(num_keyframes)
    
    def get_shared_vision_info(self) -> Dict[str, Any]:
        """
        获取共享视觉记忆池信息
        
        Returns:
            Dict: 共享视觉记忆池信息
        """
        return self.short_term_memory.get_shared_vision_info()
