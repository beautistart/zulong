#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙 (ZULONG) 系统 - 视觉短期记忆管理器

文件：zulong/l1a/vision_short_term_memory.py

功能:
- 环形缓冲区：维护最近 5 秒的视频帧队列 (FPS=30, 约 150 帧)
- 持久化缓存：将缓冲区内的帧保存为视频文件
- 元数据记录：生成 JSON 文件记录录制时间戳、帧数、触发原因

TSD v1.7 对应:
- 4.2 L1-B: 上下文打包 (Context Packaging)
- 4.4 感知预处理：视觉短期记忆
"""

import asyncio
import logging
import cv2
import numpy as np
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

from zulong.l1a.l1a_config import SHARED_VISION_DIR

logger = logging.getLogger(__name__)


class VisionShortTermMemory:
    """
    视觉短期记忆管理器
    
    功能:
    - 环形缓冲区：维护最近 5 秒的视频帧队列
    - 持久化缓存：将缓冲区内的帧保存为视频文件
    - 元数据记录：生成 JSON 文件记录录制信息
    
    使用示例:
    ```python
    memory = VisionShortTermMemory(duration=5, fps=30)
    
    # 持续添加帧
    memory.add_frame(frame)
    
    # 收到请求时保存视频
    video_path, metadata = memory.save_to_file(trigger_reason="user_request")
    ```
    """
    
    def __init__(self, duration: int = 5, fps: int = 30, cache_dir: str = None):
        """
        初始化视觉短期记忆管理器
        
        Args:
            duration: 保留时长 (秒)，默认 5 秒
            fps: 帧率，默认 30 FPS
            cache_dir: 缓存目录 (默认使用 SHARED_VISION_DIR)
        """
        self.duration = duration
        self.fps = fps
        self.max_frames = duration * fps  # 总帧数 (5 秒 * 30 帧 = 150 帧)
        
        # 环形缓冲区：使用 deque 自动丢弃最旧的帧
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=self.max_frames)
        
        # 时间戳缓冲区：记录每帧的时间戳
        self.timestamp_buffer: deque[float] = deque(maxlen=self.max_frames)
        
        # 🎯 缓存目录 (动态路径)
        self.cache_dir = Path(cache_dir) if cache_dir else SHARED_VISION_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 视频文件路径 (共享视觉记忆池)
        self.video_path = self.cache_dir / "recent_clip.mp4"  # 最近 5 秒视频
        self.frame_path = self.cache_dir / "latest_frame.jpg"  # 最新单帧
        self.metadata_path = self.cache_dir / "meta.json"  # 元数据
        
        # 状态标记
        self.is_recording = False
        
        # 🎯 运动检测标志 (事件驱动保存)
        self.motion_detected = False
        self.last_motion_time = 0.0
        
        # 👁️ 后台持久化线程
        self._persist_thread = None
        self._stop_persist = False
        
        logger.info(f"👁️ 视觉短期记忆管理器初始化完成")
        logger.info(f"   - 保留时长：{duration}秒")
        logger.info(f"   - 帧率：{fps} FPS")
        logger.info(f"   - 最大帧数：{self.max_frames}帧")
        logger.info(f"   - 缓存目录：{self.cache_dir}")
        logger.info(f"   - 共享视觉记忆池：{self.video_path}, {self.frame_path}")
    
    def add_frame(self, frame: np.ndarray, timestamp: float = None):
        """
        添加视频帧到环形缓冲区
        
        Args:
            frame: 视频帧 (BGR 格式)
            timestamp: 时间戳 (默认使用当前时间)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 添加到缓冲区 (deque 会自动丢弃最旧的帧)
        self.frame_buffer.append(frame)
        self.timestamp_buffer.append(timestamp)
        
        # 调试日志 (每 30 帧记录一次)
        if len(self.frame_buffer) % 30 == 0:
            logger.debug(f"👁️ 缓冲区帧数：{len(self.frame_buffer)}/{self.max_frames}")
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """
        获取缓冲区信息
        
        Returns:
            Dict: 缓冲区信息
        """
        if not self.frame_buffer:
            return {
                "frame_count": 0,
                "duration_seconds": 0.0,
                "fps": self.fps,
                "is_ready": False
            }
        
        # 计算实际时长
        if len(self.frame_buffer) >= 2:
            duration = self.timestamp_buffer[-1] - self.timestamp_buffer[0]
        else:
            duration = 0.0
        
        return {
            "frame_count": len(self.frame_buffer),
            "duration_seconds": duration,
            "fps": self.fps,
            "is_ready": len(self.frame_buffer) >= 10,  # 至少 10 帧才认为可用
            "oldest_timestamp": self.timestamp_buffer[0] if self.timestamp_buffer else 0,
            "newest_timestamp": self.timestamp_buffer[-1] if self.timestamp_buffer else 0
        }
    
    def save_to_file(self, trigger_reason: str = "manual") -> tuple[Optional[str], Optional[Dict]]:
        """
        将缓冲区内的帧保存为视频文件
        
        Args:
            trigger_reason: 触发原因 ("user_request", "motion_detected", "scheduled")
        
        Returns:
            tuple: (视频文件路径，元数据字典)，如果失败返回 (None, None)
        """
        if len(self.frame_buffer) < 10:
            logger.warning(f"👁️ 缓冲区帧数不足 ({len(self.frame_buffer)}/10)，无法保存视频")
            return None, None
        
        try:
            logger.info(f"👁️ 开始保存视频片段 (触发原因：{trigger_reason})")
            start_time = time.time()
            
            # 获取帧列表
            frames = list(self.frame_buffer)
            timestamps = list(self.timestamp_buffer)
            
            if not frames:
                logger.error("👁️ 缓冲区为空，无法保存")
                return None, None
            
            # 获取帧尺寸
            height, width = frames[0].shape[:2]
            
            # 创建视频编码器 (使用 H.264 编码)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(self.video_path),
                fourcc,
                self.fps,
                (width, height)
            )
            
            if not out.isOpened():
                logger.error("👁️ 无法创建视频写入器")
                return None, None
            
            # 写入帧
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            # 计算实际时长
            duration = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0
            
            # 生成元数据
            metadata = {
                "video_path": str(self.video_path),
                "trigger_reason": trigger_reason,
                "frame_count": len(frames),
                "duration_seconds": duration,
                "fps": self.fps,
                "resolution": {
                    "width": width,
                    "height": height
                },
                "start_timestamp": timestamps[0],
                "end_timestamp": timestamps[-1],
                "created_at": time.time(),
                "created_at_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            
            # 保存元数据到 JSON 文件
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            save_time = time.time() - start_time
            logger.info(f"✅ 视频保存成功：{self.video_path}")
            logger.info(f"   - 帧数：{len(frames)}帧")
            logger.info(f"   - 时长：{duration:.2f}秒")
            logger.info(f"   - 分辨率：{width}x{height}")
            logger.info(f"   - 保存耗时：{save_time:.2f}秒")
            logger.info(f"   - 元数据：{self.metadata_path}")
            
            return str(self.video_path), metadata
            
        except Exception as e:
            logger.error(f"❌ 视频保存失败：{e}", exc_info=True)
            return None, None
    
    def is_fresh(self, max_age: float = 10.0) -> bool:
        """
        检查缓存的视频是否新鲜
        
        Args:
            max_age: 最大年龄 (秒)，默认 10 秒
        
        Returns:
            bool: 是否新鲜
        """
        if not self.metadata_path.exists():
            return False
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            created_at = metadata.get("created_at", 0)
            age = time.time() - created_at
            
            return age <= max_age
        
        except Exception as e:
            logger.error(f"❌ 检查视频新鲜度失败：{e}")
            return False
    
    def get_frames_list(self, num_frames: int = 10) -> List[np.ndarray]:
        """
        获取最近的 N 帧
        
        Args:
            num_frames: 帧数
        
        Returns:
            List: 帧列表
        """
        if not self.frame_buffer:
            return []
        
        # 获取最后的 N 帧
        frames = list(self.frame_buffer)[-num_frames:]
        return frames
    
    def get_frames_list_with_metadata(self, num_frames: int = 30) -> List[Tuple[np.ndarray, dict]]:
        """
        获取最近的帧列表及其元数据
        
        Args:
            num_frames: 获取的帧数量
        
        Returns:
            List[Tuple[np.ndarray, dict]]: (帧，元数据) 对列表
        """
        if not self.frame_buffer:
            return []
        
        # 获取最近的帧和对应的时间戳
        frames = list(self.frame_buffer)[-num_frames:]
        timestamps = list(self.timestamp_buffer)[-num_frames:]
        
        # 组合成 (frame, metadata) 对
        result = []
        for i, frame in enumerate(frames):
            timestamp = timestamps[i] if i < len(timestamps) else time.time()
            result.append((frame, {"timestamp": timestamp, "source": "ring_buffer"}))
        
        return result
    
    def save_motion_frames_to_video(self, motion_frames: List[Tuple[np.ndarray, dict]], 
                                    trigger_reason: str = "motion_detected") -> tuple[Optional[str], Optional[Dict]]:
        """
        将变动帧保存为视频文件 (回溯功能)
        
        Args:
            motion_frames: 变动帧列表，每项包含 (帧，元数据)
            trigger_reason: 触发原因
        
        Returns:
            tuple: (视频文件路径，元数据字典)，如果失败返回 (None, None)
        
        TSD v1.7 对应:
        - 4.4 感知预处理：变动帧回溯保存
        """
        if not motion_frames or len(motion_frames) < 3:
            logger.warning(f"👁️ 变动帧不足 ({len(motion_frames)}/3)，无法保存视频")
            return None, None
        
        try:
            logger.info(f"👁️ 开始保存变动帧视频 (触发原因：{trigger_reason}, 帧数：{len(motion_frames)})")
            start_time = time.time()
            
            # 提取帧和元数据
            frames = [f[0] for f in motion_frames]
            frame_metadata = [f[1] for f in motion_frames]
            
            # 获取帧尺寸
            height, width = frames[0].shape[:2]
            
            # 创建视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(self.video_path),
                fourcc,
                self.fps,
                (width, height)
            )
            
            if not out.isOpened():
                logger.error("👁️ 无法创建视频写入器")
                return None, None
            
            # 写入帧
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            # 计算时长
            timestamps = [m.get("timestamp", 0) for m in frame_metadata]
            duration = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
            
            # 生成元数据
            metadata = {
                "video_path": str(self.video_path),
                "trigger_reason": trigger_reason,
                "frame_count": len(frames),
                "duration_seconds": duration,
                "fps": self.fps,
                "resolution": {
                    "width": width,
                    "height": height
                },
                "start_timestamp": timestamps[0],
                "end_timestamp": timestamps[-1],
                "created_at": time.time(),
                "created_at_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "motion_events": frame_metadata,  # 每个变动帧的详细信息
                "is_motion_backtrack": True  # 标记为变动回溯视频
            }
            
            # 保存元数据
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            save_time = time.time() - start_time
            logger.info(f"✅ 变动帧视频保存成功：{self.video_path}")
            logger.info(f"   - 帧数：{len(frames)}帧")
            logger.info(f"   - 时长：{duration:.2f}秒")
            logger.info(f"   - 保存耗时：{save_time:.2f}秒")
            
            return str(self.video_path), metadata
            
        except Exception as e:
            logger.error(f"❌ 变动帧视频保存失败：{e}", exc_info=True)
            return None, None
    
    def clear(self):
        """清空缓冲区"""
        self.frame_buffer.clear()
        self.timestamp_buffer.clear()
        logger.info("👁️ 视觉短期记忆缓冲区已清空")
    
    def clear_buffer(self):
        """清空缓冲区 (别名方法)"""
        self.clear()
    
    def save_latest_frame(self):
        """
        保存最新帧到共享视觉记忆池 (后台线程调用)
        
        TSD v1.7 对应:
        - 4.4 感知预处理：实时帧缓存
        """
        if not self.frame_buffer:
            logger.warning(f"⚠️ 无法保存最新帧：缓冲区为空")
            return
        
        try:
            # 获取最新帧
            latest_frame = self.frame_buffer[-1]
            
            # 🔥 关键修复：帧质量检查
            if latest_frame is None:
                logger.error(f"❌ 最新帧为 None，无法保存")
                return
            
            if latest_frame.size == 0:
                logger.error(f"❌ 最新帧为空数组，无法保存")
                return
            
            # 🎯 关键调试：检查画面质量
            frame_brightness = np.mean(latest_frame)
            frame_shape = latest_frame.shape
            
            # 🔥 关键修复：增强调试信息
            logger.info(f"📸 [保存最新帧] 亮度={frame_brightness:.2f}, 形状={frame_shape}, 缓冲区帧数={len(self.frame_buffer)}")
            
            if frame_brightness < 20:
                logger.warning(f"⚠️ 保存的画面过暗！平均亮度：{frame_brightness:.2f}")
            elif frame_brightness > 250:
                logger.warning(f"⚠️ 保存的画面过曝！平均亮度：{frame_brightness:.2f}")
            
            # 🔥 关键修复：检查文件路径
            logger.debug(f"📸 保存路径：{self.frame_path}")
            
            # 保存到共享文件夹
            success = cv2.imwrite(str(self.frame_path), latest_frame)
            
            if success:
                logger.info(f"✅ [最新帧] 已保存：{self.frame_path} (亮度={frame_brightness:.2f}, 形状={frame_shape})")
            else:
                logger.error(f"❌ cv2.imwrite 返回 False，保存失败")
            
        except Exception as e:
            logger.error(f"❌ 保存最新帧失败：{e}", exc_info=True)
    
    def save_clip_async(self):
        """
        异步保存视频片段到共享视觉记忆池 (后台线程调用)
        
        TSD v1.7 对应:
        - 4.2 L1-B: 视听信息流回溯
        """
        if len(self.frame_buffer) < 10:
            logger.debug(f"👁️ 缓冲区帧数不足 ({len(self.frame_buffer)}/10)，跳过保存")
            return
        
        try:
            logger.debug(f"👁️ 开始保存视频片段：{self.video_path}")
            
            # 获取帧列表
            frames = list(self.frame_buffer)
            timestamps = list(self.timestamp_buffer)
            
            if not frames:
                return
            
            # 获取帧尺寸
            height, width = frames[0].shape[:2]
            
            # 创建视频编码器 (使用 H.264 编码)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(self.video_path),
                fourcc,
                self.fps,
                (width, height)
            )
            
            if not out.isOpened():
                logger.error("👁️ 无法创建视频写入器")
                return
            
            # 写入帧
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            # 计算实际时长
            duration = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0
            
            # 生成元数据
            metadata = {
                "video_path": str(self.video_path),
                "frame_path": str(self.frame_path),
                "frame_count": len(frames),
                "duration_seconds": duration,
                "fps": self.fps,
                "resolution": {
                    "width": width,
                    "height": height
                },
                "start_timestamp": timestamps[0],
                "end_timestamp": timestamps[-1],
                "created_at": time.time(),
                "created_at_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "trigger_source": "auto_background_save"
            }
            
            # 保存元数据到 JSON 文件
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 视频片段已保存：{self.video_path} ({len(frames)}帧，{duration:.2f}秒)")
            
        except Exception as e:
            logger.error(f"❌ 保存视频片段失败：{e}", exc_info=True)
    
    def _background_persist_loop(self):
        """
        后台持久化循环 (运行在独立线程)
        
        功能:
        - 仅在检测到运动时保存最新帧和视频
        - 完全事件驱动，静止时不保存
        """
        import time
        
        while not self._stop_persist:
            try:
                # 检查是否有运动检测标志
                if self.motion_detected:
                    # 🎯 事件驱动：检测到运动时才保存
                    logger.info(f"👁️ [事件驱动] 检测到运动，保存最新帧和视频")
                    self.save_latest_frame()
                    self.save_clip_async()
                    
                    # 重置标志 (避免重复保存)
                    self.motion_detected = False
                    logger.debug("👁️ 运动保存完成，重置标志")
                
                # 短暂休眠，快速响应运动事件
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"❌ 后台持久化线程错误：{e}")
                time.sleep(0.2)
        
        logger.info("👁️ 后台持久化线程已停止")
    
    def trigger_motion_save(self):
        """
        触发运动保存标志 (由外部调用，当检测到运动时)
        
        TSD v1.7 对应:
        - 4.4 感知预处理：事件驱动保存
        """
        self.motion_detected = True
        self.last_motion_time = time.time()
        logger.info(f"👁️ [VSTM] 触发运动保存标志！motion_detected={self.motion_detected}")
    
    def start_background_persist(self):
        """
        启动后台持久化线程
        
        TSD v1.7 对应:
        - 4.4 感知预处理：实时数据持久化
        """
        if self._persist_thread is not None:
            logger.warning("👁️ 后台持久化线程已在运行")
            return
        
        import threading
        
        self._stop_persist = False
        self._persist_thread = threading.Thread(
            target=self._background_persist_loop,
            name="VisionPersistThread",
            daemon=True
        )
        self._persist_thread.start()
        
        logger.info("👁️ 后台持久化线程已启动")
    
    def stop_background_persist(self):
        """停止后台持久化线程"""
        self._stop_persist = True
        if self._persist_thread:
            self._persist_thread.join(timeout=2.0)
            self._persist_thread = None
        logger.info("👁️ 后台持久化线程已停止")
    
    def extract_keyframes(self, num_keyframes: int = 5) -> List[np.ndarray]:
        """
        提取关键帧序列 (供多模态模型使用)
        
        策略:
        1. 第一帧 (起始状态)
        2. 中间帧 (均匀采样)
        3. 最后一帧 (最新状态)
        4. 运动最大帧 (变化最剧烈的帧)
        
        Args:
            num_keyframes: 关键帧数量 (默认 5 帧，避免显存溢出)
        
        Returns:
            List[np.ndarray]: 关键帧列表
        
        TSD v1.7 对应:
        - 5.2 显存约束：关键帧提取 (3-5 帧)
        """
        if not self.frame_buffer or len(self.frame_buffer) < 2:
            return []
        
        frames = list(self.frame_buffer)
        
        if len(frames) <= num_keyframes:
            # 帧数不足，返回所有帧
            return frames
        
        keyframes = []
        
        # 1. 第一帧
        keyframes.append(frames[0])
        
        # 2. 均匀采样中间帧
        step = len(frames) // (num_keyframes - 1)
        for i in range(1, num_keyframes - 1):
            idx = i * step
            if idx < len(frames):
                keyframes.append(frames[idx])
        
        # 3. 最后一帧
        keyframes.append(frames[-1])
        
        # 4. 运动最大帧 (简化版：使用帧差法检测)
        max_motion_idx = -1
        max_motion_score = 0
        
        for i in range(1, min(len(frames), 30)):  # 只检查前 30 帧
            # 计算帧差
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # 帧差
            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_score = np.sum(diff > 30)  # 阈值 30
            
            if motion_score > max_motion_score:
                max_motion_score = motion_score
                max_motion_idx = i
        
        # 如果检测到显著运动，添加运动最大帧
        if max_motion_idx > 0 and max_motion_idx not in [len(frames) // (num_keyframes - 1) * i for i in range(num_keyframes)]:
            keyframes.append(frames[max_motion_idx])
            logger.debug(f"👁️ 检测到运动最大帧：索引{max_motion_idx}，分数{max_motion_score}")
        
        logger.info(f"👁️ 提取关键帧：{len(keyframes)}帧 (原始{len(frames)}帧)")
        
        return keyframes
    
    def is_fresh(self, max_age: float = 10.0) -> bool:
        """
        检查共享视觉记忆是否新鲜
        
        Args:
            max_age: 最大年龄 (秒)，默认 10 秒
        
        Returns:
            bool: 是否新鲜
        
        TSD v1.7 对应:
        - 4.2 L1-B: 上下文新鲜度检查
        """
        if not self.metadata_path.exists():
            return False
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            created_at = metadata.get("created_at", 0)
            age = time.time() - created_at
            
            is_fresh = age <= max_age
            logger.debug(f"👁️ 视觉记忆新鲜度检查：年龄{age:.1f}秒，最大{max_age}秒 → {'新鲜' if is_fresh else '过期'}")
            
            return is_fresh
        
        except Exception as e:
            logger.error(f"❌ 检查视觉记忆新鲜度失败：{e}")
            return False
    
    def get_shared_vision_info(self) -> Dict[str, Any]:
        """
        获取共享视觉记忆池信息
        
        Returns:
            Dict: 共享视觉记忆池信息
        """
        info = {
            "buffer_info": self.get_buffer_info(),
            "video_exists": self.video_path.exists(),
            "frame_exists": self.frame_path.exists(),
            "metadata_exists": self.metadata_path.exists(),
            "video_path": str(self.video_path),
            "frame_path": str(self.frame_path),
            "metadata_path": str(self.metadata_path)
        }
        
        # 读取元数据
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    info["metadata"] = json.load(f)
            except:
                pass
        
        return info
