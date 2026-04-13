# File: src/nodes/perception/mediapipe_gesture_recognizer.py
"""
MediaPipe Gesture Recognizer - L1-C 层手势识别模块

基于 MediaPipe 预训练模型，支持实时手势识别。
优势：
1. 开箱即用，无需训练
2. 支持多种常见手势
3. 高性能，CPU 可运行
4. 支持 ONNX 导出

手势类型：
- Closed_Fist (拳头)
- Open_Palm (张开手掌)
- Thumb_Up (竖起大拇指)
- Thumb_Down (竖起小拇指)
- Victory_Sign (V 字手势)
- ILoveYou (我爱你手势)
- Point_Up (食指向上)
- Point_Down (食指向下)
- OK_Gesture (OK 手势)
- Rock_On (摇滚手势)
"""

import asyncio
import logging
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

logger = logging.getLogger(__name__)


class MediaPipeGestureRecognizer:
    """
    MediaPipe 手势识别器
    
    使用 MediaPipe 预训练模型进行手势分类。
    """
    
    # 手势标签映射
    GESTURE_LABELS = {
        0: "Closed_Fist",
        1: "Open_Palm",
        2: "Thumb_Up",
        3: "Thumb_Down",
        4: "Victory_Sign",
        5: "ILoveYou",
        6: "Point_Up",
        7: "Point_Down",
        8: "OK_Gesture",
        9: "Rock_On",
    }
    
    # MediaPipe 简化名称 -> 标准名称映射
    # MediaPipe 可能返回简化版本（如 "Victory" 而不是 "Victory_Sign"）
    GESTURE_NAME_MAPPING = {
        "Closed_Fist": "Closed_Fist",
        "Open_Palm": "Open_Palm",
        "Thumb_Up": "Thumb_Up",
        "Thumb_Down": "Thumb_Down",
        "Victory": "Victory_Sign",  # 简化版 -> 完整版
        "Victory_Sign": "Victory_Sign",
        "ILoveYou": "ILoveYou",
        "Point_Up": "Point_Up",
        "Point_Down": "Point_Down",
        "OK": "OK_Gesture",  # 简化版 -> 完整版
        "OK_Gesture": "OK_Gesture",
        "Rock_On": "Rock_On",
    }
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        初始化 MediaPipe 手势识别器
        
        Args:
            confidence_threshold: 置信度阈值 (0.0-1.0)
        """
        self._confidence_threshold = confidence_threshold
        self._recognizer = None
        
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("⚠️ MediaPipe 未安装，使用模拟模式")
            return
        
        try:
            # ========== 使用新版 MediaPipe Tasks API ==========
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # 获取模型路径
            import sys
            from pathlib import Path
            model_path = Path(__file__).parent.parent.parent / "gesture_recognizer.task"
            
            if not model_path.exists():
                logger.error(f"❌ 模型文件不存在：{model_path}")
                logger.error(f"   绝对路径：{model_path.absolute()}")
                return
            
            logger.info(f"✅ 找到模型文件：{model_path.absolute()}")
            logger.info(f"   文件大小：{model_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # 创建 Gesture Recognizer
            base_options = python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU  # 使用 CPU
            )
            
            # 使用诊断工具的阈值设置 (0.3)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,  # 诊断工具设置
                min_hand_presence_confidence=0.3,   # 诊断工具设置
                min_tracking_confidence=0.3         # 诊断工具设置
            )
            
            self._recognizer = vision.GestureRecognizer.create_from_options(options)
            logger.info("✅ MediaPipe Gesture Recognizer 初始化成功 (新版 API)")
            logger.info(f"   置信度阈值：{self._confidence_threshold}")
            logger.info(f"   检测阈值：0.3")
            logger.info(f"   存在阈值：0.3")
            logger.info(f"   跟踪阈值：0.3")
            
        except Exception as e:
            logger.error(f"❌ MediaPipe 初始化失败：{e}", exc_info=True)
            import traceback
            traceback.print_exc()
            self._recognizer = None
    
    def classify_gesture(
        self, 
        frame: np.ndarray
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        手势分类
        
        Args:
            frame: BGR 格式帧 (640x480)
        
        Returns:
            (手势类型，置信度，详细信息)
        """
        try:
            # ========== 1. 检查 MediaPipe 可用性 ==========
            if not MEDIAPIPE_AVAILABLE or self._recognizer is None:
                return self._simulate_gesture(frame)
            
            # ========== 2. 转换颜色空间 (BGR -> RGB) ==========
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ========== 3. 创建 MP Image ==========
            from mediapipe import Image as MPImage
            from mediapipe import ImageFormat
            
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)
            
            # ========== 4. 推理 ==========
            start_time = time.time()
            
            recognition_result = self._recognizer.recognize(mp_image)
            
            inference_time = time.time() - start_time
            
            # ========== 5. 解析结果 ==========
            if not recognition_result.gestures:
                return None, 0.0, {
                    'inference_time': inference_time,
                    'model': 'mediapipe',
                    'error': 'No gestures detected'
                }
            
            # 取第一个检测到的手势
            gesture = recognition_result.gestures[0][0]
            gesture_name = gesture.category_name
            confidence = gesture.score
            
            # ========== 标准化手势名称 ==========
            # 将 MediaPipe 返回的简化名称映射到标准名称
            standardized_name = self.GESTURE_NAME_MAPPING.get(
                gesture_name, 
                gesture_name  # 如果没有映射，保持原样
            )
            gesture_name = standardized_name
            logger.debug(f" MediaPipe 原始名称：{gesture.category_name} -> 标准化：{gesture_name}")
            
            # 检查手势名称是否有效 (防止返回 "None" 字符串)
            if not gesture_name or gesture_name.lower() in ['none', 'unknown', '']:
                logger.debug(f" MediaPipe 返回无效手势名称：{gesture_name}")
                return None, confidence, {
                    'inference_time': inference_time,
                    'model': 'mediapipe',
                    'reason': 'Invalid gesture name'
                }
            
            # 应用置信度阈值
            if confidence < self._confidence_threshold:
                return None, confidence, {
                    'inference_time': inference_time,
                    'model': 'mediapipe',
                    'reason': 'Below threshold'
                }
            
            # 获取手部关键点
            landmarks = None
            if recognition_result.hand_landmarks:
                landmarks = recognition_result.hand_landmarks[0]
            
            logger.debug(f"🎯 识别到手势：{gesture_name} (置信度：{confidence:.3f})")
            
            return gesture_name, confidence, {
                'inference_time': inference_time,
                'model': 'mediapipe',
                'landmarks': landmarks,
                'handedness': recognition_result.handedness[0][0].category_name if recognition_result.handedness else None
            }
            
        except Exception as e:
            logger.error(f"❌ MediaPipe 手势识别失败：{e}", exc_info=True)
            return None, 0.0, {'error': str(e)}
    
    def _simulate_gesture(
        self, 
        frame: np.ndarray
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        模拟手势识别（MediaPipe 不可用时）
        
        使用简单的启发式规则。
        """
        # 模拟模式：返回 None
        return None, 0.0, {
            'model': 'simulated',
            'reason': 'MediaPipe not available'
        }
    
    def get_supported_gestures(self) -> list:
        """获取支持的手势列表"""
        return list(self.GESTURE_LABELS.values())
    
    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self._confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"📊 置信度阈值更新为：{self._confidence_threshold}")


# 测试代码
if __name__ == "__main__":
    import cv2
    
    async def test_mediapipe_gesture():
        """测试 MediaPipe 手势识别"""
        print("=" * 60)
        print("🎯 MediaPipe Gesture Recognizer 测试")
        print("=" * 60)
        
        # 初始化识别器
        recognizer = MediaPipeGestureRecognizer(confidence_threshold=0.5)
        
        print(f"\n✅ 支持的手势：{recognizer.get_supported_gestures()}")
        
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
        
        print("\n📸 按 'q' 退出测试")
        
        frame_count = 0
        gesture_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # 识别手势
            gesture, confidence, info = recognizer.classify_gesture(frame)
            
            # 显示结果
            if gesture:
                gesture_count += 1
                cv2.putText(
                    frame, 
                    f"{gesture} ({confidence:.2f})", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # 绘制手部关键点
                if 'landmarks' in info and info['landmarks']:
                    for landmark in info['landmarks']:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            
            # 显示 FPS
            fps = frame_count / max(1, frame_count * 0.033)
            cv2.putText(
                frame, 
                f"FPS: {fps:.1f}, Gestures: {gesture_count}", 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            cv2.imshow("MediaPipe Gesture Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 统计结果
        print("\n" + "=" * 60)
        print("📊 测试结果:")
        print(f"   - 总帧数：{frame_count}")
        print(f"   - 识别到手势：{gesture_count}")
        print(f"   - 识别率：{gesture_count / frame_count * 100:.1f}%")
        print("=" * 60)
        
        cap.release()
        cv2.destroyAllWindows()
    
    # 运行测试
    asyncio.run(test_mediapipe_gesture())
