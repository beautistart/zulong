# File: zulong/l1c/__init__.py
# L1-C 层：静默视觉注意层 (Silent Visual Attention)

"""
L1-C 层：静默视觉注意层

职责 (TSD v1.8 三层注意力机制):
- 持续运行光流法检测
- 默认不生成事件（静默模式）
- 仅在状态翻转时触发 AttentionEvent
- 持续写入共享内存数据

核心组件:
- OptimizedVisionProcessor: 优化视觉处理器（四层架构）
- ActionClassifier: MobileNetV4-TSM 动作分类
- MediaPipeGestureRecognizer: MediaPipe 手势识别
"""

from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor, init_vision_processor, get_vision_processor
from zulong.l1c.action_classifier import MobileNetV4_TSM
from zulong.l1c.mediapipe_gesture_recognizer import MediaPipeGestureRecognizer

__all__ = [
    'OptimizedVisionProcessor',
    'init_vision_processor',
    'get_vision_processor',
    'MobileNetV4_TSM',
    'MediaPipeGestureRecognizer',
]
