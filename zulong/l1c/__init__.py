# File: zulong/l1c/__init__.py
# L1-C Silent Visual Attention Layer - Community Edition

"""
ZULONG L1-C Layer - Silent Visual Attention

Community Edition includes:
- Action Classifier (MobileNetV4-TSM)
- MediaPipe Gesture Recognizer

Enterprise Edition adds:
- Optimized Vision Processor (four-layer architecture)
"""

# --- Community Edition ---
from zulong.l1c.action_classifier import MobileNetV4_TSM
from zulong.l1c.mediapipe_gesture_recognizer import MediaPipeGestureRecognizer

__all__ = [
    'MobileNetV4_TSM',
    'MediaPipeGestureRecognizer',
]

# --- Enterprise Edition (optional) ---
try:
    from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor, init_vision_processor, get_vision_processor
    __all__.extend(['OptimizedVisionProcessor', 'init_vision_processor', 'get_vision_processor'])
except ImportError:
    pass
