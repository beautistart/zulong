# File: tests/test_mediapipe_diagnosis.py
"""
MediaPipe 手势识别诊断工具

用于检查 MediaPipe 是否正确安装和加载模型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import logging

logging.basicConfig(level=logging.DEBUG)

print("\n" + "=" * 80)
print(" MediaPipe 手势识别诊断工具")
print("=" * 80)

# ========== 1. 检查 MediaPipe 是否安装 ==========
print("\n[1/5] 检查 MediaPipe 安装...")
try:
    import mediapipe as mp
    print(f"✅ MediaPipe 已安装")
    print(f"   版本：{mp.__version__ if hasattr(mp, '__version__') else '未知'}")
except ImportError as e:
    print(f"❌ MediaPipe 未安装：{e}")
    print("\n请运行：pip install mediapipe")
    sys.exit(1)

# ========== 2. 检查模型文件是否存在 ==========
print("\n[2/5] 检查模型文件...")
from pathlib import Path
model_path = Path(__file__).parent.parent / "gesture_recognizer.task"
print(f"   模型路径：{model_path}")
print(f"   绝对路径：{model_path.absolute()}")

if model_path.exists():
    print(f"✅ 模型文件存在")
    print(f"   文件大小：{model_path.stat().st_size / 1024 / 1024:.2f} MB")
else:
    print(f"❌ 模型文件不存在")
    sys.exit(1)

# ========== 3. 尝试初始化 MediaPipe Gesture Recognizer ==========
print("\n[3/5] 初始化 MediaPipe Gesture Recognizer...")
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    print("✅ MediaPipe Tasks API 可用")
    
    # 创建 Gesture Recognizer
    base_options = python.BaseOptions(
        model_asset_path=str(model_path.absolute()),
        delegate=python.BaseOptions.Delegate.CPU
    )
    
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    recognizer = vision.GestureRecognizer.create_from_options(options)
    print("✅ MediaPipe Gesture Recognizer 初始化成功！")
    
except Exception as e:
    print(f"❌ 初始化失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 4. 测试手势识别 ==========
print("\n[4/5] 测试手势识别...")
print("   请在摄像头前比出手势（V 字、OK、点赞等）")
print("   按 'q' 退出\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    sys.exit(1)

print("✅ 摄像头已打开")

test_count = 0
success_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break
        
        test_count += 1
        
        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建 MP Image
        from mediapipe import Image as MPImage
        from mediapipe import ImageFormat
        
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)
        
        # 推理
        recognition_result = recognizer.recognize(mp_image)
        
        # 解析结果
        if recognition_result.gestures:
            success_count += 1
            gesture = recognition_result.gestures[0][0]
            gesture_name = gesture.category_name
            confidence = gesture.score
            
            # 在视频上显示
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Conf: {confidence:.3f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            print(f"🎯 检测到手势：{gesture_name} (置信度：{confidence:.3f})")
        else:
            cv2.putText(frame, "Gesture: None", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # 显示帧数
        cv2.putText(frame, f"Frame: {test_count}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Success: {success_count}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示视频窗口
        cv2.imshow('MediaPipe Diagnosis - Press q to exit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()

# ========== 5. 打印统计信息 ==========
print("\n[5/5] 测试统计")
print("=" * 80)
print(f"总帧数：{test_count}")
print(f"成功检测：{success_count}")
print(f"检测成功率：{success_count / test_count * 100:.1f}%" if test_count > 0 else "N/A")

if success_count > 0:
    print("\n✅ MediaPipe 手势识别工作正常！")
else:
    print("\n⚠️  MediaPipe 未检测到手势")
    print("\n可能原因:")
    print("  1. 光线不足")
    print("  2. 手部距离太远")
    print("  3. 手势不清晰")
    print("  4. 模型文件损坏")

print("\n" + "=" * 80)
