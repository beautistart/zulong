# File: tests/debug_mediapipe_output.py
"""
MediaPipe 输出调试

检查 MediaPipe 实际返回的手势名称和置信度。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from zulong.l1c.mediapipe_gesture_recognizer import MediaPipeGestureRecognizer


def debug_mediapipe_output():
    """调试 MediaPipe 输出"""
    print("=" * 60)
    print(" MediaPipe 输出调试")
    print("=" * 60)
    
    # 初始化识别器
    print("\n 初始化 MediaPipe Gesture Recognizer...")
    recognizer = MediaPipeGestureRecognizer(confidence_threshold=0.5)
    
    if not recognizer._recognizer:
        print(" MediaPipe 不可用")
        return
    
    print(" MediaPipe 已加载")
    print(f" 支持的手势：{recognizer.get_supported_gestures()}")
    
    # 打开摄像头
    print("\n 打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" 无法打开摄像头")
        return
    
    print(" 摄像头已启动")
    print("\n 请展示以下手势:")
    print("   - Open_Palm (张开手掌) ✋")
    print("   - Closed_Fist (握拳) ✊")
    print("   - Victory_Sign (V 字手势) ✌️")
    print("   - Thumb_Up (竖起大拇指) 👍")
    print("   - OK_Gesture (OK 手势) 👌")
    print("\n 按 'q' 退出，按 'p' 打印详细信息")
    
    frame_count = 0
    last_gesture = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # 识别手势
        gesture_name, confidence, details = recognizer.classify_gesture(frame)
        
        # 显示结果
        if gesture_name:
            if gesture_name != last_gesture:
                print(f"\n帧 {frame_count}:")
                print(f"  手势名称：{gesture_name}")
                print(f"  置信度：{confidence:.4f}")
                print(f"  详细信息：{details.keys()}")
                last_gesture = gesture_name
            
            status = f"{gesture_name} ({confidence:.2f})"
            color = (0, 255, 0)
        else:
            status = "No Gesture"
            color = (0, 0, 255)
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("MediaPipe Debug", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"\n帧 {frame_count} 详细信息:")
            print(f"  手势名称：{gesture_name}")
            print(f"  置信度：{confidence}")
            print(f"  详细信息：{details}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n 调试完成")


if __name__ == "__main__":
    try:
        debug_mediapipe_output()
    except KeyboardInterrupt:
        print("\n\n 调试中断")
    except Exception as e:
        print(f"\n 调试失败：{e}")
        import traceback
        traceback.print_exc()
