# File: tests/test_mediapipe_direct.py
"""
MediaPipe 直接测试

直接在 OptimizedVisionProcessor 中使用 MediaPipe 进行手势识别，
跳过 Layer 1-3 的复杂逻辑。
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from zulong.l1c.mediapipe_gesture_recognizer import MediaPipeGestureRecognizer


async def test_mediapipe_direct():
    """
    直接测试 MediaPipe 手势识别
    """
    print("=" * 60)
    print("🎯 MediaPipe 直接测试（跳过复杂逻辑）")
    print("=" * 60)
    
    # 初始化 MediaPipe
    print("\n📦 初始化 MediaPipe Gesture Recognizer...")
    recognizer = MediaPipeGestureRecognizer(confidence_threshold=0.5)
    
    if not recognizer._recognizer:
        print("❌ MediaPipe 不可用")
        return False
    
    print("✅ MediaPipe 已加载")
    print(f"📋 支持的手势：{recognizer.get_supported_gestures()}")
    
    # 打开摄像头
    print("\n📷 打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 摄像头已启动")
    
    # 测试 60 秒
    print("\n📸 开始测试 (60 秒)...")
    print("\n📋 测试指导:")
    print("   1. 将手放在摄像头前 0.3-0.6 米（更近一些）")
    print("   2. 依次展示以下手势:")
    print("      - Open_Palm (张开手掌) ✋")
    print("      - Closed_Fist (握拳) ✊")
    print("      - Victory_Sign (V 字手势) ✌️")
    print("      - Thumb_Up (竖起大拇指) 👍")
    print("      - OK_Gesture (OK 手势) 👌")
    print("   3. 每个手势保持 5-10 秒")
    print("   4. 确保手部光线充足")
    print("\n⏳ 3 秒后开始...")
    await asyncio.sleep(1)
    print("⏳ 2 秒后开始...")
    await asyncio.sleep(1)
    print("⏳ 1 秒后开始...")
    await asyncio.sleep(1)
    
    start_time = time.time()
    frame_count = 0
    gesture_results = {}
    last_gesture = None
    gesture_start_time = None
    
    while (time.time() - start_time) < 60.0:
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_count += 1
        
        # 识别手势
        gesture, confidence, info = recognizer.classify_gesture(frame)
        
        # 统计结果
        if gesture and gesture != "None":
            gesture_results[gesture] = gesture_results.get(gesture, 0) + 1
            
            # 检测新手势
            if gesture != last_gesture:
                if gesture_start_time and last_gesture:
                    duration = time.time() - gesture_start_time
                    print(f"   🎯 {last_gesture}: 持续 {duration:.1f}秒")
                
                last_gesture = gesture
                gesture_start_time = time.time()
        
        # 显示状态
        if gesture:
            status = f"✅ {gesture} ({confidence:.2f})"
            color = (0, 255, 0)
        else:
            status = "⏳ Waiting..."
            color = (0, 0, 255)
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 显示 FPS
        elapsed = time.time() - start_time
        fps = frame_count / max(0.001, elapsed)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示当前手势持续时间
        if gesture_start_time and last_gesture:
            duration = time.time() - gesture_start_time
            cv2.putText(frame, f"Duration: {duration:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("MediaPipe Direct Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 最终统计
    if gesture_start_time and last_gesture:
        duration = time.time() - gesture_start_time
        print(f"   🎯 {last_gesture}: 持续 {duration:.1f}秒")
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 测试结果:")
    print(f"   - 总帧数：{frame_count}")
    print(f"   - 测试时长：{elapsed:.1f}秒")
    print(f"   - 平均 FPS: {fps:.1f}")
    print("\n🎯 手势识别统计:")
    
    if gesture_results:
        total_gesture_frames = sum(gesture_results.values())
        for gesture, count in sorted(gesture_results.items(), key=lambda x: x[1], reverse=True):
            percentage = count / frame_count * 100
            gesture_percentage = count / total_gesture_frames * 100
            print(f"   - {gesture}: {count}帧 ({percentage:.1f}% of total, {gesture_percentage:.1f}% of gestures)")
        
        print("\n✅ 测试成功！MediaPipe 可以识别手势")
        success = True
    else:
        print("   - 未识别到任何手势")
        print("\n❌ 测试失败！未识别到任何手势")
        success = False
    
    print("=" * 60)
    
    cap.release()
    cv2.destroyAllWindows()
    
    return success


async def main():
    """主测试函数"""
    success = await test_mediapipe_direct()
    
    if success:
        print("\n✅ MediaPipe 直接测试通过！")
        print("\n💡 下一步:")
        print("   1. 将 MediaPipe 集成到 OptimizedVisionProcessor")
        print("   2. 调整鹰眼模式触发条件")
        print("   3. 进行完整系统集成测试")
    else:
        print("\n❌ MediaPipe 直接测试失败")
        print("\n💡 建议:")
        print("   1. 确保手部光线充足")
        print("   2. 手距离摄像头更近一些 (0.3-0.6 米)")
        print("   3. 展示标准手势姿势")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 测试中断")
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
