# File: tests/test_mediapipe_gesture.py
"""
MediaPipe Gesture Recognizer 测试脚本

测试 MediaPipe 预训练模型的手势识别能力。
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from zulong.l1c.mediapipe_gesture_recognizer import MediaPipeGestureRecognizer


async def test_mediapipe_basic():
    """
    基础功能测试
    
    验证 MediaPipe 是否能正常识别手势。
    """
    print("=" * 60)
    print("🎯 MediaPipe Gesture Recognizer 基础测试")
    print("=" * 60)
    
    # 初始化识别器
    print("\n📦 初始化 MediaPipe Gesture Recognizer...")
    recognizer = MediaPipeGestureRecognizer(confidence_threshold=0.5)
    
    # 检查 MediaPipe 是否可用
    if not recognizer._recognizer:
        print("⚠️  MediaPipe 不可用，可能未安装或模型文件缺失")
        print("\n💡 解决方案:")
        print("   1. 安装 MediaPipe: pip install mediapipe")
        print("   2. 下载模型：https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task")
        print("   3. 将模型放到项目根目录")
        return False
    
    print("✅ MediaPipe 初始化成功")
    print(f"📋 支持的手势：{recognizer.get_supported_gestures()}")
    
    # 打开摄像头
    print("\n📷 打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 摄像头已启动")
    
    # 测试 30 秒
    print("\n📸 开始测试 (30 秒)...")
    print("\n📋 测试指导:")
    print("   1. 站在摄像头前 0.5-1 米")
    print("   2. 依次展示以下手势:")
    print("      - Open_Palm (张开手掌)")
    print("      - Closed_Fist (握拳)")
    print("      - Victory_Sign (V 字手势)")
    print("      - Thumb_Up (竖起大拇指)")
    print("      - OK_Gesture (OK 手势)")
    print("   3. 每个手势保持 3-5 秒")
    print("\n⏳ 3 秒后开始...")
    await asyncio.sleep(1)
    print("⏳ 2 秒后开始...")
    await asyncio.sleep(1)
    print("⏳ 1 秒后开始...")
    await asyncio.sleep(1)
    
    start_time = time.time()
    frame_count = 0
    gesture_results = {}
    
    while (time.time() - start_time) < 30.0:
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_count += 1
        
        # 识别手势
        gesture, confidence, info = recognizer.classify_gesture(frame)
        
        # 统计结果
        if gesture:
            gesture_results[gesture] = gesture_results.get(gesture, 0) + 1
        
        # 显示结果
        if gesture:
            status = f"✅ {gesture} ({confidence:.2f})"
            color = (0, 255, 0)
        else:
            status = "❌ No Gesture"
            color = (0, 0, 255)
        
        cv2.putText(
            frame, 
            status, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            color, 
            2
        )
        
        # 显示 FPS
        elapsed = time.time() - start_time
        fps = frame_count / max(0.001, elapsed)
        cv2.putText(
            frame, 
            f"FPS: {fps:.1f}, Frames: {frame_count}", 
            (10, frame.shape[0] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        cv2.imshow("MediaPipe Gesture Recognition Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 测试结果:")
    print(f"   - 总帧数：{frame_count}")
    print(f"   - 测试时长：{elapsed:.1f}秒")
    print(f"   - 平均 FPS: {fps:.1f}")
    print("\n🎯 手势识别统计:")
    
    for gesture, count in sorted(gesture_results.items(), key=lambda x: x[1], reverse=True):
        percentage = count / frame_count * 100
        print(f"   - {gesture}: {count}帧 ({percentage:.1f}%)")
    
    print("=" * 60)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 判断是否成功
    if gesture_results:
        print("\n✅ 测试成功！MediaPipe 可以识别手势")
        return True
    else:
        print("\n❌ 测试失败！未识别到任何手势")
        return False


async def test_mediapipe_all_gestures():
    """
    全手势测试
    
    测试所有支持的手势类型。
    """
    print("=" * 60)
    print("🎯 MediaPipe 全手势测试")
    print("=" * 60)
    
    recognizer = MediaPipeGestureRecognizer(confidence_threshold=0.5)
    
    if not recognizer._recognizer:
        print("❌ MediaPipe 不可用")
        return
    
    supported_gestures = recognizer.get_supported_gestures()
    
    print(f"\n📋 支持的手势 ({len(supported_gestures)}种):")
    for i, gesture in enumerate(supported_gestures, 1):
        print(f"   {i}. {gesture}")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("\n📋 测试指导:")
    print("   依次展示每个手势，系统会自动记录识别结果")
    print("   按 'n' 跳过当前手势，按 'q' 结束测试")
    
    current_gesture_idx = 0
    results = {}
    
    while current_gesture_idx < len(supported_gestures):
        target_gesture = supported_gestures[current_gesture_idx]
        
        print(f"\n 请展示：{target_gesture}")
        print("   (按 'n' 跳过，按 'q' 结束)")
        
        # 采集 5 秒
        start_time = time.time()
        detected_count = 0
        total_frames = 0
        
        while (time.time() - start_time) < 5.0:
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            total_frames += 1
            
            gesture, confidence, info = recognizer.classify_gesture(frame)
            
            if gesture:
                detected_count += 1
                
                if gesture == target_gesture:
                    cv2.putText(frame, f"✅ MATCH!", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"❌ {gesture}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Gesture: {target_gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Detected: {detected_count}/{total_frames}", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("All Gestures Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                break
        
        # 记录结果
        accuracy = detected_count / total_frames * 100 if total_frames > 0 else 0
        results[target_gesture] = {
            'detected': detected_count,
            'total': total_frames,
            'accuracy': accuracy
        }
        
        print(f"   结果：{detected_count}/{total_frames} ({accuracy:.1f}%)")
        
        current_gesture_idx += 1
        
        # 休息 2 秒
        if current_gesture_idx < len(supported_gestures):
            print("⏳ 休息 2 秒...")
            await asyncio.sleep(2)
    
    # 总结结果
    print("\n" + "=" * 60)
    print("📊 全手势测试结果:")
    print("=" * 60)
    
    for gesture, data in results.items():
        status = "✅" if data['accuracy'] > 50 else "⚠️"
        print(f"{status} {gesture}: {data['accuracy']:.1f}% ({data['detected']}/{data['total']})")
    
    print("=" * 60)
    
    cap.release()
    cv2.destroyAllWindows()


async def main():
    """主测试函数"""
    print("\n请选择测试模式:")
    print("1. 基础测试 (30 秒自由展示)")
    print("2. 全手势测试 (逐个测试所有手势)")
    print("q. 退出")
    
    choice = input("\n请输入选择 (1/2/q): ").strip()
    
    if choice == '1':
        await test_mediapipe_basic()
    elif choice == '2':
        await test_mediapipe_all_gestures()
    else:
        print("👋 退出测试")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 测试中断")
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
