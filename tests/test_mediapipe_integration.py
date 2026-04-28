# File: tests/test_mediapipe_integration.py
"""
MediaPipe 集成测试

验证 MediaPipe Gesture Recognizer 在 OptimizedVisionProcessor 中的集成效果。
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor


async def test_mediapipe_integration():
    """
    集成测试
    
    验证 MediaPipe 在完整视觉处理流程中的表现。
    """
    print("=" * 60)
    print("🎯 MediaPipe 集成测试")
    print("=" * 60)
    
    # 初始化视觉处理器
    print("\n📦 初始化 OptimizedVisionProcessor...")
    processor = OptimizedVisionProcessor()
    
    try:
        await processor.initialize(load_models=True)
        print("✅ 处理器初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 检查手势分类器类型
    if processor._gesture_classifier:
        classifier_type = type(processor._gesture_classifier).__name__
        print(f"📋 手势分类器类型：{classifier_type}")
        
        if classifier_type == "MediaPipeGestureRecognizer":
            print("✅ MediaPipe Gesture Recognizer 已加载")
            print(f"📋 支持的手势：{processor._gesture_classifier.get_supported_gestures()}")
        else:
            print(f"⚠️  未使用 MediaPipe，而是：{classifier_type}")
    else:
        print("❌ 手势分类器未加载")
        return False
    
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
    print("      - Open_Palm (张开手掌) ✋")
    print("      - Closed_Fist (握拳) ✊")
    print("      - Victory_Sign (V 字手势) ✌️")
    print("      - Thumb_Up (竖起大拇指) 👍")
    print("      - OK_Gesture (OK 手势) 👌")
    print("   3. 每个手势保持 3-5 秒，并挥动手臂")
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
        
        # 获取最新帧
        timestamp = time.time()
        
        # 推送帧到处理器
        processor.feed_frame(frame, timestamp)
        
        # 等待处理完成
        await asyncio.sleep(0.1)
        
        # 获取处理结果
        human_detected = processor.shared_memory.get('human_detected', False)
        human_bbox = processor.shared_memory.get('human_bbox')
        motion_pixels = processor.shared_memory.get('motion_pixels', 0)
        gesture_type = processor.shared_memory.get('gesture_type')
        action_score = processor.shared_memory.get('action_score', 0.0)
        
        # 统计手势结果
        if gesture_type and gesture_type != "UNKNOWN":
            gesture_results[gesture_type] = gesture_results.get(gesture_type, 0) + 1
        
        # 显示状态
        status_lines = []
        status_lines.append(f"Human: {'✅' if human_detected else '❌'}")
        status_lines.append(f"Motion: {'✅' if motion_pixels > 0 else '❌'} ({motion_pixels})")
        status_lines.append(f"Action: {action_score:.2f}")
        status_lines.append(f"Gesture: {gesture_type if gesture_type else 'None'}")
        
        # 绘制检测框
        if human_detected and human_bbox:
            x1, y1, x2, y2 = [int(coord) for coord in human_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ROI 区域
            margin = 50
            roi_x1, roi_y1 = max(0, x1 - margin), max(0, y1 - margin)
            roi_x2, roi_y2 = min(640, x2 + margin), min(480, y2 + margin)
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 1)
        
        # 显示状态
        for i, line in enumerate(status_lines):
            y = 30 + (i * 30)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示 FPS
        elapsed = time.time() - start_time
        fps = frame_count / max(0.001, elapsed)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("MediaPipe Integration Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 测试结果:")
    print(f"   - 总帧数：{frame_count}")
    print(f"   - 测试时长：{elapsed:.1f}秒")
    print(f"   - 平均 FPS: {fps:.1f}")
    print("\n🎯 手势识别统计:")
    
    if gesture_results:
        for gesture, count in sorted(gesture_results.items(), key=lambda x: x[1], reverse=True):
            percentage = count / frame_count * 100
            print(f"   - {gesture}: {count}帧 ({percentage:.1f}%)")
        
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
    success = await test_mediapipe_integration()
    
    if success:
        print("\n✅ MediaPipe 集成测试通过！")
        print("\n💡 下一步:")
        print("   1. 调整摄像头距离和光线条件")
        print("   2. 增加手势展示时间")
        print("   3. 挥动手臂提高运动检测敏感度")
    else:
        print("\n❌ MediaPipe 集成测试失败")
        print("\n💡 建议:")
        print("   1. 检查摄像头是否正常工作")
        print("   2. 确保人体在摄像头范围内")
        print("   3. 增加运动幅度和手势展示时间")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 测试中断")
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
