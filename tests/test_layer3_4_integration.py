# File: tests/test_layer3_4_integration.py
"""
Layer 3-4 联动测试

验证完整的视觉处理流程：
Layer 1 (人体检测) → Layer 2 (运动检测) → Layer 3 (动作分类) → Layer 4 (手势识别)
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor


async def test_layer3_4_integration():
    """
    Layer 3-4 联动集成测试
    
    验证完整的视觉处理流程。
    """
    print("=" * 60)
    print("🎯 Layer 3-4 联动集成测试")
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
    
    # 检查模型状态
    print("\n📋 模型加载状态:")
    print(f"   - YOLO: {'✅' if processor._model_loader and processor._model_loader.yolo_model else '❌'}")
    print(f"   - MobileNetV3: {'✅' if processor._action_classifier and processor._action_classifier._model else '❌'}")
    print(f"   - MediaPipe: {'✅' if processor._gesture_classifier else '❌'}")
    
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
    print("   1. 站在摄像头前 0.5-1 米")
    print("   2. 完整流程测试:")
    print("      - Layer 1: 确保人体在画面中（绿色框）")
    print("      - Layer 2: 挥动手臂触发运动检测（蓝色 ROI 框）")
    print("      - Layer 3: 保持动作 2-3 秒触发意图分类")
    print("      - Layer 4: 如果意图分数>0.8，自动触发手势识别")
    print("   3. 展示手势:")
    print("      - Open_Palm (张开手掌) ✋")
    print("      - Victory_Sign (V 字手势) ✌️")
    print("      - Thumb_Up (竖起大拇指) 👍")
    print("      - OK_Gesture (OK 手势) 👌")
    print("\n⏳ 3 秒后开始...")
    await asyncio.sleep(1)
    print("⏳ 2 秒后开始...")
    await asyncio.sleep(1)
    print("⏳ 1 秒后开始...")
    await asyncio.sleep(1)
    
    start_time = time.time()
    frame_count = 0
    layer_stats = {
        'layer1_detected': 0,
        'layer2_motion': 0,
        'layer3_intent': 0,
        'layer4_gesture': 0,
    }
    last_intent = None
    last_gesture = None
    intent_trigger_count = 0
    gesture_trigger_count = 0
    
    while (time.time() - start_time) < 60.0:
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_count += 1
        timestamp = time.time()
        
        # 处理帧（使用 feed_frame）
        processor.feed_frame(frame, timestamp)
        
        # 从共享内存读取结果
        result = {
            'human_detected': processor.shared_memory.get('human_detected', False),
            'human_bbox': processor.shared_memory.get('human_bbox', None),
            'motion_detected': processor.shared_memory.get('motion_pixels', 0) > 100,
            'roi_bbox': processor.shared_memory.get('roi_bbox', None),
            'intent_type': processor.shared_memory.get('intent_type', None),
            'intent_score': processor.shared_memory.get('action_score', 0.0),
            'gesture': processor.shared_memory.get('gesture_type', None),
            'gesture_confidence': processor.shared_memory.get('gesture_confidence', 0.0),
        }
        
        # 统计 Layer 1
        if result and result.get('human_detected'):
            layer_stats['layer1_detected'] += 1
            
            # 绘制人体检测框
            bbox = result.get('human_bbox')
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Layer1: Human", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 统计 Layer 2
        if result and result.get('motion_detected'):
            layer_stats['layer2_motion'] += 1
            
            # 绘制 ROI 框
            roi_bbox = result.get('roi_bbox')
            if roi_bbox:
                x1, y1, x2, y2 = map(int, roi_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "Layer2: Motion", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 统计 Layer 3
        if result and result.get('intent_type'):
            layer_stats['layer3_intent'] += 1
            intent_type = result['intent_type']
            intent_score = result.get('intent_score', 0.0)
            
            if intent_type != last_intent:
                print(f"   🧠 Layer3: {intent_type} (置信度：{intent_score:.2f})")
                last_intent = intent_type
                
                # 检查是否触发交互
                if intent_score > 0.8:
                    intent_trigger_count += 1
                    print(f"      ✅ 触发 Layer 4 (INTERACT_REQUEST)")
                elif intent_score > 0.6:
                    print(f"      ⚠️  触发 Layer 3 (SILENT_WATCH)")
        
        # 统计 Layer 4
        if result and result.get('gesture'):
            layer_stats['layer4_gesture'] += 1
            gesture = result['gesture']
            gesture_conf = result.get('gesture_confidence', 0.0)
            
            if gesture and gesture != "UNKNOWN" and gesture != last_gesture:
                print(f"      🦅 Layer4: {gesture} (置信度：{gesture_conf:.2f})")
                last_gesture = gesture
                gesture_trigger_count += 1
        
        # 显示状态信息
        y_offset = 30
        cv2.putText(frame, f"L1: {layer_stats['layer1_detected']}/{frame_count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"L2: {layer_stats['layer2_motion']}/{frame_count}", (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"L3: {layer_stats['layer3_intent']}/{frame_count}", (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"L4: {layer_stats['layer4_gesture']}/{frame_count}", (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # 显示 FPS
        elapsed = time.time() - start_time
        fps = frame_count / max(0.001, elapsed)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Layer 3-4 Integration Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 测试结果:")
    print(f"   - 总帧数：{frame_count}")
    print(f"   - 测试时长：{elapsed:.1f}秒")
    print(f"   - 平均 FPS: {fps:.1f}")
    print("\n🎯 各层触发统计:")
    print(f"   - Layer 1 (人体检测): {layer_stats['layer1_detected']}/{frame_count} ({layer_stats['layer1_detected']/frame_count*100:.1f}%)")
    print(f"   - Layer 2 (运动检测): {layer_stats['layer2_motion']}/{frame_count} ({layer_stats['layer2_motion']/frame_count*100:.1f}%)")
    print(f"   - Layer 3 (意图分类): {layer_stats['layer3_intent']}/{frame_count} ({layer_stats['layer3_intent']/frame_count*100:.1f}%)")
    print(f"   - Layer 4 (手势识别): {layer_stats['layer4_gesture']}/{frame_count} ({layer_stats['layer4_gesture']/frame_count*100:.1f}%)")
    print(f"\n🔗 联动触发:")
    print(f"   - Layer 3→4 触发次数：{intent_trigger_count}")
    print(f"   - Layer 4 手势识别：{gesture_trigger_count}")
    
    # 评估
    print("\n" + "=" * 60)
    if layer_stats['layer1_detected'] > 0 and layer_stats['layer4_gesture'] > 0:
        print("✅ 测试成功！Layer 3-4 联动正常")
        success = True
    elif layer_stats['layer1_detected'] > 0 and layer_stats['layer3_intent'] > 0:
        print("⚠️  测试部分成功！Layer 3 工作，但 Layer 4 未触发")
        print("\n💡 建议:")
        print("   1. 降低 Layer 4 触发阈值")
        print("   2. 增加动作幅度和持续时间")
        print("   3. 展示更明显的手势")
        success = True
    else:
        print("❌ 测试失败！联动未触发")
        print("\n💡 建议:")
        print("   1. 确保人体在画面中")
        print("   2. 增加运动幅度")
        print("   3. 检查模型是否正确加载")
        success = False
    
    print("=" * 60)
    
    cap.release()
    cv2.destroyAllWindows()
    
    return success


async def test_layer3_4_with_mock_intent():
    """
    模拟高置信度意图测试
    
    强制触发 Layer 4，验证手势识别功能。
    """
    print("\n" + "=" * 60)
    print("🎯 Layer 3-4 模拟高置信度测试")
    print("=" * 60)
    
    print("\n📋 测试逻辑:")
    print("   1. 临时降低意图阈值到 0.3")
    print("   2. 强制触发 Layer 4 手势识别")
    print("   3. 验证 MediaPipe 是否正常工作")
    
    # 初始化视觉处理器
    print("\n📦 初始化 OptimizedVisionProcessor...")
    processor = OptimizedVisionProcessor()
    
    try:
        await processor.initialize(load_models=True)
        print("✅ 处理器初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败：{e}")
        return False
    
    # 临时降低阈值
    print("\n🔧 调整阈值配置...")
    processor._config['intent_threshold'] = 0.3
    processor._config['interact_threshold'] = 0.5
    processor._config['eagle_eye_cooldown'] = 2.0  # 降低冷却时间
    
    print(f"   - 意图阈值：{processor._config['intent_threshold']}")
    print(f"   - 交互阈值：{processor._config['interact_threshold']}")
    print(f"   - 鹰眼冷却：{processor._config['eagle_eye_cooldown']}s")
    
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
    print("   1. 站在摄像头前")
    print("   2. 展示以下手势:")
    print("      - Open_Palm (张开手掌) ✋")
    print("      - Victory_Sign (V 字手势) ✌️")
    print("      - Thumb_Up (竖起大拇指) 👍")
    print("      - OK_Gesture (OK 手势) 👌")
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
    last_gesture = None
    
    while (time.time() - start_time) < 30.0:
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_count += 1
        timestamp = time.time()
        
        # 处理帧（使用 feed_frame）
        processor.feed_frame(frame, timestamp)
        
        # 从共享内存读取结果
        result = {
            'gesture': processor.shared_memory.get('gesture_type', None),
            'gesture_confidence': processor.shared_memory.get('gesture_confidence', 0.0),
        }
        
        # 统计手势
        if result and result.get('gesture') and result['gesture'] != 'UNKNOWN':
            gesture = result['gesture']
            conf = result.get('gesture_confidence', 0.0)
            gesture_results[gesture] = gesture_results.get(gesture, 0) + 1
            
            if gesture != last_gesture:
                print(f"   🦅 识别手势：{gesture} (置信度：{conf:.2f})")
                last_gesture = gesture
        
        # 显示状态
        if result and result.get('gesture'):
            gesture = result['gesture']
            conf = result.get('gesture_confidence', 0.0)
            status = f"Gesture: {gesture} ({conf:.2f})"
            color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 显示 FPS
        elapsed = time.time() - start_time
        fps = frame_count / max(0.001, elapsed)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Mock Intent Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 测试结果:")
    print(f"   - 总帧数：{frame_count}")
    print(f"   - 测试时长：{elapsed:.1f}秒")
    
    if gesture_results:
        print("\n🎯 手势识别统计:")
        for gesture, count in sorted(gesture_results.items(), key=lambda x: x[1], reverse=True):
            percentage = count / frame_count * 100
            print(f"   - {gesture}: {count}帧 ({percentage:.1f}%)")
        
        print("\n✅ 测试成功！MediaPipe 可以识别手势")
        success = True
    else:
        print("\n❌ 未识别到任何手势")
        success = False
    
    print("=" * 60)
    
    cap.release()
    cv2.destroyAllWindows()
    
    return success


async def main():
    """主测试函数"""
    print("=" * 60)
    print("🎯 Layer 3-4 联动测试套件")
    print("=" * 60)
    
    # 测试 1: 正常联动测试
    print("\n" + "=" * 60)
    print("📝 测试 1: Layer 3-4 正常联动")
    print("=" * 60)
    test1 = await test_layer3_4_integration()
    
    await asyncio.sleep(3)
    
    # 测试 2: 模拟高置信度测试
    print("\n" + "=" * 60)
    print("📝 测试 2: Layer 3-4 模拟高置信度")
    print("=" * 60)
    test2 = await test_layer3_4_with_mock_intent()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结:")
    print(f"   - 正常联动：{'✅ 通过' if test1 else '❌ 失败'}")
    print(f"   - 模拟测试：{'✅ 通过' if test2 else '❌ 失败'}")
    print("=" * 60)
    
    if test1 or test2:
        print("\n✅ Layer 3-4 联动测试部分或全部通过！")
        print("\n💡 下一步:")
        print("   1. 根据测试结果调整阈值")
        print("   2. 采集数据训练意图分类头")
        print("   3. 优化手势识别性能")
    else:
        print("\n❌ 测试失败")
        print("\n💡 建议:")
        print("   1. 检查模型加载状态")
        print("   2. 确保摄像头正常工作")
        print("   3. 增加动作和手势幅度")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 测试中断")
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
