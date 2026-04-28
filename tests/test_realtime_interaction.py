# File: tests/test_realtime_interaction.py
"""
实时交互测试 - 等待异步处理完成

这个测试会：
1. 同步等待每帧处理完成
2. 显示详细的 Layer 状态
3. 测试完整 L0-L4 流程
"""

import cv2
import time
import asyncio
from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor

async def test_realtime():
    """实时交互测试"""
    
    print("=" * 80)
    print(" 实时交互测试 - 完整 L0-L4 流程")
    print("=" * 80)
    
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("\n✅ 初始化完成")
    print("\n请执行以下动作：")
    print("1. 坐在摄像头前（Layer 1 会检测到你）")
    print("2. 挥动手臂（Layer 2 会检测到运动）")
    print("3. 继续挥手 2-3 秒（Layer 3 会识别意图）")
    print("4. 比出 V 字或 OK 手势（Layer 4 会识别手势）")
    print("\n按 'q' 退出，测试时间 60 秒\n")
    
    await asyncio.sleep(2)
    
    start_time = time.time()
    frame_count = 0
    last_print_time = 0
    
    try:
        while time.time() - start_time < 60:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = time.time()
            
            # 喂食帧
            processor.feed_frame(frame, timestamp)
            
            # 等待 50ms 让异步处理完成
            await asyncio.sleep(0.05)
            
            # 获取状态
            state = processor.state_machine
            mem = processor.shared_memory
            
            # 每 0.5 秒打印一次状态
            if timestamp - last_print_time >= 0.5:
                last_print_time = timestamp
                
                # 格式化输出
                l1_status = f"L1:{state['layer1_state']:<15}"
                l2_status = f"L2:{state['layer2_state']:<15}"
                l3_status = f"L3:{state['layer3_state']:<15}"
                l4_status = f"L4:{state['layer4_state']:<15}"
                
                print(f"\n[{frame_count}] {l1_status} | {l2_status} | {l3_status} | {l4_status}")
                
                if mem['human_detected']:
                    bbox = mem.get('human_bbox')
                    if bbox:
                        print(f"    └─ 人体位置：[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                
                if mem.get('motion_pixels', 0) > 0:
                    print(f"    └─ 运动像素：{mem['motion_pixels']}")
                
                if mem.get('action_score', 0) > 0:
                    intent = mem.get('intent_type', 'UNKNOWN')
                    score = mem['action_score']
                    print(f"    └─ 意图：{intent} (置信度：{score:.2f})")
                
                if mem.get('gesture_type'):
                    gesture = mem['gesture_type']
                    print(f"    └─ 手势：{gesture}")
            
            # 绘制可视化
            h, w = frame.shape[:2]
            
            # Layer 1
            if mem['human_detected'] and mem.get('human_bbox'):
                x1, y1, x2, y2 = [int(c) for c in mem['human_bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "L1: DETECTED", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Layer 2
            if mem.get('motion_pixels', 0) > 0:
                cv2.putText(frame, f"L2: {mem['motion_pixels']}px", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Layer 3
            score = mem.get('action_score', 0)
            intent = mem.get('intent_type', 'UNKNOWN')
            color = (0, 0, 255) if score < 0.6 else (0, 255, 255) if score < 0.8 else (0, 255, 0)
            cv2.putText(frame, f"L3: {intent} ({score:.2f})", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Layer 4
            gesture = mem.get('gesture_type', '')
            if gesture:
                cv2.putText(frame, f"L4: {gesture}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # FPS
            fps = 1.0 / (time.time() - timestamp) if time.time() - timestamp > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Realtime Interaction Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\n测试完成：处理了 {frame_count} 帧")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(test_realtime())
    except KeyboardInterrupt:
        print("\n\n测试中断")
