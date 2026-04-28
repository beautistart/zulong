# File: tests/test_simple_vision_debug.py
"""
简化版视觉调试测试

直接测试 OptimizedVisionProcessor，显示所有中间状态
"""

import cv2
import time
import asyncio
import logging
from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

async def test():
    print("=" * 80)
    print(" 简化视觉调试测试")
    print("=" * 80)
    
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("✅ 初始化完成，请坐在摄像头前并挥手")
    print("按 'q' 退出\n")
    
    await asyncio.sleep(2)
    
    start_time = time.time()
    frame_count = 0
    last_print = 0
    
    try:
        while time.time() - start_time < 60:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = time.time()
            
            # 喂食帧
            processor.feed_frame(frame, timestamp)
            
            # 等待处理
            await asyncio.sleep(0.1)
            
            # 获取状态
            mem = processor.shared_memory
            state = processor.state_machine
            
            # 每 1 秒打印详细状态
            if timestamp - last_print >= 1.0:
                last_print = timestamp
                print(f"\n[帧 {frame_count}]")
                print(f"  L1: {state['layer1_state']} | human={mem['human_detected']}, bbox={mem.get('human_bbox')}")
                print(f"  L2: {state['layer2_state']} | motion_pixels={mem.get('motion_pixels', 0)}")
                print(f"  L3: {state['layer3_state']} | action_score={mem.get('action_score', 0):.2f}, intent={mem.get('intent_type', 'UNKNOWN')}")
                print(f"  L4: {state['layer4_state']} | gesture={mem.get('gesture_type', 'NONE')}")
            
            # 绘制
            h, w = frame.shape[:2]
            
            # L1
            if mem['human_detected'] and mem.get('human_bbox'):
                x1, y1, x2, y2 = [int(c) for c in mem['human_bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "L1: DETECTED", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "L1: NO HUMAN", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # L2
            motion = mem.get('motion_pixels', 0)
            color = (0, 255, 0) if motion > 0 else (0, 0, 255)
            cv2.putText(frame, f"L2: {motion}px", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # L3
            score = mem.get('action_score', 0)
            intent = mem.get('intent_type', 'UNKNOWN')
            color = (0, 0, 255) if score < 0.6 else (0, 255, 255) if score < 0.8 else (0, 255, 0)
            cv2.putText(frame, f"L3: {intent} ({score:.2f})", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # L4
            gesture = mem.get('gesture_type', '')
            if gesture:
                cv2.putText(frame, f"L4: {gesture}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.imshow("Debug Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\n测试完成：{frame_count}帧")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(test())
    except KeyboardInterrupt:
        print("\n\n测试中断")
