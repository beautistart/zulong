# File: tests/test_visualize_pipeline.py
"""
可视化视觉检测流程

显示每一层的处理结果和中间数据
"""

import cv2
import numpy as np
import time
import asyncio
from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor

async def visualize_pipeline():
    """可视化四层视觉处理流程"""
    
    print("=" * 60)
    print(" 视觉检测流程可视化")
    print("=" * 60)
    
    # 初始化
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("\n✅ 初始化完成，按 'q' 退出\n")
    
    # 等待模型加载
    await asyncio.sleep(2)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = time.time()
            processor.feed_frame(frame, timestamp)
            
            # ========== 绘制可视化信息 ==========
            h, w = frame.shape[:2]
            
            # 1. Layer 1 状态
            layer1_state = processor.state_machine['layer1_state']
            human_bbox = processor.shared_memory.get('human_bbox')
            
            if human_bbox:
                x1, y1, x2, y2 = [int(c) for c in human_bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "L1: HUMAN", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 2. Layer 2 状态
            motion_pixels = processor.shared_memory.get('motion_pixels', 0)
            layer2_state = processor.state_machine['layer2_state']
            
            if motion_pixels > 0:
                info_text = f"L2: {motion_pixels}px"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 3. Layer 3 状态
            action_score = processor.shared_memory.get('action_score', 0.0)
            intent_type = processor.shared_memory.get('intent_type', 'UNKNOWN')
            layer3_state = processor.state_machine['layer3_state']
            
            score_color = (0, 0, 255) if action_score < 0.6 else (0, 255, 255) if action_score < 0.8 else (0, 255, 0)
            info_text = f"L3: {intent_type} ({action_score:.2f})"
            cv2.putText(frame, info_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
            
            # 4. Layer 4 状态
            gesture_type = processor.shared_memory.get('gesture_type', 'NONE')
            layer4_state = processor.state_machine['layer4_state']
            
            if gesture_type:
                info_text = f"L4: {gesture_type}"
                cv2.putText(frame, info_text, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # 5. 性能信息
            fps = 1.0 / (time.time() - timestamp) if time.time() - timestamp > 0 else 0
            yolo_freq = processor._frame_counter - processor._last_yolo_inference_frame
            yolo_status = f"YOLO: {yolo_freq}/{processor._config['yolo_inference_frequency']}"
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, yolo_status, (w-150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 6. 状态机
            status_text = f"L1:{layer1_state} | L2:{layer2_state} | L3:{layer3_state} | L4:{layer4_state}"
            cv2.putText(frame, status_text, (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Visual Pipeline", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(visualize_pipeline())
    except KeyboardInterrupt:
        print("\n\n可视化已退出")
