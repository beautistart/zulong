# File: tests/test_debug_layers.py
"""
调试版视觉测试 - 显示每一层的详细日志
"""

import cv2
import time
import asyncio
import logging
from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

# 只显示关键模块的 DEBUG 日志
logging.getLogger('OptimizedVisionProcessor').setLevel(logging.DEBUG)
logging.getLogger('Layer1').setLevel(logging.DEBUG)
logging.getLogger('Layer2').setLevel(logging.DEBUG)
logging.getLogger('Layer3').setLevel(logging.DEBUG)
logging.getLogger('Layer4').setLevel(logging.DEBUG)

async def test_debug():
    """调试测试"""
    
    print("=" * 80)
    print(" 视觉系统调试测试 - 详细日志")
    print("=" * 80)
    
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("\n✅ 初始化完成，开始测试...")
    print("\n请执行以下动作：")
    print("1. 坐在摄像头前")
    print("2. 挥动手臂（大幅度）")
    print("3. 继续挥手并比手势")
    print("\n按 'q' 退出\n")
    
    await asyncio.sleep(2)
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 30:  # 30 秒测试
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = time.time()
            
            # 喂食帧
            processor.feed_frame(frame, timestamp)
            
            # 每 10 帧打印一次状态
            if frame_count % 10 == 0:
                print(f"\n--- 帧 {frame_count} ---")
                print(f"  L1: {processor.state_machine['layer1_state']}")
                print(f"  L2: {processor.state_machine['layer2_state']}")
                print(f"  L3: {processor.state_machine['layer3_state']}")
                print(f"  L4: {processor.state_machine['layer4_state']}")
                print(f"  shared_memory: human={processor.shared_memory['human_detected']}, motion={processor.shared_memory.get('motion_pixels', 0)}, action={processor.shared_memory.get('action_score', 0):.2f}")
            
            # 显示画面
            cv2.imshow("Debug Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\n测试完成：处理了 {frame_count} 帧")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(test_debug())
    except KeyboardInterrupt:
        print("\n\n测试中断")
