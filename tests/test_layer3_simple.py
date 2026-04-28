# File: tests/test_layer3_simple.py
"""
Layer 3 动作分类简单测试
"""

import asyncio
import time
import cv2
import logging

from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor

logging.basicConfig(level=logging.WARNING)

async def test():
    print("\n" + "=" * 80)
    print(" Layer 3: 动作分类简单测试")
    print("=" * 80)
    print("\n请在摄像头前持续挥手 2-3 秒")
    print("按 'q' 退出\n")
    
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("✅ 初始化完成，请开始挥手！\n")
    await asyncio.sleep(2)
    
    start_time = time.time()
    max_score = 0
    best_intent = "UNKNOWN"
    
    try:
        while time.time() - start_time < 15:
            ret, frame = cap.read()
            if not ret:
                break
            
            processor.feed_frame(frame, time.time())
            await asyncio.sleep(0.05)
            
            mem = processor.shared_memory
            score = mem.get('action_score', 0)
            intent = mem.get('intent_type', 'UNKNOWN')
            
            if score > max_score:
                max_score = score
                best_intent = intent
            
            # 每 0.5 秒显示一次
            if int((time.time() - start_time) * 2) > int((time.time() - start_time - 0.05) * 2):
                print(f"\r[{time.time() - start_time:4.1f}s] Score: {score:.3f} | Intent: {intent}", end='', flush=True)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("\n\n" + "=" * 80)
        print(" 测试结果")
        print("=" * 80)
        print(f"最大动作分数：{max_score:.3f}")
        print(f"最佳意图：{best_intent}")
        
        if max_score > 0.5:
            print("\n✅ Layer 3 通过！动作分类工作正常！")
        elif max_score > 0.3:
            print("\n⚠️  Layer 3 检测到动作，但分数较低")
            print("提示：动作幅度更大一些，持续挥手 2-3 秒")
        else:
            print("\n❌ Layer 3 未检测到明显动作")
            print("提示：")
            print("  - 确保光线充足")
            print("  - 手臂完全伸展")
            print("  - 持续挥手 2-3 秒")
        
    finally:
        cap.release()
        processor.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(test())
