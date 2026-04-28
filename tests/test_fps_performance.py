# File: tests/test_fps_performance.py
"""
视觉系统 - FPS 性能测试

测试 YOLO 推理频率优化后的 FPS 性能：
- 目标 FPS: 15-20
- 对比优化前：~11-12 FPS
- 预期提升：30-50%
"""

import asyncio
import time
import cv2
import logging
from typing import Dict, Any

from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 只显示警告和错误
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("TestFPS")


async def test_fps_performance():
    """测试优化后的 FPS 性能"""
    
    print("=" * 60)
    print(" 视觉系统 - FPS 性能测试")
    print("=" * 60)
    
    # 初始化视觉处理器
    print("\n初始化 OptimizedVisionProcessor...")
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    logger.info("✅ 视觉处理器初始化完成")
    
    # 打开摄像头
    print("\n打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("❌ 无法打开摄像头")
        return
    
    logger.info("✅ 摄像头已打开")
    
    # 等待模型加载
    print("\n等待 3 秒让模型完全加载...")
    await asyncio.sleep(3)
    
    # 开始性能测试
    print("\n" + "=" * 60)
    print(" 开始性能测试 (10 秒)")
    print("=" * 60)
    print("请坐在摄像头前，偶尔挥动手臂...\n")
    
    # 性能统计
    start_time = time.time()
    frame_count = 0
    yolo_inference_count = 0
    test_duration = 10  # 秒
    
    fps_samples = []  # FPS 采样
    current_second_frames = 0
    last_second_time = start_time
    
    try:
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_second_frames += 1
            timestamp = time.time()
            
            # 每秒记录一次 FPS
            if timestamp - last_second_time >= 1.0:
                fps_samples.append(current_second_frames)
                current_second_frames = 0
                last_second_time = timestamp
            
            # 喂食帧到视觉处理器
            processor.feed_frame(frame, timestamp)
            
            # 统计 YOLO 推理次数
            if processor._frame_counter == processor._last_yolo_inference_frame:
                yolo_inference_count += 1
            
            # 显示实时 FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {test_duration - elapsed:.0f}s", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示 YOLO 推理状态
            frames_since_yolo = processor._frame_counter - processor._last_yolo_inference_frame
            yolo_status = f"YOLO: {frames_since_yolo}/3"
            cv2.putText(frame, yolo_status, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("FPS Performance Test", frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 测试完成，打印统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print("\n" + "=" * 60)
        print(" 测试结果")
        print("=" * 60)
        print(f"\n总帧数：{frame_count}")
        print(f"总时间：{total_time:.2f}秒")
        print(f"平均 FPS: {avg_fps:.2f}")
        
        if fps_samples:
            min_fps = min(fps_samples)
            max_fps = max(fps_samples)
            print(f"FPS 范围：{min_fps} - {max_fps}")
            print(f"FPS 采样：{fps_samples}")
        
        # YOLO 推理统计
        yolo_frequency = processor._config['yolo_inference_frequency']
        expected_yolo_inferences = frame_count // yolo_frequency
        actual_yolo_inferences = processor._last_yolo_inference_frame
        
        print(f"\nYOLO 推理频率：1/{yolo_frequency}")
        print(f"预期 YOLO 推理次数：~{expected_yolo_inferences}")
        print(f"实际 YOLO 推理次数：{actual_yolo_inferences}")
        
        # Layer 3 统计
        layer3_executions = frame_count // 2  # 每 2 帧执行一次
        print(f"\nLayer 3 动作分类频率：1/2")
        print(f"预期 Layer 3 执行次数：~{layer3_executions}")
        
        # 验证结果
        print("\n" + "=" * 60)
        print(" 性能评估")
        print("=" * 60)
        
        target_fps_min = 15
        target_fps_max = 20
        
        if avg_fps >= target_fps_min:
            print(f"✅ FPS 达标！({avg_fps:.2f} >= {target_fps_min})")
            
            if avg_fps >= target_fps_max:
                print(f"🎉 性能优秀！超过目标上限 ({avg_fps:.2f} >= {target_fps_max})")
            else:
                print(f"✨ 性能良好！达到目标范围 ({target_fps_min} <= {avg_fps:.2f} < {target_fps_max})")
        else:
            print(f"❌ FPS 未达标！({avg_fps:.2f} < {target_fps_min})")
            print(f"💡 建议：尝试降低 YOLO 推理频率 (当前 1/{yolo_frequency} -> 1/5 或 1/10)")
        
        # 计算性能提升
        baseline_fps = 11.5  # 优化前 FPS
        improvement = ((avg_fps - baseline_fps) / baseline_fps) * 100
        
        print(f"\n性能提升：{improvement:+.1f}% (基准：{baseline_fps:.1f} FPS)")
        
        if improvement > 30:
            print("🎉 性能提升显著！超过 30% 目标！")
        elif improvement > 15:
            print("✨ 性能提升良好！")
        else:
            print("⚠️ 性能提升有限，可能需要进一步优化")
        
        return avg_fps >= target_fps_min
        
    except Exception as e:
        logger.error(f"❌ 测试失败：{e}", exc_info=True)
        return False
    
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        logger.info("✅ 资源已释放")


if __name__ == "__main__":
    try:
        result = asyncio.run(test_fps_performance())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        exit(0)
