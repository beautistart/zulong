# File: tests/test_layer3_confidence_stats.py
"""
Layer 3 动作分类置信度统计测试

详细记录动作分数分布，帮助确定合适的阈值
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import cv2
import logging
from datetime import datetime
from collections import defaultdict

from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor

logging.basicConfig(level=logging.INFO)

async def test():
    print("\n" + "=" * 80)
    print(" Layer 3: 动作分类置信度统计测试")
    print("=" * 80)
    print("\n说明:")
    print("  - 详细记录每个动作的置信度")
    print("  - 统计置信度分布")
    print("  - 视频窗口实时显示检测结果")
    print("  - 按 'q' 键退出测试")
    print("\n测试动作:")
    print("  - 静止不动")
    print("  - 挥手")
    print("  - 走近摄像头")
    print("  - 坐下/站起")
    print("  - 身体前倾交互")
    print("\n准备...")
    
    # 初始化
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 初始化完成，请开始做动作！\n")
    await asyncio.sleep(2)
    
    start_time = time.time()
    frame_count = 0
    
    # 统计信息
    intent_stats = defaultdict(list)  # 每个意图的置信度列表
    all_scores = []
    max_score = 0
    best_intent = "UNKNOWN"
    
    # 按时间段记录
    time_segments = []
    current_segment_start = start_time
    current_segment_scores = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            frame_count += 1
            timestamp = time.time()
            processor.feed_frame(frame, timestamp)
            await asyncio.sleep(0.05)
            
            # 获取状态
            mem = processor.shared_memory
            
            action_score = mem.get('action_score', 0)
            intent_type = mem.get('intent_type', 'UNKNOWN')
            
            # 记录数据
            all_scores.append(action_score)
            intent_stats[intent_type].append(action_score)
            
            if action_score > max_score:
                max_score = action_score
                best_intent = intent_type
            
            # 每 30 秒记录一个时间段
            if timestamp - current_segment_start >= 30:
                avg_score = sum(current_segment_scores) / len(current_segment_scores) if current_segment_scores else 0
                time_segments.append({
                    'start': current_segment_start - start_time,
                    'end': timestamp - start_time,
                    'avg_score': avg_score,
                    'frame_count': len(current_segment_scores)
                })
                current_segment_start = timestamp
                current_segment_scores = []
            
            current_segment_scores.append(action_score)
            
            # 在视频上绘制信息
            display_frame = frame.copy()
            
            # Layer 3 信息（大字体）
            score_color = (0, 255, 0) if action_score > 0.5 else (0, 255, 255) if action_score > 0.3 else (0, 0, 255)
            cv2.putText(display_frame, f"L3 Score: {action_score:.3f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 3)
            cv2.putText(display_frame, f"Intent: {intent_type}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 统计信息
            elapsed = time.time() - start_time
            cv2.putText(display_frame, f"Time: {elapsed:4.1f}s", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示视频窗口
            cv2.imshow('Layer 3 Confidence Stats - Press q to exit', display_frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n👋 用户请求退出...")
                break
        
        # 打印详细统计
        print("\n\n" + "=" * 80)
        print(" Layer 3 置信度统计报告")
        print("=" * 80)
        
        print(f"\n测试时长：{time.time() - start_time:.1f}秒")
        print(f"总帧数：{frame_count}帧")
        print(f"平均帧率：{frame_count / (time.time() - start_time):.1f} FPS")
        
        print("\n" + "-" * 80)
        print(" 1. 整体统计")
        print("-" * 80)
        print(f"最高分数：{max_score:.3f} ({best_intent})")
        print(f"平均分数：{sum(all_scores) / len(all_scores):.3f}" if all_scores else "N/A")
        print(f"分数范围：0.000 - {max(all_scores):.3f}" if all_scores else "N/A")
        
        print("\n" + "-" * 80)
        print(" 2. 各意图置信度分布")
        print("-" * 80)
        
        for intent, scores in sorted(intent_stats.items(), key=lambda x: len(x[1]), reverse=True):
            if len(scores) < 5:  # 跳过样本太少的
                continue
            
            avg = sum(scores) / len(scores)
            min_s = min(scores)
            max_s = max(scores)
            count = len(scores)
            
            print(f"\n{intent}:")
            print(f"  样本数：{count}帧")
            print(f"  平均分数：{avg:.3f}")
            print(f"  分数范围：{min_s:.3f} - {max_s:.3f}")
            
            # 分数段分布
            ranges = [
                (0, 0.2, "0.00-0.20"),
                (0.2, 0.4, "0.20-0.40"),
                (0.4, 0.6, "0.40-0.60"),
                (0.6, 0.8, "0.60-0.80"),
                (0.8, 1.0, "0.80-1.00")
            ]
            
            print(f"  分布:", end=" ")
            dist_parts = []
            for low, high, label in ranges:
                count_in_range = sum(1 for s in scores if low <= s < high)
                pct = count_in_range / len(scores) * 100
                if pct > 5:  # 只显示超过 5% 的
                    dist_parts.append(f"{label}: {pct:.0f}%")
            print(", ".join(dist_parts))
        
        print("\n" + "-" * 80)
        print(" 3. 时间段分析")
        print("-" * 80)
        
        for i, seg in enumerate(time_segments):
            print(f"时间段 {i+1}: {seg['start']:.0f}s - {seg['end']:.0f}s, "
                  f"平均分数：{seg['avg_score']:.3f}, 帧数：{seg['frame_count']}")
        
        print("\n" + "-" * 80)
        print(" 4. 阈值建议")
        print("-" * 80)
        
        # 分析不同阈值下的检测率
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for thresh in thresholds:
            above_thresh = sum(1 for s in all_scores if s >= thresh)
            pct = above_thresh / len(all_scores) * 100 if all_scores else 0
            print(f"阈值 {thresh:.1f}: {above_thresh}帧 ({pct:.1f}%) 超过阈值")
        
        print("\n" + "=" * 80)
        print(" 建议配置:")
        print("=" * 80)
        
        # 根据统计给出建议
        if max_score < 0.3:
            print("⚠️  检测到的动作分数普遍较低")
            print("建议:")
            print("  - 降低 intent_threshold 到 0.2-0.3")
            print("  - 降低 interact_threshold 到 0.5-0.6")
        elif max_score < 0.5:
            print("✅ 动作检测正常，分数中等")
            print("建议配置:")
            print("  - intent_threshold: 0.3-0.4")
            print("  - interact_threshold: 0.6-0.7")
        else:
            print("✅ 动作检测良好，分数较高")
            print("建议配置:")
            print("  - intent_threshold: 0.4-0.5")
            print("  - interact_threshold: 0.7-0.8")
        
        print("\n" + "=" * 80)
        
        # 保存截图
        screenshot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'screenshots')
        os.makedirs(screenshot_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(screenshot_dir, f"layer3_stats_{timestamp}.png")
        cv2.imwrite(screenshot_path, display_frame)
        print(f"\n📸 截图已保存：{screenshot_path}")
        
        return True
        
    finally:
        cap.release()
        processor.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(test())
