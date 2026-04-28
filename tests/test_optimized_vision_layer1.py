# File: tests/test_optimized_vision_layer1.py
"""
优化视觉处理器 Phase 1 测试：YOLO-Nano 人体检测 + ROI 增益放大

测试目标:
1. 验证 YOLO-Nano 人体检测功能
2. 验证 ROI 增益放大逻辑 (3 米外手部微动检测)
3. 对比优化前后的检测灵敏度

TSD v1.7 对应:
- 4.4 感知预处理
- 5.2 显存约束
"""

import asyncio
import numpy as np
import cv2
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接导入优化版本，避免 zulong.l1c.__init__.py 的依赖问题
from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor


def create_test_frame(
    frame_size: tuple = (640, 480),
    has_human: bool = True,
    hand_position: tuple = None,
    hand_moving: bool = False,
    timestamp: float = 0.0
) -> np.ndarray:
    """
    创建测试帧 (模拟模式)
    
    Args:
        frame_size: 帧尺寸 (宽，高)
        has_human: 是否包含人体
        hand_position: 手部位置 (x, y)
        hand_moving: 手部是否移动
        timestamp: 时间戳 (用于模拟运动)
    
    Returns:
        BGR 格式测试帧
    """
    w, h = frame_size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 绘制背景
    cv2.rectangle(frame, (0, 0), (w, h), (50, 50, 50), -1)
    
    if has_human:
        # 绘制人体矩形框 (模拟 3 米外的人)
        person_w = int(w * 0.2)  # 3 米外人体宽度约占 20%
        person_h = int(h * 0.5)
        person_x = w // 2 - person_w // 2
        person_y = h // 2 - person_h // 2
        
        cv2.rectangle(frame, (person_x, person_y), 
                     (person_x + person_w, person_y + person_h), 
                     (0, 100, 200), -1)
        
        # 绘制手部区域
        if hand_position:
            hand_x, hand_y = hand_position
            
            # 模拟手部微动 (正弦波)
            if hand_moving:
                offset = int(10 * np.sin(timestamp * 5))  # 5Hz 振动
                hand_y += offset
            
            hand_size = 15  # 3 米外手部大小 (像素)
            cv2.circle(frame, (hand_x, hand_y), hand_size, (0, 255, 0), -1)
    
    return frame


async def test_layer1_human_detection():
    """测试 Layer 1: YOLO-Nano 人体检测"""
    print("\n" + "="*60)
    print("🧪 测试 Layer 1: YOLO-Nano 人体检测")
    print("="*60)
    
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=False)  # 使用模拟模式
    
    # 测试帧 1: 有人体
    frame_with_human = create_test_frame(has_human=True)
    bboxes = processor._layer1_human_detection(frame_with_human)
    
    print(f"✅ 测试帧 1 (有人体): 检测到 {len(bboxes)} 个人体")
    assert len(bboxes) > 0, "应该检测到人体"
    print(f"   检测框：{bboxes[0]}")
    
    # 测试帧 2: 无人体
    frame_no_human = create_test_frame(has_human=False)
    bboxes = processor._layer1_human_detection(frame_no_human)
    
    print(f"✅ 测试帧 2 (无人体): 检测到 {len(bboxes)} 个人体")
    assert len(bboxes) == 0, "不应该检测到人体"
    
    print("\n✅ Layer 1 测试通过")
    return True


async def test_layer2_roi_gain_amplification():
    """测试 Layer 2: ROI 增益放大"""
    print("\n" + "="*60)
    print("🧪 测试 Layer 2: ROI 增益放大 (关键测试)")
    print("="*60)
    
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=False)
    
    # 模拟人体检测框
    human_bbox = [200, 150, 400, 400]  # [x1, y1, x2, y2]
    
    # ========== 测试场景 1: 无增益 (传统光流法) ==========
    print("\n📊 场景 1: 传统全图光流法 (无 ROI 增益)")
    processor._config['roi_gain_coefficient'] = 1.0  # 无增益
    
    # 创建连续帧 (模拟 3 米外手部微动)
    base_time = time.time()
    frame1 = create_test_frame(
        has_human=True,
        hand_position=(300, 200),
        hand_moving=False,
        timestamp=base_time
    )
    frame2 = create_test_frame(
        has_human=True,
        hand_position=(300, 200),
        hand_moving=True,
        timestamp=base_time + 0.033  # 30 FPS
    )
    
    # 添加到缓冲区
    processor.frame_buffer.append(frame1)
    processor.frame_buffer.append(frame2)
    
    motion_detected, motion_pixels = processor._layer2_roi_motion_detection(
        frame2, human_bbox
    )
    
    print(f"   运动像素：{motion_pixels}")
    print(f"   检测结果：{'✅ 检测到' if motion_detected else '❌ 未检测到'}")
    
    # ========== 测试场景 2: ROI 增益 (优化方案) ==========
    print("\n📊 场景 2: ROI 增益放大 (优化方案)")
    processor._config['roi_gain_coefficient'] = 3.0  # 3 倍增益
    
    motion_detected_enhanced, motion_pixels_enhanced = processor._layer2_roi_motion_detection(
        frame2, human_bbox
    )
    
    print(f"   运动像素：{motion_pixels_enhanced}")
    print(f"   检测结果：{'✅ 检测到' if motion_detected_enhanced else '❌ 未检测到'}")
    
    # ========== 对比分析 ==========
    print("\n📈 对比分析:")
    print(f"   增益倍数：{processor._config['roi_gain_coefficient']}x")
    print(f"   运动像素提升：{motion_pixels_enhanced - motion_pixels} ({(motion_pixels_enhanced/motion_pixels - 1)*100:.0f}%)" if motion_pixels > 0 else "   运动像素提升：N/A")
    print(f"   检测灵敏度：{'✅ 提升' if motion_detected_enhanced and not motion_detected else '➡️ 保持'}")
    
    # 断言：ROI 增益应该提高检测灵敏度
    assert motion_pixels_enhanced >= motion_pixels, "ROI 增益应该增加运动像素"
    
    print("\n✅ Layer 2 测试通过")
    return True


async def test_3meter_hand_gesture_detection():
    """测试 3 米外手势检测 (综合场景)"""
    print("\n" + "="*60)
    print("🧪 测试场景：3 米外手势识别")
    print("="*60)
    
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=False)
    
    # 配置优化参数
    processor._config['roi_gain_coefficient'] = 3.0
    processor._config['roi_motion_threshold'] = 200  # 降低阈值
    
    # 模拟 3 米外场景
    human_bbox = [200, 150, 400, 400]  # 3 米外人体
    
    # 生成连续帧序列 (模拟挥手动作)
    print("\n🎬 生成挥手动作序列 (10 帧)...")
    base_time = time.time()
    
    for i in range(10):
        t = base_time + i * 0.033  # 30 FPS
        
        # 模拟挥手 (正弦波上下运动)
        hand_y_offset = int(20 * np.sin(i * 0.5))
        
        frame = create_test_frame(
            has_human=True,
            hand_position=(300, 200 + hand_y_offset),
            hand_moving=True,
            timestamp=t
        )
        
        # 输入处理器
        processor.feed_frame(frame, t)
        
        # 等待处理
        await asyncio.sleep(0.05)
    
    # 检查共享内存
    shared_mem = processor.get_shared_memory()
    
    print("\n📊 检测结果:")
    print(f"   人体检测：{'✅' if shared_mem['human_detected'] else '❌'}")
    print(f"   运动像素：{shared_mem['motion_pixels']}")
    print(f"   动作分数：{shared_mem['action_score']:.2f}")
    print(f"   手势类型：{shared_mem['gesture_type']}")
    
    # 断言
    assert shared_mem['human_detected'], "应该检测到人体"
    assert shared_mem['motion_pixels'] > 0, "应该检测到运动"
    
    print("\n✅ 3 米外手势检测测试通过")
    return True


async def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🚀 优化视觉处理器 Phase 1 测试")
    print("方案：人体锚点驱动 (YOLO-Nano + ROI 增益)")
    print("="*60)
    
    tests = [
        ("Layer 1: 人体检测", test_layer1_human_detection),
        ("Layer 2: ROI 增益", test_layer2_roi_gain_amplification),
        ("3 米外手势检测", test_3meter_hand_gesture_detection),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 测试失败：{e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
        
        await asyncio.sleep(0.5)  # 冷却时间
    
    # 汇总报告
    print("\n" + "="*60)
    print("📊 测试汇总报告")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计：{passed}/{total} 测试通过 ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！Phase 1 完成")
    else:
        print("\n⚠️ 部分测试失败，请检查日志")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
