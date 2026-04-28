# File: tests/test_vision_processor_bootstrap.py
"""
测试视觉处理器在 bootstrap 模式下是否正常工作

测试目标:
1. 验证视觉处理器已正确初始化
2. 验证 feed_frame 方法能正常接收帧
3. 验证四层处理逻辑是否执行
4. 验证日志输出是否正常
"""

import sys
import time
import numpy as np
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置日志级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("TestVisionProcessor")


def test_vision_processor_initialization():
    """测试 1: 验证视觉处理器初始化"""
    print("\n" + "=" * 80)
    print(" 测试 1: 验证视觉处理器初始化")
    print("=" * 80)
    
    try:
        from zulong.l1c.optimized_vision_processor import get_vision_processor
        
        vp = get_vision_processor()
        
        if vp is None:
            print("❌ 视觉处理器未初始化")
            return False
        
        print(f"✅ 视觉处理器已初始化：{vp}")
        print(f"   - is_running: {vp.is_running}")
        print(f"   - is_initialized: {vp.is_initialized}")
        print(f"   - ROI Gain: {vp._config['roi_gain_coefficient']}x")
        print(f"   - Digital Zoom: {vp._config['digital_zoom_factor']}x")
        print(f"   - YOLO 模型：{vp._yolo_model is not None}")
        print(f"   - 动作分类器：{vp._action_classifier is not None}")
        print(f"   - 手势识别器：{vp._gesture_classifier is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ 初始化测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_feed_frame():
    """测试 2: 验证 feed_frame 方法能正常接收帧"""
    print("\n" + "=" * 80)
    print(" 测试 2: 验证 feed_frame 方法")
    print("=" * 80)
    
    try:
        from zulong.l1c.optimized_vision_processor import get_vision_processor
        
        vp = get_vision_processor()
        
        if vp is None:
            print("❌ 视觉处理器未初始化")
            return False
        
        # 创建模拟帧 (640x480x3 BGR)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        timestamp = time.time()
        
        print(f"📥 调用 feed_frame (帧大小：{test_frame.shape})")
        vp.feed_frame(test_frame, timestamp)
        
        # 检查帧缓冲区
        if len(vp.frame_buffer) > 0:
            print(f"✅ 帧已添加到缓冲区：{len(vp.frame_buffer)} 帧")
        else:
            print("❌ 帧未添加到缓冲区")
            return False
        
        # 等待处理完成
        time.sleep(0.5)
        
        # 检查共享内存
        print(f"\n📊 共享内存状态:")
        print(f"   - human_detected: {vp.shared_memory['human_detected']}")
        print(f"   - human_bbox: {vp.shared_memory['human_bbox']}")
        print(f"   - motion_pixels: {vp.shared_memory['motion_pixels']}")
        print(f"   - action_score: {vp.shared_memory.get('action_score', 'N/A')}")
        print(f"   - intent_type: {vp.shared_memory.get('intent_type', 'N/A')}")
        print(f"   - gesture_type: {vp.shared_memory.get('gesture_type', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ feed_frame 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_layer1_human_detection():
    """测试 3: 验证 Layer 1 人体检测"""
    print("\n" + "=" * 80)
    print(" 测试 3: 验证 Layer 1 人体检测")
    print("=" * 80)
    
    try:
        from zulong.l1c.optimized_vision_processor import get_vision_processor
        
        vp = get_vision_processor()
        
        if vp is None:
            print("❌ 视觉处理器未初始化")
            return False
        
        # 创建模拟帧
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print(f"🔍 调用 _layer1_human_detection")
        bboxes = vp._layer1_human_detection(test_frame)
        
        print(f"📊 检测结果：{len(bboxes)} 个人体")
        if bboxes:
            print(f"   - 检测框：{bboxes[0]}")
        
        # 检查状态机
        print(f"   - Layer1 状态：{vp.state_machine['layer1_state']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Layer 1 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_machine():
    """测试 4: 验证状态机"""
    print("\n" + "=" * 80)
    print(" 测试 4: 验证状态机")
    print("=" * 80)
    
    try:
        from zulong.l1c.optimized_vision_processor import get_vision_processor
        
        vp = get_vision_processor()
        
        if vp is None:
            print("❌ 视觉处理器未初始化")
            return False
        
        print(f"📊 状态机状态:")
        for state, value in vp.state_machine.items():
            print(f"   - {state}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 状态机测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print(" 🧪 视觉处理器 Bootstrap 模式测试")
    print("=" * 80)
    
    # 等待系统启动
    print("\n⏳ 等待系统启动...")
    time.sleep(2.0)
    
    # 运行测试
    results = []
    
    results.append(("初始化测试", test_vision_processor_initialization()))
    results.append(("feed_frame 测试", test_feed_frame()))
    results.append(("Layer 1 人体检测", test_layer1_human_detection()))
    results.append(("状态机测试", test_state_machine()))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print(" 📊 测试结果汇总")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计：{passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！视觉处理器工作正常！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查日志")
        return 1


if __name__ == "__main__":
    exit(main())
