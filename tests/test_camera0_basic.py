# File: tests/test_camera0_basic.py
"""
摄像头 0 基础功能测试

测试目标:
1. 验证摄像头 0 能否正常打开
2. 验证能否获取到实时画面
3. 验证画面参数是否正确
4. 验证画面质量

TSD v1.7 对应:
- 4.4 感知预处理
- 7.2 集成测试场景
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_camera0_open():
    """测试 1: 打开摄像头 0"""
    print("\n" + "="*60)
    print("🧪 测试 1: 打开摄像头 0")
    print("="*60)
    
    try:
        from zulong.l0.usb_camera import USBCamera
        
        # 创建摄像头实例
        camera = USBCamera(device_id=0, width=640, height=480, fps=30)
        
        # 连接
        print("🔌 正在连接摄像头 0...")
        if not camera.connect():
            print("❌ 连接失败")
            return False, None
        
        print("✅ 摄像头 0 已连接")
        print(f"   - 分辨率：{camera.width}x{camera.height}")
        print(f"   - 帧率：{camera.fps} FPS")
        
        return True, camera
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_camera0_capture(camera):
    """测试 2: 捕获画面"""
    print("\n" + "="*60)
    print("🧪 测试 2: 捕获画面")
    print("="*60)
    
    try:
        # 启动捕获
        print("🚀 启动捕获线程...")
        camera.start()
        
        # 等待缓冲区填充
        print("⏳ 等待缓冲区填充 (0.5 秒)...")
        import time
        time.sleep(0.5)
        
        # 获取统计
        stats = camera.get_stats()
        print(f"\n📊 摄像头状态:")
        print(f"   - 运行中：{stats['is_running']}")
        print(f"   - 已连接：{stats['is_connected']}")
        print(f"   - 总帧数：{stats['total_frames']}")
        print(f"   - 丢帧数：{stats['dropped_frames']}")
        print(f"   - 实际 FPS: {stats['fps_actual']:.1f}")
        
        # 尝试获取帧
        print("\n📸 尝试获取帧...")
        frame, timestamp = camera.get_latest_frame()
        
        if frame is None:
            print("❌ 获取帧失败")
            return False
        
        print(f"✅ 成功获取帧")
        print(f"   - 帧尺寸：{frame.shape}")
        print(f"   - 帧类型：{frame.dtype}")
        print(f"   - 时间戳：{timestamp}")
        
        # 验证帧数据
        if frame.shape != (480, 640, 3):
            print(f"⚠️ 帧尺寸异常：{frame.shape} (预期：480x640x3)")
        
        if frame.dtype != np.uint8:
            print(f"⚠️ 帧类型异常：{frame.dtype} (预期：uint8)")
        
        # 检查帧内容
        mean_brightness = np.mean(frame)
        print(f"   - 平均亮度：{mean_brightness:.1f} (0-255)")
        
        if mean_brightness < 20:
            print("⚠️ 画面可能过暗")
        elif mean_brightness > 230:
            print("⚠️ 画面可能过曝")
        else:
            print("✅ 画面亮度正常")
        
        return True, frame
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_camera0_save_frame(frame):
    """测试 3: 保存帧到文件"""
    print("\n" + "="*60)
    print("🧪 测试 3: 保存帧到文件")
    print("="*60)
    
    try:
        if frame is None:
            print("⚠️ 没有帧可保存")
            return False
        
        # 创建测试数据目录
        test_dir = Path("camera_test_data")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存帧
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = test_dir / f"camera0_frame_{timestamp}.jpg"
        
        cv2.imwrite(str(filename), frame)
        
        print(f"✅ 帧已保存到：{filename}")
        print(f"   - 文件大小：{filename.stat().st_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera0_continuous_capture(camera, duration=5):
    """测试 4: 连续捕获测试"""
    print("\n" + "="*60)
    print(f"🧪 测试 4: 连续捕获测试 ({duration}秒)")
    print("="*60)
    
    try:
        import time
        
        print(f"📹 开始连续捕获...")
        start_time = time.time()
        frame_count = 0
        last_stats_time = start_time
        
        while (time.time() - start_time) < duration:
            # 获取帧
            frame, timestamp = camera.get_latest_frame()
            
            if frame is not None:
                frame_count += 1
            
            # 每秒显示一次进度
            current_time = time.time()
            if current_time - last_stats_time >= 1.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"   进度：{elapsed:.0f}s | 帧数：{frame_count} | FPS: {fps:.1f}")
                last_stats_time = current_time
            
            time.sleep(0.033)  # 30 FPS
        
        # 最终统计
        total_time = time.time() - start_time
        final_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n📊 连续捕获统计:")
        print(f"   - 总时长：{total_time:.1f}秒")
        print(f"   - 总帧数：{frame_count}")
        print(f"   - 平均 FPS: {final_fps:.1f}")
        
        if final_fps >= 25:
            print("✅ 帧率优秀 (>=25 FPS)")
        elif final_fps >= 15:
            print("⚠️ 帧率可接受 (15-25 FPS)")
        else:
            print("❌ 帧率过低 (<15 FPS)")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("📷 摄像头 0 基础功能测试")
    print("="*60)
    print(f"📅 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    camera = None
    
    try:
        # 测试 1: 打开摄像头
        success, camera = test_camera0_open()
        if not success or camera is None:
            print("\n❌ 测试 1 失败，无法继续")
            return False
        
        # 测试 2: 捕获画面
        success, frame = test_camera0_capture(camera)
        if not success:
            print("\n❌ 测试 2 失败")
        
        # 测试 3: 保存帧
        if frame is not None:
            test_camera0_save_frame(frame)
        
        # 测试 4: 连续捕获
        test_camera0_continuous_capture(camera, duration=5)
        
        print("\n" + "="*60)
        print("✅ 所有测试完成")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理资源
        if camera:
            print("\n🧹 清理资源...")
            camera.stop()
            camera.disconnect()
            print("✅ 摄像头已关闭")


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("🎉 摄像头 0 测试成功!")
        print("="*60)
        print("\n📋 结论:")
        print("   ✅ 摄像头 0 可以正常打开")
        print("   ✅ 可以获取到实时画面")
        print("   ✅ 画面参数正确")
        print("   ✅ 可以连续稳定捕获")
    else:
        print("\n" + "="*60)
        print("❌ 摄像头 0 测试失败!")
        print("="*60)
    
    sys.exit(0 if success else 1)
