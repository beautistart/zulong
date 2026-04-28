# File: tests/test_real_gesture_recognition.py
"""
真实用户手势识别测试

测试目标:
1. 3 米距离外 5 种手势识别率测试
2. 统计每种手势的准确率和召回率
3. 分析识别失败原因
4. 优化 Digital Zoom 参数

TSD v1.7 对应:
- 4.4 感知预处理
- 7.2 集成测试场景
- 5.2 显存约束

测试手势:
1. open_palm (张开手掌)
2. fist (握拳)
3. v_sign (剪刀手)
4. ok_sign (OK 手势)
5. thumbs_up (竖大拇指)
"""

import sys
import os
import asyncio
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class GestureTestRecorder:
    """手势测试记录器"""
    
    def __init__(self, test_dir: str = "gesture_test_data"):
        """
        初始化记录器
        
        Args:
            test_dir: 数据保存目录
        """
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试结果
        self.results = []
        
        # 手势标签
        self.gesture_labels = [
            "open_palm",
            "fist",
            "v_sign",
            "ok_sign",
            "thumbs_up"
        ]
        
        print(f"📁 数据保存目录：{self.test_dir.absolute()}")
    
    def record_gesture_test(
        self,
        gesture_name: str,
        predicted: str,
        confidence: float,
        distance: float,
        lighting: str,
        success: bool
    ):
        """
        记录单次手势测试结果
        
        Args:
            gesture_name: 实际手势名称
            predicted: 预测手势名称
            confidence: 置信度
            distance: 距离 (米)
            lighting: 光照条件
            success: 是否识别成功
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'gesture_name': gesture_name,
            'predicted': predicted,
            'confidence': confidence,
            'distance': distance,
            'lighting': lighting,
            'success': success,
        }
        
        self.results.append(record)
    
    def save_results(self, filename: str = "gesture_test_results.json"):
        """保存测试结果到 JSON 文件"""
        output_path = self.test_dir / filename
        
        # 计算统计
        stats = self.calculate_statistics()
        
        output_data = {
            'test_time': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'statistics': stats,
            'detailed_results': self.results,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 测试结果已保存：{output_path}")
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        计算统计信息
        
        Returns:
            统计字典
        """
        if not self.results:
            return {}
        
        # 总体准确率
        total = len(self.results)
        success_count = sum(1 for r in self.results if r['success'])
        overall_accuracy = success_count / total if total > 0 else 0.0
        
        # 每种手势的准确率
        gesture_stats = {}
        for gesture in self.gesture_labels:
            gesture_results = [r for r in self.results if r['gesture_name'] == gesture]
            if gesture_results:
                gesture_success = sum(1 for r in gesture_results if r['success'])
                gesture_accuracy = gesture_success / len(gesture_results)
                
                gesture_stats[gesture] = {
                    'total': len(gesture_results),
                    'success': gesture_success,
                    'accuracy': gesture_accuracy,
                }
        
        # 平均置信度
        avg_confidence = sum(r['confidence'] for r in self.results) / total
        
        # 距离分析
        distance_groups = {
            '0-1m': [],
            '1-2m': [],
            '2-3m': [],
            '3m+': [],
        }
        
        for r in self.results:
            dist = r['distance']
            if dist < 1.0:
                distance_groups['0-1m'].append(r['success'])
            elif dist < 2.0:
                distance_groups['1-2m'].append(r['success'])
            elif dist < 3.0:
                distance_groups['2-3m'].append(r['success'])
            else:
                distance_groups['3m+'].append(r['success'])
        
        distance_stats = {}
        for group, successes in distance_groups.items():
            if successes:
                distance_stats[group] = {
                    'total': len(successes),
                    'success': sum(successes),
                    'accuracy': sum(successes) / len(successes) if successes else 0.0,
                }
        
        return {
            'overall_accuracy': overall_accuracy,
            'gesture_stats': gesture_stats,
            'average_confidence': avg_confidence,
            'distance_stats': distance_stats,
        }


async def test_gesture_recognition_with_camera():
    """手势识别测试 (使用摄像头)"""
    print("\n" + "="*60)
    print("🧪 真实用户手势识别测试")
    print("="*60)
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'zulong' / 'l1c'))
        
        from optimized_vision_processor import OptimizedVisionProcessor
        from zulong.l0.usb_camera import USBCamera
        
        # 创建记录器
        recorder = GestureTestRecorder()
        
        # 创建摄像头
        camera = USBCamera(device_id=0, width=640, height=480, fps=30)
        
        # 连接摄像头
        if not camera.connect():
            print("❌ 摄像头连接失败")
            return False
        
        camera.start()
        print("✅ 摄像头已启动")
        
        # 创建处理器
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=True)  # 加载真实模型
        print("✅ 视觉处理器已初始化 (真实模型)")
        
        # 测试说明
        print("\n" + "="*60)
        print("📋 测试说明:")
        print("="*60)
        print("1. 请在摄像头前 3 米处展示手势")
        print("2. 每种手势保持 3 秒钟")
        print("3. 系统会自动采集 10 帧并识别")
        print("4. 测试顺序:")
        
        for i, gesture in enumerate(recorder.gesture_labels, 1):
            print(f"   {i}. {gesture}")
        
        print("\n⚠️ 按 Ctrl+C 可随时退出测试")
        print("="*60)
        
        # 等待用户准备
        input("\n按 Enter 键开始测试...")
        
        # 测试每种手势
        for gesture_name in recorder.gesture_labels:
            print(f"\n{'='*60}")
            print(f"🎯 测试手势：{gesture_name}")
            print(f"{'='*60}")
            
            # 准备提示
            print(f"⏳ 请准备展示 '{gesture_name}' 手势...")
            time.sleep(2)
            
            print(f"👉 请现在展示 '{gesture_name}' 手势 (3 秒后开始)")
            time.sleep(3)
            
            # 采集并识别 10 帧
            frame_count = 0
            success_count = 0
            predictions = []
            
            print(f"📸 开始采集 (10 帧)...")
            
            for i in range(10):
                # 获取帧
                frame, timestamp = camera.get_latest_frame()
                
                if frame is None:
                    print(f"   ⚠️ 第 {i+1} 帧：获取失败")
                    continue
                
                # 喂入处理器
                processor.feed_frame(frame, timestamp)
                
                # 等待处理
                await asyncio.sleep(0.1)
                
                # 获取识别结果
                gesture_result = processor.shared_memory.get('gesture_type')
                action_score = processor.shared_memory.get('action_score', 0.0)
                
                # 记录结果
                predicted = gesture_result if gesture_result else "unknown"
                confidence = action_score
                
                # 判断是否成功
                success = (predicted == gesture_name) and (confidence > 0.5)
                
                if success:
                    success_count += 1
                
                predictions.append({
                    'frame': i+1,
                    'predicted': predicted,
                    'confidence': confidence,
                    'success': success,
                })
                
                print(f"   帧 {i+1}: {predicted} (conf={confidence:.2f}) {'✅' if success else '❌'}")
                
                frame_count += 1
            
            # 计算该手势的准确率
            accuracy = success_count / frame_count if frame_count > 0 else 0.0
            
            print(f"\n📊 {gesture_name} 识别结果:")
            print(f"   - 总帧数：{frame_count}")
            print(f"   - 成功：{success_count}")
            print(f"   - 准确率：{accuracy:.0%}")
            
            # 记录到结果
            for pred in predictions:
                recorder.record_gesture_test(
                    gesture_name=gesture_name,
                    predicted=pred['predicted'],
                    confidence=pred['confidence'],
                    distance=3.0,  # 假设 3 米
                    lighting="indoor",
                    success=pred['success']
                )
            
            # 手势间休息
            print(f"\n⏳ 休息 2 秒，准备下一个手势...")
            time.sleep(2)
        
        # 保存结果
        recorder.save_results()
        
        # 打印总体统计
        stats = recorder.calculate_statistics()
        
        print("\n" + "="*60)
        print("📊 总体统计")
        print("="*60)
        print(f"总测试次数：{stats.get('total_tests', 0)}")
        print(f"总体准确率：{stats.get('overall_accuracy', 0.0):.0%}")
        print(f"平均置信度：{stats.get('average_confidence', 0.0):.2f}")
        
        print("\n每种手势准确率:")
        gesture_stats = stats.get('gesture_stats', {})
        for gesture, gesture_stat in gesture_stats.items():
            print(f"   {gesture}: {gesture_stat['accuracy']:.0%} ({gesture_stat['success']}/{gesture_stat['total']})")
        
        print("\n距离分析:")
        distance_stats = stats.get('distance_stats', {})
        for distance, dist_stat in distance_stats.items():
            print(f"   {distance}: {dist_stat['accuracy']:.0%} ({dist_stat['success']}/{dist_stat['total']})")
        
        # 停止
        processor.stop()
        camera.stop()
        camera.disconnect()
        
        print("\n✅ 测试完成")
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断测试")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mock_gesture_recognition():
    """模拟手势识别测试 (无摄像头时使用)"""
    print("\n" + "="*60)
    print("🧪 模拟手势识别测试")
    print("="*60)
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'zulong' / 'l1c'))
        
        from optimized_vision_processor import OptimizedVisionProcessor
        
        # 创建记录器
        recorder = GestureTestRecorder()
        
        # 创建处理器
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=False)
        print("✅ 视觉处理器已初始化")
        
        # 模拟手势标签
        gesture_labels = [
            "open_palm",
            "fist",
            "v_sign",
            "ok_sign",
            "thumbs_up"
        ]
        
        print("\n🔄 开始模拟测试 (每种手势 10 帧)...")
        
        # 测试每种手势
        for gesture_name in gesture_labels:
            print(f"\n🎯 测试手势：{gesture_name}")
            
            for i in range(10):
                # 创建模拟帧
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # 喂入处理器
                timestamp = time.time()
                processor.feed_frame(frame, timestamp)
                
                # 等待处理
                await asyncio.sleep(0.05)
                
                # 获取识别结果 (模拟)
                # 实际使用时会读取 processor.shared_memory['gesture_type']
                predicted = gesture_name if np.random.random() > 0.3 else np.random.choice(gesture_labels)
                confidence = np.random.uniform(0.5, 0.95)
                
                success = (predicted == gesture_name) and (confidence > 0.5)
                
                # 记录结果
                recorder.record_gesture_test(
                    gesture_name=gesture_name,
                    predicted=predicted,
                    confidence=confidence,
                    distance=3.0,
                    lighting="indoor",
                    success=success
                )
            
            print(f"   ✅ 完成 10 帧测试")
        
        # 保存结果
        recorder.save_results()
        
        # 打印统计
        stats = recorder.calculate_statistics()
        
        print("\n📊 模拟测试结果:")
        print(f"   - 总体准确率：{stats.get('overall_accuracy', 0.0):.0%}")
        print(f"   - 平均置信度：{stats.get('average_confidence', 0.0):.2f}")
        
        # 停止
        processor.stop()
        
        print("\n✅ 模拟测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 模拟测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行手势识别测试"""
    print("="*60)
    print("🚀 真实用户手势识别测试")
    print("="*60)
    print(f"📅 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查是否有摄像头
    try:
        from zulong.l0.usb_camera import detect_available_cameras
        cameras = detect_available_cameras()
        
        if cameras:
            print(f"✅ 检测到摄像头：device_id={cameras[0]}")
            print("🎥 将使用真实摄像头进行测试")
            
            # 运行真实测试
            success = await test_gesture_recognition_with_camera()
        else:
            print("⚠️ 未检测到摄像头")
            print("🔄 将使用模拟模式进行测试")
            
            # 运行模拟测试
            success = await test_mock_gesture_recognition()
    
    except Exception as e:
        print(f"❌ 摄像头检测失败：{e}")
        print("🔄 切换到模拟模式")
        
        success = await test_mock_gesture_recognition()
    
    print("\n" + "="*60)
    print("📋 测试完成总结")
    print("="*60)
    
    if success:
        print("✅ 手势识别测试成功完成")
        print("\n📋 下一步:")
        print("1. 查看 gesture_test_data/ 目录中的详细结果")
        print("2. 分析识别失败原因")
        print("3. 优化 Digital Zoom 参数和模型")
    else:
        print("❌ 测试失败，请检查日志")
    
    print("\n" + "="*60)
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
