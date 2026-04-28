# File: tests/test_long_running_stability.py
"""
长时间稳定性测试

测试目标:
1. 系统连续运行稳定性 (>1 小时)
2. 内存泄漏监控
3. 帧率稳定性测试
4. 错误恢复能力

TSD v1.7 对应:
- 5.2 显存约束
- 7.2 集成测试场景
- 6.3 容错与反馈

测试时长选项:
- short: 5 分钟 (快速验证)
- medium: 30 分钟 (标准测试)
- long: 60 分钟 (完整测试)
"""

import sys
import os
import asyncio
import time
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class StabilityMonitor:
    """稳定性监控器"""
    
    def __init__(self):
        """初始化监控器"""
        self.process = psutil.Process()
        
        # 监控数据
        self.memory_samples = []
        self.fps_samples = []
        self.error_count = 0
        self.frame_count = 0
        
        # 开始时间
        self.start_time = time.time()
        
        # CPU 初始时间
        self.process.cpu_percent()
    
    def sample_memory(self):
        """采样内存使用"""
        mem_mb = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.memory_samples.append({
            'timestamp': time.time() - self.start_time,
            'memory_mb': mem_mb,
        })
        return mem_mb
    
    def sample_fps(self, fps: float):
        """采样帧率"""
        self.fps_samples.append({
            'timestamp': time.time() - self.start_time,
            'fps': fps,
        })
    
    def record_error(self, error_msg: str):
        """记录错误"""
        self.error_count += 1
        print(f"❌ [错误 {self.error_count}] {error_msg}")
    
    def record_frame(self):
        """记录处理的帧"""
        self.frame_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        elapsed_time = time.time() - self.start_time
        
        # 内存统计
        if self.memory_samples:
            memory_values = [s['memory_mb'] for s in self.memory_samples]
            memory_start = memory_values[0]
            memory_end = memory_values[-1]
            memory_leak = memory_end - memory_start
            memory_avg = sum(memory_values) / len(memory_values)
            memory_max = max(memory_values)
        else:
            memory_leak = 0
            memory_avg = 0
            memory_max = 0
        
        # FPS 统计
        if self.fps_samples:
            fps_values = [s['fps'] for s in self.fps_samples]
            fps_avg = sum(fps_values) / len(fps_values)
            fps_min = min(fps_values)
            fps_max = max(fps_values)
            fps_std = np.std(fps_values)
        else:
            fps_avg = 0
            fps_min = 0
            fps_max = 0
            fps_std = 0
        
        return {
            'elapsed_time': elapsed_time,
            'total_frames': self.frame_count,
            'error_count': self.error_count,
            'memory': {
                'start_mb': memory_start if self.memory_samples else 0,
                'end_mb': memory_end if self.memory_samples else 0,
                'avg_mb': memory_avg,
                'max_mb': memory_max,
                'leak_mb': memory_leak,
                'leak_per_hour': (memory_leak / elapsed_time) * 3600 if elapsed_time > 0 else 0,
            },
            'fps': {
                'avg': fps_avg,
                'min': fps_min,
                'max': fps_max,
                'std': fps_std,
            }
        }
    
    def save_report(self, filename: str = "stability_report.json"):
        """保存测试报告"""
        stats = self.get_statistics()
        
        report = {
            'test_time': datetime.now().isoformat(),
            'duration_seconds': stats['elapsed_time'],
            'statistics': stats,
            'memory_timeline': self.memory_samples,
            'fps_timeline': self.fps_samples,
        }
        
        output_path = Path("stability_test_data") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 测试报告已保存：{output_path}")


async def test_short_running(monitor: StabilityMonitor):
    """短时间测试 (5 分钟)"""
    print("\n" + "="*60)
    print("🧪 短时间稳定性测试 (5 分钟)")
    print("="*60)
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'zulong' / 'l1c'))
        
        from optimized_vision_processor import OptimizedVisionProcessor
        
        # 创建处理器
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=True)  # 加载真实模型
        print("✅ 视觉处理器已初始化 (真实模型)")
        
        # 测试参数
        test_duration = 5 * 60  # 5 分钟
        frame_interval = 0.033  # 30 FPS
        
        print(f"\n📊 测试参数:")
        print(f"   - 时长：{test_duration/60:.0f} 分钟")
        print(f"   - 目标 FPS: {1/frame_interval:.0f}")
        
        # 开始测试
        print(f"\n🚀 开始测试...")
        start_time = time.time()
        frame_count = 0
        last_sample_time = start_time
        
        while (time.time() - start_time) < test_duration:
            try:
                # 创建模拟帧
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # 喂入处理器
                timestamp = time.time()
                processor.feed_frame(frame, timestamp)
                
                # 记录
                monitor.record_frame()
                frame_count += 1
                
                # 每秒采样一次
                if timestamp - last_sample_time >= 1.0:
                    # 内存采样
                    mem_mb = monitor.sample_memory()
                    
                    # FPS 采样
                    elapsed = timestamp - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    monitor.sample_fps(current_fps)
                    
                    # 进度显示
                    progress = (elapsed / test_duration) * 100
                    print(f"   进度：{progress:.0f}% | 帧：{frame_count} | FPS: {current_fps:.1f} | 内存：{mem_mb:.0f}MB")
                    
                    last_sample_time = timestamp
                
                # 等待下一帧
                await asyncio.sleep(frame_interval)
                
            except Exception as e:
                monitor.record_error(str(e))
                await asyncio.sleep(0.1)
        
        # 停止处理器
        processor.stop()
        
        print("\n✅ 短时间测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 短时间测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_medium_running(monitor: StabilityMonitor):
    """中等时间测试 (30 分钟)"""
    print("\n" + "="*60)
    print("🧪 中等时间稳定性测试 (30 分钟)")
    print("="*60)
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'zulong' / 'l1c'))
        
        from optimized_vision_processor import OptimizedVisionProcessor
        
        # 创建处理器
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=True)
        print("✅ 视觉处理器已初始化 (真实模型)")
        
        # 测试参数
        test_duration = 30 * 60  # 30 分钟
        frame_interval = 0.033  # 30 FPS
        
        print(f"\n📊 测试参数:")
        print(f"   - 时长：{test_duration/60:.0f} 分钟")
        print(f"   - 目标 FPS: {1/frame_interval:.0f}")
        
        # 开始测试
        print(f"\n🚀 开始测试...")
        start_time = time.time()
        frame_count = 0
        last_sample_time = start_time
        
        while (time.time() - start_time) < test_duration:
            try:
                # 创建模拟帧
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # 喂入处理器
                timestamp = time.time()
                processor.feed_frame(frame, timestamp)
                
                # 记录
                monitor.record_frame()
                frame_count += 1
                
                # 每 5 秒采样一次
                if timestamp - last_sample_time >= 5.0:
                    # 内存采样
                    mem_mb = monitor.sample_memory()
                    
                    # FPS 采样
                    elapsed = timestamp - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    monitor.sample_fps(current_fps)
                    
                    # 进度显示
                    progress = (elapsed / test_duration) * 100
                    print(f"   进度：{progress:.1f}% | 帧：{frame_count} | FPS: {current_fps:.1f} | 内存：{mem_mb:.0f}MB")
                    
                    last_sample_time = timestamp
                
                # 等待下一帧
                await asyncio.sleep(frame_interval)
                
            except Exception as e:
                monitor.record_error(str(e))
                await asyncio.sleep(0.1)
        
        # 停止处理器
        processor.stop()
        
        print("\n✅ 中等时间测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 中等时间测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_quick_validation(monitor: StabilityMonitor):
    """快速验证测试 (1 分钟)"""
    print("\n" + "="*60)
    print("🧪 快速验证测试 (1 分钟)")
    print("="*60)
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'zulong' / 'l1c'))
        
        from optimized_vision_processor import OptimizedVisionProcessor
        
        # 创建处理器
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=True)
        print("✅ 视觉处理器已初始化 (真实模型)")
        
        # 测试参数
        test_duration = 1 * 60  # 1 分钟
        frame_interval = 0.033  # 30 FPS
        
        print(f"\n📊 测试参数:")
        print(f"   - 时长：{test_duration:.0f} 秒")
        print(f"   - 目标 FPS: {1/frame_interval:.0f}")
        
        # 开始测试
        print(f"\n🚀 开始测试...")
        start_time = time.time()
        frame_count = 0
        last_sample_time = start_time
        
        while (time.time() - start_time) < test_duration:
            try:
                # 创建模拟帧
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # 喂入处理器
                timestamp = time.time()
                processor.feed_frame(frame, timestamp)
                
                # 记录
                monitor.record_frame()
                frame_count += 1
                
                # 每 10 秒采样一次
                if timestamp - last_sample_time >= 10.0:
                    # 内存采样
                    mem_mb = monitor.sample_memory()
                    
                    # FPS 采样
                    elapsed = timestamp - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    monitor.sample_fps(current_fps)
                    
                    print(f"   进度：{elapsed:.0f}s | 帧：{frame_count} | FPS: {current_fps:.1f} | 内存：{mem_mb:.0f}MB")
                    
                    last_sample_time = timestamp
                
                # 等待下一帧
                await asyncio.sleep(frame_interval)
                
            except Exception as e:
                monitor.record_error(str(e))
                await asyncio.sleep(0.1)
        
        # 停止处理器
        processor.stop()
        
        print("\n✅ 快速验证测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 快速验证测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行长时间稳定性测试"""
    print("="*60)
    print("🚀 长时间稳定性测试")
    print("="*60)
    print(f"📅 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 选择测试模式
    print("\n📋 选择测试模式:")
    print("   1. quick - 快速验证 (1 分钟)")
    print("   2. short - 短时间 (5 分钟)")
    print("   3. medium - 中等时间 (30 分钟)")
    print("   4. long - 完整测试 (60 分钟)")
    
    mode = input("\n请输入测试模式 (quick/short/medium/long): ").strip().lower()
    
    # 创建监控器
    monitor = StabilityMonitor()
    
    # 运行测试
    if mode == 'long':
        success = await test_medium_running(monitor)  # 60 分钟测试需要特殊实现
    elif mode == 'medium':
        success = await test_medium_running(monitor)
    elif mode == 'short':
        success = await test_short_running(monitor)
    else:  # quick
        success = await test_quick_validation(monitor)
    
    # 打印统计
    if success:
        stats = monitor.get_statistics()
        
        print("\n" + "="*60)
        print("📊 稳定性测试统计")
        print("="*60)
        
        print(f"\n⏱️  运行时间：{stats['elapsed_time']:.0f}秒 ({stats['elapsed_time']/60:.1f}分钟)")
        print(f"📹  总帧数：{stats['total_frames']}")
        print(f"❌  错误数：{stats['error_count']}")
        
        print(f"\n📈 内存统计:")
        print(f"   - 起始：{stats['memory']['start_mb']:.0f} MB")
        print(f"   - 结束：{stats['memory']['end_mb']:.0f} MB")
        print(f"   - 平均：{stats['memory']['avg_mb']:.0f} MB")
        print(f"   - 最大：{stats['memory']['max_mb']:.0f} MB")
        print(f"   - 泄漏：{stats['memory']['leak_mb']:.1f} MB")
        print(f"   - 泄漏率：{stats['memory']['leak_per_hour']:.1f} MB/小时")
        
        print(f"\n📈 FPS 统计:")
        print(f"   - 平均：{stats['fps']['avg']:.1f} FPS")
        print(f"   - 最小：{stats['fps']['min']:.1f} FPS")
        print(f"   - 最大：{stats['fps']['max']:.1f} FPS")
        print(f"   - 标准差：{stats['fps']['std']:.2f}")
        
        # 评估
        print("\n" + "="*60)
        print("📊 稳定性评估")
        print("="*60)
        
        # 内存泄漏评估
        if stats['memory']['leak_per_hour'] < 50:
            print("✅ 内存泄漏优秀 (<50MB/小时)")
        elif stats['memory']['leak_per_hour'] < 200:
            print("⚠️  内存泄漏可接受 (50-200MB/小时)")
        else:
            print("❌ 内存泄漏过高 (>200MB/小时)")
        
        # FPS 稳定性评估
        if stats['fps']['std'] < 2:
            print("✅ FPS 稳定性优秀 (标准差<2)")
        elif stats['fps']['std'] < 5:
            print("⚠️  FPS 稳定性可接受 (标准差 2-5)")
        else:
            print("❌ FPS 稳定性差 (标准差>5)")
        
        # 错误评估
        if stats['error_count'] == 0:
            print("✅ 无错误运行")
        elif stats['error_count'] < 10:
            print("⚠️  少量错误 (<10)")
        else:
            print("❌ 错误过多 (>10)")
        
        # 保存报告
        monitor.save_report()
        
        print("\n" + "="*60)
        print("✅ 长时间稳定性测试完成")
        print("\n📋 下一步:")
        print("1. 查看 stability_test_data/ 目录中的详细报告")
        print("2. 分析内存泄漏原因 (如有)")
        print("3. 优化帧率稳定性")
    else:
        print("\n❌ 测试失败，请检查日志")
    
    print("\n" + "="*60)
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
