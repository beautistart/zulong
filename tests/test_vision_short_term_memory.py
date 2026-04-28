#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
祖龙 (ZULONG) 系统 - 视觉短期记忆功能测试脚本

文件：tests/test_vision_short_term_memory.py

功能:
- 测试 VisionShortTermMemory 的环形缓冲区功能
- 测试视频片段保存功能
- 测试事件路由 (SENSOR_VISION_REQUEST -> VISION_DATA_READY)

TSD v1.7 对应:
- 4.2 上下文打包 (Context Packaging)
- 4.4 感知预处理 (视觉短期记忆)
"""

import asyncio
import time
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from zulong.l1a.vision_short_term_memory import VisionShortTermMemory
from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus
from zulong.l1a.reflex.vision_node import VisionNode


class VisionMemoryTester:
    """视觉短期记忆测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.test_passed = 0
        self.test_failed = 0
        self.results = []
        
        print("=" * 80)
        print("🧪 祖龙系统 - 视觉短期记忆功能测试")
        print("=" * 80)
    
    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """记录测试结果"""
        if passed:
            self.test_passed += 1
            print(f"✅ [PASS] {test_name}")
        else:
            self.test_failed += 1
            print(f"❌ [FAIL] {test_name}: {message}")
        
        self.results.append({
            "test_name": test_name,
            "passed": passed,
            "message": message
        })
    
    def test_memory_initialization(self):
        """测试 1: 视觉短期记忆初始化"""
        test_name = "视觉短期记忆初始化"
        
        try:
            memory = VisionShortTermMemory(
                duration=5,
                fps=30,
                cache_dir="./data/shared_vision_test"
            )
            
            # 验证参数
            assert memory.duration == 5, "时长设置错误"
            assert memory.fps == 30, "帧率设置错误"
            assert memory.max_frames == 150, "最大帧数计算错误"
            assert memory.cache_dir.exists(), "缓存目录未创建"
            
            self.log_result(test_name, True)
            return memory
            
        except Exception as e:
            self.log_result(test_name, False, str(e))
            return None
    
    def test_frame_buffer(self, memory: VisionShortTermMemory):
        """测试 2: 帧缓冲区功能"""
        test_name = "帧缓冲区功能"
        
        try:
            # 创建模拟视频帧 (100 帧，模拟 3.3 秒@30FPS)
            num_frames = 100
            height, width = 480, 640
            
            for i in range(num_frames):
                # 创建渐变颜色的帧
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :, 0] = (i * 2) % 256  # B 通道渐变
                frame[:, :, 1] = (i * 3) % 256  # G 通道渐变
                frame[:, :, 2] = (i * 5) % 256  # R 通道渐变
                
                timestamp = time.time() - (num_frames - i) * 0.033  # 模拟 30FPS
                memory.add_frame(frame, timestamp)
            
            # 验证缓冲区
            info = memory.get_buffer_info()
            assert info["frame_count"] == num_frames, f"帧数错误：{info['frame_count']}"
            assert info["duration_seconds"] > 3.0, f"时长不足：{info['duration_seconds']}"
            assert info["is_ready"] == True, "缓冲区未就绪"
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, str(e))
    
    def test_ring_buffer_overflow(self, memory: VisionShortTermMemory):
        """测试 3: 环形缓冲区溢出 (自动丢弃旧帧)"""
        test_name = "环形缓冲区溢出处理"
        
        try:
            # 添加 200 帧 (超过最大 150 帧)
            for i in range(200):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                memory.add_frame(frame, time.time())
            
            # 验证只保留最新 150 帧
            info = memory.get_buffer_info()
            assert info["frame_count"] == 150, f"缓冲区未正确丢弃旧帧：{info['frame_count']}"
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, str(e))
    
    def test_video_save(self, memory: VisionShortTermMemory):
        """测试 4: 视频保存功能"""
        test_name = "视频保存功能"
        
        try:
            # 保存视频
            video_path, metadata = memory.save_to_file(trigger_reason="test")
            
            # 验证文件存在
            assert video_path is not None, "视频路径为空"
            assert os.path.exists(video_path), f"视频文件不存在：{video_path}"
            assert os.path.exists(memory.metadata_path), "元数据文件不存在"
            
            # 验证元数据
            assert metadata is not None, "元数据为空"
            assert metadata["frame_count"] > 0, "帧数为 0"
            assert metadata["duration_seconds"] > 0, "时长为 0"
            assert metadata["trigger_reason"] == "test", "触发原因错误"
            
            # 验证视频文件可读
            cap = cv2.VideoCapture(video_path)
            assert cap.isOpened(), "无法打开视频文件"
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"   📊 视频信息：{frame_count}帧，{fps}FPS, {duration:.2f}秒")
            
            cap.release()
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, str(e))
    
    async def test_vision_node_integration(self):
        """测试 5: VisionNode 集成测试"""
        test_name = "VisionNode 集成测试"
        
        try:
            # 创建 VisionNode 实例
            vision_node = VisionNode()
            
            # 验证短期记忆已初始化
            assert hasattr(vision_node, 'short_term_memory'), "VisionNode 缺少 short_term_memory"
            
            # 模拟推送帧
            for i in range(30):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                await vision_node.push_frame(frame)
            
            # 验证缓冲区
            info = vision_node.get_buffer_info()
            assert info["frame_count"] == 30, f"帧数错误：{info['frame_count']}"
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, str(e))
    
    def test_freshness_check(self, memory: VisionShortTermMemory):
        """测试 5: 新鲜度检查"""
        test_name = "新鲜度检查"
        
        try:
            # 刚保存的视频应该是新鲜的
            assert memory.is_fresh(max_age=10.0) == True, "新鲜度检查失败"
            
            # 模拟旧视频 (修改元数据时间戳)
            import json
            with open(memory.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata["created_at"] = time.time() - 20  # 20 秒前
            with open(memory.metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # 现在应该不新鲜了
            assert memory.is_fresh(max_age=10.0) == False, "旧视频应标记为不新鲜"
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, str(e))
    
    def test_get_frames_list(self, memory: VisionShortTermMemory):
        """测试 6: 获取帧列表"""
        test_name = "获取帧列表"
        
        try:
            # 获取最近 10 帧
            frames = memory.get_frames_list(num_frames=10)
            
            assert len(frames) == 10, f"获取帧数错误：{len(frames)}"
            assert all(isinstance(f, np.ndarray) for f in frames), "帧类型错误"
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, str(e))
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 80)
        print("📊 测试摘要")
        print("=" * 80)
        print(f"✅ 通过：{self.test_passed}")
        print(f"❌ 失败：{self.test_failed}")
        print(f"📈 通过率：{self.test_passed / (self.test_passed + self.test_failed) * 100:.1f}%")
        print("=" * 80)
        
        if self.test_failed == 0:
            print("🎉 所有测试通过！视觉短期记忆功能正常。")
        else:
            print(f"⚠️  有 {self.test_failed} 个测试失败，请检查错误信息。")


async def run_async_tests(tester: VisionMemoryTester):
    """运行异步测试"""
    print("\n" + "=" * 80)
    print("🧪 异步集成测试")
    print("=" * 80)
    
    await tester.test_vision_node_integration()


def main():
    """主测试函数"""
    tester = VisionMemoryTester()
    
    # 测试 1: 初始化
    print("\n" + "=" * 80)
    print("🧪 基础功能测试")
    print("=" * 80)
    
    memory = tester.test_memory_initialization()
    
    if memory is None:
        print("❌ 初始化失败，无法继续测试")
        tester.print_summary()
        return
    
    # 测试 2: 帧缓冲区
    tester.test_frame_buffer(memory)
    
    # 测试 3: 环形缓冲区溢出
    tester.test_ring_buffer_overflow(memory)
    
    # 测试 4: 视频保存
    tester.test_video_save(memory)
    
    # 测试 5: 新鲜度检查
    tester.test_freshness_check(memory)
    
    # 测试 6: 获取帧列表
    tester.test_get_frames_list(memory)
    
    # 异步测试
    asyncio.run(run_async_tests(tester))
    
    # 打印摘要
    tester.print_summary()
    
    # 清理测试文件
    print("\n" + "=" * 80)
    print("🧹 清理测试文件...")
    print("=" * 80)
    
    try:
        import shutil
        test_cache_dir = Path("./data/shared_vision_test")
        if test_cache_dir.exists():
            shutil.rmtree(test_cache_dir)
            print("✅ 测试缓存目录已清理")
    except Exception as e:
        print(f"⚠️  清理失败：{e}")
    
    print("\n" + "=" * 80)
    print("🎯 测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
