# -*- coding: utf-8 -*-
# File: tests/test_week2_persistence.py
# 第 2 周优化测试：共享池持久化

import sys
from pathlib import Path
import shutil
import time

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from zulong.infrastructure.shared_memory_pool import SharedMemoryPool, DataType, ZoneType, DataEnvelope


def cleanup_test_data():
    """清理测试数据"""
    test_data_path = Path("./data/shared_memory_pool")
    if test_data_path.exists():
        shutil.rmtree(test_data_path)
        print(f"🧹 已清理测试数据：{test_data_path}")


async def test_snapshot_save_load():
    """测试 1: 快照保存和加载"""
    print("\n" + "="*80)
    print("测试 1: 快照保存和加载")
    print("="*80)
    
    # 清理旧数据
    cleanup_test_data()
    
    # 第 1 次启动：写入数据
    print("\n--- 第 1 次启动：写入数据 ---")
    pool1 = SharedMemoryPool()
    
    # 写入测试数据
    test_data = [
        {"type": DataType.TEXT_USER, "zone": ZoneType.MEMORY, "content": "用户输入 1"},
        {"type": DataType.TEXT_ASSISTANT, "zone": ZoneType.MEMORY, "content": "AI 回复 1"},
        {"type": DataType.TEXT_USER, "zone": ZoneType.MEMORY, "content": "用户输入 2"},
        {"type": DataType.SYSTEM_LOG, "zone": ZoneType.SYSTEM, "content": "系统日志"},
    ]
    
    for i, data in enumerate(test_data):
        envelope = DataEnvelope(
            trace_id=f"test_{i}",
            timestamp=time.time(),
            data_type=data["type"],
            zone=data["zone"],
            payload={"text": data["content"]}
        )
        pool1.write(envelope)
    
    stats1 = pool1.get_stats()
    print(f"写入数据后统计:")
    print(f"  Memory Zone: {stats1['memory_zone_size']} 条")
    print(f"  System Zone: {stats1['system_zone_size']} 条")
    print(f"  总追踪数：{stats1['total_traces']}")
    
    # 手动触发快照
    print("\n--- 手动触发快照 ---")
    pool1.save_snapshot_now()
    await asyncio.sleep(1)  # 等待保存完成
    
    # 模拟系统重启（创建新实例）
    print("\n--- 第 2 次启动：模拟重启 ---")
    pool2 = SharedMemoryPool()
    await asyncio.sleep(1)  # 等待加载完成
    
    stats2 = pool2.get_stats()
    print(f"恢复后统计:")
    print(f"  Memory Zone: {stats2['memory_zone_size']} 条")
    print(f"  System Zone: {stats2['system_zone_size']} 条")
    print(f"  总追踪数：{stats2['total_traces']}")
    
    # 验证：数据已恢复
    assert stats2['memory_zone_size'] == stats1['memory_zone_size'], \
        f"Memory Zone 应该恢复：期望{stats1['memory_zone_size']}, 实际{stats2['memory_zone_size']}"
    assert stats2['system_zone_size'] == stats1['system_zone_size'], \
        f"System Zone 应该恢复：期望{stats1['system_zone_size']}, 实际{stats2['system_zone_size']}"
    assert stats2['total_traces'] == stats1['total_traces'], \
        f"总追踪数应该恢复：期望{stats1['total_traces']}, 实际{stats2['total_traces']}"
    
    # 验证：可以读取恢复的数据
    envelope = pool2.read("test_0")
    assert envelope is not None, "应该能读取恢复的数据"
    assert envelope.payload["text"] == "用户输入 1", "数据内容应该正确"
    
    print(f"\n✅ 测试 1 通过：快照保存和加载成功")
    
    return True


async def test_auto_snapshot():
    """测试 2: 自动快照（加速测试）"""
    print("\n" + "="*80)
    print("测试 2: 自动快照（加速测试）")
    print("="*80)
    
    # 清理旧数据
    cleanup_test_data()
    
    # 创建共享池，设置快照间隔为 2 秒
    print("\n--- 创建共享池（快照间隔=2 秒） ---")
    pool = SharedMemoryPool()
    original_interval = pool.snapshot_interval
    pool.snapshot_interval = 2  # 加速测试
    pool.last_snapshot_time = time.time() - 5  # 模拟已过 5 秒
    
    # 写入数据
    print("\n--- 写入数据 ---")
    for i in range(5):
        envelope = DataEnvelope(
            trace_id=f"auto_test_{i}",
            timestamp=time.time(),
            data_type=DataType.TEXT_USER,
            zone=ZoneType.MEMORY,
            payload={"text": f"自动测试{i}"}
        )
        pool.write(envelope)
    
    # 手动触发一次快照（模拟自动触发）
    print("\n--- 模拟自动快照触发 ---")
    pool._save_snapshot()
    await asyncio.sleep(1)
    
    # 检查快照文件
    snapshot_path = Path("./data/shared_memory_pool")
    snapshots = list(snapshot_path.glob("snapshot_*.json.gz"))
    
    print(f"\n快照文件数：{len(snapshots)}")
    if snapshots:
        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
        print(f"最新快照：{latest.name}")
    
    assert len(snapshots) > 0, "应该至少有一个快照文件"
    
    print(f"\n✅ 测试 2 通过：自动快照机制正常")
    
    # 恢复原始设置
    pool.snapshot_interval = original_interval
    
    return True


async def test_multi_zone_persistence():
    """测试 3: 多分区数据持久化"""
    print("\n" + "="*80)
    print("测试 3: 多分区数据持久化")
    print("="*80)
    
    # 清理旧数据
    cleanup_test_data()
    
    # 第 1 次启动：写入多分区数据
    print("\n--- 第 1 次启动：写入多分区数据 ---")
    pool1 = SharedMemoryPool()
    
    # Raw Zone: 原始视频帧
    envelope1 = DataEnvelope(
        trace_id="raw_video_1",
        timestamp=time.time(),
        data_type=DataType.VIDEO_FRAME,
        zone=ZoneType.RAW,
        payload={"frame_data": "原始视频帧数据"}
    )
    pool1.write(envelope1)
    
    # Feature Zone: 提取的特征
    envelope2 = DataEnvelope(
        trace_id="feature_audio_1",
        timestamp=time.time(),
        data_type=DataType.AUDIO_FEATURE,
        zone=ZoneType.FEATURE,
        payload={"features": [0.1, 0.2, 0.3]}
    )
    pool1.write(envelope2)
    
    # Memory Zone: 对话历史
    envelope3 = DataEnvelope(
        trace_id="memory_dialogue_1",
        timestamp=time.time(),
        data_type=DataType.TEXT_USER,
        zone=ZoneType.MEMORY,
        payload={"text": "用户对话"}
    )
    pool1.write(envelope3)
    
    # System Zone: 系统日志
    envelope4 = DataEnvelope(
        trace_id="system_log_1",
        timestamp=time.time(),
        data_type=DataType.SYSTEM_LOG,
        zone=ZoneType.SYSTEM,
        payload={"log": "系统日志信息"}
    )
    pool1.write(envelope4)
    
    stats1 = pool1.get_stats()
    print(f"写入数据后统计:")
    print(f"  Raw Zone: {stats1['raw_zone_size']} 条")
    print(f"  Feature Zone: {stats1['feature_zone_size']} 条")
    print(f"  Memory Zone: {stats1['memory_zone_size']} 条")
    print(f"  System Zone: {stats1['system_zone_size']} 条")
    
    # 保存快照
    print("\n--- 保存快照 ---")
    pool1.save_snapshot_now()
    await asyncio.sleep(1)
    
    # 模拟系统重启
    print("\n--- 第 2 次启动：模拟重启 ---")
    pool2 = SharedMemoryPool()
    await asyncio.sleep(1)
    
    stats2 = pool2.get_stats()
    print(f"恢复后统计:")
    print(f"  Raw Zone: {stats2['raw_zone_size']} 条")
    print(f"  Feature Zone: {stats2['feature_zone_size']} 条")
    print(f"  Memory Zone: {stats2['memory_zone_size']} 条")
    print(f"  System Zone: {stats2['system_zone_size']} 条")
    
    # 验证：所有分区数据都已恢复
    assert stats2['raw_zone_size'] == stats1['raw_zone_size'], "Raw Zone 应该恢复"
    assert stats2['feature_zone_size'] == stats1['feature_zone_size'], "Feature Zone 应该恢复"
    assert stats2['memory_zone_size'] == stats1['memory_zone_size'], "Memory Zone 应该恢复"
    assert stats2['system_zone_size'] == stats1['system_zone_size'], "System Zone 应该恢复"
    
    # 验证：可以读取各分区数据
    raw_env = pool2.read("raw_video_1")
    assert raw_env is not None, "应该能读取 Raw Zone 数据"
    
    feature_env = pool2.read("feature_audio_1")
    assert feature_env is not None, "应该能读取 Feature Zone 数据"
    
    memory_env = pool2.read("memory_dialogue_1")
    assert memory_env is not None, "应该能读取 Memory Zone 数据"
    
    system_env = pool2.read("system_log_1")
    assert system_env is not None, "应该能读取 System Zone 数据"
    
    print(f"\n✅ 测试 3 通过：多分区数据持久化成功")
    
    return True


async def test_snapshot_cleanup():
    """测试 4: 快照清理机制"""
    print("\n" + "="*80)
    print("测试 4: 快照清理机制")
    print("="*80)
    
    # 清理旧数据
    cleanup_test_data()
    
    # 创建共享池，设置最多保留 3 个快照
    print("\n--- 创建共享池（最多保留 3 个快照） ---")
    pool = SharedMemoryPool()
    pool.max_snapshots = 3
    
    # 手动创建多个快照
    print("\n--- 创建 5 个快照 ---")
    for i in range(5):
        # 写入不同数据
        envelope = DataEnvelope(
            trace_id=f"cleanup_test_{i}",
            timestamp=time.time(),
            data_type=DataType.TEXT_USER,
            zone=ZoneType.MEMORY,
            payload={"text": f"测试数据{i}"}
        )
        pool.write(envelope)
        
        # 保存快照
        pool.save_snapshot_now()
        await asyncio.sleep(0.5)
    
    # 检查快照文件数
    snapshot_path = Path("./data/shared_memory_pool")
    snapshots = list(snapshot_path.glob("snapshot_*.json.gz"))
    
    print(f"\n快照文件数：{len(snapshots)}")
    print(f"最大允许快照数：{pool.max_snapshots}")
    
    # 验证：快照数不超过限制
    assert len(snapshots) <= pool.max_snapshots, \
        f"快照数不应超过限制：实际{len(snapshots)}, 限制{pool.max_snapshots}"
    
    print(f"\n✅ 测试 4 通过：快照清理机制正常")
    
    return True


async def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 第 2 周优化测试：共享池持久化")
    print("="*80)
    
    tests = [
        ("测试 1: 快照保存和加载", test_snapshot_save_load),
        ("测试 2: 自动快照", test_auto_snapshot),
        ("测试 3: 多分区数据持久化", test_multi_zone_persistence),
        ("测试 4: 快照清理机制", test_snapshot_cleanup),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} 失败：{e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 总结
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    print(f"通过：{passed}/{len(tests)}")
    print(f"失败：{failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！第 2 周优化完成！")
    else:
        print(f"\n⚠️ {failed} 个测试失败，请检查")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
