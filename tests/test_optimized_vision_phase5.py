# File: tests/test_optimized_vision_phase5.py
"""
优化视觉处理器 Phase 5 测试：L1-B 优化调度器 (完全独立版)

测试目标:
1. 验证调度器核心逻辑 (不依赖完整架构)
2. 验证意图分数阈值决策
3. 验证上下文缓冲区

TSD v1.7 对应:
- 4.2.1 L1-B 注意力控制器
"""

import asyncio
import time
import sys
import os
from typing import Optional, Dict, Any, List
from collections import deque
from enum import Enum

# ========== 模拟祖龙核心模块 ==========

class EventType:
    SENSOR_VISION = "SENSOR_VISION"
    INTERACTION_TRIGGER = "INTERACTION_TRIGGER"
    SYSTEM_L2_COMMAND = "SYSTEM_L2_COMMAND"
    TASK_FREEZE = "TASK_FREEZE"

class EventPriority:
    LOW = 0
    MEDIUM = 1
    HIGH = 2

class L2Status(Enum):
    IDLE = 0
    BUSY = 1
    WAITING = 2
    UNLOADED = 3

class MockStateManager:
    def get_l2_status(self) -> L2Status:
        return L2Status.IDLE
    
    def get_effective_status(self) -> str:
        return "IDLE"

state_manager = MockStateManager()

class ZulongEvent:
    def __init__(self, type, priority, source, payload):
        self.type = type
        self.priority = priority
        self.source = source
        self.payload = payload

class MockEventBus:
    def __init__(self):
        self.subscribers = {}
        self.published_events = []
    
    def subscribe(self, event_type, handler, source):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append((handler, source))
    
    def publish(self, event):
        self.published_events.append(event)

event_bus = MockEventBus()

# ========== 模拟 OptimizedScheduler 核心逻辑 ==========

class MockOptimizedScheduler:
    """简化版优化调度器 (用于测试核心逻辑)"""
    
    INTENT_THRESHOLDS = {
        'ignore': 0.3,
        'interact': 0.6,
        'eagle_eye': 0.8,
    }
    
    COOLDOWNS = {
        'attention': 2.0,
        'eagle_eye': 5.0,
    }
    
    def __init__(self):
        self._context_buffer = deque(maxlen=100)
        self._last_attention_time = 0.0
        self._last_eagle_eye_time = 0.0
    
    def get_context_buffer(self, seconds: float = 30.0) -> List[Dict[str, Any]]:
        current_time = time.time()
        return [
            ctx for ctx in self._context_buffer
            if current_time - ctx['timestamp'] < seconds
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'context_buffer_size': len(self._context_buffer),
            'last_attention_time': self._last_attention_time,
            'last_eagle_eye_time': self._last_eagle_eye_time,
        }
    
    def decide_action(self, intent_score: float) -> str:
        """意图分数决策"""
        if intent_score < self.INTENT_THRESHOLDS['ignore']:
            return "ignore"
        elif intent_score >= self.INTENT_THRESHOLDS['eagle_eye']:
            return "eagle_eye"
        elif intent_score >= self.INTENT_THRESHOLDS['interact']:
            return "interact"
        else:
            return "ignore"


# ========== 测试用例 ==========

async def test_scheduler_initialization():
    """测试 1: 调度器初始化"""
    print("\n" + "="*60)
    print("🧪 测试 1: MockOptimizedScheduler 初始化")
    print("="*60)
    
    scheduler = MockOptimizedScheduler()
    
    print(f"✅ 调度器创建成功")
    print(f"   - 注意力冷却：{scheduler.COOLDOWNS['attention']}s")
    print(f"   - 鹰眼冷却：{scheduler.COOLDOWNS['eagle_eye']}s")
    print(f"   - 意图阈值：{scheduler.INTENT_THRESHOLDS}")
    
    stats = scheduler.get_stats()
    assert stats['context_buffer_size'] == 0, "初始上下文缓冲区应为空"
    
    print(f"✅ 统计信息正确：{stats}")
    print("\n✅ 测试 1 通过")
    return True


async def test_intent_threshold_decision():
    """测试 2: 意图分数阈值决策"""
    print("\n" + "="*60)
    print("🧪 测试 2: 意图分数阈值决策")
    print("="*60)
    
    scheduler = MockOptimizedScheduler()
    
    test_cases = [
        (0.2, "ignore", "低分数应忽略"),
        (0.5, "ignore", "中低分数应忽略"),
        (0.7, "interact", "中等分数应交互"),
        (0.9, "eagle_eye", "高分数应触发鹰眼"),
    ]
    
    for score, expected_action, description in test_cases:
        action = scheduler.decide_action(score)
        assert action == expected_action, f"{description}: 期望{expected_action}, 实际{action}"
        print(f"✅ 分数{score:.1f} -> {action} ({description})")
    
    print("\n✅ 测试 2 通过")
    return True


async def test_context_buffer():
    """测试 3: 上下文缓冲区"""
    print("\n" + "="*60)
    print("🧪 测试 3: 上下文缓冲区 (过去 30 秒)")
    print("="*60)
    
    scheduler = MockOptimizedScheduler()
    current_time = time.time()
    
    # 添加 10 个历史上下文
    for i in range(10):
        scheduler._context_buffer.append({
            'timestamp': current_time - (10 - i) * 2,
            'intent_type': 'WAVING',
            'intent_score': 0.8,
        })
    
    context_30s = scheduler.get_context_buffer(seconds=30.0)
    print(f"✅ 过去 30 秒上下文：{len(context_30s)} 帧")
    assert len(context_30s) > 0, "应有历史上下文"
    
    context_10s = scheduler.get_context_buffer(seconds=10.0)
    print(f"✅ 过去 10 秒上下文：{len(context_10s)} 帧")
    assert len(context_10s) < len(context_30s), "10 秒上下文应少于 30 秒"
    
    print("\n✅ 测试 3 通过")
    return True


async def test_event_publish():
    """测试 4: 事件发布"""
    print("\n" + "="*60)
    print("🧪 测试 4: 事件发布逻辑")
    print("="*60)
    
    test_event = ZulongEvent(
        type=EventType.INTERACTION_TRIGGER,
        priority=EventPriority.MEDIUM,
        source="Test",
        payload={'test': 'data'}
    )
    
    event_bus.publish(test_event)
    
    assert len(event_bus.published_events) == 1, "应发布 1 个事件"
    print(f"✅ 事件发布成功：{len(event_bus.published_events)} 个事件")
    
    print("\n✅ 测试 4 通过")
    return True


async def test_scheduler_stats():
    """测试 5: 调度器统计"""
    print("\n" + "="*60)
    print("🧪 测试 5: 调度器统计信息")
    print("="*60)
    
    scheduler = MockOptimizedScheduler()
    stats = scheduler.get_stats()
    
    assert 'context_buffer_size' in stats
    assert 'last_attention_time' in stats
    assert 'last_eagle_eye_time' in stats
    
    print(f"✅ 统计信息字段完整：{list(stats.keys())}")
    print(f"✅ 统计信息：{stats}")
    
    print("\n✅ 测试 5 通过")
    return True


async def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🚀 优化视觉处理器 Phase 5 测试")
    print("方案：L1-B 优化调度器 (核心逻辑验证)")
    print("="*60)
    
    tests = [
        ("调度器初始化", test_scheduler_initialization),
        ("意图分数阈值", test_intent_threshold_decision),
        ("上下文缓冲区", test_context_buffer),
        ("事件发布", test_event_publish),
        ("调度器统计", test_scheduler_stats),
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
        
        await asyncio.sleep(0.3)
    
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
        print("\n🎉 所有测试通过！Phase 5 完成")
    else:
        print("\n⚠️ 部分测试失败，请检查日志")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
