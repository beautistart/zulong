#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L2管道诊断 - 直接通过EventBus追踪消息处理全流程
不依赖WebSocket，直接在进程内追踪
"""
import os
import sys
import time
import asyncio
import logging
import threading

# 设置环境
os.environ["USE_VLLM_FOR_L2"] = "true"
os.environ["USE_VLLM_FOR_L2_BACKUP"] = "true"
os.environ["VLLM_BASE_URL"] = "http://localhost:8000/v1"
os.environ["VLLM_BACKUP_BASE_URL"] = "http://localhost:8000/v1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("diagnose")


def main():
    print("=" * 80)
    print("  L2 管道诊断脚本")
    print("=" * 80)

    # 1. 初始化系统
    print("\n[1] 初始化 EventBus 和核心组件...")
    from zulong.core.event_bus import event_bus
    from zulong.core.types import ZulongEvent, EventType, EventPriority

    # 2. 收集所有事件
    collected_events = []
    event_lock = threading.Lock()

    def event_collector(event: ZulongEvent):
        with event_lock:
            collected_events.append({
                'time': time.time(),
                'type': event.type.value if hasattr(event.type, 'value') else str(event.type),
                'source': event.source,
                'payload_keys': list(event.payload.keys()) if isinstance(event.payload, dict) else str(type(event.payload)),
                'text': event.payload.get('text', '')[:200] if isinstance(event.payload, dict) else ''
            })
            evt_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
            text_preview = event.payload.get('text', '')[:80] if isinstance(event.payload, dict) else ''
            print(f"  [EVENT] {evt_type} from={event.source} text='{text_preview}'")

    # 订阅所有关键事件
    for evt in [EventType.USER_TEXT, EventType.L2_OUTPUT, EventType.ACTION_SPEAK,
                EventType.SYSTEM_L2_COMMAND, EventType.SYSTEM_STATUS]:
        try:
            event_bus.subscribe(evt, event_collector, f"Diagnose_{evt.value}")
            print(f"  [OK] 已订阅: {evt.value}")
        except Exception as e:
            print(f"  [ERR] 订阅 {evt.value} 失败: {e}")

    # 3. 初始化系统 (不启动 WebSocket 和设备)
    print("\n[2] 初始化系统组件 (bootstrap)...")
    from zulong import bootstrap
    boot = bootstrap.SystemBootstrap()
    boot.initialize()
    print("  [OK] 系统初始化完成")

    # 4. 发送测试消息
    print("\n[3] 发送测试消息: '你好，今天天气怎么样？'")
    test_event = ZulongEvent(
        type=EventType.USER_TEXT,
        source="DiagnoseScript",
        payload={"text": "你好，今天天气怎么样？", "confidence": 1.0},
        priority=EventPriority.NORMAL
    )
    event_bus.publish(test_event)
    print("  [OK] 消息已发布到 EventBus")

    # 5. 等待处理
    print("\n[4] 等待L2处理 (最多60秒)...")
    start = time.time()
    l2_found = False
    while time.time() - start < 60:
        time.sleep(1)
        with event_lock:
            for evt in collected_events:
                if evt['type'] in ('L2_OUTPUT', 'ACTION_SPEAK'):
                    l2_found = True
                    break
        if l2_found:
            print(f"\n  [PASS] 收到 L2 输出! (耗时 {time.time()-start:.1f}s)")
            break
        elapsed = int(time.time() - start)
        if elapsed % 10 == 0 and elapsed > 0:
            print(f"  ... 已等待 {elapsed}s, 当前事件数: {len(collected_events)}")

    if not l2_found:
        print(f"\n  [FAIL] 60秒内未收到 L2 输出")

    # 6. 打印事件汇总
    print("\n[5] 事件汇总:")
    with event_lock:
        for i, evt in enumerate(collected_events):
            print(f"  [{i+1}] type={evt['type']}, source={evt['source']}, text='{evt['text'][:100]}'")

    if not collected_events:
        print("  (没有收到任何事件)")

    print("\n  诊断完成")


if __name__ == "__main__":
    main()
