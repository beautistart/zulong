#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复盘机制专项测试 - 持久WebSocket连接版
保持一个连接贯穿整个测试流程，确保能收到L2回复
"""

import asyncio
import websockets
import json
import sys
import os
import io
from datetime import datetime

# Fix Windows console encoding for emoji/CJK characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PersistentReviewTester:
    """持久连接的复盘测试器 - 使用 /eventbus 双向端点"""

    def __init__(self, ws_uri="ws://127.0.0.1:5555/eventbus"):
        self.ws_uri = ws_uri
        self.ws = None
        self.test_log = []

    async def connect(self):
        """建立持久WebSocket连接到 /eventbus 端点"""
        print(f"  连接到 {self.ws_uri} ...")
        self.ws = await websockets.connect(self.ws_uri, ping_interval=None, ping_timeout=None)
        print(f"  WebSocket /eventbus 已连接 (双向事件流)")

    async def close(self):
        if self.ws:
            await self.ws.close()
            print(f"  WebSocket 已断开")

    async def send_and_wait(self, message: str, wait_time: float = 15.0, test_name: str = ""):
        """通过 EventBus 端点发送消息并等待所有响应"""
        print(f"\n{'='*80}")
        if test_name:
            print(f"  [{test_name}]")
        print(f"{'='*80}")
        print(f"\n  >>> 发送: {message}")

        # EventBus 端点消息格式: {"type": "PUBLISH", "event": {...}}
        event = {
            "type": "PUBLISH",
            "event": {
                "type": "USER_TEXT",
                "source": "review_test",
                "payload": {
                    "text": message,
                    "confidence": 1.0
                },
                "priority": "NORMAL"
            }
        }

        await self.ws.send(json.dumps(event))
        print(f"  [OK] 已发送, 等待响应 ({wait_time}s)...")

        responses = []
        try:
            while True:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=wait_time)
                data = json.loads(raw)
                resp_type = data.get('type', 'unknown')
                responses.append(data)

                if resp_type == 'ACK':
                    print(f"  <- ACK (已确认接收)")
                elif resp_type == 'SUBSCRIBE':
                    # EventBus 端点转发的事件格式: {"type": "SUBSCRIBE", "event": {...}}
                    inner = data.get('event', {})
                    evt_type = inner.get('type', 'unknown')
                    payload = inner.get('payload', {})
                    text = payload.get('text', '')
                    # 处理 text 可能是元组/列表的情况
                    if isinstance(text, (list, tuple)):
                        text = text[0] if text else ''
                    text = str(text)
                    status = payload.get('status', '')
                    if text:
                        preview = text[:300] + ('...' if len(text) > 300 else '')
                        print(f"  <- [EVENT:{evt_type}] {preview}")
                    elif status:
                        print(f"  <- [EVENT:{evt_type}] status={status}")
                    else:
                        print(f"  <- [EVENT:{evt_type}] {json.dumps(payload, ensure_ascii=False)[:200]}")
                    # 用实际事件类型记录
                    resp_type = evt_type
                elif resp_type in ('L2_OUTPUT', 'ACTION_SPEAK', 'SYSTEM_STATUS'):
                    payload = data.get('payload', {})
                    text = payload.get('text', '')
                    if isinstance(text, (list, tuple)):
                        text = text[0] if text else ''
                    text = str(text)
                    status = payload.get('status', '')
                    if text:
                        preview = text[:300] + ('...' if len(text) > 300 else '')
                        print(f"  <- [{resp_type}] {preview}")
                    elif status:
                        print(f"  <- [{resp_type}] status={status}")
                    else:
                        print(f"  <- [{resp_type}] {json.dumps(payload, ensure_ascii=False)[:200]}")
                else:
                    print(f"  <- [{resp_type}] {json.dumps(data, ensure_ascii=False)[:200]}")

                self.test_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "test_name": test_name,
                    "message": message,
                    "response_type": resp_type,
                    "response": data
                })

        except asyncio.TimeoutError:
            pass

        non_ack = [r for r in responses if r.get('type') != 'ACK']
        print(f"  [RESULT] 共 {len(responses)} 个响应 (ACK: {len(responses)-len(non_ack)}, 实际回复: {len(non_ack)})")
        return responses

    async def run_quick_review_test(self):
        """快速复盘完整流程测试"""
        print("\n\n")
        print("+" + "="*78 + "+")
        print("|" + "     测试A: 快速复盘完整流程     ".center(78) + "|")
        print("+" + "="*78 + "+")

        # 1. 前置对话 - 积累上下文
        await self.send_and_wait(
            "我最近在学习Python异步编程，遇到了很多问题，比如事件循环的概念很难理解。",
            wait_time=15.0,
            test_name="快速复盘-前置对话1"
        )
        await asyncio.sleep(2)

        await self.send_and_wait(
            "后来我发现用asyncio.run()作为入口点就简单多了，不用手动管理事件循环。",
            wait_time=15.0,
            test_name="快速复盘-前置对话2"
        )
        await asyncio.sleep(2)

        # 2. 启动复盘
        await self.send_and_wait(
            "启动复盘",
            wait_time=15.0,
            test_name="快速复盘-启动"
        )
        await asyncio.sleep(2)

        # 3. 选择快速复盘
        await self.send_and_wait(
            "快速复盘",
            wait_time=15.0,
            test_name="快速复盘-选择模式"
        )
        await asyncio.sleep(2)

        # 4. 复盘中对话
        await self.send_and_wait(
            "我觉得学习异步编程最重要的经验是：先理解同步和异步的区别，然后从简单的例子开始。",
            wait_time=15.0,
            test_name="快速复盘-对话1"
        )
        await asyncio.sleep(2)

        await self.send_and_wait(
            "另外一个教训是不要在异步函数里用time.sleep，应该用asyncio.sleep。",
            wait_time=15.0,
            test_name="快速复盘-对话2"
        )
        await asyncio.sleep(2)

        # 5. 结束复盘（触发经验提取，给更多时间）
        await self.send_and_wait(
            "结束复盘",
            wait_time=30.0,
            test_name="快速复盘-结束(经验提取)"
        )
        await asyncio.sleep(3)

        # 6. 确认经验
        await self.send_and_wait(
            "确认",
            wait_time=15.0,
            test_name="快速复盘-确认"
        )

    async def run_deep_review_test(self):
        """深度复盘流程测试"""
        print("\n\n")
        print("+" + "="*78 + "+")
        print("|" + "     测试B: 深度复盘完整流程     ".center(78) + "|")
        print("+" + "="*78 + "+")

        # 1. 前置对话
        await self.send_and_wait(
            "我在做一个机器人项目，设计了五层架构，但是各层之间的通信经常出问题。",
            wait_time=15.0,
            test_name="深度复盘-前置对话"
        )
        await asyncio.sleep(2)

        # 2. 启动复盘
        await self.send_and_wait(
            "启动复盘",
            wait_time=15.0,
            test_name="深度复盘-启动"
        )
        await asyncio.sleep(2)

        # 3. 选择深度复盘
        await self.send_and_wait(
            "深度复盘",
            wait_time=15.0,
            test_name="深度复盘-选择模式"
        )
        await asyncio.sleep(2)

        # 4. 复盘中对话
        await self.send_and_wait(
            "我发现问题出在事件总线的设计上，需要引入优先级队列来解决。",
            wait_time=15.0,
            test_name="深度复盘-对话"
        )
        await asyncio.sleep(2)

        # 5. 结束复盘（给更多时间做经验提取）
        await self.send_and_wait(
            "结束复盘",
            wait_time=30.0,
            test_name="深度复盘-结束(经验提取)"
        )
        await asyncio.sleep(3)

        # 6. 确认
        await self.send_and_wait(
            "确认",
            wait_time=15.0,
            test_name="深度复盘-确认"
        )

    def print_summary(self):
        """打印测试总结"""
        print("\n\n")
        print("+" + "="*78 + "+")
        print("|" + "     复盘机制测试结果汇总     ".center(78) + "|")
        print("+" + "="*78 + "+")

        total = len(self.test_log)
        ack_count = sum(1 for l in self.test_log if l['response_type'] == 'ACK')
        l2_count = sum(1 for l in self.test_log if l['response_type'] in ('L2_OUTPUT', 'ACTION_SPEAK'))
        other_count = total - ack_count - l2_count

        print(f"\n  总响应数: {total}")
        print(f"  ACK 确认: {ack_count}")
        print(f"  L2/语音回复: {l2_count}")
        print(f"  其他类型: {other_count}")

        print(f"\n  详细日志:")
        current_test = ""
        for log in self.test_log:
            if log['test_name'] != current_test:
                current_test = log['test_name']
                print(f"\n  --- {current_test} ---")
            resp_type = log['response_type']
            if resp_type != 'ACK':
                resp = log['response']
                # EventBus 端点的事件被包裹在 SUBSCRIBE 消息中
                if resp.get('type') == 'SUBSCRIBE':
                    inner = resp.get('event', {})
                    payload = inner.get('payload', {})
                else:
                    payload = resp.get('payload', {})
                text = payload.get('text', '')
                if isinstance(text, (list, tuple)):
                    text = text[0] if text else ''
                text = str(text)[:100]
                print(f"    [{resp_type}] {text}")

        if l2_count > 0:
            print(f"\n  [PASS] 复盘机制工作正常，收到 {l2_count} 个L2回复")
        else:
            print(f"\n  [WARN] 未收到L2实际回复，可能需要检查:")
            print(f"    1. vLLM服务是否正常响应 (curl http://localhost:8000/v1/models)")
            print(f"    2. L2处理管道是否正常工作")
            print(f"    3. WebSocket响应路由是否正确")


async def main():
    print("\n" + "=" * 80)
    print("  复盘机制专项测试 (持久连接版)")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    tester = PersistentReviewTester(ws_uri="ws://127.0.0.1:5555/eventbus")

    try:
        await tester.connect()

        # 快速复盘测试
        await tester.run_quick_review_test()
        await asyncio.sleep(5)

        # 深度复盘测试
        await tester.run_deep_review_test()

        # 总结
        tester.print_summary()

    except KeyboardInterrupt:
        print("\n\n  测试被用户中断")
    except Exception as e:
        print(f"\n\n  测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
