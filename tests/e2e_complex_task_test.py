"""
祖龙系统 — 端到端复杂任务测试
================================

通过 WebSocket 连接已启动的系统，发送复杂任务，
捕获所有事件响应，同时监控运行日志。

使用前提: 系统已在另一个终端启动 (python scripts/start_zulong.py)

运行方式: python tests/e2e_complex_task_test.py
"""

import asyncio
import json
import time
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("E2E_TEST")

# WebSocket 端点（祖龙默认）
WS_URL = "ws://127.0.0.1:5555/eventbus"

# 要发送的复杂任务
COMPLEX_TASK = (
    "请帮我分析一下 Python 3.12 和 3.13 版本的主要新特性差异，"
    "列出每个版本的 3 个最重要的新功能，并给出升级建议。"
)

# 超时配置
CONNECT_TIMEOUT = 10       # 连接超时
ACK_TIMEOUT = 10           # ACK 等待超时
RESPONSE_TIMEOUT = 180     # 响应等待超时
IDLE_TIMEOUT = 60          # 空闲超时（无新消息）


async def run_test():
    """运行端到端测试"""
    try:
        import websockets
    except ImportError:
        logger.error("需要安装 websockets: pip install websockets")
        return

    logger.info("=" * 70)
    logger.info("祖龙系统 — 端到端复杂任务测试")
    logger.info("=" * 70)
    logger.info(f"WebSocket 端点: {WS_URL}")
    logger.info(f"测试任务: {COMPLEX_TASK[:60]}...")

    # 1. 连接
    logger.info("\n--- Phase 1: 连接 WebSocket ---")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(WS_URL, max_size=10 * 1024 * 1024),
            timeout=CONNECT_TIMEOUT,
        )
        logger.info(f"已连接到 {WS_URL}")
    except asyncio.TimeoutError:
        logger.error(f"连接超时 ({CONNECT_TIMEOUT}s)，系统是否已启动？")
        return
    except Exception as e:
        logger.error(f"连接失败: {e}")
        logger.error("请确认系统已启动 (python scripts/start_zulong.py)")
        return

    # 2. 发送消息（使用 EventBus PUBLISH 协议）
    logger.info("\n--- Phase 2: 发送复杂任务 ---")
    message = {
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "priority": "NORMAL",
            "source": "DiagTestClient",
            "payload": {
                "text": COMPLEX_TASK,
                "voice_mode": "TEXT_ONLY",
            },
        },
    }
    await ws.send(json.dumps(message, ensure_ascii=False))
    logger.info(f">>> 已发送: {COMPLEX_TASK[:80]}...")

    # 3. 等待 ACK
    logger.info("\n--- Phase 3: 等待 ACK ---")
    try:
        ack_raw = await asyncio.wait_for(ws.recv(), timeout=ACK_TIMEOUT)
        logger.info(f"<<< ACK: {ack_raw[:200]}")
    except asyncio.TimeoutError:
        logger.warning(f"ACK 超时 ({ACK_TIMEOUT}s)，继续等待响应...")

    # 4. 收集响应
    logger.info("\n--- Phase 4: 收集响应事件 ---")
    events_received = []
    final_response = None
    task_graph_events = []
    thinking_steps = []
    start_time = time.time()
    last_event_time = time.time()

    while True:
        elapsed = time.time() - start_time
        idle = time.time() - last_event_time

        if elapsed > RESPONSE_TIMEOUT:
            logger.warning(f"总超时 ({RESPONSE_TIMEOUT}s)，停止等待")
            break
        if idle > IDLE_TIMEOUT:
            logger.warning(f"空闲超时 ({IDLE_TIMEOUT}s 无新消息)，停止等待")
            break

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            last_event_time = time.time()
        except asyncio.TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket 连接已关闭")
            break

        # 解析事件（EventBus 转发格式: {"type":"EVENT","event":{...}}）
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.info(f"<<< (非JSON): {raw[:200]}")
            events_received.append({"raw": raw[:500]})
            continue

        # 解包 EventBus 转发包装
        wrapper_type = data.get("type", "")
        if wrapper_type in ("EVENT", "SUBSCRIBE") and "event" in data:
            event = data["event"]
        else:
            event = data
        
        event_type = event.get("type", wrapper_type)
        payload = event.get("payload", event)

        events_received.append(data)
        elapsed_str = f"{elapsed:.1f}s"

        # 分类处理
        if event_type in ("L2_OUTPUT", "l2_output"):
            text = payload.get("text", "")
            final_response = text
            logger.info(f"<<< [{elapsed_str}] L2_OUTPUT (最终回复): {len(text)} 字符")
            logger.info(f"    回复内容前 300 字符:\n    {text[:300]}")
            if len(text) > 300:
                logger.info(f"    ...后 200 字符:\n    {text[-200:]}")
            # 收到最终回复后再等几秒看看有没有后续事件
            await asyncio.sleep(3)
            break

        elif event_type in ("L2_THINKING_STEP", "l2_thinking_step"):
            step_type = payload.get("step_type", "")
            step_data = payload.get("data", {})
            thinking_steps.append(payload)

            if "pipeline" in step_type:
                graph_data = step_data.get("graph", {})
                nodes = graph_data.get("nodes", [])
                turn = step_data.get("turn", 0)
                tool = step_data.get("tool", "")
                logger.info(
                    f"<<< [{elapsed_str}] THINKING: {step_type} "
                    f"(turn={turn}, tool={tool}, nodes={len(nodes)})"
                )
                task_graph_events.append({
                    "step_type": step_type,
                    "turn": turn,
                    "tool": tool,
                    "node_count": len(nodes),
                })
            else:
                logger.info(f"<<< [{elapsed_str}] THINKING: {step_type}")

        elif event_type in ("SUBSCRIBE_ACK", "subscribe_ack"):
            logger.info(f"<<< [{elapsed_str}] 订阅确认")

        else:
            # 其他事件
            summary = json.dumps(data, ensure_ascii=False)[:200]
            logger.info(f"<<< [{elapsed_str}] {event_type}: {summary}")

    # 5. 结果总结
    logger.info("\n" + "=" * 70)
    logger.info("测试结果总结")
    logger.info("=" * 70)

    total_time = time.time() - start_time
    logger.info(f"总耗时: {total_time:.1f}s")
    logger.info(f"收到事件数: {len(events_received)}")
    logger.info(f"思维步骤数: {len(thinking_steps)}")
    logger.info(f"任务图谱事件数: {len(task_graph_events)}")

    if task_graph_events:
        logger.info("\n--- 任务图谱事件时间线 ---")
        for evt in task_graph_events:
            logger.info(
                f"  {evt['step_type']}: turn={evt['turn']}, "
                f"tool={evt['tool']}, nodes={evt['node_count']}"
            )

    if final_response:
        logger.info(f"\n--- 最终回复 ---")
        logger.info(f"长度: {len(final_response)} 字符")
        logger.info(f"内容:\n{final_response[:1000]}")
        if len(final_response) > 1000:
            logger.info(f"\n...(截断，总长 {len(final_response)} 字符)")

        # 判定
        if len(final_response) > 50:
            logger.info("\n>>> 测试结果: PASS (收到有意义的回复)")
        else:
            logger.warning(f"\n>>> 测试结果: WARN (回复过短，可能是降级回复)")
    else:
        logger.error("\n>>> 测试结果: FAIL (未收到最终回复)")
        logger.error("可能原因:")
        logger.error("  1. FC 循环卡住（查看 zulong_runtime.log 的 [FC] 日志）")
        logger.error("  2. 模型超时（查看 [vLLM] 或 [LLM] 超时日志）")
        logger.error("  3. 事件总线断路（查看 [EventBus] 日志）")

    # 6. 保存详细日志
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"e2e_complex_result_{int(time.time())}.json"
    )
    report = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": COMPLEX_TASK,
        "total_time_sec": round(total_time, 1),
        "events_count": len(events_received),
        "thinking_steps_count": len(thinking_steps),
        "task_graph_events": task_graph_events,
        "final_response": final_response,
        "final_response_length": len(final_response) if final_response else 0,
        "passed": final_response is not None and len(final_response) > 50,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\n详细报告已保存: {report_path}")

    await ws.close()


if __name__ == "__main__":
    asyncio.run(run_test())
