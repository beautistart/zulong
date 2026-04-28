"""
测试 A: 复杂任务执行测试
========================
验证: 意图分类(COMPLEX) -> TaskGraph创建 -> FC工具调用链 -> 最终回复

独立运行: python -X utf8 tests/test_a_complex_task.py
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TEST_A")

WS_URL = "ws://127.0.0.1:5555/eventbus"

# 使用一个全新的、不同于之前的复杂任务
COMPLEX_TASK = (
    "请帮我设计一个简单的学生成绩管理方案，"
    "需要包含：1) 成绩录入的数据结构 2) 成绩统计的算法思路 3) 排名输出的格式设计。"
)

RESPONSE_TIMEOUT = 240   # 总超时 4 分钟
IDLE_TIMEOUT = 60        # 空闲超时

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def get_latest_log_lines(n=50):
    """读取最新日志文件的最后 N 行"""
    latest = os.path.join(LOG_DIR, "latest.log")
    if not os.path.exists(latest):
        return []
    try:
        with open(latest, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-n:]
    except Exception:
        return []


async def run_test():
    try:
        import websockets
    except ImportError:
        logger.error("pip install websockets")
        return

    logger.info("=" * 70)
    logger.info("TEST A: 复杂任务执行测试")
    logger.info("=" * 70)
    logger.info(f"任务: {COMPLEX_TASK[:80]}...")

    # --- 连接 ---
    logger.info("\n[1/4] 连接 WebSocket...")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(WS_URL, max_size=10 * 1024 * 1024),
            timeout=10,
        )
        logger.info(f"  OK - 已连接 {WS_URL}")
    except Exception as e:
        logger.error(f"  FAIL - 连接失败: {e}")
        return

    # --- 发送复杂任务 ---
    logger.info("\n[2/4] 发送复杂任务...")
    test_start = time.time()
    message = {
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "priority": "NORMAL",
            "source": "TestA_ComplexTask",
            "payload": {
                "text": COMPLEX_TASK,
                "voice_mode": "TEXT_ONLY",
            },
        },
    }
    await ws.send(json.dumps(message, ensure_ascii=False))
    logger.info(f"  OK - 已发送")

    # --- 收集响应 ---
    logger.info("\n[3/4] 收集响应事件...")
    events = []
    thinking_steps = []
    task_graph_events = []
    tool_calls = []
    final_response = None
    last_event_time = time.time()

    while True:
        elapsed = time.time() - test_start
        idle = time.time() - last_event_time

        if elapsed > RESPONSE_TIMEOUT:
            logger.warning(f"  !! 总超时 ({RESPONSE_TIMEOUT}s)")
            break
        if idle > IDLE_TIMEOUT and len(events) > 3:
            logger.warning(f"  !! 空闲超时 ({IDLE_TIMEOUT}s)")
            break

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            last_event_time = time.time()
        except asyncio.TimeoutError:
            continue
        except Exception:
            logger.warning("  !! WebSocket 断开")
            break

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        # 解包
        wrapper_type = data.get("type", "")
        if wrapper_type in ("EVENT", "SUBSCRIBE") and "event" in data:
            event = data["event"]
        else:
            event = data
        event_type = event.get("type", wrapper_type)
        payload = event.get("payload", event)
        events.append(data)

        if event_type in ("L2_OUTPUT", "l2_output"):
            text = payload.get("text", "")
            final_response = text
            logger.info(f"  [{elapsed:.0f}s] ** L2_OUTPUT: {len(text)} 字符 **")
            logger.info(f"  前 300 字: {text[:300]}")
            await asyncio.sleep(2)
            break

        elif event_type in ("L2_THINKING_STEP", "l2_thinking_step"):
            step_type = payload.get("step_type", "")
            step_data = payload.get("data", {})
            thinking_steps.append(payload)

            if "pipeline" in step_type:
                tool = step_data.get("tool", "")
                nodes = step_data.get("graph", {}).get("nodes", [])
                turn = step_data.get("turn", 0)
                logger.info(f"  [{elapsed:.0f}s] THINKING: {step_type} turn={turn} tool={tool} nodes={len(nodes)}")
                task_graph_events.append({"step_type": step_type, "turn": turn, "tool": tool, "node_count": len(nodes)})
                if tool:
                    tool_calls.append(tool)
            else:
                logger.info(f"  [{elapsed:.0f}s] THINKING: {step_type}")

        elif event_type in ("L2_OUTPUT_STREAM", "l2_output_stream"):
            pass  # 流式输出不打印
        elif event_type in ("SUBSCRIBE_ACK", "subscribe_ack"):
            pass
        else:
            logger.info(f"  [{elapsed:.0f}s] {event_type}")

    # --- 结果分析 ---
    logger.info("\n[4/4] 结果分析")
    logger.info("-" * 50)

    total_time = time.time() - test_start
    logger.info(f"  耗时: {total_time:.0f}s")
    logger.info(f"  事件总数: {len(events)}")
    logger.info(f"  思维步骤: {len(thinking_steps)}")
    logger.info(f"  任务图事件: {len(task_graph_events)}")
    logger.info(f"  工具调用链: {' -> '.join(tool_calls) if tool_calls else '(无)'}")

    # 检查关键指标
    checks = {
        "意图分类为 COMPLEX": any("pipeline_start" in e.get("step_type", "") for e in task_graph_events),
        "创建了 TaskGraph": any(e.get("node_count", 0) > 0 for e in task_graph_events),
        "有 FC 工具调用": len(tool_calls) > 0,
        "收到最终回复": final_response is not None and len(final_response or "") > 20,
    }

    logger.info("\n  检查项:")
    all_pass = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"    [{status}] {check_name}")
        if not passed:
            all_pass = False

    if final_response:
        logger.info(f"\n  最终回复 ({len(final_response)} 字符):")
        for line in final_response[:500].split('\n'):
            logger.info(f"    | {line}")
        if len(final_response) > 500:
            logger.info(f"    | ...(截断)")

    logger.info("\n" + "=" * 70)
    if all_pass:
        logger.info(">>> TEST A: PASS")
    else:
        logger.info(">>> TEST A: FAIL")
        # 打印最近日志帮助诊断
        logger.info("\n  最近服务器日志:")
        for line in get_latest_log_lines(30):
            line = line.strip()
            if any(kw in line for kw in ["[Intent]", "[FC]", "ERROR", "Exception", "rule_guardian", "circuit_breaker"]):
                logger.info(f"    LOG> {line[:150]}")
    logger.info("=" * 70)

    # 保存结果
    report = {
        "test": "A_complex_task",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": COMPLEX_TASK,
        "total_time_sec": round(total_time, 1),
        "events_count": len(events),
        "thinking_steps": len(thinking_steps),
        "task_graph_events": task_graph_events,
        "tool_calls": tool_calls,
        "final_response_length": len(final_response) if final_response else 0,
        "checks": {k: v for k, v in checks.items()},
        "passed": all_pass,
    }
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"test_a_result_{int(time.time())}.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\n报告已保存: {report_path}")

    await ws.close()


if __name__ == "__main__":
    asyncio.run(run_test())
