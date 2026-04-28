"""
测试 B: 旧任务恢复测试
========================
验证: 意图分类(RESUME) -> 匹配挂起任务 -> TaskGraph恢复 -> 继续执行 -> 最终回复

前置条件: 系统中有已挂起的任务（测试 A 完成后会自动挂起）

独立运行: python -X utf8 tests/test_b_resume_task.py
"""

import asyncio
import json
import time
import sys
import os
import logging
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TEST_B")

WS_URL = "ws://127.0.0.1:5555/eventbus"
SUSPENDED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "suspended_tasks")

# 恢复请求 — 明确使用"继续"意图
RESUME_REQUEST = "继续完成上次那个任务"

RESPONSE_TIMEOUT = 240
IDLE_TIMEOUT = 90

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def list_suspended_tasks():
    """列出所有挂起任务"""
    if not os.path.isdir(SUSPENDED_DIR):
        return []
    files = glob.glob(os.path.join(SUSPENDED_DIR, "task_*.json"))
    tasks = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            tasks.append({
                "file": os.path.basename(f),
                "path": f,
                "description": data.get("description", ""),
                "task_id": data.get("task_id", ""),
                "has_task_graph": data.get("task_graph") is not None,
                "mtime": os.path.getmtime(f),
            })
        except Exception as e:
            tasks.append({"file": os.path.basename(f), "error": str(e)})
    tasks.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return tasks


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
    logger.info("TEST B: 旧任务恢复测试")
    logger.info("=" * 70)

    # --- 检查挂起任务 ---
    logger.info("\n[1/5] 检查已挂起的任务...")
    suspended = list_suspended_tasks()
    if not suspended:
        logger.error("  FAIL - 没有已挂起的任务！请先运行 test_a 创建任务")
        return

    logger.info(f"  找到 {len(suspended)} 个挂起任务:")
    for i, t in enumerate(suspended[:5]):
        desc = t.get("description", "?")[:50]
        has_graph = t.get("has_task_graph", False)
        logger.info(f"    [{i}] {t.get('file', '?')} - '{desc}' graph={has_graph}")

    latest_task = suspended[0]
    logger.info(f"\n  将恢复最近的任务: {latest_task.get('description', '?')[:60]}")

    # --- 连接 ---
    logger.info("\n[2/5] 连接 WebSocket...")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(WS_URL, max_size=10 * 1024 * 1024),
            timeout=10,
        )
        logger.info(f"  OK - 已连接 {WS_URL}")
    except Exception as e:
        logger.error(f"  FAIL - 连接失败: {e}")
        return

    # --- 发送恢复请求 ---
    logger.info(f"\n[3/5] 发送恢复请求: '{RESUME_REQUEST}'")
    test_start = time.time()
    message = {
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "priority": "NORMAL",
            "source": "TestB_Resume",
            "payload": {
                "text": RESUME_REQUEST,
                "voice_mode": "TEXT_ONLY",
            },
        },
    }
    await ws.send(json.dumps(message, ensure_ascii=False))
    logger.info(f"  OK - 已发送")

    # --- 收集响应 ---
    logger.info(f"\n[4/5] 收集响应事件 (超时 {RESPONSE_TIMEOUT}s)...")
    events = []
    thinking_steps = []
    task_graph_events = []
    tool_calls = []
    final_response = None
    last_event_time = time.time()
    resume_detected = False
    task_graph_restored = False

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
                node_count = len(nodes)
                logger.info(f"  [{elapsed:.0f}s] THINKING: {step_type} turn={turn} tool={tool} nodes={node_count}")
                task_graph_events.append({"step_type": step_type, "turn": turn, "tool": tool, "node_count": node_count})
                if tool:
                    tool_calls.append(tool)

                # 检测关键信号
                if "pipeline_start" in step_type and node_count > 1:
                    task_graph_restored = True
                    logger.info(f"  ** TaskGraph 已恢复! nodes={node_count} **")

                if tool in ("task_view_overview", "task_mark_status"):
                    resume_detected = True
            else:
                logger.info(f"  [{elapsed:.0f}s] THINKING: {step_type}")

        elif event_type in ("L2_OUTPUT_STREAM", "l2_output_stream"):
            pass
        elif event_type in ("SUBSCRIBE_ACK", "subscribe_ack"):
            pass
        else:
            logger.info(f"  [{elapsed:.0f}s] {event_type}")

    # --- 结果分析 ---
    logger.info(f"\n[5/5] 结果分析")
    logger.info("-" * 50)

    total_time = time.time() - test_start
    logger.info(f"  耗时: {total_time:.0f}s")
    logger.info(f"  事件总数: {len(events)}")
    logger.info(f"  思维步骤: {len(thinking_steps)}")
    logger.info(f"  任务图事件: {len(task_graph_events)}")
    logger.info(f"  工具调用链: {' -> '.join(tool_calls) if tool_calls else '(无)'}")

    # 关键检查
    checks = {
        "意图分类为 RESUME": any("pipeline_start" in e.get("step_type", "") for e in task_graph_events),
        "TaskGraph 已恢复(nodes>1)": task_graph_restored,
        "调用了任务管理工具": resume_detected or any(t in ("task_view_overview", "task_mark_status", "task_add_node") for t in tool_calls),
        "收到最终回复": final_response is not None and len(final_response or "") > 20,
        "未陷入无限循环(耗时<200s)": total_time < 200,
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

    # 检查挂起文件是否被消费
    post_suspended = list_suspended_tasks()
    consumed = len(suspended) - len(post_suspended)
    logger.info(f"\n  挂起文件变化: {len(suspended)} -> {len(post_suspended)} (消费了 {consumed} 个)")

    logger.info("\n" + "=" * 70)
    if all_pass:
        logger.info(">>> TEST B: PASS")
    else:
        logger.info(">>> TEST B: FAIL (部分检查未通过)")
        # 打印最近日志帮助诊断
        logger.info("\n  最近服务器日志 (关键行):")
        for line in get_latest_log_lines(50):
            line = line.strip()
            if any(kw in line for kw in [
                "[Intent]", "[FC]", "RESUME", "resume", "suspend",
                "RuleGuardian", "CircuitBreaker", "ERROR", "Exception",
                "heuristic", "no_progress",
            ]):
                logger.info(f"    LOG> {line[:160]}")
    logger.info("=" * 70)

    # 保存结果
    report = {
        "test": "B_resume_task",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resume_request": RESUME_REQUEST,
        "total_time_sec": round(total_time, 1),
        "events_count": len(events),
        "thinking_steps": len(thinking_steps),
        "task_graph_events": task_graph_events,
        "tool_calls": tool_calls,
        "task_graph_restored": task_graph_restored,
        "final_response_length": len(final_response) if final_response else 0,
        "checks": {k: v for k, v in checks.items()},
        "passed": all_pass,
    }
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"test_b_result_{int(time.time())}.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\n报告已保存: {report_path}")

    await ws.close()


if __name__ == "__main__":
    asyncio.run(run_test())
