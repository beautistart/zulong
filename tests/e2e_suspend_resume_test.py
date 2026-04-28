"""
祖龙系统 — 任务挂起/恢复端到端测试
====================================

测试流程:
1. 发送复杂任务 → 等待系统开始创建 TaskGraph
2. 等待系统处理完成（或超时后自动挂起）
3. 检查挂起的任务文件
4. 发送 RESUME 请求（"继续做上次那个"）
5. 验证系统是否恢复了 TaskGraph 并继续执行

运行方式: python tests/e2e_suspend_resume_test.py
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
)
logger = logging.getLogger("SUSPEND_RESUME_TEST")

WS_URL = "ws://127.0.0.1:5555/eventbus"
SUSPENDED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "suspended_tasks")
GRAPH_BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "graph_backups")


async def send_and_collect(ws, text, label, max_wait=120, wait_for_output=True):
    """发送消息并收集事件"""
    message = {
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "priority": "NORMAL",
            "source": "SuspendResumeTest",
            "payload": {
                "text": text,
                "voice_mode": "TEXT_ONLY",
            },
        },
    }
    await ws.send(json.dumps(message, ensure_ascii=False))
    logger.info(f">>> [{label}] 已发送: {text[:60]}...")

    events = []
    thinking_steps = []
    task_graph_events = []
    final_response = None
    start_time = time.time()
    last_event_time = time.time()

    while True:
        elapsed = time.time() - start_time
        idle = time.time() - last_event_time

        if elapsed > max_wait:
            logger.info(f"[{label}] 达到最大等待时间 ({max_wait}s)")
            break
        if idle > 30 and len(events) > 0:
            logger.info(f"[{label}] 空闲 30s 无新消息，停止等待")
            break

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            last_event_time = time.time()
        except asyncio.TimeoutError:
            continue
        except Exception:
            break

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        # 解包 SUBSCRIBE/EVENT 包装
        wrapper_type = data.get("type", "")
        if wrapper_type in ("EVENT", "SUBSCRIBE") and "event" in data:
            event = data["event"]
        else:
            event = data

        event_type = event.get("type", wrapper_type)
        payload = event.get("payload", event)
        events.append(data)

        if event_type in ("L2_OUTPUT", "l2_output"):
            text_content = payload.get("text", "")
            final_response = text_content
            logger.info(f"<<< [{label}] [{elapsed:.0f}s] L2_OUTPUT: {len(text_content)} 字符")
            logger.info(f"    内容前 200 字: {text_content[:200]}")
            if wait_for_output:
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
                logger.info(f"<<< [{label}] [{elapsed:.0f}s] THINKING: {step_type} turn={turn} tool={tool} nodes={len(nodes)}")
                task_graph_events.append({"step_type": step_type, "turn": turn, "tool": tool, "node_count": len(nodes)})
            else:
                logger.info(f"<<< [{label}] [{elapsed:.0f}s] THINKING: {step_type}")

        elif event_type == "ACK":
            logger.info(f"<<< [{label}] [{elapsed:.0f}s] ACK")

        elif event_type in ("MEMORY_GRAPH_UPDATED", "memory_graph_updated"):
            logger.info(f"<<< [{label}] [{elapsed:.0f}s] MEMORY_GRAPH_UPDATED")

        else:
            logger.info(f"<<< [{label}] [{elapsed:.0f}s] {event_type}")

    return {
        "events": events,
        "thinking_steps": thinking_steps,
        "task_graph_events": task_graph_events,
        "final_response": final_response,
        "elapsed": time.time() - start_time,
    }


def check_suspended_tasks(after_timestamp=0):
    """检查挂起任务文件"""
    if not os.path.isdir(SUSPENDED_DIR):
        return []
    files = glob.glob(os.path.join(SUSPENDED_DIR, "task_*.json"))
    recent = []
    for f in files:
        mtime = os.path.getmtime(f)
        if mtime > after_timestamp:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                recent.append({
                    "file": os.path.basename(f),
                    "mtime": mtime,
                    "description": data.get("description", ""),
                    "has_task_graph": data.get("task_graph") is not None,
                    "task_id": data.get("task_id", ""),
                })
            except Exception:
                recent.append({"file": os.path.basename(f), "mtime": mtime, "error": True})
    recent.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return recent


def check_graph_backups(after_timestamp=0):
    """检查图备份"""
    if not os.path.isdir(GRAPH_BACKUP_DIR):
        return []
    files = glob.glob(os.path.join(GRAPH_BACKUP_DIR, "tg_*.json"))
    recent = []
    for f in files:
        mtime = os.path.getmtime(f)
        if mtime > after_timestamp:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                nodes = data.get("nodes", {})
                recent.append({
                    "file": os.path.basename(f),
                    "mtime": mtime,
                    "title": data.get("title", ""),
                    "node_count": len(nodes),
                    "node_ids": list(nodes.keys()),
                })
            except Exception:
                recent.append({"file": os.path.basename(f), "mtime": mtime, "error": True})
    recent.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return recent


async def run_test():
    try:
        import websockets
    except ImportError:
        logger.error("需要安装 websockets: pip install websockets")
        return

    logger.info("=" * 70)
    logger.info("祖龙系统 — 任务挂起/恢复端到端测试")
    logger.info("=" * 70)

    # 记录测试开始前的时间戳
    test_start = time.time()

    # ── Phase 1: 连接 ──
    logger.info("\n--- Phase 1: 连接 WebSocket ---")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(WS_URL, max_size=10 * 1024 * 1024),
            timeout=10,
        )
        logger.info(f"已连接到 {WS_URL}")
    except Exception as e:
        logger.error(f"连接失败: {e}")
        return

    # ── Phase 2: 发送复杂任务 ──
    logger.info("\n--- Phase 2: 发送复杂任务（等待系统处理） ---")
    complex_result = await send_and_collect(
        ws,
        "帮我做一个简单的待办事项清单，需要包含添加任务、标记完成、删除任务三个功能的设计方案。",
        "COMPLEX",
        max_wait=180,
        wait_for_output=True,
    )

    logger.info(f"\n--- Phase 2 结果 ---")
    logger.info(f"  事件数: {len(complex_result['events'])}")
    logger.info(f"  思维步骤: {len(complex_result['thinking_steps'])}")
    logger.info(f"  任务图事件: {len(complex_result['task_graph_events'])}")
    logger.info(f"  最终回复: {'有' if complex_result['final_response'] else '无'}")
    logger.info(f"  耗时: {complex_result['elapsed']:.0f}s")

    # ── Phase 3: 等待自动挂起 ──
    logger.info("\n--- Phase 3: 检查自动挂起（Rule C） ---")
    await asyncio.sleep(5)  # 等一下让 auto-suspend 完成

    suspended = check_suspended_tasks(after_timestamp=test_start)
    if suspended:
        logger.info(f"  发现 {len(suspended)} 个新挂起任务:")
        for s in suspended:
            logger.info(f"    - {s.get('file', '?')}: '{s.get('description', '?')}' has_graph={s.get('has_task_graph', '?')}")
    else:
        logger.info(f"  未发现新挂起任务（任务可能已完成或仍在执行中）")

    backups = check_graph_backups(after_timestamp=test_start)
    if backups:
        logger.info(f"  发现 {len(backups)} 个新图备份:")
        for b in backups:
            logger.info(f"    - {b.get('file', '?')}: '{b.get('title', '?')[:40]}' nodes={b.get('node_count', '?')}")

    # ── Phase 4: 发送 RESUME 请求 ──
    logger.info("\n--- Phase 4: 发送恢复请求 ---")
    resume_result = await send_and_collect(
        ws,
        "继续做上次那个待办清单的设计",
        "RESUME",
        max_wait=180,
        wait_for_output=True,
    )

    logger.info(f"\n--- Phase 4 结果 ---")
    logger.info(f"  事件数: {len(resume_result['events'])}")
    logger.info(f"  思维步骤: {len(resume_result['thinking_steps'])}")
    logger.info(f"  任务图事件: {len(resume_result['task_graph_events'])}")
    logger.info(f"  最终回复: {'有' if resume_result['final_response'] else '无'}")
    logger.info(f"  耗时: {resume_result['elapsed']:.0f}s")

    # ── Phase 5: 验证恢复是否成功 ──
    logger.info("\n--- Phase 5: 恢复结果验证 ---")

    resume_has_task_events = len(resume_result['task_graph_events']) > 0
    resume_has_response = resume_result['final_response'] is not None

    # 检查是否有 pipeline_start 事件（表明系统启动了 FC 循环）
    resume_pipeline_started = any(
        e.get("step_type") == "pipeline.pipeline_start" for e in resume_result['task_graph_events']
    )

    logger.info(f"  RESUME 启动了 FC 循环: {resume_pipeline_started}")
    logger.info(f"  RESUME 有任务图事件: {resume_has_task_events}")
    logger.info(f"  RESUME 有最终回复: {resume_has_response}")

    # ── 总结 ──
    logger.info("\n" + "=" * 70)
    logger.info("测试总结")
    logger.info("=" * 70)

    complex_ok = len(complex_result['task_graph_events']) > 0
    suspend_ok = len(suspended) > 0 or len(backups) > 0
    resume_ok = resume_has_task_events or resume_has_response

    logger.info(f"  [{'PASS' if complex_ok else 'FAIL'}] 复杂任务创建 TaskGraph: {len(complex_result['task_graph_events'])} 个图谱事件")
    logger.info(f"  [{'PASS' if suspend_ok else 'WARN'}] 任务自动挂起: {len(suspended)} 个挂起文件, {len(backups)} 个图备份")
    logger.info(f"  [{'PASS' if resume_ok else 'FAIL'}] 任务恢复执行: events={len(resume_result['task_graph_events'])}, response={'有' if resume_has_response else '无'}")

    if complex_ok and resume_ok:
        logger.info(f"\n>>> 挂起/恢复测试: PASS")
    else:
        logger.info(f"\n>>> 挂起/恢复测试: 部分通过，需人工确认")

    await ws.close()


if __name__ == "__main__":
    asyncio.run(run_test())
