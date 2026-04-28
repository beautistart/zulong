"""
祖龙系统 — 复杂任务编排 + 任务恢复 端到端调试测试
====================================================

无时间限制：模型（qwen3.5:4b）推理速度慢，本脚本不设超时。
测试分两阶段：
  Phase A: 发送复杂任务，等待完整执行完毕
  Phase B: 发送"继续任务"，验证任务恢复流程

使用前提: 系统已在另一个终端启动 (python scripts/start_zulong.py)
运行方式: python tests/test_complex_and_resume.py
"""

import asyncio
import json
import time
import os
import sys
import logging
from typing import Optional, List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("E2E_DEBUG")

# ---------- 配置 ----------
WS_URL = "ws://127.0.0.1:5555/eventbus"
CONNECT_TIMEOUT = 15  # 连接超时（唯一的超时）

# Phase A — 复杂任务
COMPLEX_TASK = (
    "帮我制定一份详细的个人健身计划，"
    "包括每周训练安排、饮食建议和注意事项。"
)

# Phase B — 任务恢复
RESUME_COMMAND = "继续任务"

# 报告目录
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------- 辅助函数 ----------
def _pretty_event(event_type: str, payload: dict, elapsed: float) -> str:
    """格式化事件输出"""
    return f"[{elapsed:>7.1f}s] {event_type}"


def _analyze_thinking_steps(steps: List[Dict]) -> Dict:
    """分析思维步骤，统计任务图操作"""
    stats = {
        "total_steps": len(steps),
        "tools_called": [],
        "task_add_node_count": 0,
        "task_mark_status_count": 0,
        "task_view_overview_count": 0,
        "task_list_suspended_count": 0,
        "other_tool_count": 0,
        "completed_nodes": [],
        "in_progress_nodes": [],
    }
    for step in steps:
        step_data = step.get("data", {})
        tool = step_data.get("tool", "")
        if not tool:
            continue
        stats["tools_called"].append(tool)
        if tool == "task_add_node":
            stats["task_add_node_count"] += 1
        elif tool == "task_mark_status":
            stats["task_mark_status_count"] += 1
            # 尝试提取状态
            args = step_data.get("args", {})
            if isinstance(args, dict):
                status = args.get("status", "")
                node_id = args.get("node_id", "")
                if status == "completed":
                    stats["completed_nodes"].append(node_id)
                elif status == "in_progress":
                    stats["in_progress_nodes"].append(node_id)
        elif tool == "task_view_overview":
            stats["task_view_overview_count"] += 1
        elif tool == "task_list_suspended":
            stats["task_list_suspended_count"] += 1
        else:
            stats["other_tool_count"] += 1
    return stats


# ---------- 核心 ----------
async def collect_until_output(ws, phase_name: str) -> Dict:
    """
    持续收集事件，直到收到 L2_OUTPUT 为止（无超时）。
    返回收集结果字典。
    """
    events_received: List[Dict] = []
    final_response: Optional[str] = None
    thinking_steps: List[Dict] = []
    task_graph_events: List[Dict] = []
    start_time = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"[{phase_name}] 开始收集事件（无超时，等待 L2_OUTPUT）...")
    logger.info(f"{'='*60}")

    while True:
        elapsed = time.time() - start_time

        # 每 60 秒打印一次心跳
        if int(elapsed) > 0 and int(elapsed) % 60 == 0:
            last_evt = events_received[-1] if events_received else None
            last_type = ""
            if last_evt:
                evt_inner = last_evt.get("event", last_evt)
                last_type = evt_inner.get("type", "")
            logger.info(
                f"[{phase_name}] 心跳: {elapsed:.0f}s 已过, "
                f"已收集 {len(events_received)} 事件, "
                f"思维步骤 {len(thinking_steps)}, "
                f"最后事件类型={last_type}"
            )

        try:
            # 用较短的 recv 超时进行轮询，但不限制总等待时间
            raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
        except asyncio.TimeoutError:
            # recv 超时只是轮询间隔，不是终止条件
            continue
        except Exception as e:
            logger.warning(f"[{phase_name}] WebSocket 异常: {e}")
            break

        # 解析
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.info(f"<<< (非JSON): {raw[:200]}")
            events_received.append({"raw": raw[:500]})
            continue

        # 解包 EventBus 转发格式
        wrapper_type = data.get("type", "")
        if wrapper_type in ("EVENT", "SUBSCRIBE") and "event" in data:
            event = data["event"]
        else:
            event = data

        event_type = event.get("type", wrapper_type)
        payload = event.get("payload", event)

        events_received.append(data)
        elapsed_str = f"{elapsed:.1f}s"

        # === L2_OUTPUT: 最终回复 ===
        if event_type in ("L2_OUTPUT", "l2_output"):
            text = payload.get("text", "")
            final_response = text
            logger.info(f"<<< [{elapsed_str}] L2_OUTPUT ({len(text)} 字符)")
            # 打印回复内容
            if text:
                preview = text[:500].replace("\n", "\n    ")
                logger.info(f"    回复:\n    {preview}")
                if len(text) > 500:
                    logger.info(f"    ... (共 {len(text)} 字符)")
            # 多等几秒收尾部事件
            await asyncio.sleep(5)
            break

        # === L2_THINKING_STEP: 推理过程 ===
        elif event_type in ("L2_THINKING_STEP", "l2_thinking_step"):
            step_type = payload.get("step_type", "")
            step_data = payload.get("data", {})
            thinking_steps.append(payload)

            if "pipeline" in step_type:
                graph_data = step_data.get("graph", {})
                nodes = graph_data.get("nodes", [])
                turn = step_data.get("turn", 0)
                tool = step_data.get("tool", "")
                tool_args = step_data.get("args", "")
                result_preview = str(step_data.get("result", ""))[:100]

                logger.info(
                    f"<<< [{elapsed_str}] THINKING: turn={turn}, "
                    f"tool={tool}, nodes={len(nodes)}"
                )
                if tool:
                    args_str = json.dumps(tool_args, ensure_ascii=False)[:150] if tool_args else ""
                    logger.info(f"    args: {args_str}")
                if result_preview:
                    logger.info(f"    result: {result_preview}")

                task_graph_events.append({
                    "step_type": step_type,
                    "turn": turn,
                    "tool": tool,
                    "node_count": len(nodes),
                    "elapsed": elapsed,
                })
            else:
                logger.info(f"<<< [{elapsed_str}] THINKING: {step_type}")

        # === ACK / 订阅确认 ===
        elif event_type in ("SUBSCRIBE_ACK", "subscribe_ack", "ACK", "ack"):
            logger.info(f"<<< [{elapsed_str}] {event_type}")

        # === 其他 ===
        else:
            summary = json.dumps(data, ensure_ascii=False)[:200]
            logger.info(f"<<< [{elapsed_str}] {event_type}: {summary}")

    total_time = time.time() - start_time

    return {
        "phase": phase_name,
        "total_time_sec": round(total_time, 1),
        "events_count": len(events_received),
        "thinking_steps_count": len(thinking_steps),
        "thinking_steps": thinking_steps,
        "task_graph_events": task_graph_events,
        "final_response": final_response,
        "final_response_length": len(final_response) if final_response else 0,
    }


def send_publish_message(text: str) -> str:
    """构造 EventBus PUBLISH 消息 JSON"""
    return json.dumps(
        {
            "type": "PUBLISH",
            "event": {
                "type": "USER_TEXT",
                "priority": "NORMAL",
                "source": "DiagTestClient",
                "payload": {
                    "text": text,
                    "voice_mode": "TEXT_ONLY",
                },
            },
        },
        ensure_ascii=False,
    )


def print_phase_report(result: Dict, phase_label: str):
    """打印阶段测试报告"""
    logger.info(f"\n{'='*60}")
    logger.info(f"{phase_label} 测试报告")
    logger.info(f"{'='*60}")
    logger.info(f"总耗时: {result['total_time_sec']}s")
    logger.info(f"收到事件数: {result['events_count']}")
    logger.info(f"思维步骤数: {result['thinking_steps_count']}")
    logger.info(f"任务图谱事件数: {len(result['task_graph_events'])}")

    # 工具调用统计
    stats = _analyze_thinking_steps(result["thinking_steps"])
    logger.info(f"\n--- 工具调用统计 ---")
    logger.info(f"  task_add_node    : {stats['task_add_node_count']}")
    logger.info(f"  task_mark_status : {stats['task_mark_status_count']}")
    logger.info(f"  task_view_overview: {stats['task_view_overview_count']}")
    logger.info(f"  task_list_suspended: {stats['task_list_suspended_count']}")
    logger.info(f"  其他工具         : {stats['other_tool_count']}")
    logger.info(f"  完成的节点       : {stats['completed_nodes']}")

    if result["task_graph_events"]:
        logger.info(f"\n--- 任务图谱事件时间线 ---")
        for evt in result["task_graph_events"]:
            logger.info(
                f"  [{evt['elapsed']:.1f}s] {evt['step_type']}: "
                f"turn={evt['turn']}, tool={evt['tool']}, nodes={evt['node_count']}"
            )

    resp = result["final_response"]
    if resp:
        logger.info(f"\n--- 最终回复 ({len(resp)} 字符) ---")
        logger.info(resp[:1500])
        if len(resp) > 1500:
            logger.info(f"...(截断，总长 {len(resp)} 字符)")
    else:
        logger.info(f"\n--- 最终回复: 无 ---")

    # 判定
    issues = []
    if not resp:
        issues.append("FAIL: 未收到最终回复")
    elif len(resp) < 50:
        issues.append(f"WARN: 回复过短 ({len(resp)} 字符), 可能是降级回复")
    if stats["task_view_overview_count"] > 2:
        issues.append(
            f"WARN: task_view_overview 调用次数过多 ({stats['task_view_overview_count']}), "
            f"应该只调用 1 次"
        )
    if stats["task_add_node_count"] == 0 and phase_label.startswith("Phase A"):
        issues.append("WARN: 未检测到 task_add_node 调用, 可能未创建任务大纲")
    if stats["task_mark_status_count"] == 0 and phase_label.startswith("Phase A"):
        issues.append("WARN: 未检测到 task_mark_status 调用, 可能未标记任务状态")
    if stats["task_list_suspended_count"] > 0 and phase_label.startswith("Phase A"):
        issues.append("WARN: 复杂任务阶段不应调用 task_list_suspended")

    if issues:
        logger.warning(f"\n--- 问题 ({len(issues)}) ---")
        for i, issue in enumerate(issues, 1):
            logger.warning(f"  {i}. {issue}")
    else:
        logger.info(f"\n>>> {phase_label} 结果: PASS")

    return issues


async def run_test():
    """主测试流程"""
    try:
        import websockets
    except ImportError:
        logger.error("需要安装 websockets: pip install websockets")
        return

    logger.info("=" * 60)
    logger.info("祖龙系统 — 复杂任务 + 恢复 调试测试")
    logger.info("=" * 60)
    logger.info(f"WebSocket: {WS_URL}")
    logger.info(f"复杂任务: {COMPLEX_TASK}")
    logger.info(f"恢复命令: {RESUME_COMMAND}")
    logger.info(f"超时策略: 无（等待到 L2_OUTPUT 为止）")

    # ========== 连接 ==========
    logger.info(f"\n--- 连接 WebSocket ---")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(
                WS_URL,
                max_size=10 * 1024 * 1024,
                ping_interval=None,   # 禁用 keepalive ping（模型推理耗时长）
                ping_timeout=None,
            ),
            timeout=CONNECT_TIMEOUT,
        )
        logger.info(f"已连接到 {WS_URL}")
    except asyncio.TimeoutError:
        logger.error(f"连接超时 ({CONNECT_TIMEOUT}s)，系统是否已启动？")
        logger.error("请先运行: python scripts/start_zulong.py")
        return
    except Exception as e:
        logger.error(f"连接失败: {e}")
        logger.error("请先运行: python scripts/start_zulong.py")
        return

    all_results = []

    # ============================================================
    # Phase A: 复杂任务编排
    # ============================================================
    logger.info(f"\n{'#'*60}")
    logger.info(f"# Phase A: 复杂任务编排测试")
    logger.info(f"{'#'*60}")
    logger.info(f">>> 发送: {COMPLEX_TASK}")
    await ws.send(send_publish_message(COMPLEX_TASK))

    # 等待 ACK
    try:
        ack_raw = await asyncio.wait_for(ws.recv(), timeout=10)
        logger.info(f"<<< ACK: {ack_raw[:200]}")
    except asyncio.TimeoutError:
        logger.warning("ACK 超时，继续...")

    result_a = await collect_until_output(ws, "PhaseA_Complex")
    issues_a = print_phase_report(result_a, "Phase A — 复杂任务编排")
    all_results.append(result_a)

    # 判断是否继续 Phase B
    # 复杂任务完成后，系统会自动挂起未完成的任务 (_auto_suspend_if_needed)
    # 等待几秒让系统完成挂起
    if result_a["final_response"]:
        logger.info("\n等待 15 秒让系统完成任务挂起...")
        await asyncio.sleep(15)

        # ============================================================
        # Phase B: 任务恢复 — 使用新连接避免长时间空闲后连接失效
        # ============================================================
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Phase B: 任务恢复测试")
        logger.info(f"{'#'*60}")

        # 关闭旧连接，建立新连接
        logger.info("--- 关闭旧 WebSocket 连接 ---")
        try:
            await ws.close()
            logger.info("旧连接已关闭")
        except Exception as e:
            logger.warning(f"关闭旧连接异常（忽略）: {e}")
        await asyncio.sleep(2)

        logger.info("--- 建立新 WebSocket 连接 ---")
        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    WS_URL,
                    max_size=10 * 1024 * 1024,
                    ping_interval=None,
                    ping_timeout=None,
                ),
                timeout=CONNECT_TIMEOUT,
            )
            logger.info(f"新连接已建立: {WS_URL}")
        except Exception as e:
            logger.error(f"Phase B 重新连接失败: {e}")
            all_results.append({
                "phase": "PhaseB_Resume",
                "total_time_sec": 0,
                "events_count": 0,
                "thinking_steps_count": 0,
                "thinking_steps": [],
                "task_graph_events": [],
                "final_response": None,
                "final_response_length": 0,
                "error": str(e),
            })
            # 跳过 Phase B
            ws = None

        if ws is not None:
            logger.info(f">>> 发送: {RESUME_COMMAND}")
            await ws.send(send_publish_message(RESUME_COMMAND))

            # 等待 ACK
            try:
                ack_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                logger.info(f"<<< ACK: {ack_raw[:200]}")
            except asyncio.TimeoutError:
                logger.warning("ACK 超时，继续...")

            result_b = await collect_until_output(ws, "PhaseB_Resume")

            # 恢复阶段的特殊检查
            issues_b = print_phase_report(result_b, "Phase B — 任务恢复")
            stats_b = _analyze_thinking_steps(result_b["thinking_steps"])
            if stats_b["task_list_suspended_count"] > 1:
                logger.warning(
                    f"  额外问题: task_list_suspended 调用 {stats_b['task_list_suspended_count']} 次, "
                    "已优化的 RESUME 流程应该 <=1 次"
                )
            all_results.append(result_b)
    else:
        logger.warning("Phase A 未收到回复，跳过 Phase B")

    # ============================================================
    # 保存报告
    # ============================================================
    report_path = os.path.join(
        REPORT_DIR, f"debug_result_{int(time.time())}.json"
    )
    report = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "complex_task": COMPLEX_TASK,
        "resume_command": RESUME_COMMAND,
        "phases": [],
    }
    for r in all_results:
        report["phases"].append({
            "phase": r["phase"],
            "total_time_sec": r["total_time_sec"],
            "events_count": r["events_count"],
            "thinking_steps_count": r["thinking_steps_count"],
            "task_graph_events": r["task_graph_events"],
            "final_response": r["final_response"],
            "final_response_length": r["final_response_length"],
            "tool_stats": _analyze_thinking_steps(r["thinking_steps"]),
        })

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\n详细报告已保存: {report_path}")

    logger.info(f"\n{'='*60}")
    logger.info("全部测试完成")
    logger.info(f"{'='*60}")

    if ws is not None:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run_test())
