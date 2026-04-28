"""
挂起/恢复 E2E 测试
流程：
1. 发送复杂任务 A（代码生成）
2. 等待 Agent 开始执行（60s）
3. 发送全新任务 B（触发挂起A + 处理B）
4. 验证磁盘上出现挂起文件
5. 等待 B 完成或超时
6. 发送 "继续之前的任务"（触发恢复A）
7. 验证恢复流程工作正常
"""
import websocket
import json
import time
import sys
import os
import glob

WS_URL = "ws://localhost:5555/eventbus"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "test_suspend_resume_output.log")
SUSPENDED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "suspended_tasks")

# 任务 A：复杂代码生成任务
TASK_A = "帮我用Python写一个简单的待办事项命令行工具，支持添加、删除、列出待办事项，使用JSON文件存储"

# 任务 B：完全不同的新任务（触发挂起）
TASK_B = "搜索一下2026年最新的人工智能发展趋势"

# 恢复请求
RESUME_MSG = "继续之前的任务"

# 阶段等待时间
WAIT_BEFORE_INTERRUPT = 60   # 等A开始执行后再中断
WAIT_FOR_B_TIMEOUT = 300     # 等B完成的超时
WAIT_BEFORE_RESUME = 10      # B完成后等一下再发恢复
TOTAL_TIMEOUT = 900          # 总超时


def send_user_text(ws, text, session_id="suspend_test", request_id=None):
    if request_id is None:
        request_id = f"sr_{int(time.time())}"
    msg = json.dumps({
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "source": "suspend_resume_test",
            "payload": {
                "text": text,
                "session_id": session_id,
                "request_id": request_id,
                "confidence": 1.0,
            }
        }
    })
    ws.send(msg)


def count_suspended_files():
    if not os.path.exists(SUSPENDED_DIR):
        return 0
    return len(glob.glob(os.path.join(SUSPENDED_DIR, "task_*.json")))


def main():
    out = open(OUTPUT_FILE, "w", encoding="utf-8")

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        out.write(line + "\n")
        out.flush()
        try:
            print(line, flush=True)
        except Exception:
            pass

    log("=== 挂起/恢复 E2E 测试 ===")
    log(f"任务 A: {TASK_A[:60]}...")
    log(f"任务 B: {TASK_B[:60]}...")
    log(f"恢复: {RESUME_MSG}")
    log(f"挂起目录: {SUSPENDED_DIR}")
    log("")

    # 记录初始挂起文件数量
    initial_suspended = count_suspended_files()
    log(f"初始挂起文件数: {initial_suspended}")

    # 连接
    try:
        ws = websocket.create_connection(WS_URL, timeout=10)
        log("WebSocket 已连接")
    except Exception as e:
        log(f"连接失败: {e}")
        out.close()
        return

    # 订阅
    subscribe_msg = json.dumps({
        "type": "SUBSCRIBE",
        "client_name": "suspend_resume_test",
        "event_types": ["L2_OUTPUT", "L2_THINKING_STEP", "L2_OUTPUT_STREAM"]
    })
    ws.send(subscribe_msg)
    log("已发送订阅请求")

    # ====== 阶段 1：发送任务 A ======
    log(f"\n{'='*60}")
    log("阶段 1: 发送任务 A (复杂代码生成)")
    log(f"{'='*60}")
    send_user_text(ws, TASK_A, request_id="task_a_001")
    log(f"已发送任务 A")

    # 跟踪状态
    phase = "WAIT_A_START"  # WAIT_A_START -> A_RUNNING -> B_RUNNING -> RESUME -> DONE
    event_count = 0
    a_events = 0
    b_events = 0
    resume_events = 0
    a_tool_calls = 0
    b_tool_calls = 0
    resume_tool_calls = 0
    a_output = ""
    b_output = ""
    resume_output = ""
    ask_user_count = 0
    suspend_detected = False
    resume_detected = False
    a_start_time = time.time()
    phase_start = time.time()
    start_time = time.time()

    ws.settimeout(3.0)

    while True:
        elapsed = time.time() - start_time
        if elapsed > TOTAL_TIMEOUT:
            log(f"\n总超时 ({TOTAL_TIMEOUT}s)，停止")
            break

        phase_elapsed = time.time() - phase_start

        # 阶段转换：等 A 开始执行一段时间后发送 B
        if phase == "WAIT_A_START" and a_events >= 3:
            phase = "A_RUNNING"
            phase_start = time.time()
            log(f"\n任务 A 已开始执行 (收到 {a_events} 个事件)")

        if phase == "A_RUNNING" and phase_elapsed >= WAIT_BEFORE_INTERRUPT:
            # 发送任务 B 中断 A
            log(f"\n{'='*60}")
            log(f"阶段 2: 发送任务 B 中断任务 A (A已运行 {phase_elapsed:.0f}s)")
            log(f"{'='*60}")
            send_user_text(ws, TASK_B, request_id="task_b_002")
            log(f"已发送任务 B: {TASK_B}")
            phase = "B_RUNNING"
            phase_start = time.time()

        # 阶段转换：B 输出收到后等待再发恢复
        if phase == "B_DONE" and phase_elapsed >= WAIT_BEFORE_RESUME:
            # 检查磁盘是否有新的挂起文件
            current_suspended = count_suspended_files()
            log(f"\n当前挂起文件数: {current_suspended} (初始: {initial_suspended})")
            if current_suspended > initial_suspended:
                log("PASS: 磁盘上发现新的挂起任务文件")
                suspend_detected = True
                # 列出挂起文件
                if os.path.exists(SUSPENDED_DIR):
                    for f in os.listdir(SUSPENDED_DIR):
                        if f.endswith(".json"):
                            log(f"  挂起文件: {f}")
            else:
                log("WARN: 磁盘上未发现新的挂起任务文件")

            log(f"\n{'='*60}")
            log(f"阶段 3: 发送恢复请求")
            log(f"{'='*60}")
            send_user_text(ws, RESUME_MSG, request_id="resume_003")
            log(f"已发送: {RESUME_MSG}")
            phase = "RESUME"
            phase_start = time.time()

        try:
            raw = ws.recv()
            data = json.loads(raw)
            event_count += 1
            evt_type = data.get("type", "")

            if evt_type == "SUBSCRIBE":
                evt = data.get("event", {})
                inner_type = evt.get("type", "")
                payload = evt.get("payload", {})

                if inner_type == "L2_THINKING_STEP":
                    step_type = payload.get("step_type", "")
                    step_data = payload.get("data", {})

                    if phase in ("WAIT_A_START", "A_RUNNING"):
                        a_events += 1

                        if step_type == "pipeline.ask_user":
                            ask_user_count += 1
                            q = step_data.get("question", "")
                            log(f"\n[ASK_USER] {q}")
                            time.sleep(2)
                            answer = "待办事项命令行工具，用JSON文件存储"
                            send_user_text(ws, answer, request_id="task_a_001")
                            log(f">>> 自动回答: {answer}")

                        elif step_type == "pipeline.agent_tool_call":
                            a_tool_calls += 1
                            tool = step_data.get("tool", "")
                            log(f"[A-TOOL #{a_tool_calls}] {tool}")

                        elif step_type == "pipeline.task_suspended":
                            suspend_detected = True
                            log(f"[SUSPENDED] 任务A已挂起: {step_data}")

                        elif step_type.startswith("pipeline."):
                            ptype = step_type.replace("pipeline.", "")
                            if ptype in ("agent_start", "agent_done", "graph_update"):
                                log(f"[A-{ptype.upper()}] {str(step_data)[:80]}")

                    elif phase == "B_RUNNING":
                        b_events += 1
                        if step_type == "pipeline.agent_tool_call":
                            b_tool_calls += 1
                            tool = step_data.get("tool", "")
                            log(f"[B-TOOL #{b_tool_calls}] {tool}")
                        elif step_type == "pipeline.task_suspended":
                            suspend_detected = True
                            log(f"[SUSPENDED] {step_data}")
                        elif step_type.startswith("pipeline.") and "agent" in step_type:
                            ptype = step_type.replace("pipeline.", "")
                            log(f"[B-{ptype.upper()}] {str(step_data)[:80]}")

                    elif phase == "RESUME":
                        resume_events += 1
                        if step_type == "pipeline.agent_tool_call":
                            resume_tool_calls += 1
                            tool = step_data.get("tool", "")
                            log(f"[R-TOOL #{resume_tool_calls}] {tool}")
                        elif step_type.startswith("pipeline.") and "agent" in step_type:
                            ptype = step_type.replace("pipeline.", "")
                            log(f"[R-{ptype.upper()}] {str(step_data)[:80]}")

                elif inner_type == "L2_OUTPUT":
                    text = payload.get("text", "")
                    is_ask = payload.get("ask_user", False)
                    is_suspended = payload.get("task_suspended", False)

                    if is_ask and phase in ("WAIT_A_START", "A_RUNNING"):
                        if ask_user_count == 0:
                            ask_user_count += 1
                            log(f"\n[ASK_USER via L2_OUTPUT] {text[:120]}")
                            time.sleep(2)
                            answer = "待办事项命令行工具，用JSON文件存储"
                            send_user_text(ws, answer, request_id="task_a_001")
                            log(f">>> 自动回答: {answer}")
                    elif is_suspended:
                        suspend_detected = True
                        log(f"[SUSPENDED via L2_OUTPUT] {text[:120]}")
                    elif text:
                        if phase == "B_RUNNING":
                            b_output = text
                            log(f"\n[B-OUTPUT] 收到! 长度={len(text)}")
                            log(f"  Preview: {text[:200]}...")
                            phase = "B_DONE"
                            phase_start = time.time()
                        elif phase == "RESUME":
                            resume_output = text
                            resume_detected = True
                            log(f"\n[RESUME-OUTPUT] 收到! 长度={len(text)}")
                            log(f"  Preview: {text[:200]}...")
                            log("恢复任务已完成，结束测试")
                            break
                        elif phase in ("WAIT_A_START", "A_RUNNING"):
                            # A 可能在中断前就完成了（任务简单时）
                            a_output = text
                            log(f"\n[A-OUTPUT] 收到 (A在中断前完成)! 长度={len(text)}")
                            log(f"  Preview: {text[:200]}...")
                            # A 已完成，没有被中断的必要
                            if phase == "WAIT_A_START" or (phase == "A_RUNNING" and phase_elapsed < WAIT_BEFORE_INTERRUPT):
                                log("任务 A 在中断前已完成，测试退出")
                                phase = "A_DONE_EARLY"
                                break

            elif evt_type == "ACK":
                pass

        except websocket.WebSocketTimeoutException:
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                log(f"  ... [{phase}] 等待中 ({int(elapsed)}s, events={event_count})")
            continue
        except Exception as e:
            log(f"[ERROR] {e}")
            break

    total_elapsed = time.time() - start_time

    # 最终磁盘检查
    final_suspended = count_suspended_files()

    log(f"\n{'='*60}")
    log("=== 测试统计 ===")
    log(f"  总耗时: {total_elapsed:.1f}s")
    log(f"  总事件数: {event_count}")
    log(f"  最终阶段: {phase}")
    log(f"  任务 A 事件: {a_events}, 工具调用: {a_tool_calls}")
    log(f"  任务 B 事件: {b_events}, 工具调用: {b_tool_calls}")
    log(f"  恢复 事件: {resume_events}, 工具调用: {resume_tool_calls}")
    log(f"  ask_user: {ask_user_count}")
    log(f"  磁盘挂起文件: 初始={initial_suspended}, 最终={final_suspended}")
    log("")
    log("=== 挂起/恢复验证 ===")

    if phase == "A_DONE_EARLY":
        log("  NOTE: 任务 A 在中断前完成，无法测试挂起/恢复")
        log("  建议使用更复杂的任务重新测试")
    else:
        log(f"  [1] 任务挂起到磁盘: {'PASS' if suspend_detected else 'FAIL'}")
        log(f"  [2] 磁盘文件出现:   {'PASS' if final_suspended > initial_suspended or suspend_detected else 'FAIL'}")
        log(f"  [3] 任务 B 完成:    {'PASS' if b_output else 'FAIL'}")
        log(f"  [4] 恢复命中:       {'PASS' if resume_detected else 'FAIL - 恢复流程未触发或未完成'}")
        log(f"  [5] 恢复输出:       {'PASS' if resume_output else 'FAIL - 未收到恢复任务输出'}")

        total_pass = sum([
            bool(suspend_detected),
            bool(final_suspended > initial_suspended or suspend_detected),
            bool(b_output),
            bool(resume_detected),
            bool(resume_output),
        ])
        log(f"\n  总分: {total_pass}/5")

    log("=== 测试完成 ===")

    ws.close()
    out.close()


if __name__ == "__main__":
    main()
