"""
测试 Pipeline 交互改进 - E2E
测试内容：
1. 复杂任务执行 + ask_user 工具调用检测
2. 任务执行中途发送消息（Supervisor 注入）
3. 结构化输出验证（summary/architecture/file_list/usage_guide 或 fallback 等效项）
"""
import websocket
import json
import time
import sys
import os

WS_URL = "ws://localhost:5555/eventbus"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "test_interaction_output.log")

# 故意模糊的任务，触发 ask_user
TASK_TEXT = "帮我写一个系统"

# 中途插话消息（在 ask_user 回答后 + 60s 后发送）
MIDWAY_MSG = "顺便加上一个用户登录功能"

# 插话延迟：ask_user回答后等待的秒数（确保Agent已开始执行）
MIDWAY_AFTER_ANSWER_DELAY = 60


def send_user_text(ws, text, session_id="interaction_test", request_id="int_req_001"):
    """发送用户文本消息"""
    msg = json.dumps({
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "source": "interaction_test",
            "payload": {
                "text": text,
                "session_id": session_id,
                "request_id": request_id,
                "confidence": 1.0,
            }
        }
    })
    ws.send(msg)


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
            pass  # GBK encoding issues on Windows

    log("=== Pipeline 交互改进 E2E 测试 ===")
    log(f"任务: {TASK_TEXT}")
    log(f"中途插话: {MIDWAY_MSG} (ask_user回答后 {MIDWAY_AFTER_ANSWER_DELAY}s)")
    log(f"连接: {WS_URL}")
    log("")

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
        "client_name": "interaction_test",
        "event_types": ["L2_OUTPUT", "L2_THINKING_STEP", "L2_OUTPUT_STREAM"]
    })
    ws.send(subscribe_msg)
    log("已发送订阅请求")

    # 发送模糊任务
    send_user_text(ws, TASK_TEXT)
    log("已发送模糊任务（预期触发 ask_user）")
    log("")

    # 监听事件
    event_count = 0
    thinking_steps = 0
    tool_calls = 0
    ask_user_count = 0
    ask_user_question = ""
    last_answer_time = None  # ask_user 最后一次回答的时间
    supervisor_injected = False
    midway_sent = False
    l2_output_text = ""
    start_time = time.time()
    timeout = 720  # 匹配 AGENT_TOTAL_TIMEOUT=600 + 余量

    ws.settimeout(3.0)

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            log(f"\n超时 ({timeout}s)，停止监听")
            break

        # 中途插话逻辑：在 ask_user 全部回答完、Agent 进入执行阶段后发送
        if (
            not midway_sent
            and last_answer_time is not None
            and (time.time() - last_answer_time) >= MIDWAY_AFTER_ANSWER_DELAY
        ):
            send_user_text(ws, MIDWAY_MSG)
            midway_sent = True
            log(f"\n>>> 已发送中途插话: {MIDWAY_MSG}")
            log("")

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
                    thinking_steps += 1
                    step_type = payload.get("step_type", "")
                    step_data = payload.get("data", {})

                    # 检测 pipeline 事件
                    if step_type.startswith("pipeline."):
                        pipeline_type = step_type.replace("pipeline.", "")

                        if pipeline_type == "ask_user":
                            ask_user_count += 1
                            q = step_data.get("question", "")
                            opts = step_data.get("options", [])
                            ask_user_question = q
                            log(f"\n{'*'*60}")
                            log(f"[ASK_USER #{ask_user_count}] Agent 向用户提问!")
                            log(f"  问题: {q}")
                            if opts:
                                log(f"  选项: {opts}")
                            log(f"{'*'*60}")

                            # 自动回答
                            time.sleep(2)
                            answer = "学生成绩管理系统，用 Python 实现，使用 CSV 文件存储"
                            send_user_text(ws, answer)
                            last_answer_time = time.time()
                            log(f">>> 自动回答: {answer}")
                            log("")

                        elif pipeline_type == "agent_tool_call":
                            tool_calls += 1
                            tool_name = step_data.get("tool", "")
                            args = step_data.get("args", {})
                            log(f"[TOOL #{tool_calls}] {tool_name}({json.dumps(args, ensure_ascii=False)[:80]})")

                        elif pipeline_type == "supervisor_inject":
                            supervisor_injected = True
                            msg = step_data.get("message", "")
                            log(f"\n[SUPERVISOR] 消息已注入: {msg[:120]}")
                            log("")

                        elif pipeline_type == "graph_update":
                            change = step_data.get("change_type", "")
                            log(f"[GRAPH] {change}")

                        elif pipeline_type == "agent_start":
                            log(f"[AGENT] 启动, tools={step_data.get('tool_count', '?')}")

                        elif pipeline_type == "agent_done":
                            log(f"[AGENT] 完成, turns={step_data.get('turns', '?')}, "
                                f"duration={step_data.get('duration', '?')}s")

                        elif pipeline_type == "task_suspended":
                            log(f"[SUSPENDED] 任务已挂起: {step_data}")

                        else:
                            log(f"[PIPELINE] {pipeline_type}: {str(step_data)[:100]}")
                    else:
                        if step_type and "tool_call" not in step_type:
                            log(f"[STEP] {step_type}")

                elif inner_type == "L2_OUTPUT":
                    text = payload.get("text", "")
                    is_ask = payload.get("ask_user", False)
                    is_suspended = payload.get("task_suspended", False)

                    if is_ask:
                        # ask_user 通过 EventBus 到达的事件（可能与 THINKING_STEP 重复）
                        if ask_user_count == 0:
                            ask_user_count += 1
                            ask_user_question = text
                            log(f"\n{'*'*60}")
                            log(f"[ASK_USER via EventBus] {text}")
                            opts = payload.get("options", [])
                            if opts:
                                log(f"  选项: {opts}")
                            log(f"{'*'*60}")

                            # 自动回答
                            time.sleep(2)
                            answer = "学生成绩管理系统，用 Python 实现，使用 CSV 文件存储"
                            send_user_text(ws, answer)
                            last_answer_time = time.time()
                            log(f">>> 自动回答: {answer}")
                            log("")
                    elif is_suspended:
                        log(f"\n[SUSPENDED] {text}")
                    else:
                        l2_output_text = text
                        ws_dir = payload.get("workspace_dir", "")
                        log(f"\n{'='*60}")
                        log(f"[RESULT] L2_OUTPUT received!")
                        log(f"  Length: {len(text)} chars")
                        if ws_dir:
                            log(f"  Workspace: {ws_dir}")

                        # 检查结构化输出（兼容 submit_final_answer 和 fallback 两种格式）
                        has_summary = "## 项目概述" in text
                        has_arch = "## 架构说明" in text
                        has_files = "## 文件清单" in text
                        has_guide = "## 使用指南" in text
                        has_graph = "## 任务完成情况" in text
                        has_work = "## 已完成工作" in text

                        log(f"  结构化输出检查:")
                        log(f"    项目概述: {'YES' if has_summary else 'NO'}")
                        log(f"    架构说明: {'YES' if has_arch else 'NO'}")
                        log(f"    文件清单: {'YES' if has_files else 'NO'}")
                        log(f"    使用指南: {'YES' if has_guide else 'NO'}")
                        log(f"    任务图谱: {'YES' if has_graph else 'NO'}")
                        log(f"    已完成工作: {'YES' if has_work else 'NO'}")

                        log(f"  Preview: {text[:400]}...")
                        log(f"{'='*60}")
                        break

                else:
                    log(f"[EVENT] {inner_type}")

            elif evt_type == "ACK":
                pass  # skip ACK noise

        except websocket.WebSocketTimeoutException:
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                log(f"  ... 等待中 ({int(elapsed)}s, events={event_count})")
            continue
        except Exception as e:
            log(f"[ERROR] {e}")
            break

    elapsed = time.time() - start_time

    log(f"\n{'='*60}")
    log(f"=== 测试统计 ===")
    log(f"  总耗时: {elapsed:.1f}s")
    log(f"  总事件数: {event_count}")
    log(f"  思考步骤: {thinking_steps}")
    log(f"  工具调用: {tool_calls}")
    log(f"  ask_user 次数: {ask_user_count}")
    log(f"")
    log(f"=== 新功能验证 ===")
    log(f"  [Issue 3] ask_user 触发: {'PASS (' + str(ask_user_count) + '次)' if ask_user_count > 0 else 'NOT TRIGGERED'}")
    if ask_user_question:
        log(f"    最后一次问题: {ask_user_question[:100]}")
    log(f"  [Issue 1] 中途插话发送: {'YES' if midway_sent else 'NO (ask_user回答后不足{0}s)'.format(MIDWAY_AFTER_ANSWER_DELAY)}")
    log(f"  [Issue 1] Supervisor 注入: {'PASS' if supervisor_injected else 'NOT DETECTED'}")
    log(f"  [Issue 4] 结构化输出:")
    if l2_output_text:
        # 兼容两种输出格式：Agent 主动 submit_final_answer 或 fallback 自动生成
        checks = {
            "项目概述": "## 项目概述" in l2_output_text,
            "架构/已完成": "## 架构说明" in l2_output_text or "## 已完成工作" in l2_output_text,
            "文件清单": "## 文件清单" in l2_output_text,
            "指南/任务图": "## 使用指南" in l2_output_text or "## 任务完成情况" in l2_output_text,
        }
        passed = sum(1 for v in checks.values() if v)
        log(f"    {passed}/4 通过 ({checks})")
    else:
        log(f"    无输出")
    log(f"=== 测试完成 ===")

    ws.close()
    out.close()


if __name__ == "__main__":
    main()
