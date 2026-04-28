"""
测试动态注意力机制 - 通过 WebSocket 发送复杂任务
"""
import websocket
import json
import time
import sys
import os

WS_URL = "ws://localhost:5555/eventbus"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "test_attention_output.log")

TASK_TEXT = "设计并实现一个学生成绩管理系统，要求包含：1）学生信息的增删改查 2）成绩录入和统计 3）按科目排名 4）导出成绩报告为文本文件。请用Python实现完整代码。"

def main():
    out = open(OUTPUT_FILE, "w", encoding="utf-8")
    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        out.write(line + "\n")
        out.flush()
        print(line, flush=True)

    log(f"=== 动态注意力机制测试 ===")
    log(f"任务: {TASK_TEXT}")
    log(f"连接: {WS_URL}")
    log("")

    try:
        ws = websocket.create_connection(WS_URL, timeout=10)
        log("WebSocket 已连接")
    except Exception as e:
        log(f"连接失败: {e}")
        out.close()
        return

    # 订阅事件
    subscribe_msg = json.dumps({
        "type": "SUBSCRIBE",
        "client_name": "attention_test",
        "event_types": ["L2_OUTPUT", "L2_THINKING_STEP", "L2_OUTPUT_STREAM"]
    })
    ws.send(subscribe_msg)
    log("已发送订阅请求")

    # 发送任务
    task_msg = json.dumps({
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "source": "attention_test",
            "payload": {
                "text": TASK_TEXT,
                "session_id": "attn_test_001",
                "request_id": "attn_req_001",
                "confidence": 1.0
            }
        }
    })
    ws.send(task_msg)
    log("已发送任务")
    log("")

    # 监听事件
    event_count = 0
    thinking_steps = 0
    tool_calls = 0
    node_updates = 0
    attention_modes = set()
    start_time = time.time()
    timeout = 600  # 10 分钟超时

    ws.settimeout(5.0)

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            log(f"\n超时 ({timeout}s)，停止监听")
            break

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
                    tool_name = payload.get("tool_name", "")
                    node_id = payload.get("node_id", "")
                    attn_mode = payload.get("attention_mode", "")
                    
                    if attn_mode:
                        attention_modes.add(attn_mode)

                    if step_type == "tool_call":
                        tool_calls += 1
                        log(f"[TOOL #{tool_calls}] {tool_name}({json.dumps(payload.get('tool_args',{}), ensure_ascii=False)[:80]})")
                    elif step_type == "node_update":
                        node_updates += 1
                        status = payload.get("status", "")
                        label = payload.get("label", "")
                        log(f"[NODE] {node_id} [{status}] {label}")
                    elif step_type == "mode_change":
                        old_m = payload.get("old_mode", "")
                        new_m = payload.get("new_mode", "")
                        log(f"[ATTENTION] {old_m} -> {new_m}")
                    elif step_type == "graph_snapshot":
                        nodes = payload.get("nodes", [])
                        log(f"[GRAPH] snapshot: {len(nodes)} nodes")
                    else:
                        log(f"[STEP] {step_type}: {str(payload)[:100]}")

                elif inner_type == "L2_OUTPUT":
                    text = payload.get("text", "")
                    ws_dir = payload.get("workspace_dir", "")
                    log(f"\n{'='*60}")
                    log(f"[RESULT] L2_OUTPUT received!")
                    log(f"  Length: {len(text)} chars")
                    log(f"  Workspace: {ws_dir}")
                    log(f"  Preview: {text[:200]}...")
                    log(f"{'='*60}")
                    # 收到最终结果，结束
                    break
                else:
                    log(f"[EVENT] {inner_type}: {str(payload)[:80]}")

            elif evt_type == "ACK":
                log(f"[ACK] {data.get('event_type', '')}")

        except websocket.WebSocketTimeoutException:
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                log(f"  ... 等待中 ({int(elapsed)}s, events={event_count})")
            continue
        except Exception as e:
            log(f"[ERROR] {e}")
            break

    elapsed = time.time() - start_time
    log(f"\n=== 测试统计 ===")
    log(f"  总耗时: {elapsed:.1f}s")
    log(f"  总事件数: {event_count}")
    log(f"  思考步骤: {thinking_steps}")
    log(f"  工具调用: {tool_calls}")
    log(f"  节点更新: {node_updates}")
    log(f"  注意力模式: {attention_modes or 'N/A'}")
    log(f"=== 测试完成 ===")

    ws.close()
    out.close()

if __name__ == "__main__":
    main()
