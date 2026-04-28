"""Circuit Breaker 集成测试 - 验证 FC Loop 收敛修复"""
import websocket
import json
import time
import sys

WS_URL = "ws://localhost:5555/eventbus"

def run_test(test_name, message, timeout=180):
    """运行单个测试用例"""
    print(f"\n{'='*60}")
    print(f"  TEST: {test_name}")
    print(f"  INPUT: {message}")
    print(f"  TIMEOUT: {timeout}s")
    print(f"{'='*60}")
    
    ws = websocket.create_connection(WS_URL, timeout=10)
    
    # 订阅事件
    for event_type in ["L2_OUTPUT", "L2_OUTPUT_STREAM", "L2_THINKING_STEP"]:
        ws.send(json.dumps({
            "type": "SUBSCRIBE",
            "address": event_type
        }))
    time.sleep(0.3)
    
    # 发送消息
    ws.send(json.dumps({
        "type": "PUBLISH",
        "address": "USER_TEXT",
        "body": {"text": message, "voice_mode": "TEXT_ONLY"}
    }))
    print(f"[{time.strftime('%H:%M:%S')}] Message sent")
    
    start = time.time()
    response_text = ""
    thinking_steps = 0
    stream_chunks = 0
    cb_events = []  # Circuit Breaker 相关事件
    
    ws.settimeout(5.0)
    
    while time.time() - start < timeout:
        try:
            raw = ws.recv()
            data = json.loads(raw)
            
            if data.get("type") == "MESSAGE":
                addr = data.get("address", "")
                body = data.get("body", {})
                
                if addr == "L2_OUTPUT":
                    response_text = body.get("text", "")
                    elapsed = time.time() - start
                    print(f"[{time.strftime('%H:%M:%S')}] L2_OUTPUT received ({elapsed:.1f}s)")
                    print(f"  Length: {len(response_text)} chars")
                    print(f"  Preview: {response_text[:200]}")
                    break
                    
                elif addr == "L2_THINKING_STEP":
                    thinking_steps += 1
                    step_type = body.get("step_type", "")
                    elapsed = time.time() - start
                    print(f"[{time.strftime('%H:%M:%S')}] THINKING_STEP #{thinking_steps} ({elapsed:.1f}s): {step_type}")
                    
                elif addr == "L2_OUTPUT_STREAM":
                    stream_chunks += 1
                    
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as e:
            print(f"[ERROR] {e}")
            break
    
    ws.close()
    elapsed = time.time() - start
    
    # 结果判定
    passed = len(response_text) > 0
    status = "PASS" if passed else "FAIL"
    
    print(f"\n--- RESULT: {status} ---")
    print(f"  Response: {len(response_text)} chars")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Thinking steps: {thinking_steps}")
    print(f"  Stream chunks: {stream_chunks}")
    
    return {
        "test": test_name,
        "status": status,
        "response_len": len(response_text),
        "elapsed": round(elapsed, 1),
        "thinking_steps": thinking_steps,
        "stream_chunks": stream_chunks,
        "response_preview": response_text[:300] if response_text else "",
    }


def main():
    print("Circuit Breaker Integration Test")
    print(f"Target: {WS_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: 基础闲聊 (不触发工具调用，CB 保持 GREEN)
    r1 = run_test(
        "基础闲聊 (回归)",
        "你好，今天心情怎么样？",
        timeout=60
    )
    results.append(r1)
    
    # 等待系统冷却
    print("\n--- Waiting 5s before next test ---")
    time.sleep(5)
    
    # Test 2: 复杂任务 (之前超时，现在应由 CB 强制收敛)
    r2 = run_test(
        "复杂任务 (CB 收敛)",
        "帮我制定一个Python学习计划",
        timeout=180
    )
    results.append(r2)
    
    # 等待系统冷却
    print("\n--- Waiting 5s before next test ---")
    time.sleep(5)
    
    # Test 3: 记忆回忆 (之前超时，现在应由 CB 强制收敛)
    r3 = run_test(
        "记忆回忆 (CB 收敛)",
        "我们之前聊过什么话题",
        timeout=180
    )
    results.append(r3)
    
    # 汇总
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  [{r['status']}] {r['test']}: {r['response_len']} chars, {r['elapsed']}s, {r['thinking_steps']} steps")
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    print(f"\n  Total: {passed}/{len(results)} passed")
    
    # 保存结果
    with open("_test_cb_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results saved to _test_cb_results.json")


if __name__ == "__main__":
    main()
