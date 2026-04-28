"""WebSocket multi-test script for Zulong system"""
import asyncio
import websockets
import json
import time

async def send_and_receive(ws, message, timeout_sec=120):
    """Send a message and wait for L2_OUTPUT response"""
    chat_msg = {
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "source": "test/qoder",
            "payload": {
                "text": message,
                "confidence": 1.0
            }
        }
    }
    await ws.send(json.dumps(chat_msg))
    print(f"\n{'='*60}")
    print(f"[SENT] {message}")
    print(f"{'='*60}")
    
    start = time.time()
    full_text = ""
    stream_chunks = []
    
    try:
        while time.time() - start < timeout_sec:
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=90)
                data = json.loads(response)
                
                msg_type = data.get("type", "")
                
                if msg_type == "SUBSCRIBE":
                    event = data.get("event", {})
                    etype = event.get("type", "")
                    payload = event.get("payload", {})
                    
                    if etype == "L2_OUTPUT_STREAM":
                        text = payload.get("text", "")
                        stream_chunks.append(text)
                    elif etype == "L2_OUTPUT":
                        full_text = payload.get("text", "")
                        break
                    elif etype == "L2_THINKING_STEP":
                        step = payload.get("step", "")
                        print(f"  [THINK] {str(step)[:200]}")
                elif msg_type == "ACK":
                    pass
                    
            except asyncio.TimeoutError:
                print("  [TIMEOUT] No response in 90s")
                break
                
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    elapsed = time.time() - start
    
    # Write response to file to avoid encoding issues
    result = {
        "message": message,
        "response": full_text,
        "response_length": len(full_text),
        "stream_chunks": len(stream_chunks),
        "elapsed_seconds": round(elapsed, 1),
        "success": bool(full_text)
    }
    
    print(f"  [RESULT] length={len(full_text)}, time={elapsed:.1f}s, stream_chunks={len(stream_chunks)}")
    
    return result

async def run_tests():
    url = "ws://localhost:5555/eventbus"
    print(f"Connecting to {url}...")
    
    async with websockets.connect(url, ping_interval=None) as ws:
        # Subscribe
        subscribe_msg = {
            "type": "SUBSCRIBE",
            "event_types": ["L2_OUTPUT", "L2_OUTPUT_STREAM", "L2_THINKING_STEP"]
        }
        await ws.send(json.dumps(subscribe_msg))
        print("Subscribed to events\n")
        
        results = []
        
        # Test 1: Basic chat
        r = await send_and_receive(ws, "你好，今天天气怎么样？")
        results.append(r)
        
        await asyncio.sleep(2)
        
        # Test 2: Complex task - ask to create a plan
        r = await send_and_receive(ws, "帮我制定一个学习Python的计划，包括基础语法、数据结构和项目实践三个阶段")
        results.append(r)
        
        await asyncio.sleep(2)
        
        # Test 3: Memory recall
        r = await send_and_receive(ws, "你还记得我们之前聊过什么话题吗？")
        results.append(r)
        
    # Write results to file
    with open("_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"ALL TESTS COMPLETE - {len(results)} tests")
    print(f"Results saved to _test_results.json")
    for i, r in enumerate(results):
        status = "PASS" if r["success"] else "FAIL"
        print(f"  Test {i+1}: [{status}] {r['message'][:30]}... -> {r['response_length']} chars in {r['elapsed_seconds']}s")
    print(f"{'='*60}")

asyncio.run(run_tests())
