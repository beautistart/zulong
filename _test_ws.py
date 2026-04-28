"""WebSocket test script for Zulong system"""
import asyncio
import websockets
import json
import time

async def test_chat(message="你好，请简单介绍一下你自己", timeout_sec=120):
    url = "ws://localhost:5555/eventbus"
    print(f"[TEST] Connecting to {url}...")
    
    async with websockets.connect(url) as ws:
        print("[TEST] Connected!")
        
        # Subscribe to response events
        subscribe_msg = {
            "type": "SUBSCRIBE",
            "event_types": ["L2_OUTPUT", "L2_OUTPUT_STREAM", "L2_THINKING_STEP", "ACTION_SPEAK"]
        }
        await ws.send(json.dumps(subscribe_msg))
        print("[TEST] Subscribed to events")
        
        # Send chat message
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
        print(f"[TEST] Sent: {message}")
        print(f"[TEST] Waiting for response (max {timeout_sec}s)...")
        
        # Listen for responses
        start = time.time()
        collected = []
        got_full_response = False
        full_text = ""
        
        try:
            while time.time() - start < timeout_sec:
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=60)
                    data = json.loads(response)
                    
                    msg_type = data.get("type", "")
                    
                    if msg_type == "SUBSCRIBE":
                        event = data.get("event", {})
                        etype = event.get("type", "")
                        payload = event.get("payload", {})
                        
                        if etype == "L2_OUTPUT_STREAM":
                            text = payload.get("text", "")
                            print(text, end="", flush=True)
                            collected.append(text)
                        elif etype == "L2_OUTPUT":
                            full_text = payload.get("text", "")
                            print(f"\n[TEST] === Full L2_OUTPUT ({len(full_text)} chars) ===")
                            print(full_text[:800])
                            if len(full_text) > 800:
                                print(f"... (truncated)")
                            got_full_response = True
                            break
                        elif etype == "L2_THINKING_STEP":
                            step = payload.get("step", "")
                            print(f"\n[THINK] {str(step)[:300]}")
                        elif etype == "ACTION_SPEAK":
                            text = payload.get("text", "")
                            print(f"\n[SPEAK] {text[:300]}")
                    elif msg_type == "ACK":
                        ack_msg = data.get("message", "")
                        print(f"[ACK] {ack_msg}")
                        
                except asyncio.TimeoutError:
                    print("\n[TEST] Timeout waiting for message")
                    break
                    
        except Exception as e:
            print(f"\n[TEST] Error: {e}")
        
        elapsed = time.time() - start
        print(f"\n[TEST] Elapsed: {elapsed:.1f}s")
        print(f"[TEST] Got full response: {got_full_response}")
        if collected:
            print(f"[TEST] Stream chunks: {len(collected)}")
        
        return got_full_response, full_text

if __name__ == "__main__":
    asyncio.run(test_chat())
