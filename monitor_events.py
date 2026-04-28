"""监控 Zulong EventBus 事件，特别关注 start_task_plan 工具调用"""
import asyncio
import websockets
import json
import time
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


async def monitor():
    uri = "ws://localhost:5555/eventbus"
    print(f"[MONITOR] Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as ws:
            print("[MONITOR] Connected! Listening for events...")
            print("=" * 70)
            
            start = time.time()
            event_count = 0
            tool_calls = []
            
            while True:
                elapsed = time.time() - start
                if elapsed > 180:  # 3分钟超时
                    print(f"\n[TIMEOUT] 3 min elapsed, stopping")
                    break
                    
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(raw)
                    msg_type = data.get("type", "")
                    
                    if msg_type == "SUBSCRIBE":
                        evt = data.get("event", {})
                        evt_type = evt.get("type", "?")
                        payload = evt.get("payload", {})
                        event_count += 1
                        
                        if evt_type == "L2_THINKING_STEP":
                            step_type = payload.get("step_type", "")
                            step_data = payload.get("data", {})
                            
                            if "tool_call" in step_type:
                                tool_name = step_data.get("tool_name", "?")
                                args = step_data.get("arguments", "")[:200]
                                tool_calls.append(tool_name)
                                print(f"  [TOOL_CALL] {tool_name}")
                                print(f"    args: {args}")
                            elif "tool_result" in step_type:
                                tool_name = step_data.get("tool_name", "?")
                                success = step_data.get("success", "?")
                                summary = step_data.get("summary", "")[:300]
                                print(f"  [TOOL_RESULT] {tool_name} success={success}")
                                if summary:
                                    print(f"    summary: {summary}")
                            elif "pipeline" in step_type:
                                print(f"  [PIPELINE] {step_type}")
                                if step_data:
                                    print(f"    data_keys: {list(step_data.keys())[:10]}")
                            elif "iteration_start" in step_type:
                                iteration = payload.get("iteration", "?")
                                print(f"  [ITER] iteration={iteration}")
                            elif "cb_evaluation" in step_type:
                                state = step_data.get("state", "?")
                                reason = step_data.get("reason", "")[:100]
                                print(f"  [CB] state={state} reason={reason}")
                            elif "inference_complete" in step_type:
                                iters = step_data.get("total_iterations", "?")
                                print(f"  [DONE] total_iterations={iters}")
                            else:
                                print(f"  [THINK] {step_type}: {str(step_data)[:100]}")
                                
                        elif evt_type == "L2_OUTPUT_STREAM":
                            # 只报告第一个和每50个chunk
                            if event_count % 50 == 1:
                                chunk = str(payload.get("chunk", ""))[:80]
                                print(f"  [STREAM] chunk #{event_count}: {chunk}")
                                
                        elif evt_type == "L2_OUTPUT":
                            text = str(payload.get("text", ""))[:300]
                            print(f"  [L2_OUTPUT] {text}")
                            
                        elif evt_type == "MEMORY_GRAPH_UPDATED":
                            print(f"  [MEM_UPD] MemoryGraph updated")
                            
                        elif evt_type == "ACTION_SPEAK":
                            text = str(payload.get("text", ""))[:200]
                            print(f"  [SPEAK] {text}")
                            
                        elif evt_type == "SYSTEM_STATUS":
                            status = str(payload.get("status", ""))
                            print(f"  [STATUS] {status}")
                            
                        else:
                            print(f"  [{evt_type}] {str(payload)[:100]}")
                    
                    elif msg_type == "ACK":
                        pass  # skip ACKs
                        
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("[WARN] Connection closed")
                    break
            
            print("\n" + "=" * 70)
            print(f"[SUMMARY] Total events: {event_count}")
            print(f"[SUMMARY] Tool calls: {tool_calls}")
            if "start_task_plan" in tool_calls:
                print("[SUMMARY] start_task_plan WAS CALLED!")
            else:
                print("[SUMMARY] start_task_plan was NOT called")
                
    except ConnectionRefusedError:
        print("[ERROR] Connection refused - Zulong not running?")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    asyncio.run(monitor())
