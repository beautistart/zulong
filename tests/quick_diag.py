"""快速诊断脚本：发送一条消息，打印所有收到的 WebSocket 事件"""
import asyncio, json, sys, time
sys.path.insert(0, ".")
try:
    import websockets
except ImportError:
    print("ERROR: pip install websockets"); sys.exit(1)

WS_URL = "ws://localhost:8080/ws"
TIMEOUT = 90  # 总超时

async def main():
    print(f"[DIAG] 连接 {WS_URL} ...")
    ws = await websockets.connect(WS_URL, max_size=10*1024*1024)

    # 读 WELCOME + 可能的快照
    for _ in range(3):
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            d = json.loads(raw)
            t = d.get("type","?")
            if t == "MEMORY_GRAPH_UPDATE":
                print(f"[DIAG] <- {t}: {len(d.get('nodes',[]))} 节点, {len(d.get('edges',[]))} 边")
            else:
                print(f"[DIAG] <- {t}: {json.dumps(d, ensure_ascii=False)[:200]}")
        except asyncio.TimeoutError:
            break

    # 发送简单消息
    msg = {
        "type": "CHAT_MESSAGE",
        "text": "你好，请回复一句话测试。",
        "session_id": "diag-001",
        "request_id": "diag-req-001",
    }
    print(f"\n[DIAG] -> 发送: {msg['text']}")
    await ws.send(json.dumps(msg))

    # 收集所有响应
    start = time.time()
    responses = []
    while time.time() - start < TIMEOUT:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            d = json.loads(raw)
            t = d.get("type", "?")
            elapsed = time.time() - start
            
            # 打印关键信息
            if t == "STREAMING_RESPONSE":
                chunk = d.get("chunk", "")
                text = d.get("text", "")
                print(f"  [{elapsed:5.1f}s] STREAMING: chunk='{chunk[:50]}' total_len={len(text)}")
                responses.append(d)
            elif t == "CHAT_RESPONSE":
                text = d.get("text", "")
                print(f"  [{elapsed:5.1f}s] CHAT_RESPONSE: '{text[:200]}'")
                responses.append(d)
                print("\n[DIAG] ✅ 收到完成响应!")
                break
            elif t == "THINKING_STEP":
                desc = d.get("description", "")
                print(f"  [{elapsed:5.1f}s] THINKING: step={d.get('step_number')} {desc[:80]}")
            elif t == "MEMORY_GRAPH_UPDATE":
                n = len(d.get("nodes", []))
                e = len(d.get("edges", []))
                print(f"  [{elapsed:5.1f}s] MEMORY_GRAPH_UPDATE: {n} 节点, {e} 边")
            elif t == "RECOVERY_TASKS":
                tasks = d.get("tasks", [])
                print(f"  [{elapsed:5.1f}s] RECOVERY_TASKS: {len(tasks)} 个可恢复任务")
            else:
                preview = json.dumps(d, ensure_ascii=False)[:200]
                print(f"  [{elapsed:5.1f}s] {t}: {preview}")
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            print(f"  [{elapsed:5.1f}s] (10秒无消息)")
            if elapsed > 60:
                print("[DIAG] ❌ 超时，无完成响应")
                break

    await ws.close()
    
    if not responses:
        print("\n[DIAG] ❌ 未收到任何响应数据！问题可能在:")
        print("  1. L1-B Gatekeeper 未处理 USER_TEXT 事件")
        print("  2. L2 推理引擎未启动或阻塞")
        print("  3. L2_OUTPUT 事件未转发回 Bridge")
    else:
        print(f"\n[DIAG] 收到 {len(responses)} 条响应")

if __name__ == "__main__":
    asyncio.run(main())
