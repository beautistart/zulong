"""
FC 架构下复杂任务执行 + 中断恢复测试

测试流程:
1. 通过 EventBus WebSocket 发送复杂任务
2. 监控 L2_OUTPUT / MemoryGraph 事件
3. 中途发送新消息模拟"继续"
4. 检查 MemoryGraph 中是否出现重复节点
"""

import asyncio
import websockets
import json
import time
import sys
import os

# 确保可以导入 zulong 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WS_URI = "ws://127.0.0.1:5555/eventbus"

# 收集到的事件
collected_events = []
graph_events = []


async def send_user_input(ws, text: str):
    """通过 EventBus 发送用户输入"""
    event = {
        "type": "PUBLISH",
        "event": {
            "type": "USER_SPEECH",
            "priority": "NORMAL",
            "source": "TestClient",
            "payload": {
                "text": text,
                "confidence": 1.0,
                "timestamp": time.time()
            }
        }
    }
    await ws.send(json.dumps(event))
    print(f"\n{'='*60}")
    print(f"[SENT] {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"{'='*60}")


async def listen_events(ws, duration: float, label: str = ""):
    """监听事件指定时间"""
    print(f"\n[LISTEN] 开始监听事件 ({label}, {duration}s)...")
    start = time.time()
    local_events = []
    
    try:
        while time.time() - start < duration:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)
                event_type = data.get("type", "?")
                
                if event_type == "SUBSCRIBE":
                    # 这是从服务端转发的本地事件
                    inner = data.get("event", {})
                    inner_type = inner.get("type", "?")
                    payload = inner.get("payload", {})
                    
                    collected_events.append({
                        "type": inner_type,
                        "payload": payload,
                        "time": time.time(),
                        "label": label,
                    })
                    local_events.append(inner_type)
                    
                    if inner_type == "MEMORY_GRAPH_UPDATED":
                        graph_events.append(payload)
                        print(f"  [GRAPH] MemoryGraph 更新: {json.dumps(payload, ensure_ascii=False)[:120]}")
                    elif inner_type == "L2_OUTPUT":
                        text = payload.get("text", payload.get("message", ""))
                        print(f"  [L2] 输出: {text[:120]}{'...' if len(text) > 120 else ''}")
                    elif inner_type == "L2_OUTPUT_STREAM":
                        chunk = payload.get("chunk", payload.get("text", ""))
                        if chunk:
                            print(chunk, end="", flush=True)
                    elif inner_type == "L2_THINKING_STEP":
                        step = payload.get("step", payload.get("text", ""))
                        print(f"  [THINK] {step[:100]}")
                    elif inner_type == "ACTION_SPEAK":
                        text = payload.get("text", "")
                        print(f"  [SPEAK] {text[:120]}")
                    else:
                        print(f"  [{inner_type}] {json.dumps(payload, ensure_ascii=False)[:80]}")
                        
                elif event_type == "ACK":
                    pass  # 忽略确认消息
                else:
                    print(f"  [?] {event_type}: {json.dumps(data, ensure_ascii=False)[:80]}")
                    
            except asyncio.TimeoutError:
                pass
            except websockets.exceptions.ConnectionClosed:
                print("[WARN] WebSocket 连接已断开")
                break
                
    except Exception as e:
        print(f"[ERROR] 监听异常: {e}")
    
    print(f"\n[LISTEN] {label} 结束，收到 {len(local_events)} 个事件: {local_events}")
    return local_events


async def run_test():
    """执行完整测试流程"""
    print("="*60)
    print("  FC 架构: 复杂任务执行 + 恢复测试")
    print("="*60)
    
    try:
        ws = await websockets.connect(WS_URI, ping_interval=None, ping_timeout=None)
    except Exception as e:
        print(f"[FATAL] 无法连接 WebSocket: {e}")
        print("请确保祖龙系统已启动 (python zulong/bootstrap.py)")
        return
    
    print("[OK] WebSocket 连接成功")
    
    try:
        # ── Phase 1: 发送复杂任务 ──
        complex_task = (
            "请帮我设计一个智能家居控制系统，包含以下内容：\n"
            "1. 系统架构设计（前端、后端、嵌入式）\n"
            "2. 核心功能模块列表（至少5个）\n"
            "3. 技术选型建议\n"
            "4. 数据库表设计（至少3张表）\n"
            "5. API 接口设计（至少5个接口）"
        )
        
        await send_user_input(ws, complex_task)
        
        # 等待 L2 处理并收集事件 (30秒)
        phase1_events = await listen_events(ws, 30.0, "Phase1-复杂任务执行")
        
        # ── Phase 2: 记录当前图谱节点状态 ──
        print(f"\n{'='*60}")
        print(f"  Phase 2: 记录图谱节点快照")
        print(f"{'='*60}")
        print(f"  Graph events so far: {len(graph_events)}")
        for i, ge in enumerate(graph_events):
            print(f"    [{i}] {json.dumps(ge, ensure_ascii=False)[:100]}")
        
        phase1_graph_count = len(graph_events)
        
        # ── Phase 3: 发送"继续"指令 ──
        print(f"\n{'='*60}")
        print(f"  Phase 3: 发送恢复/继续指令")
        print(f"{'='*60}")
        
        await send_user_input(ws, "继续上面的任务，接着完成数据库表设计和API接口设计部分")
        
        # 等待处理 (30秒)
        phase3_events = await listen_events(ws, 30.0, "Phase3-继续任务")
        
        phase3_graph_count = len(graph_events) - phase1_graph_count
        
        # ── Phase 4: 分析结果 ──
        print(f"\n{'='*60}")
        print(f"  测试结果分析")
        print(f"{'='*60}")
        
        print(f"\n  Phase 1 事件数: {len(phase1_events)}")
        print(f"  Phase 3 事件数: {len(phase3_events)}")
        print(f"  Graph 事件总数: {len(graph_events)}")
        print(f"  Phase 1 图谱事件: {phase1_graph_count}")
        print(f"  Phase 3 图谱事件: {phase3_graph_count}")
        
        # 检查是否有重复的 session/round 创建
        session_creates = []
        round_creates = []
        for ge in graph_events:
            node_type = ge.get("node_type", "")
            action = ge.get("action", "")
            node_id = ge.get("node_id", "")
            if "session" in str(node_type).lower() or "session" in str(node_id).lower():
                session_creates.append(ge)
            if "round" in str(node_type).lower() or "round" in str(node_id).lower():
                round_creates.append(ge)
        
        print(f"\n  Session 节点创建: {len(session_creates)}")
        for sc in session_creates:
            print(f"    {json.dumps(sc, ensure_ascii=False)[:100]}")
        
        print(f"  Round 节点创建: {len(round_creates)}")
        for rc in round_creates:
            print(f"    {json.dumps(rc, ensure_ascii=False)[:100]}")
        
        # 检查所有 L2 输出
        l2_outputs = [e for e in collected_events if e["type"] in ("L2_OUTPUT", "ACTION_SPEAK")]
        print(f"\n  L2 输出/语音总数: {len(l2_outputs)}")
        for lo in l2_outputs:
            text = lo["payload"].get("text", lo["payload"].get("message", ""))
            label = lo["label"]
            print(f"    [{label}] {text[:100]}...")
        
        # 判定
        print(f"\n{'='*60}")
        if len(phase1_events) > 0 and len(phase3_events) > 0:
            print("  [PASS] 两个阶段都收到了事件响应")
        else:
            print("  [WARN] 某个阶段没有收到事件响应")
        
        if len(l2_outputs) > 0:
            print("  [PASS] 系统产生了 L2 输出")
        else:
            print("  [WARN] 未检测到 L2 输出")
        
        print(f"{'='*60}")
        
    except Exception as e:
        import traceback
        print(f"[ERROR] 测试异常: {e}")
        traceback.print_exc()
    finally:
        await ws.close()
        print("[DONE] WebSocket 已关闭")


if __name__ == "__main__":
    asyncio.run(run_test())
