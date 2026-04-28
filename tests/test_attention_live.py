# -*- coding: utf-8 -*-
"""
祖龙注意力系统实时测试脚本

测试维度:
1. 初始化记忆 - 系统启动后的记忆状态
2. 闲聊记忆 - 简单对话中的短期记忆
3. 复杂任务记忆 - 任务执行中的注意力窗口管理
4. 注意力自主调整 - 模式切换 (GLOBAL → FOCUS → SINGLE_CHAIN)

连接: ws://localhost:5555/eventbus
协议: EventBus PUBLISH/SUBSCRIBE
"""
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

WS_URI = "ws://localhost:5555/eventbus"
COLLECT_TIMEOUT = 90
IDLE_TIMEOUT = 15


def log(tag, msg):
    ts = time.strftime("%H:%M:%S")
    try:
        print(f"[{ts}][{tag}] {msg}")
    except UnicodeEncodeError:
        print(f"[{ts}][{tag}] {msg.encode('ascii', errors='replace').decode('ascii')}")


async def send_and_collect(ws, text, label="", max_seconds=COLLECT_TIMEOUT, idle_timeout=IDLE_TIMEOUT):
    """发送消息并收集所有响应事件"""
    log("TEST", f"{'='*60}")
    log("TEST", f"{label}")
    log("SEND", f"{text[:120]}")
    log("TEST", f"{'='*60}")

    msg = json.dumps({
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "payload": {"text": text},
            "source": "AttentionTest",
            "priority": "NORMAL",
        }
    })
    await ws.send(msg)

    events = []
    stream_parts = []
    start = time.time()
    last_event_time = time.time()

    while True:
        elapsed = time.time() - start
        idle = time.time() - last_event_time

        if elapsed > max_seconds:
            log("TIMEOUT", f"{max_seconds}s reached")
            break
        if idle > idle_timeout and len(events) > 0:
            log("IDLE", f"{idle_timeout}s no events, done")
            break

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=3)
            data = json.loads(raw)
            last_event_time = time.time()
            msg_type = data.get("type", "")

            if msg_type == "ACK":
                log("ACK", data.get("event_type", "?"))
                continue

            if msg_type == "SUBSCRIBE":
                evt = data.get("event", {})
                evt_type = evt.get("type", "UNKNOWN")
                payload = evt.get("payload", {})
                events.append(evt)

                if evt_type == "ACTION_SPEAK":
                    log("SPEAK", payload.get("text", "")[:200])
                elif evt_type == "L2_OUTPUT":
                    log("L2_OUT", payload.get("text", "")[:200])
                elif evt_type == "L2_OUTPUT_STREAM":
                    chunk = payload.get("chunk", payload.get("text", ""))
                    if chunk:
                        stream_parts.append(str(chunk))
                        if len(stream_parts) % 30 == 1:
                            log("STREAM", f"({len(stream_parts)} chunks received)")
                elif evt_type == "L2_THINKING_STEP":
                    log("THINK", str(payload.get("step", ""))[:100])
                elif evt_type == "MEMORY_GRAPH_UPDATED":
                    log("MEM", "MemoryGraph updated")
                elif evt_type == "ATTENTION_MODE_CHANGED":
                    log("ATTN", f"Mode: {payload}")
                elif evt_type == "TASK_GRAPH_EVENT":
                    log("GRAPH", f"TaskGraph: {str(payload)[:120]}")
                elif evt_type == "SYSTEM_STATUS":
                    log("STATUS", payload.get("status", ""))
                else:
                    log(evt_type[:12], json.dumps(payload, ensure_ascii=False)[:120])
                continue

            log("OTHER", f"{msg_type}: {json.dumps(data, ensure_ascii=False)[:150]}")

        except asyncio.TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosed:
            log("WARN", "Connection closed")
            break

    # 打印流式完整文本
    if stream_parts:
        full = "".join(stream_parts)
        log("FULL", f"({len(stream_parts)} chunks, {len(full)} chars)")
        display = full[:600] + ("..." if len(full) > 600 else "")
        log("FULL", display)

    log("RESULT", f"Events: {len(events)}, Duration: {time.time()-start:.1f}s")
    return events


async def run_all_tests():
    log("START", "=" * 60)
    log("START", "祖龙注意力系统实时测试")
    log("START", f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("START", "=" * 60)

    try:
        ws = await asyncio.wait_for(
            websockets.connect(WS_URI, max_size=2**22, ping_interval=None, ping_timeout=None),
            timeout=10,
        )
    except Exception as e:
        log("ERROR", f"无法连接到祖龙 EventBus ({WS_URI}): {e}")
        log("ERROR", "请确认祖龙系统已启动 (python zulong/bootstrap.py)")
        return

    log("CONN", f"已连接到 {WS_URI}")

    # 订阅所有感兴趣的事件
    subscribe_msg = {
        "type": "SUBSCRIBE",
        "event_types": [
            "L2_OUTPUT", "L2_OUTPUT_STREAM", "L2_THINKING_STEP",
            "ACTION_SPEAK", "MEMORY_GRAPH_UPDATED",
            "ATTENTION_MODE_CHANGED", "TASK_GRAPH_EVENT",
            "SYSTEM_STATUS",
        ]
    }
    await ws.send(json.dumps(subscribe_msg))
    log("SUB", "Subscribed to events")
    await asyncio.sleep(1)

    # ========== Phase 1: 初始化记忆测试 ==========
    log("PHASE", ">>> Phase 1: 初始化记忆测试 <<<")
    await send_and_collect(
        ws,
        "你好，你现在的状态是什么？你记得我们之前聊过什么吗？",
        label="Test 1.1: 初始状态探测 (检查系统初始记忆加载)",
        max_seconds=45,
    )
    await asyncio.sleep(3)

    # ========== Phase 2: 闲聊记忆测试 ==========
    log("PHASE", ">>> Phase 2: 闲聊记忆测试 <<<")
    await send_and_collect(
        ws,
        "我叫李明，今年28岁，是一名全栈工程师，最近在研究大模型的注意力机制。请记住这些。",
        label="Test 2.1: 信息注入 (短期记忆写入)",
        max_seconds=30,
    )
    await asyncio.sleep(3)

    await send_and_collect(
        ws,
        "我刚才跟你说过我的名字和职业，你还记得吗？另外，你觉得注意力机制最关键的设计是什么？",
        label="Test 2.2: 记忆回忆 + 话题延续 (验证短期记忆保持和话题关联)",
        max_seconds=45,
    )
    await asyncio.sleep(3)

    # ========== Phase 3: 复杂任务记忆测试 ==========
    log("PHASE", ">>> Phase 3: 复杂任务记忆测试 <<<")
    await send_and_collect(
        ws,
        (
            "请帮我设计一个内存缓存系统，要求如下：\n"
            "1. 支持 LRU 淘汰策略\n"
            "2. 支持 TTL 过期\n"
            "3. 线程安全\n"
            "4. 提供 get/set/delete 接口\n"
            "5. 用 Python 实现核心代码"
        ),
        label="Test 3.1: 复杂任务触发 (验证TaskGraph创建 + 注意力模式切换)",
        max_seconds=90,
    )
    await asyncio.sleep(5)

    # 任务执行过程中追加需求
    await send_and_collect(
        ws,
        "刚才设计的缓存系统，能不能再加一个统计功能，统计命中率和未命中率？",
        label="Test 3.2: 任务追加 (验证任务上下文保持 + 图谱追加节点)",
        max_seconds=60,
    )
    await asyncio.sleep(3)

    # ========== Phase 4: 注意力自主调整测试 ==========
    log("PHASE", ">>> Phase 4: 注意力自主调整测试 <<<")
    await send_and_collect(
        ws,
        "回顾一下我们今天的对话，你从初始状态开始，经历了闲聊、任务执行，你的注意力是怎么切换的？",
        label="Test 4.1: 注意力自省 (验证模型能感知自己的注意力状态变化)",
        max_seconds=45,
    )
    await asyncio.sleep(3)

    # 测试突然话题切换（应触发新 session）
    await send_and_collect(
        ws,
        "突然想问一下，明天北京的天气怎么样？",
        label="Test 4.2: 话题突变 (验证 ensure_session 是否检测到话题边界)",
        max_seconds=30,
    )
    await asyncio.sleep(2)

    # 然后切回原话题
    await send_and_collect(
        ws,
        "算了天气的事不重要。回到之前的缓存系统，命中率统计那部分代码写好了吗？",
        label="Test 4.3: 话题回跳 (验证回到原 session 的能力)",
        max_seconds=45,
    )

    await ws.close()
    log("DONE", "=" * 60)
    log("DONE", "所有测试完成! 请检查祖龙终端窗口的以下日志:")
    log("DONE", "  1. [DialogueAdapter] 对话节点创建 / session 管理")
    log("DONE", "  2. [AttentionWindow] 模式切换日志 (GLOBAL/FOCUS/SINGLE_CHAIN)")
    log("DONE", "  3. [Agent] 已恢复 / 已注入 记忆相关日志")
    log("DONE", "  4. [MemoryGraph] 焦点更新 / 节点创建")
    log("DONE", "=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
