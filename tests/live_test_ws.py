"""WebSocket 实时环境测试脚本"""
import asyncio
import websockets
import json
import time
import sys

WS_URL = "ws://localhost:5555/eventbus"
SUBSCRIBE_EVENTS = [
    "L2_OUTPUT", "L2_OUTPUT_STREAM", "L2_THINKING_STEP",
    "ACK", "MEMORY_GRAPH_UPDATED", "ACTION_SPEAK"
]


class ZulongLiveTestClient:
    def __init__(self):
        self.ws = None
        self.events = []
        self._listener_task = None

    async def connect(self):
        self.ws = await asyncio.wait_for(
            websockets.connect(WS_URL, max_size=2**22),
            timeout=10
        )
        print("[OK] WebSocket 连接成功")
        # 订阅
        await self.ws.send(json.dumps({
            "type": "SUBSCRIBE",
            "event_types": SUBSCRIBE_EVENTS
        }))
        self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self):
        try:
            async for raw in self.ws:
                try:
                    data = json.loads(raw)
                    self.events.append(data)
                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass

    async def send_user_text(self, text, request_id=None):
        rid = request_id or f"test_{int(time.time()*1000)}"
        msg = {
            "type": "PUBLISH",
            "event": {
                "type": "USER_TEXT",
                "source": "live_test",
                "payload": {
                    "text": text,
                    "confidence": 1.0,
                    "request_id": rid
                },
                "priority": "NORMAL"
            }
        }
        await self.ws.send(json.dumps(msg))
        print(f"[SENT] '{text}' (request_id={rid})")
        return rid

    async def wait_for_output(self, timeout=120):
        """等待 L2_OUTPUT 事件"""
        start = time.time()
        while time.time() - start < timeout:
            for ev in self.events:
                evt = ev.get("event", {})
                if evt.get("type") == "L2_OUTPUT":
                    text = evt.get("payload", {}).get("text", "")
                    if text:
                        return text
            await asyncio.sleep(0.5)
        return None

    def get_thinking_steps(self):
        steps = []
        for ev in self.events:
            evt = ev.get("event", {})
            if evt.get("type") == "L2_THINKING_STEP":
                step = evt.get("payload", {}).get("step", "")
                tool = evt.get("payload", {}).get("tool_name", "")
                steps.append({"step": step[:100], "tool": tool})
        return steps

    def get_events_summary(self):
        """统计收到的各类事件"""
        summary = {}
        for ev in self.events:
            evt = ev.get("event", {})
            etype = evt.get("type", ev.get("type", "unknown"))
            summary[etype] = summary.get(etype, 0) + 1
        return summary

    def clear_events(self):
        self.events.clear()

    async def close(self):
        if self._listener_task:
            self._listener_task.cancel()
        if self.ws:
            await self.ws.close()


async def test_1_basic_connection():
    """测试 1: 基础连接和简单对话"""
    print("\n" + "="*60)
    print("测试 1: 基础连接 + 简单对话")
    print("="*60)

    client = ZulongLiveTestClient()
    await client.connect()

    # 发送简单消息
    await client.send_user_text("你好，我是测试用户张三")

    # 等待回复
    output = await client.wait_for_output(timeout=90)
    if output:
        print(f"[OK] 收到回复 ({len(output)} 字符):")
        print(f"  前200字: {output[:200]}")
    else:
        print("[WARN] 90秒内未收到 L2_OUTPUT")

    # 事件统计
    summary = client.get_events_summary()
    print(f"[INFO] 事件统计: {json.dumps(summary, ensure_ascii=False)}")

    # 思考步骤
    steps = client.get_thinking_steps()
    if steps:
        print(f"[INFO] 思考步骤数: {len(steps)}")
        for s in steps[:5]:
            print(f"  - tool={s['tool'] or 'N/A'}: {s['step'][:80]}")

    await client.close()
    return bool(output)


async def test_2_memory_retrieval():
    """测试 2: 记忆检索（身份信息存储+检索）"""
    print("\n" + "="*60)
    print("测试 2: 记忆检索 - 身份信息存储")
    print("="*60)

    client = ZulongLiveTestClient()
    await client.connect()

    # 发送包含身份信息的消息（触发 DialogueAdapter 的 IDENTITY 重要度检测）
    await client.send_user_text("帮我记住：我叫张三，我是一名Python后端工程师，常用FastAPI框架")

    output = await client.wait_for_output(timeout=90)
    if output:
        print(f"[OK] 收到回复: {output[:200]}")
    else:
        print("[WARN] 90秒内未收到回复")

    summary = client.get_events_summary()
    print(f"[INFO] 事件统计: {json.dumps(summary, ensure_ascii=False)}")

    steps = client.get_thinking_steps()
    if steps:
        print(f"[INFO] 思考步骤数: {len(steps)}")
        for s in steps[:5]:
            print(f"  - tool={s['tool'] or 'N/A'}: {s['step'][:80]}")

    await client.close()
    return bool(output)


async def test_3_complex_task():
    """测试 3: 复杂任务 - 触发完整管线"""
    print("\n" + "="*60)
    print("测试 3: 复杂任务（触发 TaskGraph + BFS + AttentionWindow + CB）")
    print("="*60)

    client = ZulongLiveTestClient()
    await client.connect()

    # 发送复杂任务
    task_text = (
        "帮我用Python写一个简单的命令行TODO应用，要求：\n"
        "1. 支持添加、删除、列出待办事项\n"
        "2. 数据用JSON文件持久化\n"
        "3. 支持按优先级排序\n"
        "请先分析需求，然后逐步实现"
    )
    await client.send_user_text(task_text)

    # 复杂任务需要更长超时
    output = await client.wait_for_output(timeout=180)
    if output:
        print(f"[OK] 收到回复 ({len(output)} 字符):")
        print(f"  前300字: {output[:300]}")
    else:
        print("[WARN] 180秒内未收到 L2_OUTPUT")

    summary = client.get_events_summary()
    print(f"[INFO] 事件统计: {json.dumps(summary, ensure_ascii=False)}")

    steps = client.get_thinking_steps()
    print(f"[INFO] 思考步骤数: {len(steps)}")
    tool_names = [s["tool"] for s in steps if s["tool"]]
    print(f"[INFO] 使用的工具: {list(set(tool_names))}")
    for s in steps[:10]:
        print(f"  - tool={s['tool'] or 'N/A'}: {s['step'][:80]}")

    await client.close()
    return bool(output)


async def test_4_suspend_resume():
    """测试 4: 任务中断与恢复"""
    print("\n" + "=" * 60)
    print("测试 4: 任务中断与恢复（suspend + resume）")
    print("=" * 60)

    client = ZulongLiveTestClient()
    await client.connect()

    # Step 1: 发送一个复杂任务（会创建 TaskGraph）
    print("\n--- Step 1: 发送复杂任务 ---")
    task_text = (
        "帮我设计一个在线图书管理系统的数据库方案，需要包含：\n"
        "1. 图书信息表\n"
        "2. 用户借阅记录表\n"
        "3. 分类管理表\n"
        "请分析需求并给出详细的ER图设计"
    )
    await client.send_user_text(task_text)

    output1 = await client.wait_for_output(timeout=120)
    if output1:
        print(f"[OK] Step 1 收到回复 ({len(output1)} 字符)")
        print(f"  前200字: {output1[:200]}")
    else:
        print("[WARN] Step 1 120秒内未收到回复")
        await client.close()
        return False

    summary1 = client.get_events_summary()
    print(f"[INFO] Step 1 事件统计: {json.dumps(summary1, ensure_ascii=False)}")

    # 检查是否有 TaskGraph 创建（通过 THINKING_STEP 中的 task_add_node）
    steps1 = client.get_thinking_steps()
    print(f"[INFO] Step 1 思考步骤数: {len(steps1)}")

    # Step 2: 暂停当前任务
    print("\n--- Step 2: 请求暂停任务 ---")
    client.clear_events()
    await asyncio.sleep(2)

    await client.send_user_text("先暂停一下这个图书管理系统的任务，我有个急事要处理")
    output2 = await client.wait_for_output(timeout=90)
    if output2:
        print(f"[OK] Step 2 收到暂停确认 ({len(output2)} 字符):")
        print(f"  前200字: {output2[:200]}")
        # 检查是否包含暂停相关字样
        suspend_keywords = ["暂停", "保存", "挂起", "suspend", "待命", "恢复"]
        found = [kw for kw in suspend_keywords if kw in output2]
        if found:
            print(f"[OK] 回复中包含暂停关键词: {found}")
        else:
            print("[WARN] 回复中未检测到暂停确认关键词")
    else:
        print("[WARN] Step 2 90秒内未收到回复")

    summary2 = client.get_events_summary()
    print(f"[INFO] Step 2 事件统计: {json.dumps(summary2, ensure_ascii=False)}")

    # Step 3: 恢复任务
    print("\n--- Step 3: 请求恢复任务 ---")
    client.clear_events()
    await asyncio.sleep(3)

    await client.send_user_text("继续之前的图书管理系统数据库设计任务")
    output3 = await client.wait_for_output(timeout=120)
    if output3:
        print(f"[OK] Step 3 收到恢复回复 ({len(output3)} 字符):")
        print(f"  前200字: {output3[:200]}")
        # 检查是否能回忆之前的任务上下文
        context_keywords = ["图书", "数据库", "ER", "借阅", "管理"]
        found = [kw for kw in context_keywords if kw in output3]
        if found:
            print(f"[OK] 恢复后回复包含原任务上下文: {found}")
        else:
            print("[WARN] 恢复后回复未检测到原任务上下文关键词")
    else:
        print("[WARN] Step 3 120秒内未收到回复")

    summary3 = client.get_events_summary()
    print(f"[INFO] Step 3 事件统计: {json.dumps(summary3, ensure_ascii=False)}")

    await client.close()
    return bool(output1) and bool(output2) and bool(output3)


async def main():
    test_name = sys.argv[1] if len(sys.argv) > 1 else "1"

    if test_name == "1":
        await test_1_basic_connection()
    elif test_name == "2":
        await test_2_memory_retrieval()
    elif test_name == "3":
        await test_3_complex_task()
    elif test_name == "4":
        await test_4_suspend_resume()
    elif test_name == "all":
        r1 = await test_1_basic_connection()
        if r1:
            await asyncio.sleep(3)
            r2 = await test_2_memory_retrieval()
            if r2:
                await asyncio.sleep(3)
                r3 = await test_3_complex_task()
                if r3:
                    await asyncio.sleep(3)
                    await test_4_suspend_resume()
    else:
        print(f"用法: python live_test_ws.py [1|2|3|4|all]")


if __name__ == "__main__":
    asyncio.run(main())
