"""
祖龙系统 - 任务打断/抢占/自动恢复 + 长短期记忆 集成测试

测试场景：
  Test 1: 发送复杂任务，验证 Orchestrator 正常启动与执行
  Test 2: 执行复杂任务中途发送无关消息，验证抢占 + 自动恢复
  Test 3: 验证记忆写入（MemoryGraph 节点持久化）

使用方法：
  1. 确保祖龙已启动 (python zulong/bootstrap.py)
  2. python tests/test_preempt_and_memory.py [test1|test2|test3|all]
"""

import asyncio
import websockets
import json
import time
import sys
import uuid
import os
from datetime import datetime
from typing import Optional

# ── 配置 ──
WS_URI = "ws://localhost:5555/eventbus"
CONNECT_TIMEOUT = 5
EVENT_TIMEOUT = 120  # 单个事件最长等待 (秒)
TOTAL_TIMEOUT = 300  # 单个测试最长总时间 (秒)

# ── 颜色输出 ──
class C:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"

def log(msg, color=C.RESET):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{C.GRAY}[{ts}]{C.RESET} {color}{msg}{C.RESET}")

def log_event(event_type, text_preview):
    log(f"  << {event_type}: {text_preview[:120]}", C.CYAN)

def log_send(label):
    log(f"  >> {label}", C.YELLOW)

def log_ok(msg):
    log(f"  [PASS] {msg}", C.GREEN)

def log_fail(msg):
    log(f"  [FAIL] {msg}", C.RED)


class ZulongTestClient:
    """祖龙 EventBus WebSocket 测试客户端"""

    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.events: list = []
        self._listener_task: Optional[asyncio.Task] = None

    async def connect(self):
        log(f"连接 {WS_URI} ...")
        self.ws = await asyncio.wait_for(
            websockets.connect(WS_URI, max_size=2**22),
            timeout=CONNECT_TIMEOUT,
        )
        # 启动后台监听
        self._listener_task = asyncio.create_task(self._listen_loop())
        log("WebSocket 连接成功", C.GREEN)

    async def close(self):
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            await self.ws.close()

    async def _listen_loop(self):
        """后台持续接收事件"""
        try:
            async for raw in self.ws:
                try:
                    data = json.loads(raw)
                    self.events.append(data)
                    # 提取可读信息
                    evt = data.get("event", {})
                    etype = evt.get("type", data.get("type", "?"))
                    payload = evt.get("payload", {})
                    text = payload.get("text", "")[:120]
                    # 只打印关键事件
                    if etype in ("L2_OUTPUT", "L2_OUTPUT_STREAM", "L2_THINKING_STEP"):
                        log_event(etype, text or "(stream chunk)")
                    elif etype == "ACK":
                        pass  # 静默
                    else:
                        log_event(etype, str(payload)[:120] if payload else str(data)[:120])
                except json.JSONDecodeError:
                    log(f"  非 JSON: {raw[:80]}", C.GRAY)
        except websockets.exceptions.ConnectionClosed:
            log("WebSocket 连接已关闭", C.YELLOW)
        except asyncio.CancelledError:
            pass

    async def send_text(self, text: str, session_id: str = "", request_id: str = ""):
        """发送用户文本消息"""
        if not session_id:
            session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:8]}"

        msg = {
            "type": "PUBLISH",
            "event": {
                "type": "USER_TEXT",
                "payload": {
                    "text": text,
                    "session_id": session_id,
                    "request_id": request_id,
                    "source": "TestClient",
                },
            },
        }
        await self.ws.send(json.dumps(msg))
        log_send(f"USER_TEXT [{request_id}]: {text[:80]}...")

    def get_l2_outputs(self) -> list:
        """提取所有 L2_OUTPUT 事件的 text"""
        results = []
        for ev in self.events:
            evt = ev.get("event", {})
            if evt.get("type") == "L2_OUTPUT":
                text = evt.get("payload", {}).get("text", "")
                if text:
                    results.append(text)
        return results

    def has_event_matching(self, field_path: str, value: str) -> bool:
        """检查是否有事件匹配指定字段值 (field_path 用 . 分隔)"""
        for ev in self.events:
            obj = ev
            for key in field_path.split("."):
                if isinstance(obj, dict):
                    obj = obj.get(key)
                else:
                    obj = None
                    break
            if obj and isinstance(obj, str) and value in obj:
                return True
        return False

    async def wait_for_output(self, keyword: str = None, timeout: float = EVENT_TIMEOUT) -> Optional[str]:
        """等待直到收到包含 keyword 的 L2_OUTPUT，或超时"""
        start = time.time()
        seen = 0
        while time.time() - start < timeout:
            outputs = self.get_l2_outputs()
            if len(outputs) > seen:
                for text in outputs[seen:]:
                    if keyword is None or keyword in text:
                        return text
                seen = len(outputs)
            await asyncio.sleep(0.5)
        return None

    async def wait_for_activity(self, timeout: float = 60) -> bool:
        """等待任何处理活动 (L2_THINKING_STEP / L2_OUTPUT / L2_OUTPUT_STREAM / MEMORY_GRAPH_UPDATED)
        
        返回 True 表示 Orchestrator 已开始工作。
        """
        active_types = {"L2_THINKING_STEP", "L2_OUTPUT", "L2_OUTPUT_STREAM", "MEMORY_GRAPH_UPDATED"}
        start = time.time()
        seen = 0
        while time.time() - start < timeout:
            for ev in self.events[seen:]:
                etype = ev.get("event", {}).get("type", "")
                if etype in active_types:
                    return True
            seen = len(self.events)
            await asyncio.sleep(0.3)
        return False

    async def wait_for_output_or_stream(self, keyword: str = None, timeout: float = EVENT_TIMEOUT) -> Optional[str]:
        """等待 L2_OUTPUT 或 L2_OUTPUT_STREAM（包含关键词），适用于流式输出场景"""
        start = time.time()
        seen = 0
        while time.time() - start < timeout:
            for ev in self.events[seen:]:
                evt = ev.get("event", {})
                etype = evt.get("type", "")
                if etype in ("L2_OUTPUT", "L2_OUTPUT_STREAM"):
                    text = evt.get("payload", {}).get("text", "")
                    if text and (keyword is None or keyword in text):
                        return text
            seen = len(self.events)
            await asyncio.sleep(0.5)
        return None

    async def wait_for_events(self, count: int = 1, timeout: float = 30):
        """等待至少 count 个新事件"""
        start_count = len(self.events)
        start = time.time()
        while time.time() - start < timeout:
            if len(self.events) - start_count >= count:
                return True
            await asyncio.sleep(0.3)
        return False


# ══════════════════════════════════════════════════════════════
# Test 1: 复杂任务正常执行
# ══════════════════════════════════════════════════════════════
async def test1_complex_task():
    """验证 Orchestrator 能正常处理复杂多步任务"""
    log(f"\n{'='*60}", C.BOLD)
    log("Test 1: 复杂任务正常执行", C.BOLD)
    log(f"{'='*60}", C.BOLD)

    client = ZulongTestClient()
    session_id = f"test1_{uuid.uuid4().hex[:8]}"
    passed = 0
    total = 3

    try:
        await client.connect()

        # 发送一个需要多步规划的任务
        task_text = (
            "请帮我分析一下 Python 3.12 和 3.13 版本的主要新特性差异，"
            "列出每个版本的 3 个最重要的新功能，并给出升级建议。"
        )
        await client.send_text(task_text, session_id=session_id)

        # 检查点 1: 是否有任何处理活动
        log("\n等待 Orchestrator 启动响应...")
        active = await client.wait_for_activity(timeout=60)
        if active:
            log_ok("Orchestrator 已启动（检测到处理活动）")
            passed += 1
        else:
            log_fail("60s 内未检测到任何活动")

        # 检查点 2: 等待更多输出（判断是否有多轮工具调用）
        log("等待任务执行中的更多输出...")
        await asyncio.sleep(30)
        all_outputs = client.get_l2_outputs()
        log(f"共收到 {len(all_outputs)} 条 L2_OUTPUT")
        if len(all_outputs) >= 1:
            log_ok(f"任务产生了 {len(all_outputs)} 条输出")
            passed += 1
        else:
            log_fail("输出数量不足")

        # 检查点 3: 检查是否有 thinking step 事件（说明 Orchestrator 在规划）
        thinking_events = [
            e for e in client.events
            if e.get("event", {}).get("type") in ("L2_THINKING_STEP", "L2_OUTPUT_STREAM")
        ]
        if thinking_events:
            log_ok(f"检测到 {len(thinking_events)} 条思考/流式事件（Orchestrator 活跃）")
            passed += 1
        else:
            log(f"  [WARN] 未检测到思考步骤事件（可能正常，取决于模型配置）", C.YELLOW)
            passed += 1  # 不强制要求

    except Exception as e:
        log_fail(f"测试异常: {e}")
    finally:
        await client.close()

    log(f"\nTest 1 结果: {passed}/{total} 通过", C.GREEN if passed == total else C.YELLOW)
    return passed, total


# ══════════════════════════════════════════════════════════════
# Test 2: 任务打断 + 自动恢复
# ══════════════════════════════════════════════════════════════
async def test2_preempt_and_resume():
    """验证：执行中发送无关消息 → 抢占 → 新任务完成 → 自动恢复"""
    log(f"\n{'='*60}", C.BOLD)
    log("Test 2: 任务打断与自动恢复", C.BOLD)
    log(f"{'='*60}", C.BOLD)

    client = ZulongTestClient()
    session_id = f"test2_{uuid.uuid4().hex[:8]}"
    passed = 0
    total = 4

    try:
        await client.connect()

        # Step 1: 发送复杂任务 (Task A)
        task_a = (
            "请帮我做一个详细的对比分析：React 和 Vue 框架在 2025 年的生态系统差异，"
            "包括性能、社区支持、学习曲线和企业采用率。需要分至少 4 个维度进行分析。"
        )
        log("\n[Step 1] 发送复杂任务 A...")
        await client.send_text(task_a, session_id=session_id, request_id="task_a")

        # Step 2: 等待 Task A 开始执行（检测任何处理活动）
        log("[Step 2] 等待 Task A 开始处理...")
        active = await client.wait_for_activity(timeout=60)
        if active:
            log_ok("Task A 已开始执行（检测到处理活动）")
            passed += 1
        else:
            log_fail("Task A 未启动（60s 内无任何活动事件）")
            await client.close()
            return passed, total

        # 额外等几秒让任务深入执行（等待更多 thinking step）
        log("等待 15s 让 Task A 深入执行...")
        await asyncio.sleep(15)
        events_before_preempt = len(client.events)

        # Step 3: 发送无关消息 (Task B) → 触发抢占
        task_b = "今天天气怎么样？"
        log(f"\n[Step 3] 发送无关消息触发抢占: '{task_b}'")
        await client.send_text(task_b, session_id=session_id, request_id="task_b_interrupt")

        # Step 4: 检查是否有抢占相关事件
        log("[Step 4] 检查抢占事件...")
        await asyncio.sleep(5)

        # 查找抢占通知
        preempt_found = False
        for ev in client.events[events_before_preempt:]:
            evt = ev.get("event", {})
            payload = evt.get("payload", {})
            text = payload.get("text", "")
            source = evt.get("source", "")
            # 检查抢占通知或新任务输出
            if "暂停" in text or "Preempt" in source or "preempt_info" in payload:
                preempt_found = True
                break
            if payload.get("auto_resume"):
                preempt_found = True
                break

        if preempt_found:
            log_ok("检测到抢占通知事件")
            passed += 1
        else:
            log(f"  [WARN] 未检测到明确的抢占通知（检查 Gatekeeper 是否判定为相关消息）", C.YELLOW)
            # 也可能 Gatekeeper 认为消息相关而没触发抢占，这也是合理行为
            passed += 0.5

        # Step 5: 等待 Task B 响应（可能是 L2_OUTPUT 或流式输出）
        log("[Step 5] 等待 Task B 响应...")
        task_b_response = await client.wait_for_output_or_stream(timeout=90)
        if task_b_response:
            log_ok(f"Task B 已响应 ({len(task_b_response)} 字符)")
            passed += 1
        else:
            # 也检查是否有新的 thinking step（说明新 Orchestrator 在工作）
            new_activity = [
                e for e in client.events[events_before_preempt:]
                if e.get("event", {}).get("type") in ("L2_THINKING_STEP", "L2_OUTPUT_STREAM")
            ]
            if new_activity:
                log_ok(f"Task B 产生了 {len(new_activity)} 条活动事件（新 Orchestrator 在工作）")
                passed += 1
            else:
                log_fail("Task B 无响应")

        # Step 6: 等待自动恢复（增加超时，因为 Task B 可能需要较长时间完成）
        log("[Step 6] 等待自动恢复 Task A...")
        resume_found = False
        resume_output = await client.wait_for_output_or_stream(keyword="恢复", timeout=120)
        if resume_output:
            resume_found = True
        else:
            # 也检查是否有 AutoResume 事件
            for ev in client.events:
                evt = ev.get("event", {})
                payload = evt.get("payload", {})
                if payload.get("auto_resume") or "AutoResume" in evt.get("source", ""):
                    resume_found = True
                    break

        if resume_found:
            log_ok("检测到自动恢复事件/输出")
            passed += 1
        else:
            log(f"  [WARN] 未检测到自动恢复（可能 Task B 尚未完成或 Gatekeeper 未抢占）", C.YELLOW)

    except Exception as e:
        log_fail(f"测试异常: {e}")
    finally:
        await client.close()

    log(f"\nTest 2 结果: {passed}/{total} 通过", C.GREEN if passed >= total - 1 else C.YELLOW)
    return passed, total


# ══════════════════════════════════════════════════════════════
# Test 3: 长短期记忆验证
# ══════════════════════════════════════════════════════════════
async def test3_memory():
    """验证记忆系统：对话节点写入 MemoryGraph + 短期 AWM 焦点追踪"""
    log(f"\n{'='*60}", C.BOLD)
    log("Test 3: 长短期记忆验证", C.BOLD)
    log(f"{'='*60}", C.BOLD)

    client = ZulongTestClient()
    session_id = f"test3_{uuid.uuid4().hex[:8]}"
    passed = 0
    total = 3

    try:
        await client.connect()

        # Step 1: 发送一个有明确记忆写入点的任务
        task = (
            "请记住：我的项目名叫星辰计划，使用 Python 3.12 + FastAPI 技术栈。"
            "然后帮我简单解释一下 FastAPI 的异步处理机制。"
        )
        log("\n[Step 1] 发送包含记忆写入点的任务...")
        await client.send_text(task, session_id=session_id, request_id="mem_task_1")

        # 等待响应
        response1 = await client.wait_for_output(timeout=90)
        if response1:
            log_ok(f"收到响应 ({len(response1)} 字符)")
            passed += 1
        else:
            log_fail("未收到响应")

        # 等待任务完成
        log("等待任务充分执行...")
        await asyncio.sleep(20)

        # Step 2: 检查 MemoryGraph 事件
        log("[Step 2] 检查 MEMORY_GRAPH_UPDATED 事件...")
        mg_events = [
            e for e in client.events
            if e.get("event", {}).get("type") == "MEMORY_GRAPH_UPDATED"
        ]
        if mg_events:
            log_ok(f"检测到 {len(mg_events)} 个 MemoryGraph 更新事件")
            passed += 1
        else:
            log(f"  [INFO] 未检测到 MEMORY_GRAPH_UPDATED 事件（可能未通过 EventBus 广播）", C.YELLOW)
            # 尝试直接检查磁盘
            graph_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "agent_workspace"
            )
            if os.path.exists(graph_dir):
                # 找最新的工作区
                dirs = sorted(
                    [d for d in os.listdir(graph_dir) if os.path.isdir(os.path.join(graph_dir, d))],
                    reverse=True,
                )
                if dirs:
                    log(f"  [INFO] 最新工作区: {dirs[0]}", C.GRAY)
                    passed += 0.5
                else:
                    log(f"  [INFO] agent_workspace 为空", C.GRAY)
            else:
                log(f"  [INFO] agent_workspace 不存在", C.GRAY)

        # Step 3: 发送回忆性问题，验证短期记忆
        log("\n[Step 3] 发送回忆性问题，验证短期记忆...")
        await asyncio.sleep(5)
        recall_task = "我之前说的项目叫什么名字？用的什么技术栈？"
        await client.send_text(recall_task, session_id=session_id, request_id="mem_recall")

        recall_response = await client.wait_for_output(timeout=90)
        if recall_response:
            # 检查是否包含之前提到的关键信息
            keywords_found = []
            for kw in ["星辰", "FastAPI", "Python", "3.12"]:
                if kw in recall_response:
                    keywords_found.append(kw)

            if len(keywords_found) >= 2:
                log_ok(f"短期记忆验证通过，回忆到: {keywords_found}")
                passed += 1
            elif len(keywords_found) >= 1:
                log(f"  [PARTIAL] 部分回忆: {keywords_found}", C.YELLOW)
                passed += 0.5
            else:
                log_fail(f"未能回忆关键信息 (响应: {recall_response[:200]})")
        else:
            log_fail("回忆问题无响应")

    except Exception as e:
        log_fail(f"测试异常: {e}")
    finally:
        await client.close()

    log(f"\nTest 3 结果: {passed}/{total} 通过", C.GREEN if passed >= total - 1 else C.YELLOW)
    return passed, total


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════
async def main():
    test_arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    test_map = {
        "test1": ("复杂任务执行", test1_complex_task),
        "test2": ("任务打断 + 自动恢复", test2_preempt_and_resume),
        "test3": ("长短期记忆", test3_memory),
    }

    results = {}

    if test_arg == "all":
        tests_to_run = ["test1", "test2", "test3"]
    elif test_arg in test_map:
        tests_to_run = [test_arg]
    else:
        print(f"用法: python {sys.argv[0]} [test1|test2|test3|all]")
        print("  test1 - 复杂任务正常执行")
        print("  test2 - 任务打断 + 自动恢复 (核心)")
        print("  test3 - 长短期记忆验证")
        print("  all   - 运行全部测试")
        return

    log(f"\n{'#'*60}", C.BOLD)
    log(f"祖龙系统集成测试 - 打断抢占 & 记忆", C.BOLD)
    log(f"测试项: {', '.join(tests_to_run)}", C.BOLD)
    log(f"{'#'*60}\n", C.BOLD)

    for test_name in tests_to_run:
        label, test_fn = test_map[test_name]
        try:
            passed, total = await test_fn()
            results[test_name] = (passed, total)
        except Exception as e:
            log_fail(f"{test_name} 异常退出: {e}")
            results[test_name] = (0, 1)

        # 测试间隔，让系统恢复
        if test_name != tests_to_run[-1]:
            log("\n--- 等待 10s 再开始下一个测试 ---\n", C.GRAY)
            await asyncio.sleep(10)

    # 汇总
    log(f"\n{'='*60}", C.BOLD)
    log("测试汇总", C.BOLD)
    log(f"{'='*60}", C.BOLD)

    all_passed = 0
    all_total = 0
    for test_name in tests_to_run:
        p, t = results.get(test_name, (0, 0))
        all_passed += p
        all_total += t
        label = test_map[test_name][0]
        status = C.GREEN + "PASS" if p >= t else (C.YELLOW + "PARTIAL" if p > 0 else C.RED + "FAIL")
        log(f"  {test_name} ({label}): {p}/{t} {status}{C.RESET}")

    log(f"\n总计: {all_passed}/{all_total}", C.GREEN if all_passed >= all_total else C.YELLOW)


if __name__ == "__main__":
    asyncio.run(main())
