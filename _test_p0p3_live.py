"""
P0-P3 实时 WebSocket 测试
=========================
场景 A: 发送复杂任务 → 等待完成 → 验证 data/completed_tasks/ 有归档文件 (P0)
场景 B: 发送不同任务 → 然后说 "修改之前的XXX" → 验证恢复历史图谱 (P1)
场景 C: "继续之前的XXX" → 验证 RESUME 搜索已完成归档 (P2)

用法:
  python _test_p0p3_live.py           # 运行全部场景
  python _test_p0p3_live.py A         # 只运行场景 A
  python _test_p0p3_live.py B         # 只运行场景 B
  python _test_p0p3_live.py C         # 只运行场景 C
  python _test_p0p3_live.py connect   # 只测试连通性
"""
import asyncio
import websockets
import json
import time
import sys
import io
import os
import glob

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

WS_URL = "ws://localhost:5555/eventbus"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COMPLETED_DIR = os.path.join(DATA_DIR, "completed_tasks")
BACKUP_DIR = os.path.join(DATA_DIR, "graph_backups")

# 每步最长等待（Ollama 冷启动可能很慢）
TIMEOUT_SEC = 300


def list_completed_archives():
    """列出 data/completed_tasks/ 下的归档文件"""
    if not os.path.isdir(COMPLETED_DIR):
        return []
    return sorted(glob.glob(os.path.join(COMPLETED_DIR, "*.json")))


def list_graph_backups():
    """列出 data/graph_backups/ 下的备份文件"""
    if not os.path.isdir(BACKUP_DIR):
        return []
    return sorted(glob.glob(os.path.join(BACKUP_DIR, "*.json")))


async def send_and_wait(ws, message, timeout_sec=TIMEOUT_SEC, label=""):
    """发送消息并等待完整响应 (L2_OUTPUT)，返回 (success, full_text, thinking_steps)"""
    tag = f"[{label}]" if label else "[TEST]"

    chat_msg = {
        "type": "PUBLISH",
        "event": {
            "type": "USER_TEXT",
            "source": "test/qoder",
            "payload": {"text": message, "confidence": 1.0}
        }
    }
    await ws.send(json.dumps(chat_msg))
    print(f"{tag} >>> 发送: {message}")

    start = time.time()
    collected_stream = []
    thinking_steps = []
    full_text = ""
    got_full = False

    try:
        while time.time() - start < timeout_sec:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=90)
                data = json.loads(raw)
                msg_type = data.get("type", "")

                if msg_type == "SUBSCRIBE":
                    event = data.get("event", {})
                    etype = event.get("type", "")
                    payload = event.get("payload", {})

                    if etype == "L2_OUTPUT_STREAM":
                        text = payload.get("text", "")
                        print(text, end="", flush=True)
                        collected_stream.append(text)
                    elif etype == "L2_OUTPUT":
                        full_text = payload.get("text", "")
                        print(f"\n{tag} <<< 收到完整响应 ({len(full_text)} chars)")
                        got_full = True
                        break
                    elif etype == "L2_THINKING_STEP":
                        step = str(payload.get("step", ""))[:200]
                        thinking_steps.append(step)
                        print(f"\n{tag} [THINK] {step}")
                    elif etype == "ACTION_SPEAK":
                        text = payload.get("text", "")[:200]
                        print(f"\n{tag} [SPEAK] {text}")
                elif msg_type == "ACK":
                    pass  # 忽略 ACK

            except asyncio.TimeoutError:
                print(f"\n{tag} 等待消息超时 (90s)")
                break
    except Exception as e:
        print(f"\n{tag} 异常: {e}")

    elapsed = time.time() - start
    print(f"{tag} 耗时: {elapsed:.1f}s | 有完整响应: {got_full}")
    return got_full, full_text, thinking_steps


async def connect_and_subscribe():
    """连接 WebSocket 并订阅事件"""
    ws = await websockets.connect(WS_URL, ping_interval=None, close_timeout=10, max_size=2**22)
    subscribe_msg = {
        "type": "SUBSCRIBE",
        "event_types": [
            "L2_OUTPUT", "L2_OUTPUT_STREAM",
            "L2_THINKING_STEP", "ACTION_SPEAK",
            "TASK_GRAPH_UPDATED", "MEMORY_GRAPH_UPDATED",
        ]
    }
    await ws.send(json.dumps(subscribe_msg))
    # 等一下 ACK
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(raw)
        if data.get("type") == "ACK":
            print("[CONN] 订阅成功")
    except asyncio.TimeoutError:
        print("[CONN] 未收到 ACK（继续）")
    return ws


# ──────────────────────────────────────────────────────────────
# 场景 A: 复杂任务 → 完成 → 检查归档 (P0)
# ──────────────────────────────────────────────────────────────
async def scenario_a():
    """P0: 完成复杂任务后自动归档"""
    print("\n" + "=" * 60)
    print("场景 A: 复杂任务完成 → P0 自动归档验证")
    print("=" * 60)

    archives_before = list_completed_archives()
    print(f"[A] 测试前归档数: {len(archives_before)}")

    ws = await connect_and_subscribe()
    try:
        # 发送一个简短的复杂任务
        ok, text, steps = await send_and_wait(
            ws,
            "帮我写一个Python函数，计算斐波那契数列的第n项",
            label="A"
        )
        if not ok:
            print("[A] WARN: 未收到完整响应")

        # 等待一会让系统处理完（归档是异步的）
        print("[A] 等待 5s 让归档完成...")
        await asyncio.sleep(5)

        archives_after = list_completed_archives()
        print(f"[A] 测试后归档数: {len(archives_after)}")
        new_archives = [a for a in archives_after if a not in archives_before]

        if new_archives:
            print(f"[A] PASS: 新增 {len(new_archives)} 个归档文件:")
            for a in new_archives:
                print(f"  - {os.path.basename(a)}")
        else:
            # 也可能任务不够复杂，没有创建 TaskGraph
            print("[A] INFO: 未新增归档（可能任务未创建 TaskGraph，不属于 P0 触发范围）")
            print("[A] 检查 thinking_steps 中是否有 COMPLEX 意图...")
            has_complex = any("complex" in s.lower() or "COMPLEX" in s for s in steps)
            if has_complex:
                print("[A] WARN: 有 COMPLEX 意图但未归档，需要检查")
            else:
                print("[A] INFO: 任务被分类为 CHAT，跳过归档检查")

        return True
    finally:
        await ws.close()


# ──────────────────────────────────────────────────────────────
# 场景 B: 已完成 → 新任务 → "修改之前的XXX" (P1)
# ──────────────────────────────────────────────────────────────
async def scenario_b():
    """P1: 对已完成历史任务说'修改之前的XXX'，验证从归档恢复"""
    print("\n" + "=" * 60)
    print("场景 B: 修改已完成历史任务 → P1 历史搜索恢复")
    print("=" * 60)

    ws = await connect_and_subscribe()
    try:
        # 步骤 1: 先完成一个任务
        print("\n[B-1] 发送第一个任务...")
        ok1, text1, steps1 = await send_and_wait(
            ws,
            "帮我设计一个简单的学生成绩管理系统的数据库表结构",
            label="B-1"
        )
        if not ok1:
            print("[B-1] WARN: 未收到完整响应")

        await asyncio.sleep(3)

        # 步骤 2: 然后要求修改（触发 P1 搜索）
        print("\n[B-2] 请求修改之前的任务...")
        ok2, text2, steps2 = await send_and_wait(
            ws,
            "修改之前的学生成绩管理系统，给数据库表加一个班级字段",
            label="B-2"
        )

        # 检查 thinking_steps 是否有恢复历史图谱的迹象
        has_restore = any("restored" in s.lower() or "归档" in s or "恢复" in s or "already_exists" in s for s in steps2)
        if has_restore:
            print("[B] PASS: thinking_steps 中发现历史恢复迹象")
        else:
            print("[B] INFO: thinking_steps 中未明确发现恢复迹象")
            print(f"[B] steps2 摘要: {steps2[:5]}")

        # 检查响应是否提到了"成绩管理"或修改
        if ok2 and ("成绩" in text2 or "班级" in text2 or "管理" in text2):
            print("[B] PASS: 响应包含成绩管理/班级相关内容")
        elif ok2:
            print(f"[B] INFO: 响应前200字: {text2[:200]}")

        return True
    finally:
        await ws.close()


# ──────────────────────────────────────────────────────────────
# 场景 C: RESUME 搜索已完成归档 (P2)
# ──────────────────────────────────────────────────────────────
async def scenario_c():
    """P2: '继续之前的XXX' 从已完成归档恢复"""
    print("\n" + "=" * 60)
    print("场景 C: RESUME 恢复已完成任务 → P2 归档搜索")
    print("=" * 60)

    # 先确认有归档可搜
    archives = list_completed_archives()
    print(f"[C] 当前归档数: {len(archives)}")
    if archives:
        # 读取第一个归档看描述
        try:
            with open(archives[-1], 'r', encoding='utf-8') as f:
                arch_data = json.load(f)
            desc = arch_data.get("description", "未知")
            print(f"[C] 最新归档描述: {desc}")
        except:
            desc = "未知"

    ws = await connect_and_subscribe()
    try:
        # 发送 RESUME 类型请求
        ok, text, steps = await send_and_wait(
            ws,
            "继续之前的学生成绩管理系统任务",
            label="C"
        )

        has_resume = any("resume" in s.lower() or "恢复" in s or "restored" in s.lower() for s in steps)
        if has_resume:
            print("[C] PASS: thinking_steps 中发现 RESUME 恢复迹象")
        else:
            print("[C] INFO: thinking_steps 中未明确发现 RESUME 恢复")
            print(f"[C] steps 摘要: {steps[:5]}")

        if ok and ("成绩" in text or "管理" in text or "学生" in text):
            print("[C] PASS: 响应与学生成绩管理系统相关")
        elif ok:
            print(f"[C] INFO: 响应前200字: {text[:200]}")

        return True
    finally:
        await ws.close()


# ──────────────────────────────────────────────────────────────
# 连通性测试
# ──────────────────────────────────────────────────────────────
async def test_connectivity():
    """简单连通性测试"""
    print("\n" + "=" * 60)
    print("连通性测试")
    print("=" * 60)

    ws = await connect_and_subscribe()
    try:
        ok, text, _ = await send_and_wait(ws, "你好", timeout_sec=120, label="PING")
        if ok:
            print(f"[PING] PASS: 收到响应 ({len(text)} chars)")
            print(f"[PING] 响应前200字: {text[:200]}")
        else:
            print("[PING] FAIL: 未收到完整响应")
        return ok
    finally:
        await ws.close()


async def main():
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"

    print(f"[MAIN] 模式: {mode}")
    print(f"[MAIN] data 目录: {DATA_DIR}")
    print(f"[MAIN] completed_tasks 目录: {COMPLETED_DIR}")
    print(f"[MAIN] 已有归档: {len(list_completed_archives())}")
    print(f"[MAIN] 已有备份: {len(list_graph_backups())}")

    if mode == "CONNECT":
        await test_connectivity()
    elif mode == "A":
        await scenario_a()
    elif mode == "B":
        await scenario_b()
    elif mode == "C":
        await scenario_c()
    elif mode == "ALL":
        await test_connectivity()
        await scenario_a()
        await scenario_b()
        await scenario_c()
    else:
        print(f"未知模式: {mode}")
        print("用法: python _test_p0p3_live.py [A|B|C|ALL|CONNECT]")


if __name__ == "__main__":
    asyncio.run(main())
