"""
MemoryGraph 运行时诊断脚本

通过 EventBus WebSocket 端点向运行中的祖龙发送诊断指令，
触发 MemoryGraph 保存并读取持久化数据分析。

用法:
  python tests/diagnose_memory_graph.py [--save] [--task "任务文本"]

功能:
  1. 默认: 连接 → 发送简单对话 → 等待处理 → 检查磁盘数据
  2. --save: 发送自定义事件触发 MemoryGraph.save()
  3. --task: 指定测试任务文本
"""

import asyncio
import json
import sys
import os
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GraphDiagnose")

try:
    import websockets
except ImportError:
    print("ERROR: pip install websockets")
    sys.exit(1)

WS_URI = "ws://127.0.0.1:5555/eventbus"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "memory_graph")


def check_persisted_graph():
    """检查磁盘上的 MemoryGraph 持久化数据"""
    logger.info(f"\n{'='*56}")
    logger.info("MemoryGraph 磁盘数据检查")
    logger.info(f"{'='*56}")
    logger.info(f"目录: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        logger.info("  目录不存在 (MemoryGraph 可能未初始化或从未保存)")
        return False

    files = os.listdir(DATA_DIR)
    logger.info(f"  文件: {files if files else '(空目录)'}")

    graph_file = os.path.join(DATA_DIR, "graph.json")
    if not os.path.exists(graph_file):
        logger.info("  graph.json 不存在 (图谱在内存中运行，尚未持久化)")
        logger.info("  提示: 在祖龙终端运行 /graph save 可触发保存")
        return False

    try:
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        meta = data.get("metadata", {})

        logger.info(f"\n  === 图谱概览 ===")
        logger.info(f"  保存时间: {meta.get('saved_at', 'N/A')}")
        logger.info(f"  节点数: {len(nodes)}")
        logger.info(f"  边数: {len(edges)}")

        # 节点类型分布
        type_counts = {}
        for n in nodes:
            t = n.get("node_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info(f"  节点类型分布: {type_counts}")

        # 边类型分布
        edge_type_counts = {}
        protected_count = 0
        for e in edges:
            t = e.get("edge_type", "unknown")
            edge_type_counts[t] = edge_type_counts.get(t, 0) + 1
            if e.get("protected"):
                protected_count += 1
        logger.info(f"  边类型分布: {edge_type_counts}")
        logger.info(f"  受保护边: {protected_count}")

        # 列出所有节点
        if nodes:
            logger.info(f"\n  === 节点列表 ({len(nodes)}) ===")
            for n in nodes:
                nid = n.get("node_id", "?")
                ntype = n.get("node_type", "?")
                label = n.get("label", "")
                backend = n.get("backend_ref", "")
                meta_keys = list(n.get("metadata", {}).keys())
                logger.info(f"    [{ntype:10s}] {nid}")
                logger.info(f"               label={label}")
                if backend:
                    logger.info(f"               backend={backend}")
                if meta_keys:
                    logger.info(f"               metadata_keys={meta_keys}")

        # 列出所有边
        if edges:
            logger.info(f"\n  === 边列表 ({len(edges)}) ===")
            for e in edges:
                src = e.get("source", "?")
                dst = e.get("target", "?")
                etype = e.get("edge_type", "?")
                weight = e.get("weight", 0)
                prot = " [PROTECTED]" if e.get("protected") else ""
                logger.info(f"    {src} --[{etype} w={weight:.3f}{prot}]--> {dst}")

        # 验证记忆图谱关键特性
        logger.info(f"\n  === 关键特性验证 ===")

        # 检查 TaskGraph 同步
        task_nodes = [n for n in nodes if n.get("node_type") == "task"]
        logger.info(f"  [{'OK' if task_nodes else 'MISS'}] TaskGraph 同步: {len(task_nodes)} 个任务节点")

        # 检查层级边
        h_edges = [e for e in edges if e.get("edge_type") == "hierarchy"]
        logger.info(f"  [{'OK' if h_edges else 'MISS'}] 层级边 (HIERARCHY): {len(h_edges)} 条")

        # 检查依赖边
        d_edges = [e for e in edges if e.get("edge_type") == "dependency"]
        logger.info(f"  [{'OK' if d_edges else '----'}] 依赖边 (DEPENDENCY): {len(d_edges)} 条")

        # 检查受保护边
        logger.info(f"  [{'OK' if protected_count > 0 else 'MISS'}] 受保护结构边: {protected_count} 条")

        # 检查对话节点
        dialog_nodes = [n for n in nodes if n.get("node_type") == "dialogue"]
        logger.info(f"  [{'OK' if dialog_nodes else '----'}] 对话节点 (DIALOGUE): {len(dialog_nodes)} 个")

        # 检查知识节点
        kg_nodes = [n for n in nodes if n.get("node_type") in ("knowledge", "person", "concept")]
        logger.info(f"  [{'OK' if kg_nodes else '----'}] 知识图谱节点: {len(kg_nodes)} 个")

        return True

    except Exception as e:
        logger.error(f"  读取 graph.json 失败: {e}")
        return False


async def send_task_and_wait(text: str, wait_time: float = 60):
    """发送对话任务并等待处理"""
    logger.info(f"\n{'='*56}")
    logger.info(f"发送测试任务: {text[:60]}...")
    logger.info(f"{'='*56}")

    try:
        ws = await websockets.connect(
            WS_URI, ping_interval=None, ping_timeout=None, max_size=None,
        )
    except Exception as e:
        logger.error(f"连接失败: {e}")
        logger.info("请确认祖龙系统已启动 (python zulong/bootstrap.py)")
        return False

    try:
        # 发送 USER_SPEECH 事件
        event = {
            "type": "PUBLISH",
            "event": {
                "type": "USER_SPEECH",
                "priority": "NORMAL",
                "source": "GraphDiagnose",
                "payload": {
                    "text": text,
                    "confidence": 1.0,
                    "timestamp": time.time(),
                },
            },
        }
        await ws.send(json.dumps(event))
        logger.info("事件已发送")

        # 接收响应
        graph_updates = 0
        tool_calls = 0
        l2_outputs = 0
        thinking_steps = 0
        start = time.time()

        while time.time() - start < wait_time:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
                data = json.loads(raw)
                event_data = data.get("event", {})
                payload = event_data.get("payload", {})

                msg_type = data.get("type", "")
                event_type = event_data.get("type", msg_type)

                if event_type == "L2_THINKING_STEP":
                    thinking_steps += 1
                    step_type = payload.get("step_type", "")
                    if "graph_update" in step_type:
                        graph_updates += 1
                        change_data = payload.get("data", {})
                        change_type = change_data.get("change_type", "")
                        node_data = change_data.get("data", {})
                        label = node_data.get("label", node_data.get("node_id", ""))
                        logger.info(f"  [GRAPH_UPDATE] {change_type}: {label}")
                    elif "tool_call" in step_type:
                        tool_calls += 1
                        tool_name = payload.get("data", {}).get("tool_name", "?")
                        if tool_calls <= 5:
                            logger.info(f"  [TOOL_CALL] {tool_name}")
                elif event_type == "L2_OUTPUT":
                    l2_outputs += 1
                    text_out = payload.get("text", "")[:150]
                    logger.info(f"  [L2_OUTPUT] {text_out}")
                elif event_type == "ACTION_SPEAK":
                    text_out = payload.get("text", "")[:100]
                    logger.info(f"  [SPEAK] {text_out}")
                elif msg_type == "ACK":
                    logger.info(f"  [ACK] {data.get('message', '')}")

            except asyncio.TimeoutError:
                elapsed = time.time() - start
                if elapsed > 30:
                    break
                continue

        logger.info(f"\n  统计: thinking={thinking_steps}, graph_updates={graph_updates}, "
                    f"tool_calls={tool_calls}, l2_outputs={l2_outputs}")

        return graph_updates > 0

    finally:
        await ws.close()


def try_import_graph():
    """尝试直接导入 MemoryGraph 查看状态(仅当与祖龙在同一进程时有效)"""
    try:
        from zulong.memory.memory_graph import MemoryGraph
        mg = MemoryGraph._instance
        if mg:
            logger.info(f"\n  [IN-PROCESS] MemoryGraph 单例存在:")
            logger.info(f"    节点: {mg.stats['total_nodes']}")
            logger.info(f"    边: {mg.stats['total_edges']}")
            return True
        else:
            logger.info("  [IN-PROCESS] MemoryGraph._instance = None (可能在另一个进程)")
            return False
    except ImportError:
        logger.info("  [IN-PROCESS] 无法导入 MemoryGraph")
        return False


async def main():
    parser = argparse.ArgumentParser(description="MemoryGraph 运行时诊断")
    parser.add_argument("--save", action="store_true",
                        help="提示用户在祖龙终端执行 /graph save")
    parser.add_argument("--task", type=str, default=None,
                        help="发送测试任务文本")
    parser.add_argument("--wait", type=float, default=90,
                        help="等待任务完成的时间(秒)")
    args = parser.parse_args()

    logger.info("=" * 56)
    logger.info("MemoryGraph 运行时诊断")
    logger.info("=" * 56)

    # 1. 检查磁盘数据
    has_data = check_persisted_graph()

    # 2. 发送任务 (如果指定)
    if args.task:
        had_updates = await send_task_and_wait(args.task, wait_time=args.wait)
        if had_updates:
            logger.info("\n检测到 graph_update 事件 -> MemoryGraph 正在同步 TaskGraph 变更")
    elif not has_data:
        # 默认发送一个简单任务
        logger.info("\n磁盘无数据，发送测试任务触发图谱构建...")
        had_updates = await send_task_and_wait(
            "你好，请简单介绍一下你是谁，你有什么能力？",
            wait_time=args.wait,
        )
        if had_updates:
            logger.info("\n检测到 graph_update 事件 -> MemoryGraph 正在同步 TaskGraph 变更")

    # 3. 提示保存
    if args.save or not has_data:
        logger.info(f"\n{'='*56}")
        logger.info("下一步操作:")
        logger.info(f"{'='*56}")
        logger.info("  请在祖龙调试终端中执行以下命令来保存图谱:")
        logger.info("    /graph save")
        logger.info("  然后重新运行此脚本查看完整数据:")
        logger.info("    python tests/diagnose_memory_graph.py")

    # 4. 再次检查磁盘
    if has_data:
        logger.info("\n诊断完成。图谱数据有效。")
    else:
        logger.info("\n诊断完成。等待 /graph save 后再次运行本工具查看完整数据。")


if __name__ == "__main__":
    asyncio.run(main())
