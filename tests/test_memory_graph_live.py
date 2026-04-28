"""
记忆图谱 (MemoryGraph) 实时集成测试

通过 WebSocket 连接运行中的祖龙系统，发送对话任务，
验证记忆图谱的初始化、适配器同步、图注意力、赫布学习等效果。

用法:
  python tests/test_memory_graph_live.py

前置条件:
  - 祖龙系统已在独立终端启动 (python zulong/bootstrap.py)
  - WebSocket 服务器运行在 ws://127.0.0.1:5555
"""

import asyncio
import json
import sys
import os
import time
import logging
from datetime import datetime

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MemoryGraphLiveTest")

try:
    import websockets
except ImportError:
    print("ERROR: websockets 未安装，请运行: pip install websockets")
    sys.exit(1)


# ============================================================
# 测试配置
# ============================================================
WS_URI = "ws://127.0.0.1:5555/eventbus"
RESPONSE_TIMEOUT = 300  # 等待响应超时(秒)
MAX_RESPONSES = 300     # 最大接收消息数


# ============================================================
# 测试任务
# ============================================================
TEST_TASKS = [
    {
        "name": "多步骤分析任务",
        "text": "请帮我分析一下人工智能在教育领域的应用现状，列出3个具体方向，并简要说明每个方向的优势和挑战。",
        "description": "测试任务分解 → TaskGraph 生成 → MemoryGraph 同步 → 图注意力激活",
    },
]


class MemoryGraphLiveTest:
    """记忆图谱实时测试客户端"""

    def __init__(self):
        self.ws = None
        self.responses = []
        self.events_received = {
            "L2_OUTPUT": [],
            "L2_OUTPUT_STREAM": [],
            "L2_THINKING_STEP": [],
            "ACTION_SPEAK": [],
            "ACK": [],
            "OTHER": [],
        }
        self.test_results = {}

    async def connect(self):
        """连接到祖龙 WebSocket 服务器"""
        logger.info(f"连接到 {WS_URI} ...")
        try:
            self.ws = await websockets.connect(
                WS_URI,
                ping_interval=None,
                ping_timeout=None,
                max_size=None,
            )
            logger.info("WebSocket 连接成功")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    async def send_user_speech(self, text: str):
        """发送 USER_SPEECH 事件"""
        event = {
            "type": "PUBLISH",
            "event": {
                "type": "USER_SPEECH",
                "priority": "NORMAL",
                "source": "MemoryGraphTest",
                "payload": {
                    "text": text,
                    "confidence": 1.0,
                    "timestamp": time.time(),
                },
            },
        }
        await self.ws.send(json.dumps(event))
        logger.info(f"已发送 USER_SPEECH: {text[:60]}...")

    async def receive_responses(self, timeout: float = RESPONSE_TIMEOUT):
        """接收并分类响应消息"""
        start = time.time()
        count = 0

        while time.time() - start < timeout and count < MAX_RESPONSES:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
                count += 1
                data = json.loads(raw)
                self.responses.append(data)

                # 分类
                msg_type = data.get("type", "UNKNOWN")
                event = data.get("event", {})
                event_type = event.get("type", msg_type)

                if event_type in self.events_received:
                    self.events_received[event_type].append(data)
                else:
                    self.events_received["OTHER"].append(data)

                # 简要日志
                payload = event.get("payload", {})
                if event_type == "L2_OUTPUT":
                    text = payload.get("text", "")[:200]
                    logger.info(f"  [L2_OUTPUT] {text}")
                elif event_type == "L2_OUTPUT_STREAM":
                    token = payload.get("token", payload.get("text", ""))[:20]
                    # 流式 token 不单独打印，积累后显示
                    pass
                elif event_type == "L2_THINKING_STEP":
                    # 尝试多种可能的字段名
                    step = (payload.get("step", "")
                            or payload.get("text", "")
                            or payload.get("content", "")
                            or str(payload)[:120])
                    if step:
                        logger.info(f"  [THINKING] {step[:120]}")
                elif event_type == "ACTION_SPEAK":
                    text = payload.get("text", "")[:120]
                    logger.info(f"  [SPEAK] {text}")
                elif msg_type == "ACK":
                    logger.info(f"  [ACK] {data.get('message', '')}")
                else:
                    logger.info(f"  [{event_type}] {str(payload)[:120]}")

                # 检测任务完成信号
                if event_type == "L2_OUTPUT":
                    text = payload.get("text", "")
                    if any(kw in text for kw in ["完成", "总结", "以上就是", "希望对你有帮助"]):
                        logger.info("  >> 检测到任务完成信号")
                        # 再等几秒收尾
                        await asyncio.sleep(3)
                        break

            except asyncio.TimeoutError:
                # 10秒无新消息
                elapsed = time.time() - start
                if elapsed > 30:
                    logger.info(f"  超过 {elapsed:.0f}s 无新消息，结束接收")
                    break
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.warning("  WebSocket 连接关闭")
                break

        logger.info(f"共接收 {count} 条消息")

    async def run_task_test(self, task: dict):
        """运行单个测试任务"""
        logger.info(f"\n{'='*60}")
        logger.info(f"测试: {task['name']}")
        logger.info(f"描述: {task['description']}")
        logger.info(f"{'='*60}")

        # 清空之前的响应
        self.responses.clear()
        for k in self.events_received:
            self.events_received[k].clear()

        # 发送任务
        await self.send_user_speech(task["text"])

        # 等待响应
        await self.receive_responses()

        # 分析结果
        result = {
            "task": task["name"],
            "total_messages": len(self.responses),
            "l2_outputs": len(self.events_received["L2_OUTPUT"]),
            "thinking_steps": len(self.events_received["L2_THINKING_STEP"]),
            "action_speaks": len(self.events_received["ACTION_SPEAK"]),
            "stream_tokens": len(self.events_received["L2_OUTPUT_STREAM"]),
            "acks": len(self.events_received["ACK"]),
        }

        # 提取最终回复文本
        final_texts = []
        for msg in self.events_received["L2_OUTPUT"]:
            payload = msg.get("event", {}).get("payload", {})
            text = payload.get("text", "")
            if text:
                final_texts.append(text)
        result["final_text_length"] = sum(len(t) for t in final_texts)
        result["final_texts"] = final_texts

        self.test_results[task["name"]] = result
        return result

    async def check_graph_state(self):
        """检查 MemoryGraph 持久化状态

        尝试从磁盘加载图谱数据（如果系统已保存）
        """
        graph_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "memory_graph",
        )
        logger.info(f"\n检查图谱持久化目录: {graph_path}")

        if not os.path.exists(graph_path):
            logger.info("  目录不存在（图谱尚未持久化到磁盘，但可能在内存中运行正常）")
            return

        files = os.listdir(graph_path)
        logger.info(f"  文件列表: {files}")

        graph_file = os.path.join(graph_path, "graph.json")
        if os.path.exists(graph_file):
            with open(graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            logger.info(f"  节点数: {len(nodes)}")
            logger.info(f"  边数: {len(edges)}")

            if nodes:
                # 按类型统计
                type_counts = {}
                for n in nodes:
                    t = n.get("node_type", "unknown")
                    type_counts[t] = type_counts.get(t, 0) + 1
                logger.info(f"  节点类型分布: {type_counts}")

                # 显示前几个节点
                for n in nodes[:5]:
                    logger.info(f"    - [{n.get('node_type')}] {n.get('node_id')}: {n.get('label')}")

            if edges:
                edge_type_counts = {}
                for e in edges:
                    t = e.get("edge_type", "unknown")
                    edge_type_counts[t] = edge_type_counts.get(t, 0) + 1
                logger.info(f"  边类型分布: {edge_type_counts}")
        else:
            logger.info("  graph.json 不存在（图谱在内存中，尚未持久化）")

    def print_summary(self):
        """打印测试摘要"""
        logger.info(f"\n{'='*60}")
        logger.info("记忆图谱集成测试 - 结果摘要")
        logger.info(f"{'='*60}")

        for name, result in self.test_results.items():
            logger.info(f"\n[{name}]")
            logger.info(f"  总消息数: {result['total_messages']}")
            logger.info(f"  L2 输出: {result['l2_outputs']} 条")
            logger.info(f"  思考步骤: {result['thinking_steps']} 步")
            logger.info(f"  语音动作: {result['action_speaks']} 次")
            logger.info(f"  流式 token: {result['stream_tokens']} 个")
            logger.info(f"  最终回复长度: {result['final_text_length']} 字符")

            if result["final_texts"]:
                logger.info(f"  最终回复预览:")
                for i, text in enumerate(result["final_texts"][:2]):
                    preview = text[:200].replace("\n", " ")
                    logger.info(f"    [{i+1}] {preview}...")

        logger.info(f"\n{'='*60}")
        logger.info("验证清单 (请在祖龙终端日志中确认):")
        logger.info(f"{'='*60}")
        logger.info("  [1] MemoryGraph 初始化: 搜索 'MemoryGraph 初始化完成'")
        logger.info("  [2] 适配器注册: 搜索 '注册适配器'")
        logger.info("  [3] TaskGraph 同步: 搜索 'TaskGraphAdapter' 或 'sync'")
        logger.info("  [4] 图注意力启用: 搜索 'graph_attention' 或 'use_graph_attention'")
        logger.info("  [5] 注意力窗口: 搜索 'AttentionWindow' 初始化日志中 graph_attention=True")
        logger.info("  [6] 降级运行: 如果有错误，搜索 'MemoryGraph' 相关 warning")


async def main():
    """主测试流程"""
    logger.info("=" * 60)
    logger.info("记忆图谱 (MemoryGraph) 实时集成测试")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    tester = MemoryGraphLiveTest()

    # 1. 连接
    if not await tester.connect():
        logger.error("无法连接到祖龙 WebSocket 服务器，请确认系统已启动")
        return

    try:
        # 2. 运行测试任务
        for task in TEST_TASKS:
            result = await tester.run_task_test(task)

            # 任务间间隔
            if task != TEST_TASKS[-1]:
                logger.info("等待 5 秒后执行下一个任务...")
                await asyncio.sleep(5)

        # 3. 检查图谱持久化
        await tester.check_graph_state()

        # 4. 打印摘要
        tester.print_summary()

    finally:
        if tester.ws:
            await tester.ws.close()
            logger.info("WebSocket 连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
