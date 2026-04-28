"""
祖龙图式记忆架构 - 实时集成测试
=============================
通过 WebSocket 连接到运行中的祖龙系统，验证：
  1. 短期记忆（同一会话内上下文记忆）
  2. 长期记忆（知识沉淀与持久化）
  3. 跨会话记忆（新会话回忆旧会话信息）
  4. 复杂任务中断恢复
  5. 复杂任务结束后访问和修改

使用方法：
  1. 确保祖龙系统和 OpenClaw Bridge 均已启动
  2. python tests/test_memory_architecture_live.py
"""

import asyncio
import json
import uuid
import time
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# websockets 库
try:
    import websockets
except ImportError:
    print("[ERROR] 缺少 websockets 库，请运行: pip install websockets")
    sys.exit(1)

# ============================================================
# 配置
# ============================================================
WS_URL = "ws://localhost:8080/ws"
HTTP_URL = "http://localhost:8080"
RESPONSE_TIMEOUT = 120   # 等待响应最大秒数（简单任务）
COMPLEX_RESPONSE_TIMEOUT = 240  # 复杂任务（文档生成等）等待秒数
INTER_MSG_DELAY = 3      # 消息间隔秒数（避免过载）

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"tests/test_memory_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("MemoryTest")


# ============================================================
# 数据结构
# ============================================================
class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    detail: str = ""
    response_text: str = ""
    elapsed_sec: float = 0.0
    memory_graph_nodes: int = 0
    memory_graph_edges: int = 0


@dataclass
class CollectedResponse:
    """收集 WebSocket 返回的所有消息"""
    final_text: str = ""
    chunks: List[str] = field(default_factory=list)
    thinking_steps: List[dict] = field(default_factory=list)
    memory_graph: Optional[dict] = None
    raw_messages: List[dict] = field(default_factory=list)
    completed: bool = False


# ============================================================
# WebSocket 助手
# ============================================================
class ZulongWSClient:
    """祖龙 WebSocket 测试客户端"""

    def __init__(self, url: str = WS_URL):
        self.url = url
        self.ws = None
        self._session_id = str(uuid.uuid4())
        self._request_counter = 0

    async def connect(self) -> bool:
        """建立连接"""
        try:
            self.ws = await websockets.connect(self.url, max_size=10 * 1024 * 1024)
            # 等待 WELCOME 消息
            welcome = await asyncio.wait_for(self.ws.recv(), timeout=10)
            data = json.loads(welcome)
            if data.get("type") == "WELCOME":
                logger.info(f"[WS] 已连接: {data.get('message', '')}")
            # 可能还会收到 MEMORY_GRAPH_UPDATE 快照
            try:
                snap = await asyncio.wait_for(self.ws.recv(), timeout=5)
                snap_data = json.loads(snap)
                if snap_data.get("type") == "MEMORY_GRAPH_UPDATE":
                    n = len(snap_data.get("nodes", []))
                    e = len(snap_data.get("edges", []))
                    logger.info(f"[WS] 收到记忆图谱快照: {n} 节点, {e} 边")
            except asyncio.TimeoutError:
                pass
            return True
        except Exception as e:
            logger.error(f"[WS] 连接失败: {e}")
            return False

    async def close(self):
        """关闭连接"""
        if self.ws:
            await self.ws.close()
            self.ws = None

    @property
    def session_id(self) -> str:
        return self._session_id

    def new_session(self) -> str:
        """切换到新会话"""
        self._session_id = str(uuid.uuid4())
        logger.info(f"[WS] 新会话: {self._session_id[:8]}...")
        return self._session_id

    async def send_message(
        self, text: str, referenced_nodes: list = None, timeout: int = None
    ) -> CollectedResponse:
        """
        发送消息并收集所有响应

        Args:
            text: 消息文本
            referenced_nodes: 引用节点
            timeout: 自定义超时秒数（默认使用 RESPONSE_TIMEOUT）

        Returns:
            CollectedResponse 包含最终文本、思考步骤、记忆图谱等
        """
        response_timeout = timeout or RESPONSE_TIMEOUT
        self._request_counter += 1
        request_id = f"test-req-{self._request_counter}-{uuid.uuid4().hex[:8]}"

        msg = {
            "type": "CHAT_MESSAGE",
            "text": text,
            "session_id": self._session_id,
            "request_id": request_id,
        }
        if referenced_nodes:
            msg["referenced_nodes"] = referenced_nodes

        logger.info(f"[TX] >>> {text[:80]}...")
        await self.ws.send(json.dumps(msg))

        # 收集响应
        collected = CollectedResponse()
        start = time.time()
        last_activity = time.time()

        while time.time() - start < response_timeout:
            try:
                raw = await asyncio.wait_for(
                    self.ws.recv(), timeout=min(30, response_timeout - (time.time() - start))
                )
                data = json.loads(raw)
                collected.raw_messages.append(data)
                msg_type = data.get("type", "")
                last_activity = time.time()

                if msg_type == "STREAMING_RESPONSE":
                    chunk = data.get("chunk", "")
                    if chunk:
                        collected.chunks.append(chunk)
                    collected.final_text = data.get("text", collected.final_text)

                elif msg_type == "CHAT_RESPONSE":
                    collected.final_text = data.get("text", "")
                    collected.completed = True

                elif msg_type == "THINKING_STEP":
                    step = {
                        "step_number": data.get("step_number"),
                        "description": data.get("description", ""),
                    }
                    collected.thinking_steps.append(step)
                    logger.info(
                        f"  [思考] 步骤{step['step_number']}: {step['description'][:60]}..."
                    )

                elif msg_type == "MEMORY_GRAPH_UPDATE":
                    collected.memory_graph = data
                    n = len(data.get("nodes", []))
                    e = len(data.get("edges", []))
                    logger.info(f"  [记忆图谱] 更新: {n} 节点, {e} 边")

                # 如果收到 CHAT_RESPONSE 就算完成
                if collected.completed:
                    # 额外等 3 秒看看有没有记忆图谱更新
                    extra_deadline = time.time() + 3
                    while time.time() < extra_deadline:
                        try:
                            extra = await asyncio.wait_for(self.ws.recv(), timeout=2)
                            edata = json.loads(extra)
                            collected.raw_messages.append(edata)
                            if edata.get("type") == "MEMORY_GRAPH_UPDATE":
                                collected.memory_graph = edata
                                n = len(edata.get("nodes", []))
                                e = len(edata.get("edges", []))
                                logger.info(f"  [记忆图谱] 后续更新: {n} 节点, {e} 边")
                        except asyncio.TimeoutError:
                            break
                    break

            except asyncio.TimeoutError:
                # 如果有流式文本且距上次活动超过 15 秒，视为流式完成
                idle = time.time() - last_activity
                if collected.final_text and idle > 12:
                    logger.warning(f"[WS] 流式响应静默 {idle:.0f}s，视为完成")
                    collected.completed = True
                    break
                if not collected.final_text and idle > 25:
                    logger.warning("[WS] 无响应超时")
                    break
                continue

        elapsed = time.time() - start
        
        # 兜底：如果有流式块但 final_text 为空，用 chunks 拼接
        if not collected.final_text and collected.chunks:
            collected.final_text = "".join(collected.chunks)
            logger.info(f"[WS] 使用 {len(collected.chunks)} 个流式块拼接最终文本 (len={len(collected.final_text)})")
        
        text_preview = collected.final_text[:120] if collected.final_text else "(无响应)"
        logger.info(f"[RX] <<< [{elapsed:.1f}s] {text_preview}...")

        return collected


# ============================================================
# 测试用例
# ============================================================
async def test_short_term_memory(client: ZulongWSClient) -> List[TestResult]:
    """
    测试 1: 短期记忆（同一会话内上下文记忆）
    
    验证点:
    - 系统能记住本轮对话早期提到的信息
    - 系统能在后续问题中引用之前的上下文
    """
    results = []
    logger.info("\n" + "=" * 70)
    logger.info("测试 1: 短期记忆 - 同一会话内上下文记忆")
    logger.info("=" * 70)

    client.new_session()

    # 1-1: 建立基础事实
    t0 = time.time()
    resp1 = await client.send_message(
        "你好，我叫张明，我在一家叫做「星辰科技」的公司担任首席架构师，我们的主要产品是一个叫做 CloudX 的云计算平台。请记住这些信息。"
    )
    r = TestResult(
        name="1-1 建立基础事实",
        status=TestStatus.PASS if resp1.final_text else TestStatus.FAIL,
        detail="向系统介绍个人信息",
        response_text=resp1.final_text[:200],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value} ({r.elapsed_sec:.1f}s)")

    await asyncio.sleep(INTER_MSG_DELAY)

    # 1-2: 验证即时记忆
    t0 = time.time()
    resp2 = await client.send_message("请问我的名字叫什么？我在哪家公司工作？")
    has_name = "张明" in resp2.final_text
    has_company = "星辰" in resp2.final_text
    r = TestResult(
        name="1-2 验证即时记忆",
        status=TestStatus.PASS if (has_name and has_company) else TestStatus.FAIL,
        detail=f"提到张明={has_name}, 提到星辰={has_company}",
        response_text=resp2.final_text[:200],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    await asyncio.sleep(INTER_MSG_DELAY)

    # 1-3: 追加信息后，验证上下文连贯性
    t0 = time.time()
    resp3 = await client.send_message(
        "我们的 CloudX 平台最近在做一次重大升级，要从微服务架构迁移到服务网格 (Service Mesh) 架构。预计需要3个月完成。"
    )
    await asyncio.sleep(INTER_MSG_DELAY)

    resp4 = await client.send_message(
        "请帮我总结一下我之前告诉你的所有信息，包括我的个人信息和项目情况。"
    )
    has_name2 = "张明" in resp4.final_text
    has_product = "CloudX" in resp4.final_text or "云" in resp4.final_text
    has_upgrade = "服务网格" in resp4.final_text or "Service Mesh" in resp4.final_text or "升级" in resp4.final_text
    all_ok = has_name2 and has_product and has_upgrade
    r = TestResult(
        name="1-3 上下文连贯性总结",
        status=TestStatus.PASS if all_ok else (TestStatus.WARN if (has_name2 or has_product) else TestStatus.FAIL),
        detail=f"张明={has_name2}, CloudX/云={has_product}, 升级={has_upgrade}",
        response_text=resp4.final_text[:300],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    # 检查记忆图谱
    if resp4.memory_graph:
        nodes = len(resp4.memory_graph.get("nodes", []))
        edges = len(resp4.memory_graph.get("edges", []))
        r.memory_graph_nodes = nodes
        r.memory_graph_edges = edges
        logger.info(f"  记忆图谱: {nodes} 节点, {edges} 边")

    return results


async def test_long_term_memory(client: ZulongWSClient) -> List[TestResult]:
    """
    测试 2: 长期记忆（知识沉淀）
    
    验证点:
    - 系统会将重要信息沉淀到记忆图谱
    - 记忆图谱中应产生 KNOWLEDGE/PERSON/CONCEPT 节点
    - 应有 SEMANTIC 和 REFERENCE 边
    """
    results = []
    logger.info("\n" + "=" * 70)
    logger.info("测试 2: 长期记忆 - 知识沉淀与图谱构建")
    logger.info("=" * 70)

    # 沿用当前会话，继续丰富知识
    t0 = time.time()
    resp1 = await client.send_message(
        "关于我们 CloudX 的技术栈，我补充一下：后端使用 Go 语言和 gRPC，前端用 React + TypeScript，"
        "数据库用 PostgreSQL 和 Redis，消息队列用 Kafka，容器编排用 Kubernetes。"
        "我们团队有 15 名工程师，其中 5 名是后端，4 名前端，3 名 SRE，3 名数据工程师。"
    )
    r = TestResult(
        name="2-1 丰富技术知识",
        status=TestStatus.PASS if resp1.final_text else TestStatus.FAIL,
        detail="灌入技术栈和团队信息",
        response_text=resp1.final_text[:200],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value}")

    await asyncio.sleep(INTER_MSG_DELAY)

    # 2-2: 验证记忆图谱是否有节点
    t0 = time.time()
    resp2 = await client.send_message(
        "请你查看一下你的记忆图谱，告诉我你目前记住了关于我和我项目的哪些关键信息？"
    )
    # 检查回复中是否提到关键信息
    keywords = ["张明", "CloudX", "星辰", "Go", "Kubernetes", "架构"]
    matches = [kw for kw in keywords if kw.lower() in resp2.final_text.lower()]
    r = TestResult(
        name="2-2 知识沉淀验证",
        status=TestStatus.PASS if len(matches) >= 3 else (TestStatus.WARN if len(matches) >= 1 else TestStatus.FAIL),
        detail=f"命中关键词: {matches} ({len(matches)}/{len(keywords)})",
        response_text=resp2.final_text[:300],
        elapsed_sec=time.time() - t0,
    )
    if resp2.memory_graph:
        r.memory_graph_nodes = len(resp2.memory_graph.get("nodes", []))
        r.memory_graph_edges = len(resp2.memory_graph.get("edges", []))
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    await asyncio.sleep(INTER_MSG_DELAY)

    # 2-3: 检查图谱边类型
    if resp2.memory_graph:
        edges = resp2.memory_graph.get("edges", [])
        edge_types = set()
        for e in edges:
            edge_types.add(e.get("type", "unknown").lower())
        has_semantic = "semantic" in edge_types
        has_reference = "reference" in edge_types
        has_hierarchy = "hierarchy" in edge_types
        r = TestResult(
            name="2-3 图谱边类型检查",
            status=TestStatus.PASS if (has_hierarchy and (has_semantic or has_reference)) else TestStatus.WARN,
            detail=f"边类型: {edge_types}, SEMANTIC={has_semantic}, REFERENCE={has_reference}, HIERARCHY={has_hierarchy}",
            elapsed_sec=0,
        )
        results.append(r)
        logger.info(f"  结果: {r.status.value} - {r.detail}")
    else:
        results.append(TestResult(
            name="2-3 图谱边类型检查",
            status=TestStatus.WARN,
            detail="未收到 MEMORY_GRAPH_UPDATE，无法检查边类型",
        ))
        logger.info("  结果: WARN - 未收到图谱更新")

    return results


async def test_cross_session_memory(client: ZulongWSClient) -> List[TestResult]:
    """
    测试 3: 跨会话记忆
    
    验证点:
    - 切换到新会话后，系统仍能回忆之前会话的关键信息
    - 跨会话检索路径（FAISS summary sidecar）正常工作
    """
    results = []
    logger.info("\n" + "=" * 70)
    logger.info("测试 3: 跨会话记忆 - 新会话回忆旧会话信息")
    logger.info("=" * 70)

    old_session = client.session_id
    logger.info(f"  旧会话: {old_session[:8]}...")

    # 切换到新会话
    new_sid = client.new_session()
    logger.info(f"  新会话: {new_sid[:8]}...")

    # 需要重新连接以确保干净的新会话
    await client.close()
    await asyncio.sleep(2)
    connected = await client.connect()
    if not connected:
        results.append(TestResult(
            name="3-0 重新连接",
            status=TestStatus.FAIL,
            detail="无法重新连接 WebSocket",
        ))
        return results

    await asyncio.sleep(INTER_MSG_DELAY)

    # 3-1: 在新会话中询问上一个会话的信息
    t0 = time.time()
    resp1 = await client.send_message(
        "你好，我是之前跟你聊过的用户。请问你还记得我叫什么名字吗？我在哪个公司工作？我们的产品叫什么？"
    )
    has_name = "张明" in resp1.final_text
    has_company = "星辰" in resp1.final_text
    has_product = "CloudX" in resp1.final_text or "云" in resp1.final_text
    score = sum([has_name, has_company, has_product])
    r = TestResult(
        name="3-1 跨会话身份回忆",
        status=TestStatus.PASS if score >= 2 else (TestStatus.WARN if score >= 1 else TestStatus.FAIL),
        detail=f"张明={has_name}, 星辰={has_company}, CloudX={has_product}",
        response_text=resp1.final_text[:300],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    await asyncio.sleep(INTER_MSG_DELAY)

    # 3-2: 询问技术细节（更深层的长期记忆）
    t0 = time.time()
    resp2 = await client.send_message(
        "请回忆一下我之前告诉你的技术栈信息，比如我们用什么编程语言、什么数据库？"
    )
    tech_keywords = ["Go", "gRPC", "React", "TypeScript", "PostgreSQL", "Redis", "Kafka", "Kubernetes"]
    tech_matches = [kw for kw in tech_keywords if kw.lower() in resp2.final_text.lower()]
    r = TestResult(
        name="3-2 跨会话技术细节回忆",
        status=TestStatus.PASS if len(tech_matches) >= 3 else (TestStatus.WARN if len(tech_matches) >= 1 else TestStatus.FAIL),
        detail=f"命中: {tech_matches} ({len(tech_matches)}/{len(tech_keywords)})",
        response_text=resp2.final_text[:300],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    return results


async def test_complex_task_interrupt_resume(client: ZulongWSClient) -> List[TestResult]:
    """
    测试 4: 复杂任务中断与恢复
    
    验证点:
    - 发起复杂任务后系统开始执行
    - 发送"暂停"/"停下来"后任务被挂起
    - 发送"继续之前的任务"后任务恢复
    - 恢复后上下文完整
    """
    results = []
    logger.info("\n" + "=" * 70)
    logger.info("测试 4: 复杂任务中断与恢复")
    logger.info("=" * 70)

    client.new_session()
    await asyncio.sleep(INTER_MSG_DELAY)

    # 4-1: 发起复杂任务（使用更长的超时）
    t0 = time.time()
    resp1 = await client.send_message(
        "请帮我写一份关于微服务架构迁移到服务网格架构的技术方案，要求包含以下章节：\n"
        "1. 背景与现状分析\n"
        "2. 目标架构设计\n"
        "3. 迁移步骤规划\n"
        "4. 风险评估与应对\n"
        "5. 资源与时间估算\n"
        "请详细撰写每个章节。",
        timeout=COMPLEX_RESPONSE_TIMEOUT,
    )
    task_started = len(resp1.final_text) > 100
    r = TestResult(
        name="4-1 发起复杂任务",
        status=TestStatus.PASS if task_started else TestStatus.FAIL,
        detail=f"响应长度={len(resp1.final_text)}",
        response_text=resp1.final_text[:200],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    await asyncio.sleep(INTER_MSG_DELAY)

    # 4-2: 尝试中断/暂停
    t0 = time.time()
    resp2 = await client.send_message(
        "先暂停一下这个任务，我有个紧急问题要问：Python 的 asyncio.gather 和 asyncio.wait 有什么区别？"
    )
    # 检查是否回答了临时问题
    has_asyncio = "asyncio" in resp2.final_text.lower() or "gather" in resp2.final_text.lower() or "wait" in resp2.final_text.lower()
    r = TestResult(
        name="4-2 任务中断（插入问题）",
        status=TestStatus.PASS if has_asyncio else TestStatus.WARN,
        detail=f"回答了 asyncio 问题={has_asyncio}",
        response_text=resp2.final_text[:200],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    await asyncio.sleep(INTER_MSG_DELAY)

    # 4-3: 恢复任务（使用更长超时，因为需要继续生成文档）
    t0 = time.time()
    resp3 = await client.send_message(
        "好，asyncio 的问题解决了。请继续之前那个微服务迁移方案，从你上次写到的地方接着写。",
        timeout=COMPLEX_RESPONSE_TIMEOUT,
    )
    # 检查是否继续了之前的任务内容
    task_keywords = ["微服务", "服务网格", "Service Mesh", "迁移", "架构", "方案"]
    task_matches = [kw for kw in task_keywords if kw in resp3.final_text]
    r = TestResult(
        name="4-3 恢复中断的任务",
        status=TestStatus.PASS if len(task_matches) >= 2 else TestStatus.WARN,
        detail=f"恢复关键词: {task_matches}",
        response_text=resp3.final_text[:300],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    return results


async def test_post_task_access_modify(client: ZulongWSClient) -> List[TestResult]:
    """
    测试 5: 复杂任务结束后访问和修改
    
    验证点:
    - 任务完成后能够回顾其内容
    - 能够对已完成的任务提出修改意见
    - 修改后的结果保留原始上下文
    """
    results = []
    logger.info("\n" + "=" * 70)
    logger.info("测试 5: 复杂任务结束后访问与修改")
    logger.info("=" * 70)

    # 继续当前会话（有之前的技术方案）
    await asyncio.sleep(INTER_MSG_DELAY)

    # 5-1: 回顾已完成任务（使用更长超时）
    t0 = time.time()
    resp1 = await client.send_message(
        "我之前让你写的那个微服务迁移方案，请帮我回顾一下你都写了哪些章节？每个章节的核心要点是什么？",
        timeout=COMPLEX_RESPONSE_TIMEOUT,
    )
    has_review = any(kw in resp1.final_text for kw in ["微服务", "迁移", "章节", "架构", "服务网格"])
    r = TestResult(
        name="5-1 回顾已完成任务",
        status=TestStatus.PASS if has_review else TestStatus.FAIL,
        detail=f"提到方案内容={has_review}",
        response_text=resp1.final_text[:300],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    await asyncio.sleep(INTER_MSG_DELAY)

    # 5-2: 请求修改（使用更长超时）
    t0 = time.time()
    resp2 = await client.send_message(
        "请帮我修改那个方案的「风险评估」部分，增加以下风险点：\n"
        "1. 网络延迟问题：Service Mesh 会引入额外的 sidecar proxy 延迟\n"
        "2. 可观测性成本：需要额外的监控和日志基础设施\n"
        "3. 团队学习曲线：工程师需要学习 Istio/Envoy\n"
        "请输出修改后的完整风险评估章节。",
        timeout=COMPLEX_RESPONSE_TIMEOUT,
    )
    modify_keywords = ["延迟", "sidecar", "proxy", "可观测", "监控", "学习曲线", "Istio", "Envoy"]
    modify_matches = [kw for kw in modify_keywords if kw.lower() in resp2.final_text.lower()]
    r = TestResult(
        name="5-2 修改已完成任务",
        status=TestStatus.PASS if len(modify_matches) >= 3 else (TestStatus.WARN if len(modify_matches) >= 1 else TestStatus.FAIL),
        detail=f"修改命中: {modify_matches}",
        response_text=resp2.final_text[:300],
        elapsed_sec=time.time() - t0,
    )
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    await asyncio.sleep(INTER_MSG_DELAY)

    # 5-3: 验证修改后仍保持完整上下文（使用更长超时）
    t0 = time.time()
    resp3 = await client.send_message(
        "现在请帮我总结一下我们这次对话的完整内容：我的个人信息、项目信息、我们讨论的技术方案以及修改的部分。",
        timeout=COMPLEX_RESPONSE_TIMEOUT,
    )
    summary_kw = ["张明", "星辰", "CloudX", "微服务", "服务网格", "风险"]
    summary_matches = [kw for kw in summary_kw if kw in resp3.final_text]
    r = TestResult(
        name="5-3 全局上下文完整性",
        status=TestStatus.PASS if len(summary_matches) >= 4 else (TestStatus.WARN if len(summary_matches) >= 2 else TestStatus.FAIL),
        detail=f"总结命中: {summary_matches} ({len(summary_matches)}/{len(summary_kw)})",
        response_text=resp3.final_text[:400],
        elapsed_sec=time.time() - t0,
    )
    if resp3.memory_graph:
        r.memory_graph_nodes = len(resp3.memory_graph.get("nodes", []))
        r.memory_graph_edges = len(resp3.memory_graph.get("edges", []))
    results.append(r)
    logger.info(f"  结果: {r.status.value} - {r.detail}")

    return results


# ============================================================
# 报告生成
# ============================================================
def generate_report(all_results: Dict[str, List[TestResult]]) -> str:
    """生成测试报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("  祖龙图式记忆架构 - 实时集成测试报告")
    lines.append(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    total = 0
    passed = 0
    warned = 0
    failed = 0

    for suite_name, results in all_results.items():
        lines.append(f"\n--- {suite_name} ---")
        for r in results:
            total += 1
            icon = {"PASS": "[OK]", "FAIL": "[XX]", "WARN": "[!!]", "SKIP": "[--]"}.get(r.status.value, "[??]")
            if r.status == TestStatus.PASS:
                passed += 1
            elif r.status == TestStatus.WARN:
                warned += 1
            elif r.status == TestStatus.FAIL:
                failed += 1

            lines.append(f"  {icon} {r.name}")
            lines.append(f"       {r.detail}")
            if r.elapsed_sec > 0:
                lines.append(f"       耗时: {r.elapsed_sec:.1f}s")
            if r.memory_graph_nodes:
                lines.append(f"       图谱: {r.memory_graph_nodes} 节点, {r.memory_graph_edges} 边")
            if r.response_text:
                # 打印摘要（前100字）
                preview = r.response_text.replace("\n", " ")[:100]
                lines.append(f"       响应: {preview}...")

    lines.append(f"\n{'=' * 70}")
    lines.append(f"  总计: {total} 项")
    lines.append(f"  通过: {passed} 项")
    lines.append(f"  警告: {warned} 项")
    lines.append(f"  失败: {failed} 项")
    lines.append(f"  通过率: {passed/total*100:.0f}%" if total else "  无测试执行")
    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================
async def main():
    logger.info("祖龙图式记忆架构 - 实时集成测试")
    logger.info(f"目标: {WS_URL}")

    client = ZulongWSClient(WS_URL)

    # 连接
    connected = await client.connect()
    if not connected:
        logger.error("无法连接到祖龙系统，测试终止")
        sys.exit(1)

    all_results = {}

    try:
        # 测试 1: 短期记忆
        results = await test_short_term_memory(client)
        all_results["测试1: 短期记忆（同会话上下文）"] = results

        # 测试 2: 长期记忆
        results = await test_long_term_memory(client)
        all_results["测试2: 长期记忆（知识沉淀/图谱构建）"] = results

        # 测试 3: 跨会话记忆（这里会重新连接）
        results = await test_cross_session_memory(client)
        all_results["测试3: 跨会话记忆"] = results

        # 测试 4: 复杂任务中断与恢复
        results = await test_complex_task_interrupt_resume(client)
        all_results["测试4: 复杂任务中断与恢复"] = results

        # 测试 5: 复杂任务结束后访问与修改
        results = await test_post_task_access_modify(client)
        all_results["测试5: 任务完成后访问与修改"] = results

    except Exception as e:
        logger.error(f"测试过程异常: {e}", exc_info=True)
    finally:
        await client.close()

    # 生成报告
    report = generate_report(all_results)
    logger.info("\n" + report)

    # 保存报告
    report_file = f"tests/memory_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
        # 追加详细响应
        f.write("\n\n" + "=" * 70)
        f.write("\n  详细响应记录\n")
        f.write("=" * 70 + "\n")
        for suite_name, results in all_results.items():
            f.write(f"\n### {suite_name}\n")
            for r in results:
                f.write(f"\n[{r.name}]\n")
                f.write(f"状态: {r.status.value}\n")
                f.write(f"详情: {r.detail}\n")
                f.write(f"响应全文:\n{r.response_text}\n")
                f.write("-" * 40 + "\n")

    logger.info(f"\n报告已保存到: {report_file}")

    # 返回退出码
    total_fail = sum(
        1 for rs in all_results.values() for r in rs if r.status == TestStatus.FAIL
    )
    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
