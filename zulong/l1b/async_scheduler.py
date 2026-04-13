# File: zulong/l1b/async_scheduler.py
# L1-B 异步非阻塞调度器 (TSD v2.2)
# 实现主动记忆注入、并行检索、上下文组装

import asyncio
import json
import logging
import queue
import re
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent
from zulong.memory.three_libraries import (
    get_three_library_manager,
    ThreeLibraryManager
)
from zulong.memory.tagging_engine import TaggingEngine
from zulong.storage.hot_storage import HotStorage
from zulong.memory.short_term_memory import get_short_term_memory, ShortTermMemory

logger = logging.getLogger(__name__)


@dataclass
class TaskItem:
    """任务项"""
    task_id: str
    prompt: str
    status: str  # "PENDING", "READY", "EXECUTING", "COMPLETED"
    raw_input: str
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    ready_at: Optional[float] = None


class AsyncL1BScheduler:
    """L1-B 异步非阻塞调度器
    
    特点：
    - 异步非阻塞：主线程不等待检索完成
    - 主动注入：在 L2 加载前完成上下文组装
    - 并行检索：同时查询技能库、经验库、知识库
    - 硬编码优先：使用 Python 原生代码，不调用大模型
    
    对应 TSD v2.2 第 9.4 节
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化异步调度器"""
        if not hasattr(self, '_initialized'):
            self._initialized = False
            self._running = False
            
            self.task_queue: queue.Queue = queue.Queue()
            self.ready_tasks: Dict[str, TaskItem] = {}
            self.current_task: Optional[TaskItem] = None
            
            self.library_manager: Optional[ThreeLibraryManager] = None
            self.tagging_engine: Optional[TaggingEngine] = None
            self.hot_storage: Optional[HotStorage] = None
            self.short_term_memory: Optional[ShortTermMemory] = None
            self._executor = ThreadPoolExecutor(max_workers=4)
            self._loop: Optional[asyncio.AbstractEventLoop] = None
            self._thread: Optional[threading.Thread] = None
            
            self._interrupt_handlers: List[Callable] = []
            self._query_rewrite_rules: Dict[str, str] = {}
            
            self._load_query_rewrite_rules()
            
            logger.info("[AsyncL1BScheduler] 初始化完成")
    
    def initialize(self):
        """初始化组件"""
        if self._initialized:
            return
        
        self.library_manager = get_three_library_manager()
        self.tagging_engine = TaggingEngine()
        self.hot_storage = HotStorage()
        self.short_term_memory = get_short_term_memory()  # 🔥 新增：短期记忆
        self._initialized = True
        logger.info("[AsyncL1BScheduler] 组件初始化完成 (LibraryManager, TaggingEngine, HotStorage, ShortTermMemory)")
    
    def start(self):
        """启动调度器"""
        if self._running:
            return
        
        self.initialize()
        self._running = True
        
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        
        logger.info("[AsyncL1BScheduler] 调度器已启动")
    
    def stop(self):
        """停止调度器"""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._executor.shutdown(wait=False)
        logger.info("[AsyncL1BScheduler] 调度器已停止")
    
    def _run_event_loop(self):
        """运行事件循环"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._main_loop())
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.close()
    
    async def _main_loop(self):
        """主循环"""
        while self._running:
            try:
                await asyncio.sleep(0.01)
                self._check_for_emergency_interrupts()
            except Exception as e:
                logger.error(f"[AsyncL1BScheduler] 主循环错误: {e}")
    
    def _load_query_rewrite_rules(self):
        """加载查询重写规则"""
        self._query_rewrite_rules = {
            "切蛋糕": "任务分片",
            "像切蛋糕": "任务分片",
            "蛋糕分": "任务分片",
            "一刀切": "统一处理",
            "滚雪球": "迭代积累",
            "搭积木": "模块化构建",
            "拼图": "组合集成",
            "剥洋葱": "逐层深入",
            "走迷宫": "路径搜索",
            "下棋": "策略规划",
            "打太极": "柔性处理",
            "拔河": "资源竞争",
            "接力棒": "任务传递",
            "多米诺": "连锁反应",
            "蝴蝶效应": "敏感依赖"
        }
    
    def rewrite_query(self, raw_input: str) -> str:
        """查询重写：将口语转化为技术术语
        
        Args:
            raw_input: 原始输入
            
        Returns:
            str: 重写后的技术术语
        """
        tech_query = raw_input
        for colloquial, technical in self._query_rewrite_rules.items():
            if colloquial in raw_input:
                tech_query = tech_query.replace(colloquial, technical)
                logger.info(f"[AsyncL1BScheduler] 查询重写: '{colloquial}' -> '{technical}'")
        return tech_query
    
    def handle_request(self, user_input: str, 
                       context: Optional[Dict] = None) -> str:
        """主入口：接收用户请求（非阻塞）
        
        Args:
            user_input: 用户输入
            context: 附加上下文
            
        Returns:
            str: 任务 ID
        """
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        task = TaskItem(
            task_id=task_id,
            prompt="",
            status="PENDING",
            raw_input=user_input,
            context=context or {}
        )
        
        self.task_queue.put(task)
        
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._retrieve_and_inject(task),
                self._loop
            )
        else:
            asyncio.run(self._retrieve_and_inject(task))
        
        logger.info(f"[AsyncL1BScheduler] 任务已入队: {task_id}")
        return task_id
    
    def handle_request_async(self, user_input: str,
                              context: Optional[Dict] = None) -> str:
        """异步入口：接收用户请求（用于 async 环境）
        
        Args:
            user_input: 用户输入
            context: 附加上下文
            
        Returns:
            str: 任务 ID
        """
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        task = TaskItem(
            task_id=task_id,
            prompt="",
            status="PENDING",
            raw_input=user_input,
            context=context or {}
        )
        
        self.task_queue.put(task)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._retrieve_and_inject(task))
            else:
                asyncio.run(self._retrieve_and_inject(task))
        except RuntimeError:
            asyncio.run(self._retrieve_and_inject(task))
        
        logger.info(f"[AsyncL1BScheduler] 任务已入队(异步): {task_id}")
        return task_id
    
    async def _retrieve_and_inject(self, task: TaskItem):
        """后台任务：主动记忆注入
        
        Args:
            task: 任务项
        """
        start_time = time.time()
        
        try:
            tech_query = self.rewrite_query(task.raw_input)
            
            messages = task.context.get("messages", [])
            if messages:
                cleaned_messages = self._clean_messages(messages)
                task.context["cleaned_messages"] = cleaned_messages
            
            if self.library_manager is None:
                self.initialize()
            
            results = await self.library_manager.retrieve_all(
                query=tech_query,
                experience_type="logic",
                experience_limit=5,
                knowledge_limit=10
            )
            
            # 构建记忆部分（技能、经验、知识）
            memory_prompt = self.library_manager.build_super_prompt(
                query=task.raw_input,
                skills=results.get("skills"),
                experiences=results.get("experiences"),
                knowledge=results.get("knowledge")
            )
            
            # 构建对话上下文（包含系统指令）
            context_prompt = self._build_context_from_messages(messages)
            
            # 合并：系统指令 + 对话历史 + 记忆内容
            # 系统指令必须在最前面，覆盖记忆中的任何指令
            final_prompt = context_prompt + "\n\n" + memory_prompt
            
            task.prompt = final_prompt
            task.status = "READY"
            task.ready_at = time.time()
            
            self.ready_tasks[task.task_id] = task
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"[AsyncL1BScheduler] 任务就绪: {task.task_id}, 耗时: {elapsed:.1f}ms")
            
            event_bus.publish(ZulongEvent(
                type=EventType.SYSTEM_L2_COMMAND,
                priority=EventPriority.NORMAL,
                source="L1-B-Async",
                payload={
                    "task_id": task.task_id,
                    "prompt": final_prompt,
                    "status": "READY",
                    "retrieval_time_ms": elapsed
                }
            ))
            
        except Exception as e:
            logger.error(f"[AsyncL1BScheduler] 检索失败: {e}")
            task.status = "FAILED"
            task.prompt = task.raw_input
    
    def _clean_messages(self, messages: List[Dict]) -> List[Dict]:
        """清洗消息数组
        
        提取关键内容，过滤垃圾数据
        
        Args:
            messages: 原始消息数组
            
        Returns:
            List[Dict]: 清洗后的消息数组
        """
        cleaned = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if not content or not content.strip():
                continue
            
            if role == "system":
                cleaned.append({
                    "role": "system",
                    "content": content.strip(),
                    "type": "instruction"
                })
            elif role == "user":
                extracted = self._extract_user_intent(content)
                cleaned.append({
                    "role": "user",
                    "content": content.strip(),
                    "intent": extracted.get("intent", "unknown"),
                    "entities": extracted.get("entities", [])
                })
            elif role == "assistant":
                structured = self._extract_assistant_response(content)
                cleaned.append({
                    "role": "assistant",
                    "content": content.strip(),
                    "action": structured.get("action"),
                    "data": structured.get("data")
                })
        
        return cleaned
    
    def _extract_user_intent(self, content: str) -> Dict:
        """提取用户意图
        
        Args:
            content: 用户消息内容
            
        Returns:
            Dict: 提取的意图和实体
        """
        result = {
            "intent": "unknown",
            "entities": []
        }
        
        content_lower = content.lower()
        
        intent_patterns = {
            "query": ["是什么", "怎么", "如何", "为什么", "查询", "搜索", "找"],
            "command": ["执行", "运行", "启动", "停止", "打开", "关闭"],
            "upload": ["上传", "发送", "提交", "导入"],
            "confirm": ["确认", "是的", "好的", "可以", "确定"],
            "cancel": ["取消", "不要", "不行", "拒绝"]
        }
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in content_lower:
                    result["intent"] = intent
                    break
            if result["intent"] != "unknown":
                break
        
        import re
        numbers = re.findall(r'\d+', content)
        if numbers:
            result["entities"].extend([{"type": "number", "value": n} for n in numbers])
        
        return result
    
    def _extract_assistant_response(self, content: str) -> Dict:
        """提取助手响应中的结构化数据
        
        Args:
            content: 助手响应内容
            
        Returns:
            Dict: 提取的结构化数据
        """
        result = {
            "action": None,
            "data": None
        }
        
        import re
        
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, content)
        
        if json_matches:
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        result["action"] = parsed.get("action")
                        result["data"] = parsed
                        break
                except json.JSONDecodeError:
                    continue
        
        return result
    
    def _build_context_from_messages(self, messages: List[Dict]) -> str:
        """从消息数组构建上下文
        
        Args:
            messages: 消息数组
            
        Returns:
            str: 构建的上下文字符串
        """
        parts = []
        
        # 添加自定义系统指令（覆盖 OpenClaw 的 system message）
        # 必须放在最前面，确保模型优先遵循（即使没有消息也添加）
        # 使用强约束格式，明确禁止的行为
        parts.append("""[系统指令]
重要：你必须严格遵守以下规则：
1. 直接回答问题，不要输出任何思考过程
2. 不要使用<think>或</think>标签
3. 不要复读用户的问题
4. 用中文简洁回答
5. 立即开始回答，不要说"好的"、"让我想想"等废话""")
        
        # 只保留最近的对话历史（最多 10 条）
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        has_content = False
        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # 跳过空内容
            if not content or content.strip() == "":
                continue
            
            # 跳过 system message（使用我们自己的系统指令）
            if role == "system":
                continue
            
            # 清理 OpenClaw 的 metadata
            if isinstance(content, str):
                if "[用户]" in content or "Sender (untrusted metadata)" in content:
                    # 提取实际的用户消息
                    lines = content.split("\n")
                    actual_message = ""
                    for line in lines:
                        line_stripped = line.strip()
                        if (line_stripped and 
                            not line_stripped.startswith("[") and 
                            not line_stripped.startswith("Sender") and 
                            not line_stripped.startswith("```") and
                            not line_stripped.startswith("JSON") and
                            not line_stripped.startswith("{") and
                            not line_stripped.startswith("}") and
                            not line_stripped.startswith('"label"') and
                            not line_stripped.startswith('"id"')):
                            actual_message += line + "\n"
                    content = actual_message.strip()
                    
                    # 如果清理后为空，尝试提取时间戳后的内容
                    if not content:
                        import re
                        match = re.search(r'\[.*?\]\s*(.+)', content)
                        if match:
                            content = match.group(1)
            
            if role == "user":
                if content:  # 只添加非空内容
                    parts.append(f"\n\n[用户]\n{content}")
                    has_content = True
            elif role == "assistant":
                if content:  # 只添加非空内容
                    parts.append(f"\n\n[助手]\n{content}")
                    has_content = True
        
        # 如果没有对话历史，添加默认提示
        if not has_content:
            parts.append("\n\n[用户]\n等待用户输入...")
        
        result = "\n".join(parts)
        return result
    
    def extract_response_content(self, api_response: Dict) -> Dict:
        """从 API 响应中提取内容
        
        清洗 OpenAI/DeepSeek 格式的响应
        
        Args:
            api_response: API 响应数据
            
        Returns:
            Dict: 提取的内容和元数据
        """
        result = {
            "content": "",
            "role": "assistant",
            "finish_reason": None,
            "usage": None,
            "structured_data": None
        }
        
        try:
            choices = api_response.get("choices", [])
            if choices:
                first_choice = choices[0]
                message = first_choice.get("message", {})
                
                result["content"] = message.get("content", "")
                result["role"] = message.get("role", "assistant")
                result["finish_reason"] = first_choice.get("finish_reason")
            
            result["usage"] = api_response.get("usage")
            
            if result["content"]:
                result["structured_data"] = self._extract_assistant_response(result["content"])
            
            logger.info(f"[AsyncL1BScheduler] 响应清洗完成，内容长度：{len(result['content'])}")
            
        except Exception as e:
            logger.error(f"[AsyncL1BScheduler] 响应清洗失败：{e}")
        
        return result
    
    def finalize_task(self, task: TaskItem, ai_response: str):
        """完成任务并写入记忆（修复缺陷 1：记忆写入闭环）
        
        🔥 关键修复：将 L2 的输出反向写回记忆系统
        
        Args:
            task: 任务项
            ai_response: AI 回复内容
        """
        try:
            # 1. 确保组件已初始化
            if not self._initialized:
                self.initialize()
            
            # 2. 构建完整对话
            user_input = task.raw_input
            full_dialog = f"用户：{user_input}\n助手：{ai_response}"
            
            # 3. 🔥 写入短期记忆（内存缓存，快速访问）
            self.short_term_memory.store(
                user_input=user_input,
                ai_response=ai_response,
                metadata={"task_id": task.task_id}
            )
            logger.debug(f"[AsyncL1BScheduler] ✅ 对话已存入短期记忆")
            
            # 4. 🔥 写入 TaggingEngine（打标并存储到 RAG）
            doc_id = self.tagging_engine.tag_and_store(
                content=full_dialog,
                rag_manager=self.library_manager.rag_manager
            )
            logger.info(f"[AsyncL1BScheduler] ✅ 记忆已写入 RAG: {doc_id}")
            
            # 5. 🔥 写入 HotStorage（持久化对话日志）
            log_data = {
                "trace_id": task.task_id,
                "user_input": {
                    "text": user_input,
                    "timestamp": task.created_at
                },
                "ai_response": {
                    "text": ai_response,
                    "timestamp": time.time()
                },
                "context": task.context,
                "status": "COMPLETED",
                "memory_doc_id": doc_id
            }
            log_id = self.hot_storage.store_log(log_data)
            logger.info(f"[AsyncL1BScheduler] ✅ 对话已存入 HotStorage: {log_id}")
            
            # 6. 🔥 写入环形缓冲区（为下次复盘做准备）
            # 注意：ring_buffer 需要在 scheduler 中初始化
            # 这里先记录事件，后续由 MemoryManager 统一处理
            logger.debug(f"[AsyncL1BScheduler] 📝 对话事件已记录到缓冲区：{task.task_id}")
            
        except Exception as e:
            logger.error(f"[AsyncL1BScheduler] ❌ 记忆写入失败：{e}")
            # 不抛出异常，避免影响主流程
    
    def get_ready_task(self, task_id: str) -> Optional[TaskItem]:
        """获取就绪的任务
        
        Args:
            task_id: 任务 ID
            
        Returns:
            Optional[TaskItem]: 任务项
        """
        return self.ready_tasks.get(task_id)
    
    def get_next_ready_task(self) -> Optional[TaskItem]:
        """获取下一个就绪的任务
        
        Returns:
            Optional[TaskItem]: 任务项
        """
        if not self.ready_tasks:
            return None
        
        oldest_id = min(
            self.ready_tasks.keys(),
            key=lambda tid: self.ready_tasks[tid].ready_at or 0
        )
        return self.ready_tasks.pop(oldest_id)
    
    def _check_for_emergency_interrupts(self):
        """检查紧急中断信号"""
        pass
    
    def force_preempt(self, high_priority_task: Dict) -> bool:
        """强制中断当前任务
        
        Args:
            high_priority_task: 高优先级任务
            
        Returns:
            bool: 是否成功
        """
        try:
            if self.current_task:
                logger.warning(f"[AsyncL1BScheduler] 强制中断任务: {self.current_task.task_id}")
                self.current_task.status = "PREEMPTED"
            
            task_id = self.handle_request(
                high_priority_task.get("input", ""),
                high_priority_task.get("context")
            )
            
            logger.info(f"[AsyncL1BScheduler] 高优先级任务已插入: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"[AsyncL1BScheduler] 强制中断失败: {e}")
            return False
    
    def add_interrupt_handler(self, handler: Callable):
        """添加中断处理器
        
        Args:
            handler: 处理函数
        """
        self._interrupt_handlers.append(handler)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "running": self._running,
            "queue_size": self.task_queue.qsize(),
            "ready_tasks": len(self.ready_tasks),
            "current_task": self.current_task.task_id if self.current_task else None,
            "initialized": self._initialized
        }
    
    def _restore_conversation_history(self, limit: int = 10) -> List[Dict]:
        """恢复最近 N 轮对话历史（修复缺陷 2：对话历史持久化）
        
        🔥 从短期记忆（内存）中恢复最近 N 轮对话，转换为 messages 格式
        如果短期记忆为空，则回退到 HotStorage
        
        Args:
            limit: 恢复的对话轮数（默认 10 轮）
            
        Returns:
            List[Dict]: 历史对话列表，格式为 [{"role": "user/assistant", "content": "..."}]
        """
        try:
            # 1. 确保组件已初始化
            if not self._initialized:
                self.initialize()
            
            # 2. 🔥 优先从短期记忆恢复（快速，无需数据库）
            if self.short_term_memory:
                recent_messages = self.short_term_memory.get_recent(limit=limit)
                
                if recent_messages:
                    logger.info(f"[AsyncL1BScheduler] 📚 从短期记忆恢复 {len(recent_messages)} 条历史对话")
                    return recent_messages
            
            # 3. 回退到 HotStorage（如果短期记忆为空）
            logger.debug("[AsyncL1BScheduler] 短期记忆为空，尝试从 HotStorage 恢复...")
            
            from datetime import datetime, timedelta
            recent_logs = self.hot_storage.query_logs(
                status="COMPLETED",
                limit=limit * 2,  # 每轮对话包含 user 和 assistant 两条消息
                skip=0
            )
            
            if not recent_logs:
                logger.debug("[AsyncL1BScheduler] 没有找到历史对话")
                return []
            
            # 4. 转换为 messages 格式
            messages = []
            for log in reversed(recent_logs):  # 按时间正序排列
                user_input = log.get("user_input", {})
                ai_response = log.get("ai_response", {})
                
                # 添加用户消息
                if isinstance(user_input, dict) and "text" in user_input:
                    messages.append({
                        "role": "user",
                        "content": user_input["text"],
                        "timestamp": user_input.get("timestamp")
                    })
                
                # 添加助手消息
                if isinstance(ai_response, dict) and "text" in ai_response:
                    messages.append({
                        "role": "assistant",
                        "content": ai_response["text"],
                        "timestamp": ai_response.get("timestamp")
                    })
            
            logger.info(f"[AsyncL1BScheduler] 📚 从 HotStorage 恢复 {len(messages)} 条历史对话")
            return messages
            
        except Exception as e:
            logger.error(f"[AsyncL1BScheduler] 恢复历史对话失败：{e}")
            return []


async_scheduler = AsyncL1BScheduler()


def get_async_scheduler() -> AsyncL1BScheduler:
    """获取异步调度器单例"""
    return async_scheduler
