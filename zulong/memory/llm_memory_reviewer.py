# File: zulong/memory/llm_memory_reviewer.py
# LLM 主动记忆管理 - 让 LLM 对记忆做最终审查 (TSD v2.5)
#
# 功能：
# 1. 重要内容入库前由 LLM 做最终审查（确认价值、补充标签）
# 2. MemoryGraph 边衰减到阈值时由 LLM 做剪枝前审查
# 3. 通过 L2-BACKUP 异步执行，不阻塞主对话流
#
# 集成点：
# - MemoryGraph.decay_and_prune() 标记 pending_review 边
# - MemoryGraph.submit_prune_review() 提交审查请求
# - L2BackupScheduler 异步执行 LLM 审查任务

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

class ReviewDecision(Enum):
    """LLM 审查决策"""
    KEEP = "keep"               # 保留（确认有价值）
    DISCARD = "discard"         # 丢弃（无价值）
    COMPRESS = "compress"       # 压缩（保留核心，压缩细节）
    PROMOTE = "promote"         # 晋升（价值更高，升级到更高层）
    MERGE = "merge"             # 合并（与其他记忆合并）


class ReviewTaskType(Enum):
    """审查任务类型"""
    PRE_STORE = "pre_store"           # 入库前审查
    PRE_EVICT = "pre_evict"           # 淘汰前审查
    PERIODIC_REVIEW = "periodic"      # 定期审查
    EMERGENCY_REVIEW = "emergency"    # 紧急审查（快超限时）


@dataclass
class MemoryReviewTask:
    """记忆审查任务"""
    task_id: str
    task_type: ReviewTaskType
    memories: List[Dict[str, Any]]    # 待审查的记忆列表
    context: Dict[str, Any] = field(default_factory=dict)  # 上下文信息
    priority: int = 1                 # 0=最高优先级
    created_at: float = field(default_factory=time.time)
    status: str = "pending"           # pending/running/completed/failed
    result: Optional[Dict] = None
    
    def __lt__(self, other):
        return self.priority < other.priority


@dataclass
class ReviewResult:
    """审查结果"""
    task_id: str
    decisions: List[Dict[str, Any]]   # [{memory_id, decision, reason, summary}]
    total_reviewed: int = 0
    total_kept: int = 0
    total_discarded: int = 0
    total_compressed: int = 0
    total_promoted: int = 0
    processing_time: float = 0.0


# ============================================================
# LLM 记忆审查器
# ============================================================

class LLMMemoryReviewer:
    """LLM 主动记忆管理器
    
    TSD v2.5 对应规则:
    - 让 LLM 作为记忆的"最终审查官"
    - 在临时记忆层(L1)实现主动管理
    - 通过 L2-BACKUP 异步执行，不阻塞主对话
    
    工作流程:
    1. 触发条件：记忆入库/即将淘汰/定期巡检/快超限
    2. 构建审查 Prompt：包含待审查记忆 + 上下文 + 审查标准
    3. 提交到 L2-BACKUP 异步执行
    4. 接收审查结果，执行决策（保留/丢弃/压缩/晋升）
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # 审查配置
        self.max_batch_size = 50           # 单次审查最大记忆数（从10提升以减少积压）
        self.review_cooldown = 60          # 审查冷却时间（秒），从300降为60加速吞吐
        self.last_review_time = 0.0
        
        # 紧急审查阈值
        self.emergency_threshold = 0.8     # 当容量使用率超过 80% 触发紧急审查
        
        # 审查 Prompt 模板
        self._review_prompt_template = self._build_review_prompt_template()
        self._evict_prompt_template = self._build_evict_prompt_template()
        
        # 统计信息
        self._stats = {
            "total_reviews": 0,
            "total_kept": 0,
            "total_discarded": 0,
            "total_compressed": 0,
            "total_promoted": 0,
            "avg_review_time": 0.0,
        }
        
        self._initialized = True
        logger.info("[LLMMemoryReviewer] 初始化完成")
    
    # ============================================================
    # 审查 Prompt 构建
    # ============================================================
    
    def _build_review_prompt_template(self) -> str:
        """构建入库审查 Prompt 模板"""
        return """你是祖龙机器人的记忆审查官。请审查以下对话内容，决定其记忆价值。

## 审查标准
1. **重要事实信息**（人名、地点、时间、数字等）→ KEEP 或 PROMOTE
2. **用户偏好和习惯** → KEEP
3. **任务指令和承诺** → PROMOTE（升级为长期记忆）
4. **闲聊寒暄** → DISCARD（除非包含重要信息）
5. **重复内容** → MERGE（与已有记忆合并）
6. **长对话但信息密度低** → COMPRESS（压缩为摘要）

## 待审查内容
{memories}

## 输出格式（JSON数组）
请对每条记忆输出审查决策：
```json
[
  {{
    "memory_index": 0,
    "decision": "KEEP|DISCARD|COMPRESS|PROMOTE|MERGE",
    "reason": "简要说明理由",
    "summary": "如果是COMPRESS，提供压缩后的摘要",
    "importance_score": 0.0-1.0,
    "tags": ["标签1", "标签2"]
  }}
]
```"""
    
    def _build_evict_prompt_template(self) -> str:
        """构建淘汰前审查 Prompt 模板（适配 MemoryGraph 图节点格式）"""
        return """你是祖龙机器人的记忆管理员。记忆图谱中部分边即将因衰减被剪枝，需要你做最后把关。

## 当前状态
- 记忆容量使用率：{usage_ratio:.1%}
- 需要释放的空间：至少 {target_free} 条边
- 当前活跃话题：{active_topics}

## 待审查的图谱记忆边
以下每条记录是一条图谱中即将被剪枝的边（连接两个记忆节点）：
{memories}

## 审查原则
1. **绝对保留**：用户个人信息（identity）、重要承诺（must_remember）、未完成的任务指令
2. **优先保留**：最近对话上下文、高重要度节点间的连接
3. **可以丢弃**：过时的闲聊连接、已完成任务的弱关联
4. **建议压缩**：长对话但包含少量重要信息 → 生成摘要后丢弃原文
5. **建议提升**：发现未被正确标记的重要信息 → PROMOTE

## 输出格式（JSON数组）
```json
[
  {{
    "memory_index": 0,
    "decision": "KEEP|DISCARD|COMPRESS|PROMOTE|MERGE",
    "reason": "简要说明理由",
    "summary": "如果是COMPRESS，提供压缩后的摘要"
  }}
]
```"""
    
    # ============================================================
    # 审查任务提交
    # ============================================================
    
    async def review_before_store(self, memories: List[Dict[str, Any]],
                                   context: Optional[Dict] = None) -> str:
        """入库前审查：提交到 L2-BACKUP 异步执行
        
        Args:
            memories: 待入库的记忆列表
            context: 上下文信息
            
        Returns:
            str: 审查任务 ID
        """
        task_id = f"review_store_{int(time.time() * 1000)}"
        
        # 构建审查 Prompt
        memories_text = self._format_memories_for_prompt(memories)
        prompt = self._review_prompt_template.format(memories=memories_text)
        
        # 提交到 L2-BACKUP
        await self._submit_to_backup(
            task_id=task_id,
            task_type=ReviewTaskType.PRE_STORE,
            prompt=prompt,
            memories=memories,
            context=context or {},
            priority=2,  # 普通优先级
        )
        
        self._stats["total_reviews"] += 1
        return task_id
    
    async def review_before_evict(self, memories: List[Dict[str, Any]],
                                   usage_ratio: float,
                                   target_free: int,
                                   active_topics: List[str] = None) -> str:
        """淘汰前审查：在即将丢弃记忆前让 LLM 做最后把关
        
        Args:
            memories: 即将被淘汰的记忆列表
            usage_ratio: 当前容量使用率
            target_free: 需要释放的记忆数
            active_topics: 当前活跃话题
            
        Returns:
            str: 审查任务 ID
        """
        task_id = f"review_evict_{int(time.time() * 1000)}"
        
        # 构建审查 Prompt
        memories_text = self._format_memories_for_prompt(memories)
        prompt = self._evict_prompt_template.format(
            memories=memories_text,
            usage_ratio=usage_ratio,
            target_free=target_free,
            active_topics=", ".join(active_topics or ["未知"]),
        )
        
        # 紧急审查用更高优先级
        priority = 0 if usage_ratio > self.emergency_threshold else 1
        
        await self._submit_to_backup(
            task_id=task_id,
            task_type=ReviewTaskType.PRE_EVICT if usage_ratio <= self.emergency_threshold 
                      else ReviewTaskType.EMERGENCY_REVIEW,
            prompt=prompt,
            memories=memories,
            context={
                "usage_ratio": usage_ratio,
                "target_free": target_free,
            },
            priority=priority,
        )
        
        self._stats["total_reviews"] += 1
        return task_id
    
    async def periodic_review(self, all_memories: List[Dict[str, Any]]) -> str:
        """定期巡检：低优先级后台任务
        
        Args:
            all_memories: 所有活跃记忆
            
        Returns:
            str: 审查任务 ID
        """
        # 检查冷却时间
        if time.time() - self.last_review_time < self.review_cooldown:
            return ""
        
        task_id = f"review_periodic_{int(time.time() * 1000)}"
        
        # 只审查最旧的 N 条记忆
        memories_to_review = sorted(
            all_memories, 
            key=lambda m: m.get('timestamp', 0)
        )[:self.max_batch_size]
        
        memories_text = self._format_memories_for_prompt(memories_to_review)
        prompt = self._review_prompt_template.format(memories=memories_text)
        
        await self._submit_to_backup(
            task_id=task_id,
            task_type=ReviewTaskType.PERIODIC_REVIEW,
            prompt=prompt,
            memories=memories_to_review,
            context={},
            priority=5,  # 低优先级
        )
        
        self.last_review_time = time.time()
        self._stats["total_reviews"] += 1
        return task_id
    
    async def review_importance_candidates(self, memories: List[Dict[str, Any]]) -> str:
        """审查重要性晋升候选节点

        Args:
            memories: 候选节点信息列表（含id, label, content, importance, access_count）

        Returns:
            str: 审查任务 ID
        """
        task_id = f"review_importance_{int(time.time() * 1000)}"

        candidates_text = self._format_importance_candidates_for_prompt(memories)
        prompt = self._build_importance_review_prompt(candidates_text)

        await self._submit_to_backup(
            task_id=task_id,
            task_type=ReviewTaskType.PERIODIC_REVIEW,
            prompt=prompt,
            memories=memories,
            context={"review_kind": "importance_promotion"},
            priority=3,
        )

        self._stats["total_reviews"] += 1
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg:
                with mg._candidates_lock:
                    pending_remaining = len(mg._pending_llm_candidates)
            else:
                pending_remaining = -1
        except Exception:
            pending_remaining = -1
        logger.info(f"[LLMMemoryReviewer] 已提交重要性审查任务: {task_id} (候选数={len(memories)}, 剩余待审查={pending_remaining})")
        return task_id

    def _build_importance_review_prompt(self, candidates_text: str) -> str:
        return f"""你是一个记忆管理审查专家。以下记忆节点因高访问频率被标记为重要性晋升候选。

晋升标准：
- 节点被频繁访问(≥5次)且已为IMPORTANT级别
- 如果节点内容与核心任务强相关，应提升为MUST_REMEMBER
- 如果节点内容仅为临时/辅助信息，保持IMPORTANT

待审查节点：
{candidates_text}

请以JSON数组格式输出审查结果，每个元素包含：
- "id": 节点ID
- "decision": "PROMOTE"(提升为MUST_REMEMBER) 或 "KEEP"(保持IMPORTANT)
- "reason": 决策理由(一句话)"""

    def _format_importance_candidates_for_prompt(self, memories: List[Dict]) -> str:
        lines = []
        for i, m in enumerate(memories, 1):
            lines.append(f"{i}. ID: {m.get('id', '?')}")
            lines.append(f"   标签: {m.get('label', '')}")
            content = str(m.get('content', ''))
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"   内容: {content}")
            lines.append(f"   当前重要度: {m.get('importance', 'IMPORTANT')}")
            lines.append(f"   访问次数: {m.get('access_count', 0)}")
            lines.append("")
        return "\n".join(lines)
    
    # ============================================================
    # L2-BACKUP 集成
    # ============================================================
    
    async def _submit_to_backup(self, task_id: str, task_type: ReviewTaskType,
                                 prompt: str, memories: List[Dict],
                                 context: Dict, priority: int):
        """提交审查任务到 L2-BACKUP 调度器"""
        try:
            from zulong.l2.backup_scheduler import get_l2_backup_scheduler
            scheduler = get_l2_backup_scheduler()
            
            # 将审查任务包装为复盘任务格式
            review_turns = [
                {
                    "user": prompt,
                    "assistant": "",  # 等待 L2-BACKUP 填充
                    "_meta": {
                        "task_type": task_type.value,
                        "review_task_id": task_id,
                        "memories_count": len(memories),
                        "original_memories": memories,
                        "context": context,
                    }
                }
            ]
            
            await scheduler.submit_summarization_task(
                conversation_turns=review_turns,
                priority=priority,
            )
            
            logger.info(
                f"[LLMMemoryReviewer] 已提交审查任务: {task_id} "
                f"(类型={task_type.value}, 优先级={priority}, 记忆数={len(memories)})"
            )
            
        except Exception as e:
            logger.error(f"[LLMMemoryReviewer] 提交审查任务失败: {e}")
    
    # ============================================================
    # 审查结果处理
    # ============================================================
    
    async def process_review_result(self, task_id: str, 
                                     llm_response: str,
                                     original_memories: List[Dict],
                                     context: Optional[Dict] = None) -> ReviewResult:
        """处理 LLM 审查结果
        
        Args:
            task_id: 任务 ID
            llm_response: LLM 返回的 JSON 审查结果
            original_memories: 原始记忆列表
            
        Returns:
            ReviewResult: 审查结果
        """
        import json
        start_time = time.time()
        
        result = ReviewResult(task_id=task_id)
        
        try:
            # 解析 LLM 返回的 JSON
            # 尝试提取 JSON 内容
            json_str = llm_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            decisions = json.loads(json_str.strip())
            
            for decision_data in decisions:
                idx = decision_data.get("memory_index", 0)
                decision_str = decision_data.get("decision", "KEEP").upper()
                reason = decision_data.get("reason", "")
                summary = decision_data.get("summary", "")
                
                decision = ReviewDecision.KEEP
                if decision_str == "DISCARD":
                    decision = ReviewDecision.DISCARD
                    result.total_discarded += 1
                elif decision_str == "COMPRESS":
                    decision = ReviewDecision.COMPRESS
                    result.total_compressed += 1
                elif decision_str == "PROMOTE":
                    decision = ReviewDecision.PROMOTE
                    result.total_promoted += 1
                    if context and context.get("review_kind") == "importance_promotion":
                        node_id = decision_data.get("id", "")
                        if node_id:
                            try:
                                from zulong.memory.memory_graph import get_memory_graph, Importance
                                mg = get_memory_graph()
                                if mg:
                                    mg.promote_importance(node_id, Importance.MUST_REMEMBER)
                                    logger.info(f"[LLMMemoryReviewer] 重要性晋升回填成功: {node_id}")
                            except Exception as e:
                                logger.warning(f"[LLMMemoryReviewer] 重要性晋升回填失败: {node_id} - {e}")
                elif decision_str == "MERGE":
                    decision = ReviewDecision.MERGE
                else:
                    result.total_kept += 1
                    # 审查后未提升重要性的节点，及时清理删除
                    if context and context.get("review_kind") == "importance_promotion":
                        node_id = decision_data.get("id", "")
                        if node_id:
                            try:
                                from zulong.memory.memory_graph import get_memory_graph
                                mg = get_memory_graph()
                                if mg and mg.has_node(node_id):
                                    removed = mg.remove_node(node_id)
                                    if removed:
                                        logger.info(f"[LLMMemoryReviewer] 审查未提升，清理删除节点: {node_id}")
                            except (KeyError, Exception) as e:
                                logger.warning(f"[LLMMemoryReviewer] 清理删除节点失败: {node_id} - {e}")
                
                result.decisions.append({
                    "memory_index": idx,
                    "decision": decision.value,
                    "reason": reason,
                    "summary": summary,
                    "importance_score": decision_data.get("importance_score", 0.5),
                    "tags": decision_data.get("tags", []),
                })
                result.total_reviewed += 1
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"[LLMMemoryReviewer] 解析审查结果失败: {e}")
            # 解析失败时，默认保留所有记忆
            for i, _ in enumerate(original_memories):
                result.decisions.append({
                    "memory_index": i,
                    "decision": ReviewDecision.KEEP.value,
                    "reason": "解析失败，默认保留",
                    "summary": "",
                })
                result.total_kept += 1
                result.total_reviewed += 1
        
        result.processing_time = time.time() - start_time
        
        # 更新统计
        self._stats["total_kept"] += result.total_kept
        self._stats["total_discarded"] += result.total_discarded
        self._stats["total_compressed"] += result.total_compressed
        self._stats["total_promoted"] += result.total_promoted
        
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg:
                with mg._candidates_lock:
                    pending_remaining = len(mg._pending_llm_candidates)
            else:
                pending_remaining = -1
        except Exception:
            pending_remaining = -1
        logger.info(
            f"[LLMMemoryReviewer] 审查完成: {task_id} "
            f"(保留={result.total_kept}, 丢弃={result.total_discarded}, "
            f"压缩={result.total_compressed}, 晋升={result.total_promoted}, "
            f"剩余待审查={pending_remaining})"
        )
        
        return result
    
    # ============================================================
    # 工具方法
    # ============================================================
    
    def _format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """将记忆列表格式化为 Prompt 文本
        
        同时支持两种格式：
        - 旧格式（对话格式）: {"user": ..., "assistant": ..., "timestamp": ...}
        - 新格式（图边格式）: {"source_label": ..., "target_label": ..., "source_content": ..., "importance": ...}
        """
        parts = []
        for i, mem in enumerate(memories):
            if not isinstance(mem, dict):
                continue
            
            # 图边格式（来自 MemoryGraph.submit_prune_review）
            if "source_label" in mem or "target_label" in mem:
                src_label = mem.get("source_label", "?")
                tgt_label = mem.get("target_label", "?")
                src_content = mem.get("source_content", "")[:200]
                tgt_content = mem.get("target_content", "")[:200]
                imp = mem.get("importance", 0)
                imp_names = {0: "trivial", 1: "normal", 2: "identity", 3: "fact", 4: "important", 5: "must_remember"}
                imp_label = imp_names.get(imp, f"level_{imp}")
                parts.append(
                    f"[边 {i}] 重要度={imp_label}\n"
                    f"  节点A [{src_label}]: {src_content}\n"
                    f"  节点B [{tgt_label}]: {tgt_content}"
                )
            else:
                # 旧对话格式
                user_data = mem.get("user", {})
                ai_data = mem.get("assistant", {})
                user_text = user_data.get("text", str(user_data)) if isinstance(user_data, dict) else str(user_data)
                ai_text = ai_data.get("text", str(ai_data)) if isinstance(ai_data, dict) else str(ai_data)
                timestamp = mem.get("timestamp", "")
                parts.append(
                    f"[记忆 {i}] (时间: {timestamp})\n"
                    f"  用户: {user_text[:200]}\n"
                    f"  AI: {ai_text[:200]}"
                )
        
        return "\n\n".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "review_cooldown": self.review_cooldown,
            "last_review_time": self.last_review_time,
            "emergency_threshold": self.emergency_threshold,
        }


# ============================================================
# 全局单例
# ============================================================

_llm_memory_reviewer: Optional[LLMMemoryReviewer] = None


def get_llm_memory_reviewer() -> LLMMemoryReviewer:
    """获取 LLM 记忆审查器单例"""
    global _llm_memory_reviewer
    if _llm_memory_reviewer is None:
        _llm_memory_reviewer = LLMMemoryReviewer()
    return _llm_memory_reviewer
