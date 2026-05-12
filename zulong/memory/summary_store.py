# File: zulong/memory/summary_store.py
# 双索引摘要库 - L1摘要导航 + L2详情向量 (TSD v2.5)
#
# 核心架构：
# - L1 摘要导航层：SQLite 存储摘要文本 + FAISS 存储摘要向量
# - L2 详情向量层：FAISS 存储完整对话向量
# - 混合检索：SQL 过滤 + 向量语义搜索 并行执行
#
# 参照资料：
# - 向量化双索引摘要库加记忆库索引化增强记忆原子任务.txt
# - 向量化双索引摘要库加记忆库索引化增强记忆原子任务补充.txt

import logging
import json
import time
import os
import sqlite3
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

class SummaryStatus(Enum):
    """摘要状态"""
    HOT = "hot"       # 热数据（最近活跃）
    WARM = "warm"     # 温数据（中等活跃）
    COLD = "cold"     # 冷数据（不活跃，可归档）


@dataclass
class SummaryEntry:
    """摘要条目（L1 导航层）"""
    summary_id: str                   # 摘要 ID
    summary_text: str                 # LLM 生成的摘要文本
    topic: str = ""                   # 话题标签
    keywords: List[str] = field(default_factory=list)  # 关键词
    turn_start: int = 0              # 起始轮次
    turn_end: int = 0                # 结束轮次
    turn_count: int = 0              # 包含的对话轮数
    status: SummaryStatus = SummaryStatus.HOT
    importance: float = 0.5          # 重要性 (0-1)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0            # 访问次数
    
    # 关联的详情向量 ID 列表（指向 L2 详情层）
    detail_vector_ids: List[str] = field(default_factory=list)


@dataclass
class DetailEntry:
    """详情条目（L2 详情层）"""
    detail_id: str                    # 详情 ID
    summary_id: str                   # 所属摘要 ID
    content: str                      # 完整对话内容
    turn_id: int = 0                 # 对话轮次
    user_text: str = ""              # 用户原文
    ai_text: str = ""                # AI 原文
    timestamp: float = field(default_factory=time.time)


@dataclass
class HybridSearchResult:
    """混合检索结果"""
    summary_id: str
    summary_text: str
    relevance_score: float            # 综合相关度得分
    sql_matched: bool = False         # 是否 SQL 命中
    vector_score: float = 0.0         # 向量相似度得分
    details: List[Dict] = field(default_factory=list)  # 关联的详情


# ============================================================
# SQLite 摘要索引
# ============================================================

class SummaryIndex:
    """L1 摘要导航索引 (SQLite)"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化 SQLite 数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id TEXT PRIMARY KEY,
                summary_text TEXT NOT NULL,
                topic TEXT DEFAULT '',
                keywords TEXT DEFAULT '[]',
                turn_start INTEGER DEFAULT 0,
                turn_end INTEGER DEFAULT 0,
                turn_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'hot',
                importance REAL DEFAULT 0.5,
                created_at REAL,
                updated_at REAL,
                access_count INTEGER DEFAULT 0,
                detail_vector_ids TEXT DEFAULT '[]'
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic ON summaries(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON summaries(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON summaries(importance)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON summaries(created_at)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"[SummaryIndex] SQLite 初始化完成: {self.db_path}")
    
    def add_summary(self, entry: SummaryEntry) -> bool:
        """添加摘要"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO summaries 
                (summary_id, summary_text, topic, keywords, turn_start, turn_end,
                 turn_count, status, importance, created_at, updated_at, 
                 access_count, detail_vector_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.summary_id,
                entry.summary_text,
                entry.topic,
                json.dumps(entry.keywords, ensure_ascii=False),
                entry.turn_start,
                entry.turn_end,
                entry.turn_count,
                entry.status.value,
                entry.importance,
                entry.created_at,
                entry.updated_at,
                entry.access_count,
                json.dumps(entry.detail_vector_ids),
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"[SummaryIndex] 添加摘要失败: {e}")
            return False
    
    def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[SummaryEntry]:
        """按关键词检索摘要"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 构建 LIKE 条件
            conditions = []
            params = []
            for kw in keywords:
                conditions.append("(summary_text LIKE ? OR keywords LIKE ? OR topic LIKE ?)")
                params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%"])
            
            where_clause = " OR ".join(conditions) if conditions else "1=1"
            
            cursor.execute(f"""
                SELECT * FROM summaries 
                WHERE {where_clause}
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
            """, params + [limit])
            
            results = []
            for row in cursor.fetchall():
                results.append(self._row_to_entry(row))
            
            conn.close()
            return results
        except Exception as e:
            logger.error(f"[SummaryIndex] 关键词检索失败: {e}")
            return []
    
    def search_by_topic(self, topic: str, limit: int = 10) -> List[SummaryEntry]:
        """按话题检索摘要"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM summaries 
                WHERE topic LIKE ?
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
            """, (f"%{topic}%", limit))
            
            results = [self._row_to_entry(row) for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            logger.error(f"[SummaryIndex] 话题检索失败: {e}")
            return []
    
    def search_by_time_range(self, start_time: float, end_time: float, 
                              limit: int = 10) -> List[SummaryEntry]:
        """按时间范围检索"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM summaries
                WHERE created_at >= ? AND created_at <= ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (start_time, end_time, limit))
            
            results = [self._row_to_entry(row) for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            logger.error(f"[SummaryIndex] 时间范围检索失败: {e}")
            return []
    
    def update_status(self, summary_id: str, status: SummaryStatus) -> bool:
        """更新摘要状态（Hot/Warm/Cold 转换）"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE summaries SET status = ?, updated_at = ? WHERE summary_id = ?",
                (status.value, time.time(), summary_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"[SummaryIndex] 更新状态失败: {e}")
            return False
    
    def increment_access(self, summary_id: str):
        """增加访问计数"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE summaries SET access_count = access_count + 1, updated_at = ? WHERE summary_id = ?",
                (time.time(), summary_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[SummaryIndex] 更新访问计数失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM summaries")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT status, COUNT(*) FROM summaries GROUP BY status")
            status_counts = dict(cursor.fetchall())
            
            conn.close()
            return {
                "total_summaries": total,
                "hot": status_counts.get("hot", 0),
                "warm": status_counts.get("warm", 0),
                "cold": status_counts.get("cold", 0),
            }
        except Exception as e:
            return {"total_summaries": 0, "error": str(e)}
    
    def _row_to_entry(self, row) -> SummaryEntry:
        """将数据库行转换为 SummaryEntry"""
        return SummaryEntry(
            summary_id=row[0],
            summary_text=row[1],
            topic=row[2],
            keywords=json.loads(row[3]) if row[3] else [],
            turn_start=row[4],
            turn_end=row[5],
            turn_count=row[6],
            status=SummaryStatus(row[7]),
            importance=row[8],
            created_at=row[9],
            updated_at=row[10],
            access_count=row[11],
            detail_vector_ids=json.loads(row[12]) if row[12] else [],
        )


# ============================================================
# 双索引摘要库
# ============================================================

class DualIndexSummaryStore:
    """双索引摘要库
    
    TSD v2.5 架构：
    - L1 摘要导航层：SQLite(文本+元数据) + FAISS(摘要向量)
    - L2 详情向量层：FAISS(完整对话向量)
    - 混合检索：SQL过滤 + 向量搜索 并行执行
    
    数据流：
    1. 对话产生 → LLM 生成摘要 → 存入 L1 (SQLite + 摘要向量)
    2. 完整对话 → 向量化 → 存入 L2 (详情向量)
    3. 检索时 → SQL 粗筛 + 向量精排 并行 → 融合结果
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, persist_dir: str = "./data/summary_store"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        # L1 摘要导航层
        self.summary_index = SummaryIndex(
            db_path=os.path.join(persist_dir, "summaries.db")
        )
        
        # L1 摘要向量索引 (FAISS)
        from zulong.memory.base_rag_library import FAISSVectorStore
        self.summary_vector_store = FAISSVectorStore(dimension=512)
        
        # L2 详情向量层 (FAISS)
        self.detail_vector_store = FAISSVectorStore(dimension=512)
        
        # Embedding 管理器
        from zulong.memory.embedding_manager import EmbeddingModelManager
        self.embedding_manager = EmbeddingModelManager()
        
        # 状态流转配置
        self.hot_to_warm_hours = 24       # 24 小时无访问 → Warm
        self.warm_to_cold_hours = 168     # 7 天无访问 → Cold
        
        # 统计信息
        self._stats = {
            "total_summaries_stored": 0,
            "total_details_stored": 0,
            "total_hybrid_searches": 0,
        }
        
        # 尝试加载已有索引
        self._load_vector_indices()
        
        self._initialized = True
        logger.info("[DualIndexSummaryStore] 初始化完成")
    
    # ============================================================
    # 存储操作
    # ============================================================
    
    async def store_summary(self, summary_text: str, 
                             detail_turns: List[Dict[str, str]],
                             topic: str = "",
                             keywords: List[str] = None,
                             importance: float = 0.5) -> str:
        """存储摘要及其对应的详情
        
        流程：
        1. 摘要文本 → 向量化 → 存入 L1 (SQLite + FAISS)
        2. 详情对话 → 逐条向量化 → 存入 L2 (FAISS)
        3. 建立 L1 → L2 的关联索引
        
        Args:
            summary_text: LLM 生成的摘要文本
            detail_turns: 原始对话列表 [{"user": ..., "assistant": ...}]
            topic: 话题标签
            keywords: 关键词列表
            importance: 重要性评分
            
        Returns:
            str: 摘要 ID
        """
        summary_id = f"summary_{int(time.time() * 1000)}"
        detail_vector_ids = []
        
        # 1. 向量化摘要文本
        summary_embedding = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.embedding_manager.encode_query(summary_text)
        )
        
        # 2. 存入 L1 摘要向量索引
        if summary_embedding is not None:
            self.summary_vector_store.add_vectors_with_ids(
                vectors=summary_embedding.reshape(1, -1),
                metadata=[{"summary_id": summary_id, "topic": topic}],
                vector_ids=[summary_id]
            )
        
        # 3. 逐条向量化详情并存入 L2
        for i, turn in enumerate(detail_turns):
            user_text = turn.get("user", "")
            ai_text = turn.get("assistant", "")
            content = f"{user_text} {ai_text}"
            
            detail_id = f"{summary_id}_detail_{i}"
            
            detail_embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda c=content: self.embedding_manager.encode_document(c)
            )
            
            if detail_embedding is not None:
                self.detail_vector_store.add_vectors_with_ids(
                    vectors=detail_embedding.reshape(1, -1),
                    metadata=[{
                        "detail_id": detail_id,
                        "summary_id": summary_id,
                        "user_text": user_text[:200],
                        "ai_text": ai_text[:200],
                    }],
                    vector_ids=[detail_id]
                )
                detail_vector_ids.append(detail_id)
        
        # 4. 存入 L1 SQLite 索引
        turn_ids = [turn.get("turn_id", i) for i, turn in enumerate(detail_turns)]
        entry = SummaryEntry(
            summary_id=summary_id,
            summary_text=summary_text,
            topic=topic,
            keywords=keywords or [],
            turn_start=min(turn_ids) if turn_ids else 0,
            turn_end=max(turn_ids) if turn_ids else 0,
            turn_count=len(detail_turns),
            importance=importance,
            detail_vector_ids=detail_vector_ids,
        )
        self.summary_index.add_summary(entry)
        
        # 5. 持久化向量索引
        self._save_vector_indices()
        
        self._stats["total_summaries_stored"] += 1
        self._stats["total_details_stored"] += len(detail_vector_ids)
        
        logger.info(
            f"[DualIndexSummaryStore] 存储完成: {summary_id} "
            f"(摘要={len(summary_text)}字, 详情={len(detail_vector_ids)}条)"
        )
        
        return summary_id
    
    # ============================================================
    # 混合检索
    # ============================================================
    
    async def hybrid_search(self, query: str, top_k: int = 5,
                             include_details: bool = True,
                             topic_filter: Optional[str] = None,
                             time_range: Optional[Tuple[float, float]] = None
                             ) -> List[HybridSearchResult]:
        """混合并行检索
        
        并行执行：
        1. SQL 过滤（关键词 + 话题 + 时间范围）
        2. 向量语义搜索（摘要向量相似度）
        
        融合策略：SQL 命中加权 + 向量分数排序
        
        Args:
            query: 查询文本
            top_k: 返回数量
            include_details: 是否包含原始对话详情
            topic_filter: 话题过滤
            time_range: 时间范围 (start, end)
            
        Returns:
            混合检索结果列表
        """
        self._stats["total_hybrid_searches"] += 1
        
        # 并行执行 SQL 检索和向量检索
        sql_task = asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._sql_search(query, topic_filter, time_range, top_k * 2)
        )
        vector_task = asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._vector_search(query, top_k * 2)
        )
        
        sql_results, vector_results = await asyncio.gather(sql_task, vector_task)
        
        # 融合结果
        merged = self._merge_results(sql_results, vector_results, top_k)
        
        # 如果需要详情，加载关联的详情向量
        if include_details:
            for result in merged:
                result.details = await self._load_details(result.summary_id)
        
        return merged
    
    def _sql_search(self, query: str, topic: Optional[str], 
                     time_range: Optional[Tuple], limit: int) -> List[Tuple[str, float]]:
        """SQL 检索（在线程池中执行）"""
        results = []
        
        # 提取关键词
        keywords = [w for w in query.split() if len(w) >= 2]
        if not keywords:
            keywords = [query]
        
        # 按关键词检索
        entries = self.summary_index.search_by_keywords(keywords, limit)
        for entry in entries:
            results.append((entry.summary_id, entry.importance))
        
        # 按话题检索
        if topic:
            topic_entries = self.summary_index.search_by_topic(topic, limit)
            for entry in topic_entries:
                if entry.summary_id not in [r[0] for r in results]:
                    results.append((entry.summary_id, entry.importance * 0.8))
        
        # 按时间范围检索
        if time_range:
            time_entries = self.summary_index.search_by_time_range(
                time_range[0], time_range[1], limit
            )
            for entry in time_entries:
                if entry.summary_id not in [r[0] for r in results]:
                    results.append((entry.summary_id, entry.importance * 0.6))
        
        return results
    
    def _vector_search(self, query: str, limit: int) -> List[Tuple[str, float]]:
        """向量语义搜索（在线程池中执行）"""
        results = []
        
        query_embedding = self.embedding_manager.encode_query(query)
        if query_embedding is None:
            return results
        
        indices, distances = self.summary_vector_store.search(
            query_embedding, top_k=limit
        )
        
        for idx, dist in zip(indices, distances):
            if idx == -1:
                continue
            doc_id = self.summary_vector_store.reverse_id_map.get(idx)
            if doc_id:
                # 将 L2 距离转换为相似度分数 (0-1)
                similarity = 1.0 / (1.0 + dist)
                results.append((doc_id, similarity))
        
        return results
    
    def _merge_results(self, sql_results: List[Tuple[str, float]], 
                        vector_results: List[Tuple[str, float]],
                        top_k: int) -> List[HybridSearchResult]:
        """融合 SQL 和向量检索结果"""
        scores: Dict[str, Dict] = {}
        
        # SQL 结果
        for summary_id, score in sql_results:
            if summary_id not in scores:
                scores[summary_id] = {"sql_score": 0, "vector_score": 0, "sql_matched": False}
            scores[summary_id]["sql_score"] = score
            scores[summary_id]["sql_matched"] = True
        
        # 向量结果
        for summary_id, score in vector_results:
            if summary_id not in scores:
                scores[summary_id] = {"sql_score": 0, "vector_score": 0, "sql_matched": False}
            scores[summary_id]["vector_score"] = score
        
        # 计算综合分数：SQL 权重 0.3 + 向量权重 0.7
        merged = []
        for summary_id, data in scores.items():
            relevance = data["sql_score"] * 0.3 + data["vector_score"] * 0.7
            # SQL 命中给予额外加分
            if data["sql_matched"]:
                relevance += 0.1
            
            merged.append(HybridSearchResult(
                summary_id=summary_id,
                summary_text="",  # 后续填充
                relevance_score=relevance,
                sql_matched=data["sql_matched"],
                vector_score=data["vector_score"],
            ))
        
        # 按综合分数排序
        merged.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return merged[:top_k]
    
    async def _load_details(self, summary_id: str) -> List[Dict]:
        """加载摘要关联的详情"""
        details = []
        
        # 从详情向量索引中检索
        for doc_id, meta in self.detail_vector_store.metadata_store.items():
            if meta.get("summary_id") == summary_id and not meta.get("deleted", False):
                details.append({
                    "detail_id": doc_id,
                    "user_text": meta.get("user_text", ""),
                    "ai_text": meta.get("ai_text", ""),
                })
        
        # 更新访问计数
        self.summary_index.increment_access(summary_id)
        
        return details
    
    # ============================================================
    # 状态管理 (Hot → Warm → Cold)
    # ============================================================
    
    async def update_statuses(self):
        """更新所有摘要的状态（定期调用）"""
        current_time = time.time()
        
        try:
            conn = sqlite3.connect(
                os.path.join(self.persist_dir, "summaries.db")
            )
            cursor = conn.cursor()
            
            # Hot → Warm (超过 24 小时无访问)
            warm_threshold = current_time - self.hot_to_warm_hours * 3600
            cursor.execute("""
                UPDATE summaries SET status = 'warm'
                WHERE status = 'hot' AND updated_at < ?
            """, (warm_threshold,))
            
            # Warm → Cold (超过 7 天无访问)
            cold_threshold = current_time - self.warm_to_cold_hours * 3600
            cursor.execute("""
                UPDATE summaries SET status = 'cold'
                WHERE status = 'warm' AND updated_at < ?
            """, (cold_threshold,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"[DualIndexSummaryStore] 状态更新失败: {e}")
    
    # ============================================================
    # 持久化
    # ============================================================
    
    def _save_vector_indices(self):
        """保存向量索引到磁盘"""
        try:
            self.summary_vector_store.save(
                os.path.join(self.persist_dir, "summary_vectors")
            )
            self.detail_vector_store.save(
                os.path.join(self.persist_dir, "detail_vectors")
            )
        except Exception as e:
            logger.error(f"[DualIndexSummaryStore] 保存向量索引失败: {e}")
    
    def _load_vector_indices(self):
        """从磁盘加载向量索引"""
        try:
            summary_path = os.path.join(self.persist_dir, "summary_vectors")
            detail_path = os.path.join(self.persist_dir, "detail_vectors")
            
            if os.path.exists(f"{summary_path}.index"):
                self.summary_vector_store.load(summary_path)
            if os.path.exists(f"{detail_path}.index"):
                self.detail_vector_store.load(detail_path)
        except Exception as e:
            logger.debug(f"[DualIndexSummaryStore] 加载向量索引失败(非致命): {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        sql_stats = self.summary_index.get_stats()
        return {
            **self._stats,
            "summary_index": sql_stats,
            "summary_vectors": self.summary_vector_store.get_stats(),
            "detail_vectors": self.detail_vector_store.get_stats(),
        }


# ============================================================
# 全局单例
# ============================================================

_dual_index_store: Optional[DualIndexSummaryStore] = None


def get_dual_index_summary_store(persist_dir: str = "./data/summary_store") -> DualIndexSummaryStore:
    """获取双索引摘要库单例"""
    global _dual_index_store
    if _dual_index_store is None:
        _dual_index_store = DualIndexSummaryStore(persist_dir=persist_dir)
    return _dual_index_store
