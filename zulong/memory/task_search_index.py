# File: zulong/memory/task_search_index.py
# 历史任务语义检索索引
#
# 利用项目已有的 embedding 模型 (BAAI/bge-small-zh-v1.5, 512维) + FAISS
# 将已完成归档和磁盘备份的任务标题向量化，支持语义检索历史任务。
#
# 设计决策：
# - 直接组合 FAISSVectorStore + embedding_model，不继承 BaseRAGLibrary
# - 单例模式，首次使用时从磁盘加载或重建索引
# - 增量更新：归档/备份时调用 add_entry()
# - 降级方案：embedding 不可用时返回空结果，由调用方退回文本匹配

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_COMPLETED_TASKS_DIR = os.path.join(".", "data", "completed_tasks")
_GRAPH_BACKUPS_DIR = os.path.join(".", "data", "graph_backups")
_DEFAULT_PERSIST_PATH = os.path.join(".", "data", "rag", "task_index")
_DIMENSION = 512
_MIN_TITLE_LENGTH = 3
_DEFAULT_SIMILARITY_THRESHOLD = 0.55
_AUTO_SAVE_INTERVAL = 60  # 秒
_AUTO_SAVE_DIRTY_COUNT = 10


@dataclass
class TaskIndexEntry:
    """语义索引中的一条记录"""
    entry_id: str       # task_id 或 graph_id
    title: str          # 任务标题（用于 embedding 编码和展示）
    source: str         # "completed" | "backup"
    file_path: str      # 对应磁盘 JSON 路径
    indexed_at: float = 0.0  # 索引时间戳

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskIndexEntry":
        return cls(
            entry_id=d["entry_id"],
            title=d.get("title", ""),
            source=d.get("source", ""),
            file_path=d.get("file_path", ""),
            indexed_at=d.get("indexed_at", 0.0),
        )


class HistoricalTaskIndex:
    """历史任务语义检索索引（单例）

    封装 FAISSVectorStore + embedding_model，提供：
    - add_entry(): 增量添加任务到索引
    - search(): 语义检索最相关的历史任务
    - rebuild_from_disk(): 冷启动时从磁盘文件批量重建
    - save()/load(): 索引持久化
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, persist_path: str = _DEFAULT_PERSIST_PATH,
                 dimension: int = _DIMENSION):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self._persist_path = persist_path
        self._dimension = dimension
        self._entries: Dict[str, TaskIndexEntry] = {}
        self._vector_store = None
        self._faiss_available = False
        self._dirty_count = 0
        self._last_save_time = time.time()

        # 尝试初始化 FAISS
        try:
            from zulong.memory.base_rag_library import FAISSVectorStore
            self._vector_store = FAISSVectorStore(
                dimension=dimension, index_type="Flat"
            )
            self._faiss_available = True
            logger.info("[TaskSearchIndex] FAISS 初始化成功")
        except Exception as e:
            logger.warning(f"[TaskSearchIndex] FAISS 不可用，语义检索已禁用: {e}")
            return

        # 尝试从磁盘加载已有索引
        if not self._try_load():
            # 无持久化索引，检查是否需要冷启动重建
            has_files = (
                os.path.isdir(_COMPLETED_TASKS_DIR)
                and any(f.endswith(".json") for f in os.listdir(_COMPLETED_TASKS_DIR))
            ) or (
                os.path.isdir(_GRAPH_BACKUPS_DIR)
                and any(f.endswith(".json") for f in os.listdir(_GRAPH_BACKUPS_DIR))
            )
            if has_files:
                logger.info("[TaskSearchIndex] 无持久化索引，启动冷启动重建...")
                self.rebuild_from_disk()

    def is_available(self) -> bool:
        """检查语义检索是否可用（FAISS + embedding 模型就绪）"""
        if not self._faiss_available or self._vector_store is None:
            return False
        try:
            from zulong.models.embedding_model import embedding_model
            return embedding_model.model is not None
        except Exception:
            return False

    def add_entry(self, entry: TaskIndexEntry) -> Optional[str]:
        """增量添加一条任务到索引（自动去重）

        Args:
            entry: 任务索引条目

        Returns:
            vector_id 或 None（添加失败时）
        """
        if not self._faiss_available or not entry.title or len(entry.title) < _MIN_TITLE_LENGTH:
            return None

        try:
            from zulong.models.embedding_model import embedding_model

            # 如果同 ID 已存在，先删除旧条目
            if entry.entry_id in self._entries:
                self._vector_store.delete_vectors([entry.entry_id])
                del self._entries[entry.entry_id]

            # 编码标题
            vec = embedding_model.encode_query(entry.title)
            if vec is None:
                return None
            vec = vec.reshape(1, -1).astype(np.float32)

            # 写入时间戳
            entry.indexed_at = time.time()

            # 写入 FAISS
            self._vector_store.add_vectors_with_ids(
                vec,
                metadata=[{"entry_id": entry.entry_id, "source": entry.source}],
                vector_ids=[entry.entry_id],
            )
            self._entries[entry.entry_id] = entry

            self._dirty_count += 1
            self._maybe_auto_save()

            logger.debug(
                f"[TaskSearchIndex] 已索引: {entry.entry_id} "
                f"({entry.title[:30]}..., source={entry.source})"
            )
            return entry.entry_id

        except Exception as e:
            logger.warning(f"[TaskSearchIndex] add_entry 失败: {e}")
            return None

    def remove_entry(self, entry_id: str) -> bool:
        """从索引中移除一条记录"""
        if not self._faiss_available:
            return False
        if entry_id not in self._entries:
            return False

        self._vector_store.delete_vectors([entry_id])
        del self._entries[entry_id]
        self._dirty_count += 1
        logger.debug(f"[TaskSearchIndex] 已移除: {entry_id}")
        return True

    def search(self, query: str, top_k: int = 5,
               similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
               ) -> List[Tuple[TaskIndexEntry, float]]:
        """语义检索最相关的历史任务

        Args:
            query: 查询文本（如任务标题或用户描述）
            top_k: 返回最多 top_k 个结果
            similarity_threshold: 最低相似度阈值 (0-1)

        Returns:
            [(TaskIndexEntry, similarity), ...] 按相似度降序
        """
        if not self.is_available() or not query:
            return []
        if self._vector_store.index.ntotal == 0:
            return []

        try:
            from zulong.models.embedding_model import embedding_model

            # 编码查询
            query_vec = embedding_model.encode_query(query)
            if query_vec is None:
                return []

            # FAISS 搜索
            indices, distances = self._vector_store.search(query_vec, top_k=top_k)

            # L2 距离转相似度 + 阈值过滤
            results = []
            for idx, dist in zip(indices, distances):
                sim = 1.0 / (1.0 + dist)
                if sim < similarity_threshold:
                    continue
                doc_id = self._vector_store.reverse_id_map.get(idx)
                if doc_id and doc_id in self._entries:
                    results.append((self._entries[doc_id], sim))

            # 按相似度降序排序
            results.sort(key=lambda x: x[1], reverse=True)

            if results:
                logger.info(
                    f"[TaskSearchIndex] 搜索 '{query[:40]}' → "
                    f"{len(results)} 条结果 (最高 sim={results[0][1]:.3f})"
                )
            return results

        except Exception as e:
            logger.warning(f"[TaskSearchIndex] search 失败: {e}")
            return []

    def rebuild_from_disk(self) -> int:
        """从磁盘文件冷启动重建索引

        扫描 data/completed_tasks/ 和 data/graph_backups/，
        批量编码并写入 FAISS。

        Returns:
            索引条目总数
        """
        if not self._faiss_available:
            return 0

        logger.info("[TaskSearchIndex] 开始冷启动重建...")
        entries_to_index: Dict[str, TaskIndexEntry] = {}

        # 1. 扫描已完成归档（优先）
        if os.path.isdir(_COMPLETED_TASKS_DIR):
            for fname in os.listdir(_COMPLETED_TASKS_DIR):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(_COMPLETED_TASKS_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    task_id = data.get("task_id", fname.replace(".json", ""))
                    title = data.get("description", "")
                    if title and len(title) >= _MIN_TITLE_LENGTH:
                        entries_to_index[task_id] = TaskIndexEntry(
                            entry_id=task_id,
                            title=title,
                            source="completed",
                            file_path=fpath,
                        )
                except Exception:
                    continue

        # 2. 扫描磁盘备份（不覆盖 completed 的同 ID 记录）
        if os.path.isdir(_GRAPH_BACKUPS_DIR):
            for fname in os.listdir(_GRAPH_BACKUPS_DIR):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(_GRAPH_BACKUPS_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    graph_id = data.get("id", fname.replace(".json", ""))
                    title = data.get("title", "")
                    if title and len(title) >= _MIN_TITLE_LENGTH:
                        if graph_id not in entries_to_index:
                            entries_to_index[graph_id] = TaskIndexEntry(
                                entry_id=graph_id,
                                title=title,
                                source="backup",
                                file_path=fpath,
                            )
                except Exception:
                    continue

        if not entries_to_index:
            logger.info("[TaskSearchIndex] 无可索引的历史任务")
            return 0

        # 3. 批量编码
        try:
            from zulong.models.embedding_model import embedding_model

            entry_list = list(entries_to_index.values())
            texts = [e.title for e in entry_list]
            ids = [e.entry_id for e in entry_list]

            logger.info(f"[TaskSearchIndex] 批量编码 {len(texts)} 条任务标题...")
            vectors = embedding_model.encode_documents(texts)

            # 4. 重新初始化 FAISS（清空旧索引）
            from zulong.memory.base_rag_library import FAISSVectorStore
            self._vector_store = FAISSVectorStore(
                dimension=self._dimension, index_type="Flat"
            )

            # 5. 批量写入 FAISS
            metadata_list = [
                {"entry_id": e.entry_id, "source": e.source}
                for e in entry_list
            ]
            self._vector_store.add_vectors_with_ids(
                vectors.astype(np.float32),
                metadata=metadata_list,
                vector_ids=ids,
            )

            # 6. 更新内存条目
            now = time.time()
            for e in entry_list:
                e.indexed_at = now
            self._entries = entries_to_index

            # 7. 持久化
            self.save()

            logger.info(
                f"[TaskSearchIndex] 冷启动重建完成: "
                f"{len(self._entries)} 条条目已索引"
            )
            return len(self._entries)

        except Exception as e:
            logger.error(f"[TaskSearchIndex] 冷启动重建失败: {e}")
            return 0

    def save(self) -> bool:
        """持久化索引到磁盘"""
        if not self._faiss_available:
            return False

        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)

            # 保存 FAISS 索引 + ID 映射
            self._vector_store.save(self._persist_path)

            # 保存 entries
            entries_path = f"{self._persist_path}.entries.json"
            with open(entries_path, "w", encoding="utf-8") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._entries.items()},
                    f, ensure_ascii=False, indent=2,
                )

            self._dirty_count = 0
            self._last_save_time = time.time()
            logger.info(
                f"[TaskSearchIndex] 已持久化: "
                f"{len(self._entries)} 条条目, "
                f"FAISS ntotal={self._vector_store.index.ntotal}"
            )
            return True

        except Exception as e:
            logger.error(f"[TaskSearchIndex] 持久化失败: {e}")
            return False

    def _try_load(self) -> bool:
        """尝试从磁盘加载已有索引"""
        if not self._faiss_available:
            return False

        # 检查索引文件是否存在
        index_file = f"{self._persist_path}.index"
        entries_file = f"{self._persist_path}.entries.json"
        if not os.path.exists(index_file) or not os.path.exists(entries_file):
            return False

        try:
            # 加载 FAISS
            ok = self._vector_store.load(self._persist_path)
            if not ok:
                return False

            # 验证维度一致性
            if self._vector_store.index.d != self._dimension:
                logger.warning(
                    f"[TaskSearchIndex] 维度不匹配: "
                    f"索引={self._vector_store.index.d}, 期望={self._dimension}. "
                    f"将触发重建."
                )
                return False

            # 加载 entries
            with open(entries_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._entries = {
                k: TaskIndexEntry.from_dict(v) for k, v in raw.items()
            }

            logger.info(
                f"[TaskSearchIndex] 从磁盘加载成功: "
                f"{len(self._entries)} 条条目, "
                f"FAISS ntotal={self._vector_store.index.ntotal}"
            )
            return True

        except Exception as e:
            logger.warning(f"[TaskSearchIndex] 从磁盘加载失败: {e}")
            return False

    def _maybe_auto_save(self):
        """延迟持久化策略：dirty 计数或时间间隔触发"""
        if self._dirty_count >= _AUTO_SAVE_DIRTY_COUNT:
            self.save()
        elif time.time() - self._last_save_time > _AUTO_SAVE_INTERVAL:
            self.save()

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "entries": len(self._entries),
            "faiss_available": self._faiss_available,
            "embedding_available": self.is_available(),
            "faiss_stats": (
                self._vector_store.get_stats()
                if self._vector_store else None
            ),
        }


# ── 全局单例访问 ──

_index_instance: Optional[HistoricalTaskIndex] = None


def get_task_search_index() -> HistoricalTaskIndex:
    """获取全局 HistoricalTaskIndex 单例（延迟初始化）"""
    global _index_instance
    if _index_instance is None:
        _index_instance = HistoricalTaskIndex()
    return _index_instance
