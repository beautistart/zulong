# 复盘机制：三重防重复机制

"""
功能:
- 事件级过滤 (失败必复盘、成功抽样)
- 内容级过滤 (向量查重，相似度>0.95)
- 时间级过滤 (1 小时窗口聚合)

对应 TSD v2.3 第 11.3 节
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class DeduplicationFilter:
    """三重防重复过滤器"""
    
    def __init__(self,
                 similarity_threshold: float = 0.95,
                 time_window_minutes: int = 60,
                 success_sampling_rate: float = 0.1,
                 embedding_model=None):
        """初始化过滤器
        
        Args:
            similarity_threshold: 相似度阈值 (0-1)
            time_window_minutes: 时间窗口 (分钟)
            success_sampling_rate: 成功抽样率 (0-1)
            embedding_model: Embedding 模型
        """
        self.similarity_threshold = similarity_threshold
        self.time_window = timedelta(minutes=time_window_minutes)
        self.success_sampling_rate = success_sampling_rate
        self.embedding_model = embedding_model
        
        # 事件记录
        self._recent_events: Dict[str, datetime] = {}
        self._content_hashes: Dict[str, datetime] = {}
        self._time_window_events: Dict[str, List[Dict]] = defaultdict(list)
        
        # 统计信息
        self.stats = {
            'total_events': 0,
            'filtered_by_event': 0,
            'filtered_by_content': 0,
            'filtered_by_time': 0,
            'passed': 0
        }
        
        logger.info(f"[DedupFilter] 初始化完成："
                   f"similarity={similarity_threshold}, "
                   f"time_window={time_window_minutes}min, "
                   f"sampling={success_sampling_rate}")
    
    def should_review(self,
                      event_data: Dict[str, Any],
                      event_type: str) -> Tuple[bool, str]:
        """判断是否需要复盘
        
        Args:
            event_data: 事件数据
            event_type: 事件类型 ('success'/'failure')
            
        Returns:
            Tuple[bool, str]: (是否复盘，原因)
        """
        self.stats['total_events'] += 1
        
        # 1. 事件级过滤
        event_pass, reason = self._event_level_filter(event_data, event_type)
        if not event_pass:
            self.stats['filtered_by_event'] += 1
            return False, reason
        
        # 2. 内容级过滤
        content_pass, reason = self._content_level_filter(event_data)
        if not content_pass:
            self.stats['filtered_by_content'] += 1
            return False, reason
        
        # 3. 时间级过滤
        time_pass, reason = self._time_level_filter(event_data)
        if not time_pass:
            self.stats['filtered_by_time'] += 1
            return False, reason
        
        # 通过所有过滤
        self.stats['passed'] += 1
        return True, "通过所有过滤"
    
    def _event_level_filter(self,
                            event_data: Dict[str, Any],
                            event_type: str) -> Tuple[bool, str]:
        """事件级过滤
        
        规则:
        - 失败事件：必须复盘 (除非重复)
        - 成功事件：抽样复盘
        
        Args:
            event_data: 事件数据
            event_type: 事件类型
            
        Returns:
            Tuple[bool, str]: (是否通过，原因)
        """
        # 失败事件 - 必须复盘
        if event_type == 'failure':
            return True, "失败事件必须复盘"
        
        # 成功事件 - 抽样
        import random
        if random.random() <= self.success_sampling_rate:
            return True, f"成功事件抽样 (rate={self.success_sampling_rate})"
        else:
            return False, f"成功事件未命中抽样"
    
    def _content_level_filter(self,
                               event_data: Dict[str, Any]) -> Tuple[bool, str]:
        """内容级过滤 (向量查重)
        
        Args:
            event_data: 事件数据
            
        Returns:
            Tuple[bool, str]: (是否通过，原因)
        """
        # 提取内容
        content = self._extract_content(event_data)
        
        # 计算内容哈希
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # 检查哈希重复
        if content_hash in self._content_hashes:
            last_seen = self._content_hashes[content_hash]
            if datetime.utcnow() - last_seen < self.time_window:
                return False, f"内容重复 (哈希匹配，{last_seen})"
        
        # 向量相似度检查 (如果有 embedding 模型)
        if self.embedding_model:
            similar_events = self._find_similar_events(content)
            
            if similar_events:
                max_similarity = max(sim[1] for sim in similar_events)
                
                if max_similarity > self.similarity_threshold:
                    return False, f"内容高度相似 (similarity={max_similarity:.3f})"
        
        # 更新哈希记录
        self._content_hashes[content_hash] = datetime.utcnow()
        
        return True, "内容不重复"
    
    def _time_level_filter(self,
                           event_data: Dict[str, Any]) -> Tuple[bool, str]:
        """时间级过滤 (1 小时窗口聚合)
        
        Args:
            event_data: 事件数据
            
        Returns:
            Tuple[bool, str]: (是否通过，原因)
        """
        # 提取任务描述
        task_key = self._extract_task_key(event_data)
        
        # 获取时间窗口内的事件
        now = datetime.utcnow()
        window_start = now - self.time_window
        
        # 清理过期事件
        self._time_window_events[task_key] = [
            event for event in self._time_window_events[task_key]
            if event['timestamp'] > window_start
        ]
        
        # 检查窗口内是否有类似事件
        recent_events = self._time_window_events[task_key]
        
        if recent_events:
            # 检查时间间隔
            last_event = recent_events[-1]
            time_diff = now - last_event['timestamp']
            
            # 如果间隔太短 (< 5 分钟),跳过
            if time_diff < timedelta(minutes=5):
                return False, f"距离上次复盘太近 ({time_diff.total_seconds()/60:.1f} 分钟)"
            
            # 限制窗口内事件数量 (最多 5 个)
            if len(recent_events) >= 5:
                return False, f"窗口内事件过多 ({len(recent_events)} 个)"
        
        # 添加到窗口
        self._time_window_events[task_key].append({
            'timestamp': now,
            'event_data': event_data
        })
        
        return True, "时间窗口检查通过"
    
    def _extract_content(self, event_data: Dict[str, Any]) -> str:
        """提取内容用于查重
        
        Args:
            event_data: 事件数据
            
        Returns:
            str: 内容文本
        """
        parts = []
        
        # 任务描述
        if 'task_description' in event_data:
            parts.append(event_data['task_description'])
        
        # 错误信息 (失败事件)
        if 'error_message' in event_data:
            parts.append(event_data['error_message'])
        
        # 关键步骤 (成功事件)
        if 'key_steps' in event_data:
            if isinstance(event_data['key_steps'], list):
                parts.extend(event_data['key_steps'])
            else:
                parts.append(str(event_data['key_steps']))
        
        # 组合内容
        content = "\n".join(parts)
        
        return content
    
    def _extract_task_key(self, event_data: Dict[str, Any]) -> str:
        """提取任务键 (用于时间窗口聚合)
        
        Args:
            event_data: 事件数据
            
        Returns:
            str: 任务键
        """
        # 优先使用任务描述
        if 'task_description' in event_data:
            return event_data['task_description'][:100]
        
        # 使用事件 ID
        if 'event_id' in event_data:
            return event_data['event_id']
        
        # 使用时间戳 (降级方案)
        return datetime.utcnow().isoformat()
    
    def _find_similar_events(self,
                             content: str,
                             limit: int = 5) -> List[Tuple[str, float]]:
        """查找相似事件 (向量相似度)
        
        Args:
            content: 内容文本
            limit: 返回数量
            
        Returns:
            List[Tuple[str, float]]: [(事件内容，相似度), ...]
        """
        if not self.embedding_model:
            return []
        
        try:
            # TODO: 实现向量相似度搜索
            # 这里需要集成到经验库的向量搜索功能
            
            logger.debug(f"[DedupFilter] 向量相似度搜索：{content[:50]}...")
            
            return []
            
        except Exception as e:
            logger.error(f"[DedupFilter] 向量相似度搜索失败：{e}")
            return []
    
    def cleanup_old_records(self):
        """清理过期记录"""
        now = datetime.utcnow()
        cutoff = now - self.time_window * 2
        
        # 清理内容哈希
        expired_hashes = [
            h for h, t in self._content_hashes.items()
            if t < cutoff
        ]
        for h in expired_hashes:
            del self._content_hashes[h]
        
        # 清理时间窗口事件
        expired_tasks = [
            task for task, events in self._time_window_events.items()
            if not events or all(e['timestamp'] < cutoff for e in events)
        ]
        for task in expired_tasks:
            del self._time_window_events[task]
        
        logger.debug(f"[DedupFilter] 清理完成，删除 {len(expired_hashes)} 个哈希，"
                    f"{len(expired_tasks)} 个任务窗口")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            **self.stats,
            'recent_events_count': len(self._recent_events),
            'content_hashes_count': len(self._content_hashes),
            'time_windows_count': len(self._time_window_events)
        }


# 全局单例
_dedup_filter_instance = None


def get_dedup_filter(
    similarity_threshold: float = 0.95,
    time_window_minutes: int = 60,
    success_sampling_rate: float = 0.1,
    embedding_model=None
) -> DeduplicationFilter:
    """获取防重复过滤器单例
    
    Args:
        similarity_threshold: 相似度阈值
        time_window_minutes: 时间窗口
        success_sampling_rate: 成功抽样率
        embedding_model: Embedding 模型
        
    Returns:
        DeduplicationFilter: 单例实例
    """
    global _dedup_filter_instance
    
    if _dedup_filter_instance is None:
        _dedup_filter_instance = DeduplicationFilter(
            similarity_threshold=similarity_threshold,
            time_window_minutes=time_window_minutes,
            success_sampling_rate=success_sampling_rate,
            embedding_model=embedding_model
        )
    
    return _dedup_filter_instance
