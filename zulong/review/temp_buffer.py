# File: zulong/review/temp_buffer.py
# 复盘临时缓冲区 - 上下文隔离机制

from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import uuid
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReviewConversation:
    """复盘对话记录"""
    id: str
    timestamp: float
    user_input: str
    system_response: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReviewTempBuffer:
    """复盘临时缓冲区
    
    功能：
    - 隔离复盘期间的对话，不污染主记忆池
    - 临时存储待分析的对话内容
    - 支持导出给 L2 进行结构化分析
    - 支持合并到主记忆池（经用户确认后）
    """
    
    def __init__(self, session_id: str):
        """初始化临时缓冲区
        
        Args:
            session_id: 复盘会话 ID
        """
        self.session_id = session_id
        self.created_at = time.time()
        self.conversations: List[ReviewConversation] = []
        self.metadata: Dict[str, Any] = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'conversation_count': 0
        }
        
        logger.info(f"[ReviewTempBuffer] 创建临时缓冲区，会话 ID: {session_id}")
    
    def add_conversation(self, user_input: str, system_response: str = None, tags: List[str] = None):
        """添加对话记录
        
        Args:
            user_input: 用户输入
            system_response: 系统响应（可选）
            tags: 标签
        """
        conversation = ReviewConversation(
            id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            user_input=user_input,
            system_response=system_response,
            tags=tags or []
        )
        
        self.conversations.append(conversation)
        self.metadata['conversation_count'] += 1
        
        logger.debug(f"[ReviewTempBuffer] 添加对话：{user_input[:30]}...")
    
    def add_user_input(self, record: Dict[str, Any]):
        """🔥 新增：添加用户输入记录
        
        Args:
            record: 用户输入记录字典，包含 role, content, timestamp 等
        """
        conversation = ReviewConversation(
            id=str(uuid.uuid4())[:8],
            timestamp=record.get('timestamp', time.time()),
            user_input=record.get('content', ''),
            system_response=None,
            tags=record.get('tags', [])
        )
        
        self.conversations.append(conversation)
        self.metadata['conversation_count'] += 1
        
        logger.debug(f"[ReviewTempBuffer] 添加用户输入：{record.get('content', '')[:30]}...")
    
    def add_system_response(self, record: Dict[str, Any]):
        """🔥 新增：添加系统响应记录
        
        Args:
            record: 系统响应记录字典，包含 role, content, timestamp 等
        """
        # 🔥 关键修复：找到最后一条用户输入，添加系统响应
        if self.conversations:
            last_conv = self.conversations[-1]
            # 如果最后一条是用户输入（没有系统响应），则添加
            if last_conv.system_response is None:
                last_conv.system_response = record.get('content', '')
                logger.debug(f"[ReviewTempBuffer] 已添加系统响应到最后一条对话")
            else:
                # 否则创建新的对话记录
                conversation = ReviewConversation(
                    id=str(uuid.uuid4())[:8],
                    timestamp=record.get('timestamp', time.time()),
                    user_input='',  # 系统响应没有用户输入
                    system_response=record.get('content', ''),
                    tags=record.get('tags', [])
                )
                self.conversations.append(conversation)
                self.metadata['conversation_count'] += 1
                logger.debug(f"[ReviewTempBuffer] 创建新的系统响应记录")
        else:
            # 没有用户输入，创建独立的系统响应记录
            conversation = ReviewConversation(
                id=str(uuid.uuid4())[:8],
                timestamp=record.get('timestamp', time.time()),
                user_input='',
                system_response=record.get('content', ''),
                tags=record.get('tags', [])
            )
            self.conversations.append(conversation)
            self.metadata['conversation_count'] += 1
            logger.debug(f"[ReviewTempBuffer] 创建独立的系统响应记录")
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """获取所有对话记录
        
        Returns:
            List[Dict]: 对话记录列表
        """
        return [
            {
                'id': conv.id,
                'timestamp': conv.timestamp,
                'user_input': conv.user_input,
                'system_response': conv.system_response,
                'tags': conv.tags
            }
            for conv in self.conversations
        ]
    
    def export_for_analysis(self) -> Dict[str, Any]:
        """导出给 L2 进行分析
        
        Returns:
            Dict: 包含所有对话数据的字典
        """
        return {
            'session_id': self.session_id,
            'created_at': self.metadata['created_at'],
            'duration_seconds': time.time() - self.created_at,
            'conversation_count': len(self.conversations),
            'conversations': [
                {
                    'user': conv.user_input,
                    'system': conv.system_response,
                    'timestamp': conv.timestamp
                }
                for conv in self.conversations
            ]
        }
    
    def clear(self):
        """清空缓冲区"""
        self.conversations.clear()
        logger.info(f"[ReviewTempBuffer] 已清空缓冲区，会话 ID: {self.session_id}")


class ReviewBufferManager:
    """复盘缓冲区管理器（单例）
    
    管理全局的复盘临时缓冲区
    """
    
    _instance = None
    _current_buffer: Optional[ReviewTempBuffer] = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化管理器"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            logger.info("[ReviewBufferManager] 初始化完成")
    
    def create_buffer(self, session_id: str) -> ReviewTempBuffer:
        """创建新的复盘缓冲区
        
        Args:
            session_id: 会话 ID
            
        Returns:
            ReviewTempBuffer: 新建的缓冲区
        """
        if self._current_buffer:
            logger.warning(f"[ReviewBufferManager] 发现未清理的旧缓冲区，会话 ID: {self._current_buffer.session_id}")
            # 自动清理旧缓冲区
            self.destroy_buffer()
        
        self._current_buffer = ReviewTempBuffer(session_id)
        return self._current_buffer
    
    def get_buffer(self) -> Optional[ReviewTempBuffer]:
        """获取当前缓冲区
        
        Returns:
            ReviewTempBuffer: 当前缓冲区，如果没有则返回 None
        """
        return self._current_buffer
    
    def has_buffer(self) -> bool:
        """检查是否存在活动的缓冲区
        
        Returns:
            bool: 是否存在
        """
        return self._current_buffer is not None
    
    def destroy_buffer(self) -> Optional[Dict[str, Any]]:
        """销毁当前缓冲区
        
        Returns:
            Dict: 缓冲区数据（用于归档），如果没有则返回 None
        """
        if not self._current_buffer:
            return None
        
        # 导出数据
        data = self._current_buffer.export_for_analysis()
        
        # 清空并销毁
        self._current_buffer.clear()
        self._current_buffer = None
        
        logger.info("[ReviewBufferManager] 已销毁缓冲区")
        return data
    
    def add_to_buffer(self, user_input: str, system_response: str = None, tags: List[str] = None):
        """添加数据到当前缓冲区
        
        Args:
            user_input: 用户输入
            system_response: 系统响应
            tags: 标签
        """
        if self._current_buffer:
            self._current_buffer.add_conversation(user_input, system_response, tags)
        else:
            logger.warning("[ReviewBufferManager] 尝试添加到不存在的缓冲区")
    
    def clear_buffer(self):
        """清空当前缓冲区内容（保留缓冲区实例）"""
        if self._current_buffer:
            self._current_buffer.clear()
            self._current_buffer.metadata['conversation_count'] = 0
            logger.info("[ReviewBufferManager] 已清空缓冲区内容")
        else:
            logger.warning("[ReviewBufferManager] 尝试清空不存在的缓冲区")
    
    def add_user_input(self, record: Dict[str, Any]):
        """添加用户输入到当前缓冲区（代理方法）
        
        Args:
            record: 用户输入记录
        """
        if self._current_buffer:
            self._current_buffer.add_user_input(record)
        else:
            logger.warning("[ReviewBufferManager] 尝试添加用户输入到不存在的缓冲区")
    
    def add_system_response(self, record: Dict[str, Any]):
        """添加系统响应到当前缓冲区（代理方法）
        
        Args:
            record: 系统响应记录
        """
        if self._current_buffer:
            self._current_buffer.add_system_response(record)
        else:
            logger.warning("[ReviewBufferManager] 尝试添加系统响应到不存在的缓冲区")


# 全局单例
review_buffer_manager = ReviewBufferManager()


def get_review_buffer_manager() -> ReviewBufferManager:
    """获取复盘缓冲区管理器实例
    
    Returns:
        ReviewBufferManager: 管理器实例
    """
    return review_buffer_manager
