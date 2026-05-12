# File: zulong/review/state.py
# 复盘状态管理器 - 显性化复盘流程

from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReviewState:
    """复盘状态管理器
    
    功能：
    - 管理复盘模式的全局状态
    - 提供状态反馈标记
    - 记录复盘会话信息
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化复盘状态"""
        if not hasattr(self, '_initialized'):
            # 核心状态标志
            self.is_active = False  # 是否处于复盘模式
            self.mode = None  # 'quick' | 'deep' | None
            self.session_id = None  # 当前会话 ID
            self.start_time = None  # 开始时间
            
            # 过程状态
            self.current_stage = None  # 当前阶段：'selecting' | 'analyzing' | 'reviewing' | 'confirming'
            self.status_message = ""  # 当前状态消息
            
            # 结果数据
            self.experience_count = 0  # 生成的经验数量
            self.confirmed = False  # 是否已确认
            
            self._initialized = True
            logger.info("[ReviewState] 初始化完成")
    
    def enter_review_mode(self, mode: str, session_id: str):
        """进入复盘模式
        
        Args:
            mode: 复盘模式 'quick' | 'deep'
            session_id: 会话 ID
        """
        self.is_active = True
        self.mode = mode
        self.session_id = session_id
        self.start_time = datetime.now()
        self.current_stage = 'selecting'
        self.status_message = "已进入复盘模式"
        self.confirmed = False
        
        mode_name = "快速" if mode == 'quick' else "深度"
        logger.info(f"[ReviewState] 进入{mode_name}复盘模式，会话 ID: {session_id}")
    
    def update_stage(self, stage: str, message: str = ""):
        """更新当前阶段
        
        Args:
            stage: 阶段名称
            message: 状态消息
        """
        self.current_stage = stage
        self.status_message = message
        logger.info(f"[ReviewState] 阶段更新：{stage} - {message}")
    
    def set_analyzing(self):
        """设置为分析中状态"""
        self.update_stage('analyzing', '正在检索记忆库和分析对话...')
    
    def set_generating(self):
        """设置为生成经验中"""
        self.update_stage('generating', '正在提炼经验...')
    
    def set_waiting_confirmation(self, count: int):
        """设置为等待确认状态
        
        Args:
            count: 经验数量
        """
        self.experience_count = count
        self.update_stage('confirming', f'已生成 {count} 条经验，等待用户确认')
    
    def confirm_completion(self):
        """确认完成"""
        self.confirmed = True
        self.update_stage('completed', '复盘完成')
    
    def exit_review_mode(self):
        """退出复盘模式"""
        duration = None
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
        
        self.is_active = False
        self.mode = None
        self.session_id = None
        self.current_stage = None
        self.status_message = ""
        
        if duration is not None:
            logger.info(f"[ReviewState] 退出复盘模式，耗时：{duration:.2f}秒")
        else:
            logger.info("[ReviewState] 退出复盘模式")
        return duration
    
    def get_status_indicator(self) -> str:
        """获取状态指示器（用于 UI 显示）
        
        Returns:
            str: 状态指示字符串
        """
        if not self.is_active:
            return ""
        
        indicators = {
            'selecting': '🤔 选择模式',
            'analyzing': '🔍 分析中...',
            'generating': '💡 提炼经验...',
            'confirming': '✅ 等待确认',
            'completed': '✨ 已完成'
        }
        
        return indicators.get(self.current_stage, '📊 复盘中')
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息
        
        Returns:
            Dict: 会话信息
        """
        return {
            'is_active': self.is_active,
            'mode': self.mode,
            'session_id': self.session_id,
            'stage': self.current_stage,
            'status': self.status_message,
            'experience_count': self.experience_count,
            'confirmed': self.confirmed
        }


# 全局单例
review_state = ReviewState()


def get_review_state() -> ReviewState:
    """获取复盘状态实例
    
    Returns:
        ReviewState: 状态实例
    """
    return review_state
