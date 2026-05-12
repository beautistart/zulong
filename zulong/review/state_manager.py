# zulong/review/state_manager.py
"""
复盘状态管理器 - 统一管理复盘相关状态

对应 TSD v2.3 第 11.1 节
"""

import threading
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from zulong.core.types import L2Status
from zulong.core.state_manager import state_manager

logger = logging.getLogger(__name__)


class ReviewMode(Enum):
    """复盘模式枚举"""
    QUICK = "quick"  # 快速复盘
    DEEP = "deep"    # 深度复盘


class ReviewStage(Enum):
    """复盘阶段枚举"""
    # 🔥 新增：三阶段状态机
    MODE_SELECTING = "mode_selecting"          # 模式选择阶段（等待用户选择快速/深度）
    REVIEW_ACTIVE = "review_active"            # 对话进行阶段（L2 正常对话，后台监听）
    EXPERIENCE_CONFIRMING = "experience_confirming"  # 经验确认阶段（等待用户确认/修改）
    
    # 原有阶段（保留用于子流程）
    SELECTING = "selecting"                    # 选择模式（兼容旧代码）
    ANALYZING = "analyzing"                    # 分析中
    GENERATING = "generating"                  # 生成经验中
    WAITING_CONFIRMATION = "waiting_confirmation"  # 等待确认
    CONFIRMING = "confirming"                  # 确认中
    COMPLETED = "completed"                    # 已完成
    CANCELLED = "cancelled"                    # 已取消
    FAILED = "failed"                          # 失败


class ReviewStateManager:
    """复盘状态管理器 - 🔥 单例模式
    
    功能：
    - 统一管理复盘相关的所有状态
    - 提供线程安全的状态访问
    - 记录复盘会话信息
    - 支持状态变更回调
    
    核心优势：
    - 避免分散的状态管理导致的同步问题
    - 提供显性的状态流转追踪
    - 支持并发场景下的状态一致性
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式 - 线程安全"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化复盘状态管理器"""
        if not hasattr(self, '_initialized'):
            # 🔒 线程锁
            self._state_lock = threading.Lock()
            
            # 核心状态标志
            self._is_active = False  # 是否处于复盘模式
            self._mode: Optional[ReviewMode] = None  # 复盘模式
            self._stage: Optional[ReviewStage] = None  # 当前阶段
            self._session_id: Optional[str] = None  # 当前会话 ID
            self._start_time: Optional[datetime] = None  # 开始时间
            
            # 过程状态
            self._is_processing = False  # 是否正在处理中（防重入锁）
            self._experience_count: int = 0  # 生成的经验数量
            self._confirmed: bool = False  # 是否已确认
            self._status_message: str = ""  # 当前状态消息
            
            # 数据缓存
            self._pending_experiences: Optional[list] = None  # 待确认的经验草案
            self._pending_summary: str = ""  # 待确认的总结
            self._pending_tags: list = []  # 待确认的标签
            
            # 回调函数
            self._state_change_callbacks: list = []
            
            self._initialized = True
            logger.info("[ReviewStateManager] 初始化完成")
    
    def _notify_state_change(self, action: str):
        """通知状态变更
        
        Args:
            action: 变更动作描述
        """
        try:
            # 同步到全局 state_manager
            state_manager.set_context('review_is_active', self._is_active)
            state_manager.set_context('review_stage', self._stage.value if self._stage else None)
            # 🔥 [BUG-10 修复] 只在 _mode 有值或明确退出时才同步 review_mode
            # 避免 acquire/release_processing_lock 时用 None 覆盖外部设置的 True
            if self._mode is not None:
                state_manager.set_context('review_mode', self._mode.value)
            elif not self._is_active and action in ('exit_review_mode', 'force_exit'):
                # 明确退出时才清除
                state_manager.set_context('review_mode', False)
            state_manager.set_context('review_status_message', self._status_message)
            
            # 触发回调函数
            for callback in self._state_change_callbacks:
                try:
                    callback(action, {
                        'is_active': self._is_active,
                        'stage': self._stage.value if self._stage else None,
                        'mode': self._mode.value if self._mode else None,
                        'message': self._status_message
                    })
                except Exception as e:
                    logger.error(f"状态变更回调失败：{e}")
        except Exception as e:
            logger.warning(f"同步状态到全局 state_manager 失败：{e}")
    
    # ========== 核心状态管理 ==========
    
    def enter_review_mode(self, mode: ReviewMode, session_id: str):
        """进入复盘模式
        
        Args:
            mode: 复盘模式（快速/深度）
            session_id: 会话 ID
        """
        with self._state_lock:
            self._is_active = True
            self._mode = mode
            self._session_id = session_id
            self._start_time = datetime.now()
            self._stage = ReviewStage.MODE_SELECTING  # 🔥 新增：初始进入模式选择阶段
            self._is_processing = False
            self._confirmed = False
            self._status_message = "已进入复盘模式，请选择快速或深度"
            
            # 同步到全局 state_manager
            try:
                state_manager.set_context('review_mode', True)
                state_manager.set_context('review_session_id', session_id)
                state_manager.set_context('review_type', mode.value)
                state_manager.set_context('review_stage', 'mode_selecting')  # 🔥 新增
            except Exception as e:
                logger.warning(f"同步到 state_manager 失败：{e}")
            
            mode_name = "快速" if mode == ReviewMode.QUICK else "深度"
            logger.info(f"[ReviewStateManager] 进入{mode_name}复盘模式，会话 ID: {session_id}")
            self._notify_state_change('enter_review_mode')
    
    def enter_mode_selecting(self):
        """🔥 新增：进入模式选择阶段"""
        with self._state_lock:
            self._stage = ReviewStage.MODE_SELECTING
            self._status_message = "请选择复盘模式：快速复盘 或 深度复盘"
            # 同步到全局 state_manager
            try:
                state_manager.set_context('review_stage', 'mode_selecting')
            except Exception as e:
                logger.warning(f"同步阶段状态失败：{e}")
            logger.info("[ReviewStateManager] 进入模式选择阶段")
            self._notify_state_change('enter_mode_selecting')
    
    def enter_review_active(self, review_type: str = 'quick'):
        """🔥 新增：进入对话进行阶段
        
        Args:
            review_type: 复盘类型 ('quick' | 'deep')
        """
        with self._state_lock:
            self._stage = ReviewStage.REVIEW_ACTIVE
            self._status_message = "复盘对话中，说'结束复盘'完成复盘"
            # 同步到全局 state_manager
            try:
                state_manager.set_context('review_stage', 'review_active')  # 🔥 修复：使用正确的阶段值
                state_manager.set_context('review_type', review_type)
            except Exception as e:
                logger.warning(f"同步阶段状态失败：{e}")
            logger.info(f"[ReviewStateManager] 进入对话进行阶段 ({review_type})")
            self._notify_state_change('enter_review_active')
    
    def enter_experience_confirming(self, experience_count: int):
        """🔥 新增：进入经验确认阶段
        
        Args:
            experience_count: 生成的经验数量
        """
        with self._state_lock:
            self._stage = ReviewStage.EXPERIENCE_CONFIRMING
            self._experience_count = experience_count
            self._status_message = f"已生成{experience_count}条经验，请说'确认'保存或'修改'重新对话"
            # 同步到全局 state_manager
            try:
                state_manager.set_context('review_stage', 'experience_confirming')  # 🔥 修复：使用正确的阶段值
            except Exception as e:
                logger.warning(f"同步阶段状态失败：{e}")
            logger.info(f"[ReviewStateManager] 进入经验确认阶段，经验数量：{experience_count}")
            self._notify_state_change('enter_experience_confirming')
    
    def exit_review_mode(self, reason: str = 'completed'):
        """退出复盘模式
        
        Args:
            reason: 退出原因 ('completed' | 'cancelled' | 'failed')
        """
        with self._state_lock:
            duration = None
            if self._start_time:
                duration = (datetime.now() - self._start_time).total_seconds()
            
            # 根据原因设置最终阶段
            if reason == 'completed':
                self._stage = ReviewStage.COMPLETED
            elif reason == 'cancelled':
                self._stage = ReviewStage.CANCELLED
            elif reason == 'failed':
                self._stage = ReviewStage.FAILED
            
            # 清理状态
            self._is_active = False
            self._mode = None
            self._session_id = None
            self._is_processing = False
            self._status_message = ""
            
            # 清理缓存数据
            self._pending_experiences = None
            self._pending_summary = ""
            self._pending_tags = []
            
            # 同步到全局 state_manager
            try:
                state_manager.set_context('review_mode', False)
                state_manager.set_context('review_session_id', None)
                state_manager.set_context('review_type', None)
                state_manager.set_context('_review_processing', False)
            except Exception as e:
                logger.warning(f"同步到 state_manager 失败：{e}")
            
            # 重置 L2 状态
            try:
                state_manager.set_l2_status(L2Status.IDLE)
            except Exception as e:
                logger.error(f"重置 L2 状态失败：{e}")
            
            if duration is not None:
                logger.info(f"[ReviewStateManager] 退出复盘模式，耗时：{duration:.2f}秒，原因：{reason}")
            else:
                logger.info(f"[ReviewStateManager] 退出复盘模式，原因：{reason}")
            
            self._notify_state_change('exit_review_mode')
            return duration
    
    def force_exit(self):
        """🔥 强制退出复盘模式（异常情况下使用）"""
        logger.warning("[ReviewStateManager] 强制退出复盘模式")
        
        with self._state_lock:
            # 清理所有状态
            self._is_active = False
            self._mode = None
            self._session_id = None
            self._stage = ReviewStage.FAILED
            self._is_processing = False
            self._status_message = "异常退出"
            
            # 清理缓存数据
            self._pending_experiences = None
            self._pending_summary = ""
            self._pending_tags = []
            
            # 强制同步到全局 state_manager
            try:
                state_manager.set_context('review_mode', False)
                state_manager.set_context('review_session_id', None)
                state_manager.set_context('review_type', None)
                state_manager.set_context('_review_processing', False)
                state_manager.set_l2_status(L2Status.IDLE)
                logger.info("[ReviewStateManager] ✅ 已强制重置所有状态")
            except Exception as e:
                logger.error(f"[ReviewStateManager] 强制退出时同步状态失败：{e}")
            
            self._notify_state_change('force_exit')
    
    # ========== 阶段管理 ==========
    
    def update_stage(self, stage: ReviewStage, message: str = ""):
        """更新当前阶段
        
        Args:
            stage: 阶段枚举
            message: 状态消息
        """
        with self._state_lock:
            self._stage = stage
            self._status_message = message
            logger.info(f"[ReviewStateManager] 阶段更新：{stage.name} - {message}")
            self._notify_state_change('update_stage')
    
    def set_analyzing(self):
        """设置为分析中状态"""
        self.update_stage(ReviewStage.ANALYZING, '正在检索记忆库和分析对话...')
    
    def set_generating(self):
        """设置为生成经验中"""
        self.update_stage(ReviewStage.GENERATING, '正在提炼经验...')
    
    def set_waiting_confirmation(self, count: int):
        """设置为等待确认状态
        
        Args:
            count: 生成的经验数量
        """
        with self._state_lock:
            self._experience_count = count
            self.update_stage(ReviewStage.WAITING_CONFIRMATION, f'已生成 {count} 条经验，等待用户确认')
    
    def set_confirming(self):
        """设置为确认中"""
        self.update_stage(ReviewStage.CONFIRMING, '正在确认经验...')
    
    def set_completed(self):
        """设置为已完成"""
        self.update_stage(ReviewStage.COMPLETED, '复盘完成')
    
    # ========== 防重入锁管理 ==========
    
    def acquire_processing_lock(self) -> bool:
        """获取处理锁
        
        Returns:
            bool: 是否成功获取（True=获取成功，False=已有锁）
        """
        with self._state_lock:
            if self._is_processing:
                logger.debug("[ReviewStateManager] 处理锁已被占用，拒绝重复请求")
                return False
            
            self._is_processing = True
            # 同步到全局 state_manager
            try:
                state_manager.set_context('_review_processing', True)
            except Exception as e:
                logger.warning(f"同步处理锁状态失败：{e}")
            
            logger.info("[ReviewStateManager] 🔒 已获取处理锁")
            self._notify_state_change('acquire_lock')
            return True
    
    def release_processing_lock(self):
        """释放处理锁"""
        with self._state_lock:
            self._is_processing = False
            # 同步到全局 state_manager
            try:
                state_manager.set_context('_review_processing', False)
            except Exception as e:
                logger.warning(f"同步处理锁状态失败：{e}")
            
            logger.info("[ReviewStateManager] 🔓 已释放处理锁")
            self._notify_state_change('release_lock')
    
    # ========== 数据管理 ==========
    
    def set_pending_experiences(self, experiences: list, summary: str = "", tags: list = None):
        """设置待确认的经验数据
        
        Args:
            experiences: 经验草案列表
            summary: 总结
            tags: 标签列表
        """
        with self._state_lock:
            self._pending_experiences = experiences
            self._pending_summary = summary
            self._pending_tags = tags or []
            logger.info(f"[ReviewStateManager] 设置待确认经验：{len(experiences)} 条")
    
    def confirm_experiences(self):
        """确认经验"""
        with self._state_lock:
            self._confirmed = True
            logger.info("[ReviewStateManager] 经验已确认")
    
    def get_pending_experiences(self) -> Optional[list]:
        """获取待确认的经验数据
        
        Returns:
            Optional[list]: 经验草案列表，如果没有则为 None
        """
        with self._state_lock:
            return self._pending_experiences
    
    def get_experience_count(self) -> int:
        """获取经验数量"""
        with self._state_lock:
            return self._experience_count
    
    # ========== 便捷方法 ==========
    
    # 🔥 新增：三阶段状态机便捷方法
    def is_mode_selecting(self) -> bool:
        """是否处于模式选择阶段"""
        with self._state_lock:
            return self._is_active and self._stage == ReviewStage.MODE_SELECTING
    
    def is_review_active(self) -> bool:
        """是否处于对话进行阶段"""
        with self._state_lock:
            return self._is_active and self._stage == ReviewStage.REVIEW_ACTIVE
    
    def is_experience_confirming(self) -> bool:
        """是否处于经验确认阶段"""
        with self._state_lock:
            return self._is_active and self._stage == ReviewStage.EXPERIENCE_CONFIRMING
    
    # ========== 状态查询 ==========
    
    def is_active(self) -> bool:
        """是否处于复盘模式"""
        with self._state_lock:
            return self._is_active
    
    def get_mode(self) -> Optional[ReviewMode]:
        """获取复盘模式"""
        with self._state_lock:
            return self._mode
    
    def get_stage(self) -> Optional[ReviewStage]:
        """获取当前阶段"""
        with self._state_lock:
            return self._stage
    
    def get_session_id(self) -> Optional[str]:
        """获取会话 ID"""
        with self._state_lock:
            return self._session_id
    
    def is_processing(self) -> bool:
        """是否正在处理中"""
        with self._state_lock:
            return self._is_processing
    
    def get_status_message(self) -> str:
        """获取状态消息"""
        with self._state_lock:
            return self._status_message
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息字典"""
        with self._state_lock:
            duration = 0.0
            if self._start_time:
                duration = (datetime.now() - self._start_time).total_seconds()
            
            return {
                'is_active': self._is_active,
                'mode': self._mode.value if self._mode else None,
                'stage': self._stage.value if self._stage else None,
                'session_id': self._session_id,
                'is_processing': self._is_processing,
                'experience_count': self._experience_count,
                'confirmed': self._confirmed,
                'status_message': self._status_message,
                'duration': duration,
                'start_time': self._start_time.isoformat() if self._start_time else None
            }
    
    def get_status_indicator(self) -> str:
        """获取状态指示器（用于 UI 显示）"""
        with self._state_lock:
            if not self._is_active:
                return ''
            
            stage_indicators = {
                ReviewStage.MODE_SELECTING: '🤔 选择模式',
                ReviewStage.REVIEW_ACTIVE: '💬 复盘对话中',
                ReviewStage.EXPERIENCE_CONFIRMING: '✅ 确认经验',
                ReviewStage.ANALYZING: '🔍 分析中...',
                ReviewStage.GENERATING: '💡 提炼经验...',
                ReviewStage.WAITING_CONFIRMATION: '⏳ 等待确认',
                ReviewStage.CONFIRMING: '✔️ 确认中...',
                ReviewStage.COMPLETED: '✨ 已完成',
                ReviewStage.CANCELLED: '❌ 已取消',
                ReviewStage.FAILED: '⚠️ 失败'
            }
            
            return stage_indicators.get(self._stage, '🔄 处理中')
    
    # ========== 流程控制 ==========
    
    def can_accept_input(self) -> bool:
        """是否可以接受用户输入
        
        Returns:
            bool: 是否可以接受输入
        """
        with self._state_lock:
            # 非复盘模式：可以接受
            if not self._is_active:
                return True
            
            # 处理中：不接受
            if self._is_processing:
                return False
            
            # 🔥 新增：模式选择阶段和确认阶段可以接受输入
            if self._stage in [ReviewStage.MODE_SELECTING, ReviewStage.EXPERIENCE_CONFIRMING]:
                return True
            
            # 对话进行阶段：可以接受（正常对话）
            if self._stage == ReviewStage.REVIEW_ACTIVE:
                return True
            
            # 其他阶段：不接受
            return False
    
    def should_forward_to_replay(self) -> bool:
        """是否应该转发到 ReplayIntegration
        
        Returns:
            bool: 是否应该转发
        """
        with self._state_lock:
            # 🔥 修改：复盘模式下且未处理中，应该转发
            return self._is_active and not self._is_processing


# ========== 全局单例 ==========

_review_state_manager_instance: Optional[ReviewStateManager] = None


def get_review_state_manager() -> ReviewStateManager:
    """获取复盘状态管理器单例
    
    Returns:
        ReviewStateManager: 单例实例
    """
    global _review_state_manager_instance
    if _review_state_manager_instance is None:
        _review_state_manager_instance = ReviewStateManager()
    return _review_state_manager_instance


def reset_review_state_manager():
    """重置复盘状态管理器（仅用于测试）"""
    global _review_state_manager_instance
    _review_state_manager_instance = None
    logger.info("[ReviewStateManager] 已重置单例实例")
