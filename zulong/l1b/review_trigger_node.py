# File: zulong/l1b/review_trigger_node.py
# 复盘触发节点 - 检测"启动复盘"关键词并触发复盘机制
# 🔥 修复版本：支持三阶段状态机，根据阶段分发用户输入

from typing import Dict, Any
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class ReviewTriggerNode:
    """复盘触发节点
    
    功能:
    - 检测用户输入是否包含"启动复盘"关键词
    - 精确匹配，避免误触
    - 🔥 新增：根据复盘阶段分发用户输入到不同处理方法
    - 触发复盘机制，让 L2 进入复盘状态
    """
    
    def __init__(self, review_trigger=None):
        """初始化复盘触发节点
        
        Args:
            review_trigger: ReviewTrigger 实例 (可选，延迟注入)
        """
        self.review_trigger = review_trigger
        self.trigger_keyword = "启动复盘"
        logger.info("[ReviewTriggerNode] 初始化完成 - 🔥 修复版本")
    
    def set_review_trigger(self, review_trigger):
        """注入 ReviewTrigger 实例
        
        Args:
            review_trigger: ReviewTrigger 实例
        """
        self.review_trigger = review_trigger
        logger.info("[ReviewTriggerNode] ReviewTrigger 已注入")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行复盘触发逻辑 - 🔥 修复：支持三阶段状态机
        
        Args:
            state: 当前系统状态
            
        Returns:
            Dict: 更新后的状态
        """
        try:
            # 获取用户输入
            user_input = state.get('user_input', '')
            
            # 🔥 关键修复：从全局状态管理器读取复盘模式状态
            review_mode = False
            try:
                from zulong.core.state_manager import state_manager
                review_mode = state_manager.get_context('review_mode', False)
            except Exception as e:
                logger.debug(f"[ReviewTriggerNode] 读取全局状态失败：{e}")
            
            # 如果全局状态是 False，再检查本地 state（向后兼容）
            if not review_mode:
                review_mode = state.get('review_mode', False)
            
            if review_mode:
                # 🔥 复盘模式下，根据阶段分发用户输入
                logger.info(f"[ReviewTriggerNode] 复盘模式下收到用户输入：{user_input}")
                
                # 🔥 获取当前复盘阶段
                review_stage = None
                try:
                    review_stage = state_manager.get_context('review_stage')
                except Exception as e:
                    logger.debug(f"[ReviewTriggerNode] 读取阶段状态失败：{e}")
                
                logger.info(f"[ReviewTriggerNode] 当前复盘阶段：{review_stage}")
                
                # 🔥 检查是否是退出指令（任何阶段都适用）
                if user_input.strip() in ['退出复盘', '结束复盘', '退出', '结束']:
                    logger.info("[ReviewTriggerNode] 用户退出复盘模式")
                    self._handle_exit_review_mode(state)
                    return state
                
                # 🔥 获取 ReplayIntegration 实例
                from zulong.review.integration import get_replay_integration
                replay_integration = get_replay_integration()
                
                # 🔥 根据阶段分发处理
                if review_stage == 'mode_selecting':
                    # 模式选择阶段：处理快速/深度选择
                    logger.info("[ReviewTriggerNode] 模式选择阶段，处理用户选择")
                    asyncio.create_task(
                        self._handle_mode_selection_async(replay_integration, user_input)
                    )
                elif review_stage == 'review_active':
                    # 🔥 修复版：对话进行阶段 - 检测结束复盘 + 记录对话
                    logger.info(f"[ReviewTriggerNode] 对话进行阶段，输入：{user_input}")
                    
                    # 🔥 关键修复：检测结束复盘关键词
                    if user_input.strip() in ['结束复盘', '完成复盘', '结束', '完成']:
                        logger.info("[ReviewTriggerNode] 🔍 检测到结束复盘指令！")
                        asyncio.create_task(
                            self._handle_end_review_async(replay_integration, user_input)
                        )
                    else:
                        # 🔥 修复：记录用户输入和 L2 回复到缓冲区（用于后续经验提取）
                        try:
                            self._record_to_buffer(user_input, state)
                            logger.debug(f"[ReviewTriggerNode] 已记录对话到缓冲区")
                        except Exception as e:
                            logger.warning(f"[ReviewTriggerNode] 记录对话失败：{e}")
                        
                        # 返回原始 state，不阻断后续处理（让其他节点转发给 L2）
                        return state
                        
                elif review_stage == 'experience_confirming':
                    # 经验确认阶段：处理确认/修改/退出
                    logger.info("[ReviewTriggerNode] 经验确认阶段，处理用户确认")
                    asyncio.create_task(
                        self._handle_experience_confirmation_async(replay_integration, user_input)
                    )
                else:
                    # 未知阶段：默认按确认处理（向后兼容）
                    logger.warning(f"[ReviewTriggerNode] 未知阶段 {review_stage}，按确认处理")
                    asyncio.create_task(
                        self._handle_user_confirmation_async(replay_integration, user_input)
                    )
                
                logger.info("[ReviewTriggerNode] 已分发用户输入到对应处理方法")
                return state
            
            # 正常模式：检测"启动复盘"关键词
            if user_input.strip() == self.trigger_keyword:
                logger.info(f"[ReviewTriggerNode] 检测到复盘关键词：'{user_input}'")
                
                # 设置复盘标志
                state['review_mode'] = True
                state['review_intent_detected'] = True
                
                # 设置高优先级
                from zulong.core.types import EventPriority
                state['priority'] = EventPriority.HIGH
                
                logger.info("[ReviewTriggerNode] 已设置复盘模式，L2 将进入复盘状态")
                
                # 如果 ReviewTrigger 已注入，立即触发
                if self.review_trigger:
                    # 创建异步任务触发复盘
                    asyncio.create_task(
                        self._trigger_review_async(state)
                    )
                    logger.info("[ReviewTriggerNode] 已异步触发复盘机制")
                else:
                    logger.warning("[ReviewTriggerNode] ReviewTrigger 未注入，仅设置标志位")
            else:
                # 不是复盘指令，保持正常状态
                if 'review_mode' not in state:
                    state['review_mode'] = False
                    state['review_intent_detected'] = False
            
            return state
            
        except Exception as e:
            logger.error(f"[ReviewTriggerNode] 错误：{e}", exc_info=True)
            # 异常时返回原状态，不影响系统运行
            return state
    
    def _handle_exit_review_mode(self, state: Dict[str, Any]):
        """处理退出复盘模式
        
        Args:
            state: 当前系统状态
        """
        try:
            # 清理状态
            state['review_mode'] = False
            state['review_intent_detected'] = False
            
            # 通知 ReplayIntegration 清理状态
            if self.review_trigger:
                from zulong.review.integration import get_replay_integration
                replay_integration = get_replay_integration()
                replay_integration.review_mode = False
                replay_integration.pending_experiences = None
            
            # 重置 L2 状态
            from zulong.core.types import EventType, EventPriority, ZulongEvent
            from zulong.core.event_bus import event_bus
            from zulong.core.types import L2Status
            
            event = ZulongEvent(
                type=EventType.L2_OUTPUT,
                source="ReviewTriggerNode",
                payload={
                    'text': '好的，已退出复盘模式。我们继续正常对话吧！😊',
                    'session_id': None,
                    'review_mode': False
                },
                priority=EventPriority.NORMAL
            )
            
            # 重置 L2 状态为 IDLE
            from zulong.core.state_manager import state_manager
            state_manager.set_l2_status(L2Status.IDLE)
            
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(event_bus.publish, event)
            
            logger.info("[ReviewTriggerNode] ✅ 已退出复盘模式，L2 状态已重置")
            
        except Exception as e:
            logger.error(f"[ReviewTriggerNode] 退出复盘模式失败：{e}", exc_info=True)
    
    async def _handle_mode_selection_async(self, replay_integration, user_input: str):
        """异步处理模式选择
        
        Args:
            replay_integration: ReplayIntegration 实例
            user_input: 用户输入
        """
        try:
            logger.info(f"[ReviewTriggerNode] 模式选择输入：{user_input}")
            
            # 调用 ReplayIntegration 的模式选择处理方法
            if hasattr(replay_integration, 'handle_mode_selection'):
                await replay_integration.handle_mode_selection(user_input)
            else:
                # 向后兼容：使用旧方法
                replay_integration.handle_user_confirmation(user_input)
            
            logger.info(f"[ReviewTriggerNode] 模式选择处理完成：{user_input}")
        except Exception as e:
            logger.error(f"[ReviewTriggerNode] 处理模式选择失败：{e}", exc_info=True)
    
    async def _handle_end_review_async(self, replay_integration, user_input: str):
        """🔥 新增：异步处理结束复盘指令
        
        当用户在对话进行阶段输入"结束复盘"时调用。
        触发经验提取流程，进入经验确认阶段。
        
        Args:
            replay_integration: ReplayIntegration 实例
            user_input: 用户输入（应该是"结束复盘"）
        """
        try:
            logger.info(f"[ReviewTriggerNode] 🎯 处理结束复盘指令：{user_input}")
            
            # 调用 ReplayIntegration 的结束复盘处理方法
            if hasattr(replay_integration, 'handle_end_review'):
                await replay_integration.handle_end_review()
            else:
                # 向后兼容：直接触发快速分析流程
                logger.warning("[ReviewTriggerNode] ReplayIntegration 没有 handle_end_review 方法，使用替代方案")
                
                # 从缓冲区获取数据并执行分析
                from zulong.review.temp_buffer import get_review_buffer_manager
                buffer_manager = get_review_buffer_manager()
                
                if buffer_manager.has_buffer():
                    buffer_data = buffer_manager.get_buffer().export_for_analysis()
                    if hasattr(replay_integration, 'trigger_experience_extraction'):
                        await replay_integration.trigger_experience_extraction(buffer_data)
                    else:
                        logger.error("[ReviewTriggerNode] 无法触发经验提取！")
                        
            logger.info(f"[ReviewTriggerNode] ✅ 结束复盘处理完成")
            
        except Exception as e:
            logger.error(f"[ReviewTriggerNode] 处理结束复盘失败：{e}", exc_info=True)
    
    def _record_to_buffer(self, user_input: str, state: Dict[str, Any] = None):
        """🔥 修复：记录用户输入和 L2 回复到临时缓冲区
        
        在对话进行阶段，将用户的每条输入和 L2 的回复都记录到缓冲区，
        以便后续进行经验提取。
        
        Args:
            user_input: 用户输入文本
            state: 当前系统状态（用于获取 L2 回复）
        """
        try:
            from zulong.review.temp_buffer import get_review_buffer_manager
            buffer_manager = get_review_buffer_manager()
            
            if not buffer_manager.has_buffer():
                logger.warning("[ReviewTriggerNode] 缓冲区不存在，无法记录")
                return
            
            # 🔥 记录用户输入
            user_record = {
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'review_conversation'
            }
            
            buffer_manager.add_user_input(user_record)
            logger.debug(f"[ReviewTriggerNode] 已记录用户输入到缓冲区")
            
            # 🔥 关键修复：记录 L2 回复（如果有）
            if state:
                l2_response = state.get('l2_response') or state.get('ai_response') or state.get('last_l2_output')
                
                if l2_response:
                    l2_record = {
                        'role': 'assistant',
                        'content': l2_response,
                        'timestamp': datetime.utcnow().isoformat(),
                        'source': 'review_conversation'
                    }
                    
                    buffer_manager.add_system_response(l2_record)
                    logger.debug(f"[ReviewTriggerNode] 已记录 L2 回复到缓冲区")
                else:
                    logger.debug(f"[ReviewTriggerNode] 未找到 L2 回复，仅记录用户输入")
            
        except Exception as e:
            logger.error(f"[ReviewTriggerNode] 记录到缓冲区失败：{e}", exc_info=True)
    
    async def _handle_experience_confirmation_async(self, replay_integration, user_input: str):
        """异步处理经验确认
        
        Args:
            replay_integration: ReplayIntegration 实例
            user_input: 用户输入
        """
        try:
            logger.info(f"[ReviewTriggerNode] 经验确认输入：{user_input}")
            
            # 调用 ReplayIntegration 的经验确认处理方法
            if hasattr(replay_integration, 'handle_experience_confirmation'):
                await replay_integration.handle_experience_confirmation(user_input)
            else:
                # 向后兼容：使用旧方法
                replay_integration.handle_user_confirmation(user_input)
            
            logger.info(f"[ReviewTriggerNode] 经验确认处理完成：{user_input}")
        except Exception as e:
            logger.error(f"[ReviewTriggerNode] 处理经验确认失败：{e}", exc_info=True)
    
    async def _handle_user_confirmation_async(self, replay_integration, user_input: str):
        """异步处理用户确认（向后兼容）
        
        Args:
            replay_integration: ReplayIntegration 实例
            user_input: 用户输入
        """
        try:
            replay_integration.handle_user_confirmation(user_input)
            logger.info(f"[ReviewTriggerNode] 用户确认处理完成：{user_input}")
        except Exception as e:
            logger.error(f"[ReviewTriggerNode] 处理用户确认失败：{e}", exc_info=True)
    
    async def _trigger_review_async(self, state: Dict[str, Any]):
        """异步触发复盘机制
        
        Args:
            state: 当前系统状态
        """
        try:
            if self.review_trigger:
                from zulong.review.trigger import TriggerType
                
                # 触发用户主动复盘
                result = await self.review_trigger.trigger_user_active(
                    context={
                        'trigger_keyword': self.trigger_keyword,
                        'user_input': state.get('user_input', ''),
                        'trigger_source': 'L1B_node'
                    }
                )
                
                if result:
                    logger.info("[ReviewTriggerNode] ✅ 复盘机制触发成功")
                else:
                    logger.warning("[ReviewTriggerNode] ⚠️ 复盘机制触发返回 False")
        except Exception as e:
            logger.error(f"[ReviewTriggerNode] 异步触发失败：{e}", exc_info=True)
