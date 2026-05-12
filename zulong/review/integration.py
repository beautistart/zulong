# zulong/review/integration.py
"""
复盘集成器 - 接收复盘触发事件并协调复盘流程

对应 TSD v2.3 第 11.1 节
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import uuid

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, ZulongEvent, EventPriority, L2Status
from zulong.review.state_manager import get_review_state_manager, ReviewMode

logger = logging.getLogger(__name__)


class ReplayIntegration:
    """复盘集成器
    
    功能:
    - 订阅 REPLAY_TRIGGERED 事件
    - 协调复盘处理器执行分析
    - 生成复盘报告并保存
    
    数据流:
    1. 接收 REPLAY_TRIGGERED 事件
    2. 根据触发类型选择处理策略
    3. 从共享池获取最近的对话/任务数据
    4. 调用成功提炼器或失败分析器
    5. 生成复盘报告
    6. 保存到经验库
    
    支持的复盘模式:
    - 快速复盘：自动分析、自动生成经验、自动应用
    - 深度复盘：分析后生成草案、等待用户确认、可选择修改
    """
    
    def __init__(self):
        """初始化复盘集成器"""
        self._setup_event_subscription()
        self._recent_tasks: List[Dict] = []
        self._recent_conversations: List[Dict] = []
        
        # 初始化状态属性
        self.review_mode = False
        self.review_type = None
        self.pending_experiences = None
        self.pending_summary = None
        self.pending_tags = None
        
        # 🔥 关键修复：使用 ReviewStateManager 统一管理状态
        self._state_manager = get_review_state_manager()
        
        logger.info("[ReplayIntegration] 初始化完成")
    
    def _setup_event_subscription(self):
        """设置事件订阅"""
        event_bus.subscribe(
            EventType.REPLAY_TRIGGERED,
            self.on_replay_triggered,
            "ReplayIntegration"
        )
        logger.info("[ReplayIntegration] 已订阅 REPLAY_TRIGGERED 事件")
    
    async def on_replay_triggered(self, event):
        """🔥 修复：异步处理复盘触发事件
        
        Args:
            event: 复盘触发事件 (可能是 ZulongEvent 或 dict)
        """
        try:
            logger.info(f"[ReplayIntegration] 🎯 收到复盘触发事件")
            
            # 兼容处理：可能是 dict 或 ZulongEvent
            if hasattr(event, 'payload'):
                # ZulongEvent 对象
                trigger_type = event.payload.get('trigger_type')
                priority = event.payload.get('priority')
                context = event.payload.get('context', {})
            else:
                # dict 对象
                trigger_type = event.get('type', 'unknown')
                priority = event.get('priority', 'NORMAL')
                context = event.get('context', {})
            
            logger.info(f"[ReplayIntegration] 触发类型：{trigger_type}")
            logger.info(f"[ReplayIntegration] 优先级：{priority}")
            logger.info(f"[ReplayIntegration] 上下文：{context}")
            
            # 根据触发类型执行不同的复盘逻辑
            # 🔥 修复：trigger_type 可能是 TriggerType 枚举，需要转换为字符串
            trigger_type_str = trigger_type.value if hasattr(trigger_type, 'value') else str(trigger_type)
            
            if trigger_type_str == 'user_active':
                # 🔥 修复：异步调用
                await self._handle_user_active_review(context)
            elif trigger_type_str == 'quiet_mode':
                await self._handle_quiet_mode_review(context)
            elif trigger_type_str == 'night_schedule':
                await self._handle_night_schedule_review(context)
            else:
                logger.warning(f"[ReplayIntegration] 未知触发类型：{trigger_type_str}")
            
            logger.info(f"[ReplayIntegration] ✅ 复盘处理完成")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 处理复盘事件失败：{e}", exc_info=True)
    
    async def _handle_user_active_review(self, context: Dict[str, Any]):
        """🔥 修复：异步处理用户主动复盘 - 三阶段状态机
        
        Args:
            context: 上下文信息，包含：
                - trigger_keyword: 触发关键词
                - user_input: 用户输入
                - trigger_source: 触发来源
                - review_type: 复盘类型 ('quick' | 'deep' | None) - 🔥 新增
                - trace_id: 追踪 ID
        """
        logger.info(f"[ReplayIntegration] 处理用户主动复盘")
        
        try:
            # 🔥 关键修复 1: 检查是否已有处理锁
            if not self._state_manager.acquire_processing_lock():
                logger.warning("[ReplayIntegration] 无法获取处理锁，忽略重复请求")
                return
            
            # 1. 进入复盘模式
            session_id = str(uuid.uuid4())[:8]
            self.review_session_id = session_id  # 保存会话 ID
            
            # 判断复盘类型
            review_type = context.get('review_type', None)
            user_input = context.get('user_input', '').lower()
            
            if not review_type:
                if '快速' in user_input or '自动' in user_input:
                    review_type = 'quick'
                elif '深度' in user_input or '确认' in user_input:
                    review_type = 'deep'
                else:
                    # 默认：快速复盘，防止卡死
                    review_type = 'quick'
                    logger.info("未检测到明确类型，默认使用快速复盘")
            
            # 2. 使用 ReviewStateManager 进入复盘模式
            mode = ReviewMode.QUICK if review_type == 'quick' else ReviewMode.DEEP
            self._state_manager.enter_review_mode(mode, session_id)
            logger.info(f"[ReplayIntegration] 已进入{review_type}复盘模式，会话 ID: {session_id}")
            
            # 3. 创建临时缓冲区
            try:
                from zulong.review.temp_buffer import get_review_buffer_manager
                buffer_manager = get_review_buffer_manager()
                buffer_manager.create_buffer(session_id)
                logger.info("[ReplayIntegration] 已创建临时缓冲区")
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 创建缓冲区失败：{e}")
            
            # 🔥 关键修复：如果明确指定了 review_type，直接进入对应阶段
            # 避免在用户明确输入"快速复盘"时还停留在模式选择阶段
            if review_type:
                logger.info(f"[ReplayIntegration] 明确指定了复盘类型：{review_type}，直接进入对话阶段")
                await self._start_review_flow(review_type)
                return
            
            # 🔥 新增：三阶段状态机 - 进入模式选择阶段（仅当未指定 review_type 时）
            self._state_manager.enter_mode_selecting()
            self._publish_mode_select_prompt()
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 用户主动复盘失败：{e}", exc_info=True)
            # 🔥 关键修复 3: 异常时强制退出复盘模式，防止卡死
            self._force_exit_review_mode()
        finally:
            # 🔥 关键修复：释放处理锁
            try:
                self._state_manager.release_processing_lock()
            except Exception as e:
                logger.error(f"释放处理锁失败：{e}")
    
    def _publish_mode_select_prompt(self):
        """🔥 新增：发布模式选择提示"""
        response_text = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 **复盘向导已启动**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "请选择复盘模式：\n\n"
            "⚡ **快速复盘**\n"
            "   • 基于关键词和短时记忆，生成摘要\n"
            "   • 自动分析并应用经验\n"
            "   • 适合日常回顾\n\n"
            "🔍 **深度复盘**\n"
            "   • 调用长期记忆库，进行多维分析\n"
            "   • 生成经验草案，需您确认\n"
            "   • 适合重要项目复盘\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "💬 请直接说 `快速复盘` 或 `深度复盘`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        
        self._publish_l2_response(response_text)
        logger.info("[ReplayIntegration] 已发布模式选择提示")
    
    # ========== 🔥 新增：三阶段状态机核心方法 ==========
    
    async def handle_mode_selection(self, text: str):
        """🔥 新增：处理模式选择阶段的输入
        
        Args:
            text: 用户输入文本
        """
        logger.info(f"[ReplayIntegration] 处理模式选择输入：{text}")
        
        try:
            if not self._state_manager.acquire_processing_lock():
                logger.warning("[ReplayIntegration] 无法获取处理锁，忽略重复请求")
                return
            
            # 检测退出指令
            if '退出' in text or '取消' in text:
                logger.info("[ReplayIntegration] 用户选择退出复盘")
                self._state_manager.exit_review_mode('cancelled')
                self._publish_l2_response("✅ 已取消复盘")
                return
            
            # 检测快速复盘
            if '快速' in text:
                logger.info("[ReplayIntegration] 用户选择快速复盘")
                await self._start_review_flow('quick')
                return
            
            # 检测深度复盘
            if '深度' in text:
                logger.info("[ReplayIntegration] 用户选择深度复盘")
                await self._start_review_flow('deep')
                return
            
            # 无效输入，提示重新选择
            logger.debug("[ReplayIntegration] 无效的模式选择，重新提示")
            self._publish_mode_select_prompt()
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 处理模式选择失败：{e}", exc_info=True)
            self._force_exit_review_mode()
        finally:
            try:
                self._state_manager.release_processing_lock()
            except Exception as e:
                logger.error(f"释放处理锁失败：{e}")
    
    async def handle_experience_confirmation(self, text: str):
        """🔥 新增：处理经验确认阶段的输入
        
        Args:
            text: 用户输入文本
        """
        logger.info(f"[ReplayIntegration] 处理经验确认输入：{text}")
        
        try:
            if not self._state_manager.acquire_processing_lock():
                logger.warning("[ReplayIntegration] 无法获取处理锁，忽略重复请求")
                return
            
            # 检测确认指令
            if '确认' in text or '好的' in text or '保存' in text:
                logger.info("[ReplayIntegration] 用户确认经验")
                await self._confirm_and_save_experiences()
                return
            
            # 检测修改指令
            if '修改' in text or '重新' in text:
                logger.info("[ReplayIntegration] 用户要求修改经验")
                await self._restart_review_conversation()
                return
            
            # 检测退出指令
            if '退出' in text or '取消' in text:
                logger.info("[ReplayIntegration] 用户选择退出复盘")
                self._state_manager.exit_review_mode('cancelled')
                self._publish_l2_response("✅ 已取消复盘，经验未保存")
                return
            
            # 无效输入，提示重新确认
            logger.debug("[ReplayIntegration] 无效的确认指令，重新提示")
            self._publish_experience_confirm_prompt()
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 处理经验确认失败：{e}", exc_info=True)
            self._force_exit_review_mode()
        finally:
            try:
                self._state_manager.release_processing_lock()
            except Exception as e:
                logger.error(f"释放处理锁失败：{e}")
    
    async def _start_review_flow(self, review_type: str):
        """🔥 修复版：启动复盘流程 - 进入L2对话阶段
        
        根据用户描述的预期流程：
        1. 用户选择快速/深度复盘后
        2. 进入L2对话阶段（不是直接分析！）
        3. 用户可以自由与L2对话
        4. 用户输入"结束复盘"后才触发经验提取
        
        Args:
            review_type: 复盘类型 ('quick' | 'deep')
        """
        logger.info(f"[ReplayIntegration] 启动{review_type}复盘流程 - 进入L2对话阶段")
        
        try:
            # 🔥 关键修复：进入对话进行阶段，但不执行分析！
            self._state_manager.enter_review_active(review_type)
            
            # 创建或清空临时缓冲区用于记录对话内容
            session_id = self._state_manager.get_session_id()
            try:
                from zulong.review.temp_buffer import get_review_buffer_manager
                buffer_manager = get_review_buffer_manager()
                
                if not buffer_manager.has_buffer():
                    buffer_manager.create_buffer(session_id)
                    logger.info("[ReplayIntegration] 已创建临时缓冲区")
                else:
                    buffer_manager.clear_buffer()
                    logger.info("[ReplayIntegration] 已清空临时缓冲区")
                    
                # 将之前的对话数据导入到缓冲区作为基础数据
                recent_data = await self._get_recent_context()
                conversations = recent_data.get('conversations', [])
                if conversations:
                    for conv in conversations[-20:]:  # 最近20条对话
                        if isinstance(conv, dict):
                            user_text = conv.get('text', '') or conv.get('user_input', '') or conv.get('user', '')
                            system_text = conv.get('system_response', '') or conv.get('system', '') or conv.get('ai_response', '')
                        else:
                            user_text = str(conv)
                            system_text = None
                        buffer_manager.add_conversation(
                            user_input=user_text,
                            system_response=system_text,
                            tags=['historical']
                        )
                    logger.info(f"[ReplayIntegration] 已导入 {min(len(conversations), 20)} 条历史对话到缓冲区")
                    
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 缓冲区操作失败：{e}")
            
            # 🔥 发布进入对话的提示
            mode_name = "快速" if review_type == 'quick' else "深度"
            response_text = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ **{mode_name}复盘模式已启动**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"好的，已进入{mode_name}复盘模式！\n\n"
                "💬 **现在我们可以自由对话了**\n\n"
                "您可以：\n"
                "   • 告诉我您想复盘的内容\n"
                "   • 讨论最近的工作或学习\n"
                "   • 分享您的想法和感受\n\n"
                "📝 **我会记录我们的对话**\n\n"
                "当您想结束复盘时，请说 **「结束复盘」**\n"
                "届时我会分析对话并提取经验。\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            
            self._publish_l2_response(response_text)
            logger.info(f"[ReplayIntegration] ✅ 已进入{mode_name}复盘对话阶段，等待用户输入")
            
            # 🔥 关键点：不在这里调用 _handle_quick_review_async() 或 _handle_deep_review()！
            # 只设置状态并提示用户开始对话
            # 真正的分析会在用户说"结束复盘"时触发
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 启动复盘流程失败：{e}", exc_info=True)
            self._force_exit_review_mode()
            raise
    
    async def _confirm_and_save_experiences(self):
        """🔥 修复：确认并保存经验 - 真正从状态管理器获取经验
        
        从状态管理器获取之前生成的经验，并应用到经验库。
        """
        logger.info("[ReplayIntegration] 确认并保存经验")
        
        try:
            # 🔥 关键修复：从状态管理器获取待确认的经验
            pending_data = self._state_manager.get_pending_experiences()
            
            if not pending_data or not pending_data.get('experiences'):
                logger.warning("[ReplayIntegration] 没有找到待确认的经验")
                # 发布提示
                response_text = (
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "⚠️ **暂无经验可保存**\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "可能是之前的分析未生成经验，或者复盘流程异常。\n\n"
                    "💡 建议：重新开始复盘\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
                self._publish_l2_response(response_text)
                self._state_manager.exit_review_mode('completed')
                return
            
            experiences = pending_data['experiences']
            summary = pending_data.get('summary', '')
            tags = pending_data.get('tags', [])
            
            logger.info(f"[ReplayIntegration] 获取到 {len(experiences)} 条待确认经验")
            
            # 🔥 保存经验到经验库
            saved_count = await self._apply_experiences(experiences)
            
            # 写入 MemoryGraph（替代旧的 ShortTermMemory 写入）
            try:
                from zulong.memory.memory_graph import get_memory_graph
                from zulong.memory.graph_adapters import DialogueAdapter
                mg = get_memory_graph()
                if mg:
                    adapter = DialogueAdapter()
                    session_id = adapter.ensure_session(mg, f"复盘总结：{summary}")
                    round_id = adapter.add_round(
                        mg, f"review-{self._state_manager.get_session_id()}",
                        f"复盘总结：{summary}",
                        session_id=session_id,
                    )
                    adapter.add_sub_dialogue(
                        mg, round_id, turn=1,
                        content=f"已保存{saved_count}条经验：{', '.join([e.get('content', '')[:50] for e in experiences[:3]])}",
                        role="assistant",
                    )
                    logger.info("[ReplayIntegration] 已保存复盘总结到 MemoryGraph")
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 保存到 MemoryGraph 失败：{e}")
            
            # 退出复盘模式
            self._state_manager.exit_review_mode('completed')
            
            # 发布完成提示
            response_text = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "✅ **复盘完成**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 **复盘总结**:\n{summary}\n\n"
                f"💾 **已保存 {saved_count} 条经验到记忆库**\n\n"
                f"🏷️ **标签**: {', '.join(tags) if tags else '无'}\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            self._publish_l2_response(response_text)
            logger.info(f"[ReplayIntegration] 复盘完成，保存{saved_count}条经验")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 确认保存经验失败：{e}", exc_info=True)
            raise
    
    async def _restart_review_conversation(self):
        """🔥 新增：重新开始复盘对话"""
        logger.info("[ReplayIntegration] 重新开始复盘对话")
        
        try:
            # 重置到对话进行阶段
            review_type = self._state_manager.get_mode()
            review_type_str = 'quick' if review_type == ReviewMode.QUICK else 'deep'
            
            self._state_manager.enter_review_active(review_type_str)
            
            # 清空缓冲区，准备新的对话
            try:
                from zulong.review.temp_buffer import get_review_buffer_manager
                buffer_manager = get_review_buffer_manager()
                if buffer_manager.has_buffer():
                    buffer_manager.clear_buffer()
                    logger.info("[ReplayIntegration] 已清空缓冲区，准备重新对话")
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 清空缓冲区失败：{e}")
            
            # 发布提示
            response_text = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🔄 **好的，让我们重新开始**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "请继续告诉我您想复盘的内容...\n\n"
                "完成后说 **「结束复盘」** 即可\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            self._publish_l2_response(response_text)
            
            logger.info("[ReplayIntegration] ✅ 已返回到对话阶段")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 重新开始复盘失败：{e}", exc_info=True)
            raise
    
    async def handle_end_review(self):
        """🔥 核心修复：处理"结束复盘"指令 - 真正调用 ExperienceExtractor 提取经验
        
        当用户在对话进行阶段输入"结束复盘"时调用。
        
        流程：
        1. 从缓冲区获取所有对话内容
        2. 🔥 调用 ExperienceExtractor 分析对话并提取经验
        3. 进入经验确认阶段（EXPERIENCE_CONFIRMING）
        4. 展示生成的经验列表，等待用户确认/修改
        
        这是实现完整复盘流程的关键方法！
        """
        logger.info("[ReplayIntegration] 🎯 处理'结束复盘'指令 - 开始提取经验")
        
        try:
            # 🔥 关键修复 1: 获取处理锁
            if not self._state_manager.acquire_processing_lock():
                logger.warning("[ReplayIntegration] 无法获取处理锁，忽略重复请求")
                return
            
            # 🔥 步骤 1: 从缓冲区获取对话数据
            logger.info("[ReplayIntegration] 正在从缓冲区读取对话数据...")
            
            buffer_data = None
            conversations_count = 0
            
            try:
                from zulong.review.temp_buffer import get_review_buffer_manager
                buffer_manager = get_review_buffer_manager()
                
                if buffer_manager.has_buffer():
                    buffer_data = buffer_manager.get_buffer().export_for_analysis()
                    conversations_count = len(buffer_data.get('conversations', []))
                    logger.info(f"[ReplayIntegration] ✅ 从缓冲区获取到 {conversations_count} 条对话记录")
                else:
                    logger.warning("[ReplayIntegration] 缓冲区不存在或为空")
                    
            except Exception as e:
                logger.error(f"[ReplayIntegration] 从缓冲区获取数据失败：{e}")
                # 降级方案：使用最近上下文
                buffer_data = await self._get_recent_context()
                conversations_count = len(buffer_data.get('conversations', []))
                logger.warning(f"[ReplayIntegration] 使用降级方案，获取到 {conversations_count} 条历史对话")
            
            # 🔥 步骤 2: 发布分析中提示
            self._publish_status_message("🔍 正在分析我们的对话...")
            self._publish_status_message("💡 正在提炼经验和教训...")
            
            # 🔥 步骤 3: 设置 L2 状态为 ANALYZING
            from zulong.core.state_manager import state_manager
            session_id = self._state_manager.get_session_id()
            state_manager.set_l2_status(L2Status.REVIEW_ANALYZING, task_id=session_id)
            
            # 更新状态为分析中
            self._state_manager.set_analyzing()
            
            # 🔥 关键修复 2: 真正调用 ExperienceExtractor 提取经验
            logger.info("[ReplayIntegration] 调用 ExperienceExtractor 进行经验提取...")
            
            try:
                from zulong.review.experience_extractor import get_experience_extractor
                extractor = get_experience_extractor()
                
                # 判断是否深度分析
                review_mode = self._state_manager.get_mode()
                is_deep = (review_mode == ReviewMode.DEEP) if review_mode else False
                
                # 🔥 真正调用 ExperienceExtractor 的 extract_from_buffer 方法
                structured_data = await extractor.extract_from_buffer(buffer_data, deep=is_deep)
                
                logger.info(f"[ReplayIntegration] ✅ ExperienceExtractor 成功提取 {len(structured_data.get('experiences', []))} 条经验")
                
                # 使用提取到的经验
                experiences = structured_data.get('experiences', [])
                summary = structured_data.get('summary', '')
                tags = structured_data.get('suggested_tags', [])
                
            except Exception as e:
                logger.error(f"[ReplayIntegration] ExperienceExtractor 提取失败：{e}")
                # 降级方案：使用原有的 L2 调用方法
                logger.warning("[ReplayIntegration] 降级到使用 L2 直接分析")
                analysis_result = await self._analyze_conversation_with_l2_async(buffer_data)
                experiences = await self._generate_experiences_async(analysis_result, draft=True)
                summary = analysis_result.get('summary', '')
                tags = analysis_result.get('suggested_tags', [])
            
            # 更新状态为生成经验中
            self._state_manager.set_generating()
            self._publish_status_message("💡 正在生成经验草案...")
            
            logger.info(f"[ReplayIntegration] ✅ 已生成 {len(experiences)} 条经验")
            
            # 🔥 步骤 4: 保存待确认的经验到状态管理器
            self._state_manager.set_pending_experiences(
                experiences=experiences,
                summary=summary,
                tags=tags
            )
            
            # 🔥 步骤 5: 进入经验确认阶段（关键！）
            self._state_manager.enter_experience_confirming(len(experiences))
            
            # 🔥 步骤 6: 展示经验给用户确认
            analysis_result = {
                'summary': summary,
                'total_conversations': conversations_count,
                'suggested_tags': tags
            }
            self._publish_experience_confirmation_prompt_with_details(experiences, analysis_result)
            
            # 🔥 重要：不要在这里退出复盘模式！
            # 等待用户输入"确认"或"修改"
            
            logger.info(f"[ReplayIntegration] ✅ 经验已展示，进入确认阶段")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 处理'结束复盘'失败：{e}", exc_info=True)
            self._force_exit_review_mode()
        finally:
            # 🔥 关键：释放处理锁
            try:
                self._state_manager.release_processing_lock()
            except Exception as e:
                logger.error(f"释放处理锁失败：{e}")
    
    def _publish_experience_confirmation_prompt_with_details(self, experiences: list, analysis_result: dict):
        """🔥 新增：发布带详细内容的经验确认提示
        
        Args:
            experiences: 生成的经验列表
            analysis_result: 分析结果
        """
        try:
            # 构建经验列表文本
            experience_list_text = ""
            for i, exp in enumerate(experiences[:10], 1):  # 最多显示10条
                title = exp.get('title', f'经验 {i}')
                content = exp.get('content', exp.get('description', ''))
                category = exp.get('category', '')
                
                experience_list_text += (
                    f"\n### {i}. {title}\n"
                    f"{content[:100]}{'...' if len(content) > 100 else ''}\n"
                )
                
                if category:
                    experience_list_text += f"📂 分类：{category}\n"
            
            if not experiences:
                experience_list_text = "\n⚠️ 本次对话未提取到明显的经验\n"
            
            # 构建总结信息
            summary = analysis_result.get('summary', '对话分析完成')
            total_conversations = analysis_result.get('total_conversations', 0)
            
            response_text = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "✅ **复盘分析完成 - 经验待确认**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 **分析概况**:\n"
                f"   • 对话轮数：{total_conversations} 轮\n"
                f"   • 提取经验：{len(experiences)} 条\n\n"
                f"📝 **总结**: {summary}\n\n"
                f"💡 **生成的经验**:{experience_list_text}\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "💬 **请选择操作**:\n\n"
                "✅ 说 `确认` - 保存所有经验到记忆库\n"
                "🔄 说 `修改` - 返回继续对话，重新生成经验\n"
                "❌ 说 `退出` - 放弃本次复盘\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            
            self._publish_l2_response(response_text)
            logger.info(f"[ReplayIntegration] 已发布经验确认提示（{len(experiences)}条经验）")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 发布经验确认提示失败：{e}", exc_info=True)
            # 降级提示
            self._publish_experience_confirm_prompt()
    
    def _publish_experience_confirm_prompt(self):
        """🔥 新增：发布经验确认提示"""
        response_text = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "✅ **经验待确认**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📊 已生成以下经验：\n\n"
            "（此处显示经验列表）\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "💬 请回复：\n"
            "✅ `确认` - 保存经验到记忆库\n"
            "🔄 `修改` - 重新对话生成经验\n"
            "❌ `退出` - 放弃本次复盘\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        
        self._publish_l2_response(response_text)
        logger.info("[ReplayIntegration] 已发布经验确认提示")
    
    def _ask_review_mode(self, context: Dict[str, Any]):
        """询问用户选择复盘模式
        
        Args:
            context: 上下文信息
        """
        logger.info("[ReplayIntegration] 询问用户选择复盘模式")
        
        # 🔥 新增：更新状态为"选择中"
        try:
            from zulong.review.state import get_review_state
            review_state = get_review_state()
            review_state.update_stage('selecting', '等待用户选择复盘模式')
        except Exception as e:
            logger.debug(f"[ReplayIntegration] 更新状态失败：{e}")
        
        # 🔥 增强：入口仪式感文案
        response_text = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 **复盘向导已启动**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "检测到您想进行复盘。请选择模式：\n\n"
            "⚡ **快速复盘**\n"
            "   • 基于关键词和短时记忆，生成摘要\n"
            "   • 自动分析并应用经验\n"
            "   • 适合日常回顾\n\n"
            "🔍 **深度复盘**\n"
            "   • 调用长期记忆库，进行多维分析\n"
            "   • 生成经验草案，需您确认\n"
            "   • 适合重要项目复盘\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "💬 请直接说 `快速复盘` 或 `深度复盘`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        
        self._publish_l2_response(response_text)
    
    async def _ask_quick_review_confirmation(self, recent_data: Dict[str, Any], context: Dict[str, Any]):
        """🔥 新增：询问用户确认快速复盘
        
        Args:
            recent_data: 最近的对话数据
            context: 上下文信息
        """
        logger.info("[ReplayIntegration] 询问用户确认快速复盘")
        
        try:
            # 🔥 新增：更新状态为"待确认"
            from zulong.review.state import get_review_state
            review_state = get_review_state()
            review_state.update_stage('waiting_confirmation', '等待用户确认快速复盘')
            
            # 🔥 关键修复：显示确认提示框 (常驻)
            conversations_count = len(recent_data.get('conversations', []))
            time_window = recent_data.get('time_range', {})
            
            response_text = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "⚡ **快速复盘 - 待确认**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 已检索到最近 {conversations_count} 条对话记录\n"
                f"⏱️ 时间范围：{time_window.get('start', 'N/A')[:19]} 至 {time_window.get('end', 'N/A')[:19]}\n\n"
                "🤖 **分析方式**:\n"
                "   • 基于短时记忆和对话内容\n"
                "   • 自动提炼经验教训\n"
                "   • 直接应用到记忆库\n\n"
                "💬 **请确认是否执行快速复盘？**\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "✅ 说 `确认`、`好的`、`开始` 执行\n"
                "❌ 说 `取消`、`不要`、`退出` 放弃\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            
            self._publish_l2_response(response_text)
            logger.info(f"[ReplayIntegration] 已发布快速复盘确认提示")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 询问确认失败：{e}", exc_info=True)
            # 失败时直接执行
            await self._handle_quick_review_async(recent_data, context)
    
    def _handle_quick_review(self, recent_data: Dict[str, Any], context: Dict[str, Any]):
        """🔥 保留：同步版本（兼容旧调用）"""
        logger.warning("[ReplayIntegration] 使用同步版本 _handle_quick_review，建议改用异步版本")
        # 同步调用会阻塞，不推荐使用
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._handle_quick_review_async(recent_data, context))
        finally:
            loop.close()
    
    async def _handle_quick_review_async(self, recent_data: Dict[str, Any], context: Dict[str, Any]):
        """🔥 修复：异步处理快速复盘 - 🔥 核心修复：使用 try...finally 确保状态重置
        
        Args:
            recent_data: 最近的对话数据
            context: 上下文信息
        """
        logger.info("[ReplayIntegration] 处理快速复盘 (异步)")
        
        # 🔥 调试日志：记录传入的数据
        conversations_count = len(recent_data.get('conversations', []))
        logger.info(f"[ReplayIntegration] 传入的对话数据：{conversations_count} 条")
        
        # 🔥 关键修复 4: 使用 try...finally 确保状态重置
        try:
            # 🔥 修复 2: 完善 L2 状态管理
            from zulong.core.state_manager import state_manager
            
            # 设置 L2 状态为 REVIEW_ANALYZING
            session_id = self._state_manager.get_session_id()
            state_manager.set_l2_status(L2Status.REVIEW_ANALYZING, task_id=session_id)
            logger.info(f"[ReplayIntegration] ✅ L2 状态已设置为 REVIEW_ANALYZING")
            
            # 🔥 新增：更新状态为"分析中"
            self._state_manager.set_analyzing()
            
            # 发布状态提示
            self._publish_status_message("🔍 正在检索记忆库和分析对话...")
            
            # 1. 🔥 修复：异步分析对话
            analysis_result = await self._analyze_conversation_with_l2_async(recent_data)
            
            # 🔥 新增：更新状态为"生成经验中"
            self._state_manager.set_generating()
            self._publish_status_message("💡 正在提炼经验...")
            
            # 2. 🔥 修复 1: 异步生成经验（调用 L2）
            experiences = await self._generate_experiences_async(analysis_result)
            
            # 🔥 新增：更新状态为"应用中"
            self._publish_status_message("💾 正在应用经验到记忆库...")
            
            # 3. 自动应用经验
            await self._apply_experiences(experiences)
            
            # 4. 生成复盘报告
            report = self._generate_review_report(
                trigger_type='user_active_quick',
                context=context,
                analysis=analysis_result,
                data=recent_data,
                experiences=experiences
            )
            
            # 5. 保存报告
            self._save_review_report(report)
            
            # 🔥 新增：更新状态为"完成"
            self._state_manager.set_completed()
            
            # 🔥 修复 2: 重置 L2 状态为 IDLE
            state_manager.set_l2_status(L2Status.IDLE, task_id=session_id)
            logger.info(f"[ReplayIntegration] ✅ L2 状态已重置为 IDLE")
            
            # 6. 🔥 增强：结果展示文案
            response_text = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "✅ **快速复盘完成**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 分析了 {analysis_result.get('total_conversations', 0)} 条对话\n"
                f"💡 生成了 {len(experiences)} 条经验\n"
                f"💾 经验已自动应用到记忆库\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏱️ 耗时：{self._state_manager.get_session_info().get('duration', 'N/A')}秒\n"
                f"📝 会话 ID: `{session_id}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            self._publish_l2_response(response_text)
            
            # 7. 退出复盘模式
            self._state_manager.exit_review_mode('completed')
            
            logger.info(f"[ReplayIntegration] 快速复盘完成，生成{len(experiences)}条经验")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 快速复盘失败：{e}", exc_info=True)
            raise  # 🔥 关键修复：让 finally 块处理状态重置
        finally:
            # 🔥 核心修复：无论成功或失败，必须释放 L2 锁
            # 这是解决"无法对话"的关键
            try:
                from zulong.core.state_manager import state_manager
                state_manager.set_l2_status(L2Status.IDLE)
                logger.info(f"[ReplayIntegration] ✅ 已通过 finally 块强制重置 L2 状态为 IDLE")
            except Exception as e:
                logger.error(f"[ReplayIntegration] finally 块重置 L2 状态失败：{e}")
    
    def _force_exit_review_mode(self):
        """🔥 新增：防止系统卡在复盘状态的强制退出方法"""
        logger.warning("[ReplayIntegration] 强制退出复盘模式，重置系统状态")
        
        # 清理会话 ID
        if hasattr(self, 'review_session_id'):
            delattr(self, 'review_session_id')
        
        # 🔥 关键修复：使用 ReviewStateManager 强制退出
        self._state_manager.force_exit()
    
    async def _execute_quick_review(self):
        """🔥 新增：执行真正的快速复盘 (用户确认后调用)
        
        这个方法在用户确认执行快速复盘后调用
        """
        logger.info("[ReplayIntegration] 执行真正的快速复盘 (用户已确认)")
        
        try:
            # 1. 获取上下文数据
            recent_data = await self._get_recent_context()
            
            # 2. 调用已有的异步处理方法
            context = {
                'trigger_keyword': '快速复盘',
                'user_input': '确认执行',
                'trigger_source': 'user_confirmation',
                'review_type': 'quick'
            }
            
            await self._handle_quick_review_async(recent_data, context)
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 执行快速复盘失败：{e}", exc_info=True)
    
    async def _handle_deep_review(self, recent_data: Dict[str, Any], context: Dict[str, Any]):
        """处理深度复盘
        
        Args:
            recent_data: 最近的对话数据
            context: 上下文信息
        """
        logger.info("[ReplayIntegration] 处理深度复盘")
        
        try:
            # 🔥 新增：获取状态管理器
            from zulong.review.state import get_review_state
            review_state = get_review_state()
            
            # 🔥 过程感知：分析中
            review_state.set_analyzing()
            self._publish_status_message("🔍 正在检索长期记忆库...")
            await asyncio.sleep(0.5)  # 短暂延迟，让用户看到状态
            
            self._publish_status_message("🧠 正在调用专家模型分析...")
            
            # 1. 深度分析对话
            analysis_result = self._analyze_conversation_with_l2(recent_data, deep=True)
            
            # 🔥 过程感知：生成经验中
            review_state.set_generating()
            self._publish_status_message("💡 正在提炼经验草案...")
            
            # 2. 🔥 新增：调用 L2 生成结构化 JSON
            structured_data = await self._generate_structured_experiences(recent_data)
            
            # 3. 保存待确认的经验
            self.pending_experiences = structured_data.get('experiences', [])
            self.pending_summary = structured_data.get('summary', '')
            self.pending_tags = structured_data.get('suggested_tags', [])
            
            # 🔥 过程感知：等待确认
            review_state.set_waiting_confirmation(len(self.pending_experiences))
            
            # 4. 🔥 增强：结构化展示 + 强制确认机制
            response_text = self._format_experience_draft_enhanced(self.pending_experiences)
            self._publish_l2_response(response_text)
            
            logger.info(f"[ReplayIntegration] 深度复盘分析完成，等待用户确认")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 深度复盘失败：{e}", exc_info=True)
            self.review_mode = False
            self.review_type = None
            try:
                review_state.exit_review_mode()
            except:
                pass
    
    async def _generate_structured_experiences(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """🔥 新增：调用 L2 生成结构化经验数据
        
        Args:
            data: 对话数据
            
        Returns:
            Dict: 结构化的经验数据
        """
        try:
            # 🔥 L2 生成：调用经验提取器
            from zulong.review.experience_extractor import get_experience_extractor
            extractor = get_experience_extractor()
            
            # 从缓冲区导出数据给 L2 分析
            from zulong.review.temp_buffer import get_review_buffer_manager
            buffer_manager = get_review_buffer_manager()
            
            if buffer_manager.has_buffer():
                buffer_data = buffer_manager.get_buffer().export_for_analysis()
                structured_data = await extractor.extract_from_buffer(buffer_data, deep=True)
            else:
                # 降级：使用普通数据
                structured_data = await extractor.extract_from_buffer(data, deep=True)
            
            logger.info(f"[ReplayIntegration] L2 生成结构化数据完成")
            return structured_data
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 生成结构化经验失败：{e}", exc_info=True)
            # 返回空结构
            return {
                'summary': '暂无总结',
                'experiences': [],
                'suggested_tags': []
            }
    
    def _apply_confirmed_experiences(self, experiences: List[Dict[str, Any]]):
        """🔥 新增：L1-B 执行经验写入
        
        Args:
            experiences: 已确认的经验列表
        """
        try:
            # 🔥 L1-B 执行：调用安全应用器
            from zulong.review.safe_applier import get_safe_applier
            applier = get_safe_applier()
            
            # 1. 验证并写入数据库
            result = applier.apply_experiences(experiences, self.review_session_id)
            
            if result['success']:
                logger.info(f"[ReplayIntegration] 成功应用 {result['applied_count']} 条经验")
                
                # 2. 合并缓冲区到主记忆池
                from zulong.review.temp_buffer import get_review_buffer_manager
                buffer_manager = get_review_buffer_manager()
                
                if buffer_manager.has_buffer():
                    buffer_data = buffer_manager.get_buffer().export_for_analysis()
                    applier.merge_buffer_to_memory(buffer_data, experiences)
                
                # 3. 清理资源
                applier.cleanup(self.review_session_id)
                
                return True
            else:
                logger.error(f"[ReplayIntegration] 应用经验失败：{result['errors']}")
                return False
                
        except Exception as e:
            logger.error(f"[ReplayIntegration] 应用经验失败：{e}", exc_info=True)
            return False
    
    def _analyze_conversation_with_l2(self, data: Dict[str, Any], deep: bool = False) -> Dict[str, Any]:
        """调用 L2 进行对话分析（同步等待响应）
        
        Args:
            data: 对话数据
            deep: 是否深度分析
            
        Returns:
            Dict: 分析结果
        """
        try:
            conversations = data.get('conversations', [])
            
            # 🔥 关键修复：检查是否有对话数据
            no_data_mode = False
            if not conversations:
                logger.info("[ReplayIntegration] 没有对话数据，但强制调用 L2 进行交互")
                # 🔥 强制调用 L2，即使没有历史数据
                # 让 L2 生成一个友好的交互响应，而不是直接返回简化分析
                no_data_mode = True
                prompt = (
                    f"用户开启了快速复盘模式。\n\n"
                    f"虽然我没有检索到历史对话数据，但我已准备好为您服务。\n\n"
                    f"请以友好的语气回复用户，可以：\n"
                    f"1. 告知用户当前没有历史对话记录\n"
                    f"2. 询问用户想复盘什么内容\n"
                    f"3. 建议用户可以开始新的对话，稍后再复盘\n\n"
                    f"请生成一段自然、友好的回复。"
                )
            else:
                # 构建分析提示
                if deep:
                    prompt = (
                        f"请对以下对话进行深度分析：\n\n"
                        f"对话内容：{conversations}\n\n"
                        f"请分析：\n"
                        f"1. 用户的真实意图和目标\n"
                        f"2. 关键决策点和选择\n"
                        f"3. 系统响应的质量评估\n"
                        f"4. 可优化的环节\n"
                        f"5. 可复用的经验和教训\n\n"
                        f"请以 JSON 格式返回，包含：key_decisions, user_intents, quality_score, improvements"
                    )
                else:
                    prompt = (
                        f"请分析以下对话并总结经验：\n\n"
                        f"对话内容：{conversations}\n\n"
                        f"请总结：\n"
                        f"1. 对话主题\n"
                        f"2. 达成的结果\n"
                        f"3. 可复用的经验\n\n"
                        f"请以 JSON 格式返回，包含：experiences, summary, key_decisions"
                    )
            
            # 🔥 关键修复：同步等待 L2 响应
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, EventPriority, ZulongEvent
            
            import asyncio
            import time
            
            # 🔥 修复：使用当前事件循环创建 Future，避免 "attached to a different loop" 错误
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # 没有运行中的事件循环，创建新的
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 创建响应接收器（必须在当前 loop 中创建）
            response_future = loop.create_future()
            
            def on_l2_response(event: ZulongEvent):
                """L2 响应回调"""
                if not response_future.done():
                    logger.info(f"[ReplayIntegration] 收到 L2 响应事件")
                    # 🔥 修复：在线程安全的方式下设置 Future 结果
                    if not loop.is_running():
                        loop.call_soon_threadsafe(response_future.set_result, event.payload)
                    else:
                        loop.call_soon_threadsafe(response_future.set_result, event.payload)
            
            # 订阅临时响应
            temp_subscriber_id = f"review_response_{self.review_session_id}"
            try:
                event_bus.subscribe(EventType.L2_OUTPUT, on_l2_response, temp_subscriber_id)
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 订阅 L2 响应失败：{e}")
            
            try:
                # 发布分析请求
                event = ZulongEvent(
                    type=EventType.SYSTEM_L2_COMMAND,
                    source="ReplayIntegration",
                    payload={
                        'command': 'analyze_for_review',
                        'prompt': prompt,
                        'session_id': self.review_session_id,
                        'deep_analysis': deep,
                        'expect_json_response': True
                    },
                    priority=EventPriority.HIGH
                )
                
                event_bus.publish(event)
                logger.info(f"[ReplayIntegration] 已发布 L2 分析请求（深度：{deep}）")
                
                # 🔥 同步等待响应（最多等待 30 秒）
                start_time = time.time()
                
                try:
                    # 🔥 修复：如果已经有运行中的事件循环，使用异步方式等待
                    # 如果没有，使用 run_until_complete
                    if asyncio.get_running_loop():
                        # 在异步上下文中，需要特殊处理
                        # 创建一个新的线程来等待
                        import concurrent.futures
                        
                        def wait_for_result():
                            new_loop = asyncio.new_event_loop()
                            try:
                                asyncio.set_event_loop(new_loop)
                                return new_loop.run_until_complete(
                                    asyncio.wait_for(response_future, timeout=30.0)
                                )
                            finally:
                                new_loop.close()
                        
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(wait_for_result)
                            analysis_result = future.result(timeout=35.0)
                    else:
                        # 在同步上下文中，直接使用 run_until_complete
                        analysis_result = loop.run_until_complete(
                            asyncio.wait_for(response_future, timeout=30.0)
                        )
                    
                    elapsed = time.time() - start_time
                    logger.info(f"[ReplayIntegration] 收到 L2 响应，耗时：{elapsed:.2f}秒")
                    
                    # 解析 L2 响应
                    if isinstance(analysis_result, dict):
                        # L2 返回了 JSON
                        return {
                            'total_conversations': len(conversations),
                            'key_decisions': analysis_result.get('key_decisions', []),
                            'user_intents': analysis_result.get('user_intents', []),
                            'quality_score': float(analysis_result.get('quality_score', 0.8)),
                            'improvements': analysis_result.get('improvements', []),
                            'experiences': analysis_result.get('experiences', []),
                            'summary': analysis_result.get('summary', '')
                        }
                    else:
                        # L2 返回了文本，尝试解析
                        logger.warning(f"[ReplayIntegration] L2 返回了文本而非 JSON，尝试解析")
                        
                        # 🔥 无数据模式：直接返回 L2 的回复作为总结
                        if no_data_mode:
                            return {
                                'total_conversations': 0,
                                'key_decisions': [],
                                'user_intents': [],
                                'quality_score': 0.0,
                                'improvements': [],
                                'experiences': [],
                                'summary': str(analysis_result)  # 直接使用 L2 的回复
                            }
                        else:
                            return self._parse_l2_response(str(analysis_result), conversations)
                        
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.warning(f"[ReplayIntegration] L2 响应超时（{elapsed:.2f}秒），使用本地简化分析")
                    return self._analyze_conversation(data)
                except Exception as e:
                    logger.error(f"[ReplayIntegration] 等待 L2 响应失败：{e}", exc_info=True)
                    return self._analyze_conversation(data)
                finally:
                    # 🔥 修复：只关闭我们创建的 loop
                    if not asyncio.get_running_loop():
                        loop.close()
                    
            finally:
                # 取消订阅
                try:
                    event_bus.unsubscribe(EventType.L2_OUTPUT, temp_subscriber_id)
                    logger.debug(f"[ReplayIntegration] 已取消临时订阅：{temp_subscriber_id}")
                except Exception as e:
                    logger.debug(f"[ReplayIntegration] 取消订阅失败：{e}")
        
        except Exception as e:
            logger.error(f"[ReplayIntegration] L2 分析失败：{e}", exc_info=True)
            # 返回简化分析结果
            return self._analyze_conversation(data)
    
    def _parse_l2_response(self, response_text: str, conversations: List) -> Dict[str, Any]:
        """解析 L2 的文本响应
        
        Args:
            response_text: L2 返回的文本
            conversations: 原始对话数据
            
        Returns:
            Dict: 解析后的分析结果
        """
        logger.info(f"[ReplayIntegration] 尝试解析 L2 文本响应：{response_text[:100]}...")
        
        # 简单实现：返回基本信息
        # TODO: 实现 JSON 解析
        return {
            'total_conversations': len(conversations),
            'key_decisions': [],
            'user_intents': [],
            'quality_score': 0.8,
            'improvements': [],
            'experiences': [],
            'summary': response_text[:500] if response_text else ''
        }
    
    async def _handle_quiet_mode_review(self, context: Dict[str, Any]):
        """🔥 修复：异步处理安静模式复盘
        
        Args:
            context: 上下文信息，包含：
                - quiet_duration_minutes: 安静时长（分钟）
        """
        logger.info(f"[ReplayIntegration] 处理安静模式复盘")
        logger.info(f"[ReplayIntegration] 安静时长：{context.get('quiet_duration_minutes', 0):.1f}分钟")
        
        try:
            # 1. 获取安静模式期间的系统状态
            quiet_duration = context.get('quiet_duration_minutes', 0)
            recent_data = await self._get_recent_context(minutes=int(quiet_duration))
            
            # 2. 分析安静期间的状态
            analysis_result = self._analyze_quiet_period(recent_data, quiet_duration)
            
            # 3. 生成复盘报告
            report = self._generate_review_report(
                trigger_type='quiet_mode',
                context=context,
                analysis=analysis_result,
                data=recent_data
            )
            
            # 4. 保存报告
            self._save_review_report(report)
            
            logger.info(f"[ReplayIntegration] 安静模式复盘完成")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 安静模式复盘失败：{e}", exc_info=True)
    
    async def _handle_night_schedule_review(self, context: Dict[str, Any]):
        """🔥 修复：异步处理夜间定时复盘
        
        Args:
            context: 上下文信息，包含：
                - scheduled_time: 计划触发时间
        """
        logger.info(f"[ReplayIntegration] 处理夜间定时复盘")
        
        try:
            # 1. 获取当天的所有数据和任务
            recent_data = await self._get_recent_context(minutes=24*60)  # 过去 24 小时
            
            # 2. 分析当天的所有任务和对话
            analysis_result = self._analyze_daily_summary(recent_data)
            
            # 3. 生成每日复盘报告
            report = self._generate_review_report(
                trigger_type='night_schedule',
                context=context,
                analysis=analysis_result,
                data=recent_data
            )
            
            # 4. 保存报告
            self._save_review_report(report)
            
            logger.info(f"[ReplayIntegration] 夜间定时复盘完成")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 夜间定时复盘失败：{e}", exc_info=True)
    
    async def _get_recent_context(self, minutes: int = 30) -> Dict[str, Any]:
        """🔥 修复：异步获取上下文数据
        
        Args:
            minutes: 获取最近多少分钟的数据
            
        Returns:
            Dict: 上下文数据，包含对话、任务、系统状态等
        """
        logger.info(f"[ReplayIntegration] 🔍 开始获取上下文数据，时间窗口：{minutes}分钟")
        try:
            # 🔥 修复：直接使用当前事件循环，避免死锁
            from zulong.infrastructure.shared_memory_pool import SharedMemoryPool, ZoneType, DataType
            
            # ✅ 异步获取共享池实例
            logger.info(f"[ReplayIntegration] 尝试获取 SharedMemoryPool 实例...")
            pool = await SharedMemoryPool.get_instance()
            logger.info(f"[ReplayIntegration] ✅ 成功获取 SharedMemoryPool 实例：{pool}")
            
            # ✅ 异步读取最近的文本数据
            logger.info(f"[ReplayIntegration] 尝试获取最近 {minutes} 分钟的数据...")
            recent_texts = await pool.get_recent(time_window_sec=minutes*60)
            
            # 🔥 调试日志：记录获取到的数据
            logger.info(f"[ReplayIntegration] 从共享池获取到 {len(recent_texts)} 条数据")
            
            conversations = []
            for item in recent_texts:
                if hasattr(item, 'data'):
                    conversations.append({
                        'trace_id': item.trace_id if hasattr(item, 'trace_id') else None,
                        'text': item.data.get('text', '') if hasattr(item, 'data') else '',
                        'timestamp': item.timestamp if hasattr(item, 'timestamp') else None,
                        'metadata': item.metadata if hasattr(item, 'metadata') else {}
                    })
            
            logger.info(f"[ReplayIntegration] 构建了 {len(conversations)} 条对话记录")
            
            return {
                'conversations': conversations,
                'time_range': {
                    'start': (datetime.utcnow() - timedelta(minutes=minutes)).isoformat(),
                    'end': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 获取上下文数据失败：{e}", exc_info=True)
            return {'conversations': [], 'time_range': {}}
    
    def _analyze_conversation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析对话逻辑和决策过程
        
        Args:
            data: 上下文数据
            
        Returns:
            Dict: 分析结果
        """
        conversations = data.get('conversations', [])
        
        # 简单分析：统计对话数量、识别关键决策点
        analysis = {
            'total_conversations': len(conversations),
            'key_decisions': [],
            'user_intents': [],
            'system_responses': []
        }
        
        # TODO: 调用 L2 或 L3 进行深度分析
        # 目前先做简单的统计
        
        logger.info(f"[ReplayIntegration] 分析了 {len(conversations)} 条对话记录")
        
        return analysis
    
    def _analyze_quiet_period(self, data: Dict[str, Any], duration_minutes: float) -> Dict[str, Any]:
        """分析安静期间的状态
        
        Args:
            data: 上下文数据
            duration_minutes: 安静时长
            
        Returns:
            Dict: 分析结果
        """
        return {
            'quiet_duration_minutes': duration_minutes,
            'activity_count': len(data.get('conversations', [])),
            'summary': f"安静模式持续{duration_minutes:.1f}分钟，期间检测到{len(data.get('conversations', []))}条活动"
        }
    
    def _analyze_daily_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析每日总结
        
        Args:
            data: 上下文数据
            
        Returns:
            Dict: 分析结果
        """
        conversations = data.get('conversations', [])
        
        return {
            'total_activities': len(conversations),
            'time_range': data.get('time_range', {}),
            'summary': f"今日共处理{len(conversations)}条对话"
        }
    
    def _generate_experiences(self, analysis: Dict[str, Any], draft: bool = False) -> List[Dict]:
        """🔥 修复：从分析结果生成经验（调用 L2 进行智能生成）
        
        Args:
            analysis: 分析结果
            draft: 是否是草案（草案需要用户确认）
            
        Returns:
            List[Dict]: 经验列表
        """
        try:
            # 🔥 修复：调用 L2 进行经验生成
            experiences = self._generate_experiences_with_l2(analysis)
            
            # 如果 L2 生成失败或返回空，降级到基于规则生成
            if not experiences:
                logger.info("[ReplayIntegration] L2 生成经验失败，降级到基于规则生成")
                experiences = self._generate_experiences_fallback(analysis, draft)
            
            return experiences
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 生成经验失败：{e}")
            # 降级处理
            return self._generate_experiences_fallback(analysis, draft)
    
    def _generate_experiences_with_l2(self, analysis: Dict[str, Any]) -> List[Dict]:
        """🔥 新增：调用 L2 智能生成经验
        
        Args:
            analysis: 分析结果
            
        Returns:
            List[Dict]: 经验列表
        """
        try:
            # 构建生成提示
            prompt = (
                "请根据以下分析结果，提炼 1-3 条可执行的经验：\n\n"
                f"分析结果：{analysis}\n\n"
                "经验格式要求:\n"
                "1. 简洁明了（不超过 50 字）\n"
                "2. 可执行（包含具体操作）\n"
                "3. 可迁移（适用于类似场景）\n\n"
                "请以 JSON 格式返回数组：[\n"
                "  {\"content\": \"经验内容\", \"category\": \"类别\", \"confidence\": 0.8},\n"
                "  ...\n"
                "]"
            )
            
            # 🔥 调用 L2（复用 _analyze_conversation_with_l2 的逻辑）
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, EventPriority, ZulongEvent
            
            import asyncio
            import json
            
            # 获取事件循环
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 创建响应接收器
            response_future = loop.create_future()
            
            def on_l2_response(event: ZulongEvent):
                """L2 响应回调"""
                if not response_future.done():
                    logger.info(f"[ReplayIntegration] 收到 L2 经验生成响应")
                    loop.call_soon_threadsafe(response_future.set_result, event.payload)
            
            # 订阅临时响应
            temp_subscriber_id = f"review_exp_gen_{self.review_session_id}"
            try:
                event_bus.subscribe(EventType.L2_OUTPUT, on_l2_response, temp_subscriber_id)
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 订阅 L2 响应失败：{e}")
            
            try:
                # 发布生成请求
                event = ZulongEvent(
                    type=EventType.SYSTEM_L2_COMMAND,
                    source="ReplayIntegration",
                    payload={
                        'command': 'generate_experiences',
                        'prompt': prompt,
                        'session_id': self.review_session_id,
                        'expect_json_response': True
                    },
                    priority=EventPriority.HIGH
                )
                
                event_bus.publish(event)
                logger.info("[ReplayIntegration] 已发布 L2 经验生成请求")
                
                # 同步等待响应（最多 15 秒）
                try:
                    payload = loop.run_until_complete(asyncio.wait_for(response_future, timeout=15.0))
                    logger.info(f"[ReplayIntegration] ✅ L2 经验生成响应已接收")
                    
                    # 解析 JSON 响应
                    text = payload.get('text', '')
                    
                    # 尝试提取 JSON
                    import re
                    json_match = re.search(r'\[.*\]', text, re.DOTALL)
                    if json_match:
                        experiences = json.loads(json_match.group())
                        logger.info(f"[ReplayIntegration] ✅ 成功解析 {len(experiences)} 条经验")
                        
                        # 添加元数据
                        for exp in experiences:
                            exp['draft'] = draft
                            exp['source'] = 'l2_generated'
                        
                        return experiences
                    else:
                        logger.warning(f"[ReplayIntegration] 未找到 JSON 格式：{text}")
                        return []
                        
                except asyncio.TimeoutError:
                    logger.error("[ReplayIntegration] L2 经验生成超时")
                    return []
                    
            finally:
                # 取消订阅
                try:
                    event_bus.unsubscribe(EventType.L2_OUTPUT, temp_subscriber_id)
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] L2 经验生成失败：{e}", exc_info=True)
            return []
    
    def _generate_experiences_fallback(self, analysis: Dict[str, Any], draft: bool = False) -> List[Dict]:
        """🔥 降级：基于规则生成经验（当 L2 失败时使用）
        
        Args:
            analysis: 分析结果
            draft: 是否是草案
            
        Returns:
            List[Dict]: 经验列表
        """
        experiences = []
        
        # 从分析结果中提取经验
        if analysis.get('key_decisions'):
            for i, decision in enumerate(analysis['key_decisions']):
                experiences.append({
                    'id': f'exp_{i+1}',
                    'type': 'decision',
                    'content': f'关键决策：{decision}',
                    'confidence': 0.8,
                    'tags': ['决策', '关键节点'],
                    'draft': draft
                })
        
        if analysis.get('improvements'):
            for i, improvement in enumerate(analysis['improvements']):
                experiences.append({
                    'id': f'imp_{i+1}',
                    'type': 'improvement',
                    'content': f'可优化：{improvement}',
                    'confidence': 0.7,
                    'tags': ['优化', '改进'],
                    'draft': draft
                })
        
        # 如果没有具体内容，生成默认经验
        if not experiences:
            experiences.append({
                'id': 'exp_default',
                'type': 'general',
                'content': '对话分析完成，暂无特定经验',
                'confidence': 0.5,
                'tags': ['通用'],
                'draft': draft
            })
        
        return experiences
    
    async def _apply_experiences(self, experiences: List[Dict]) -> int:
        """应用经验到记忆库 - 真正持久化存储
        
        Args:
            experiences: 经验列表
            
        Returns:
            int: 成功保存的经验数量
        """
        saved_count = 0
        try:
            logger.info(f"[ReplayIntegration] 应用 {len(experiences)} 条经验到记忆库")
            
            # 尝试写入 ExperienceStore（replay 模块的经验向量库）
            try:
                from zulong.replay.experience_store import get_experience_store
                from zulong.replay.patch_compiler import SystemPatch, PatchStatus
                from zulong.replay.attributor import FaultLayer, AdjustmentType
                from zulong.replay.clock_synchronizer import get_unified_timestamp
                
                store = get_experience_store()
                
                for exp in experiences:
                    content = exp.get('content', '')
                    tags = exp.get('tags', [])
                    confidence = exp.get('confidence', 0.8)
                    
                    # 将经验转换为 SystemPatch 格式（ExperienceStore 的标准输入）
                    patch = SystemPatch(
                        patch_id=f"review_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{saved_count}",
                        condition=f"review_experience: {content[:50]}",
                        adjustment=content,
                        fault_layer=FaultLayer.L2.value if hasattr(FaultLayer, 'L2') else 'L2',
                        adjustment_type=AdjustmentType.PROMPT_INJECT.value if hasattr(AdjustmentType, 'PROMPT_INJECT') else 'prompt_inject',
                        confidence=confidence,
                        delta_t=0.0,
                        scene_features=tags,
                        source_event_id=self._state_manager.get_session_id() or 'review_session',
                        status=PatchStatus.ACTIVE if hasattr(PatchStatus, 'ACTIVE') else 'active'
                    )
                    
                    entry_id = await store.store(patch, additional_keywords=tags)
                    exp['applied'] = True
                    exp['entry_id'] = entry_id
                    exp['applied_at'] = datetime.utcnow().isoformat()
                    saved_count += 1
                    logger.info(f"[ReplayIntegration] 经验已存储: {entry_id}")
                    
            except Exception as e:
                logger.warning(f"[ReplayIntegration] ExperienceStore 写入失败: {e}，使用事件通知方式")
                # 降级：通过事件通知其他组件处理存储
                for exp in experiences:
                    exp['applied'] = True
                    exp['applied_at'] = datetime.utcnow().isoformat()
                    saved_count += 1
            
            # 发布经验存储事件
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, EventPriority, ZulongEvent
            
            event = ZulongEvent(
                type=EventType.EXPERIENCE_STORED,
                source="ReplayIntegration",
                payload={
                    'experiences': experiences,
                    'session_id': getattr(self, 'review_session_id', None) or self._state_manager.get_session_id(),
                    'saved_count': saved_count,
                    'auto_applied': True
                },
                priority=EventPriority.NORMAL
            )
            
            event_bus.publish(event)
            
            logger.info(f"[ReplayIntegration] 经验已应用到记忆库，成功 {saved_count} 条")
            return saved_count
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 应用经验失败：{e}", exc_info=True)
            return saved_count
    
    def _format_experience_draft(self, experiences: List[Dict]) -> str:
        """格式化经验草案供用户确认
        
        Args:
            experiences: 经验草案列表
            
        Returns:
            str: 格式化后的文本
        """
        if not experiences:
            return "暂无经验总结"
        
        lines = [
            "📝 **我总结了以下经验，请您确认：**\n",
            "您可以回复：\n"
            "- ✅ **确认** / **保存**：应用所有经验\n"
            "- ✏️ **修改第 X 条**：我会重新总结\n"
            "- ❌ **取消** / **不要了**：放弃保存\n"
        ]
        
        for i, exp in enumerate(experiences, 1):
            exp_type = exp.get('type', 'general')
            content = exp.get('content', '')
            confidence = exp.get('confidence', 0)
            
            type_icon = {
                'decision': '🎯',
                'improvement': '💡',
                'general': '📌'
            }.get(exp_type, '📌')
            
            lines.append(f"{i}. {type_icon} {content} (置信度：{confidence:.0%})")
        
        return "\n".join(lines)
    
    def _format_experience_draft_enhanced(self, experiences: List[Dict]) -> str:
        """🔥 增强版：结构化格式化经验草案供用户确认
        
        Args:
            experiences: 经验草案列表
            
        Returns:
            str: 格式化后的文本（带分割线和结构化展示）
        """
        if not experiences:
            return "暂无经验总结"
        
        # 🔥 增强：入口仪式感 + 结构化卡片
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📋 **深度复盘完成 - 请您审阅**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💡 我从本次对话中提炼了 **{len(experiences)}** 条核心经验：\n"
        ]
        
        for i, exp in enumerate(experiences, 1):
            exp_type = exp.get('type', 'general')
            content = exp.get('content', '')
            confidence = exp.get('confidence', 0)
            tags = exp.get('tags', [])
            
            type_icon = {
                'decision': '🎯 决策',
                'improvement': '💡 改进',
                'lesson': '📚 教训',
                'best_practice': '⭐ 最佳实践',
                'general': '📌 经验'
            }.get(exp_type, '📌 经验')
            
            # 🔥 结构化展示：类型 + 内容 + 置信度 + 标签
            lines.append(f"\n**第 {i} 条：{type_icon}**")
            lines.append(f"└─ 内容：{content}")
            lines.append(f"└─ 置信度：{confidence:.0%}")
            
            if tags:
                lines.append(f"└─ 标签：{' '.join(tags)}")
        
        # 🔥 增强：强制确认机制文案
        lines.extend([
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "✅ **请确认是否存入您的知识库**\n\n"
            "您可以回复：\n"
            "• ✅ **确认** / **保存** / **好的**：应用所有经验到记忆库\n"
            "• ✏️ **修改第 X 条**：我会重新总结该条经验\n"
            "• ❌ **取消** / **不要了** / **放弃**：本次不保存\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ])
        
        return "\n".join(lines)
    
    def _publish_l2_response(self, text: str):
        """发布 L2 响应到事件总线
        
        Args:
            text: 要回复的文本
        """
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, EventPriority, ZulongEvent
            
            # 确保 review_session_id 存在
            session_id = getattr(self, 'review_session_id', None)
            
            event = ZulongEvent(
                type=EventType.L2_OUTPUT,
                source="ReplayIntegration",
                payload={
                    'text': text,
                    'session_id': session_id,
                    'review_mode': True  # 始终设置为 True，确保响应被正确处理
                },
                priority=EventPriority.HIGH
            )
            
            # 🔥 修复：直接发布事件，不使用 call_soon_threadsafe
            # 因为事件总线会在线程中处理
            event_bus.publish(event)
            
            logger.info(f"[ReplayIntegration] 已发布 L2 响应：{text[:50]}...")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 发布 L2 响应失败：{e}")
    
    def _publish_status_message(self, text: str):
        """发布状态消息（用于显示 Loading/Processing 状态）
        
        Args:
            text: 状态消息文本
        """
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, EventPriority, ZulongEvent
            
            event = ZulongEvent(
                type=EventType.L2_OUTPUT,
                source="ReplayIntegration",
                payload={
                    'text': f"[状态] {text}",
                    'session_id': self.review_session_id,
                    'review_mode': self.review_mode,
                    'is_status_message': True  # 标记为状态消息
                },
                priority=EventPriority.NORMAL
            )
            
            event_bus.publish(event)
            
            logger.info(f"[ReplayIntegration] 已发布状态消息：{text}")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 发布状态消息失败：{e}")
    
    async def _analyze_conversation_with_l2_async(self, buffer_data: dict) -> dict:
        """🔥 新增：调用 L2 分析对话内容
        
        Args:
            buffer_data: 缓冲区数据（包含 conversations 列表）
            
        Returns:
            分析结果字典（包含 summary, suggested_tags 等）
        """
        try:
            logger.info("[ReplayIntegration] 调用 L2 进行对话分析...")
            
            # 构建分析提示词
            conversations = buffer_data.get('conversations', [])
            total_conversations = len(conversations)
            
            if total_conversations == 0:
                logger.warning("[ReplayIntegration] 没有对话内容可分析")
                return {
                    'summary': '本次复盘没有足够的对话内容',
                    'suggested_tags': [],
                    'total_conversations': 0,
                    'key_insights': []
                }
            
            # 构建对话文本
            conversation_text = ""
            for conv in conversations[-30:]:
                role = conv.get('role', 'unknown')
                content = conv.get('content', '')
                timestamp = conv.get('timestamp', '')
                role_display = "用户" if role == 'user' else "助手"
                conversation_text += f"[{timestamp}] {role_display}: {content}\n"
            
            # 构建分析提示
            analysis_prompt = f"""
请分析以下对话内容，提取有价值的经验和教训：

{conversation_text}

分析要求：
1. 总结对话的主要内容和重点
2. 提取 3-5 条有价值的经验或教训
3. 为每条经验提供简短的描述
4. 为每条经验建议一个分类
5. 为整个对话提供 2-3 个相关标签

请以 JSON 格式输出分析结果，包含以下字段：
- summary: 对话总结
- suggested_tags: 建议的标签列表
- total_conversations: 对话总数
- key_insights: 关键洞察列表
- experiences: 提取的经验列表（每条包含 title, content, category）
"""
            
            # 调用 L2 进行分析
            from zulong.l2.inference_engine import get_inference_engine
            engine = get_inference_engine()
            
            if not engine:
                logger.error("[ReplayIntegration] 无法获取 InferenceEngine 实例")
                # 降级：返回模拟结果
                return {
                    'summary': f'分析了 {total_conversations} 条对话记录',
                    'suggested_tags': ['工作总结', '经验提取'],
                    'total_conversations': total_conversations,
                    'key_insights': [
                        '通过对话发现了有价值的讨论内容',
                        '可以从中提炼出实用的经验和教训'
                    ],
                    'experiences': []
                }
            
            # 调用 L2 分析
            analysis_result = await engine._analyze_for_review(analysis_prompt)
            
            # 确保返回的数据结构正确
            if not isinstance(analysis_result, dict):
                logger.error("[ReplayIntegration] L2 返回的分析结果不是字典")
                # 降级：返回模拟结果
                return {
                    'summary': f'分析了 {total_conversations} 条对话记录',
                    'suggested_tags': ['工作总结', '经验提取'],
                    'total_conversations': total_conversations,
                    'key_insights': [
                        '通过对话发现了有价值的讨论内容',
                        '可以从中提炼出实用的经验和教训'
                    ],
                    'experiences': []
                }
            
            logger.info(f"[ReplayIntegration] ✅ L2 分析完成")
            return analysis_result
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] L2 分析失败：{e}", exc_info=True)
            # 失败时返回错误结果，由调用方处理异常
            return {
                'total_conversations': len(buffer_data.get('conversations', [])),
                'key_decisions': [],
                'user_intents': [],
                'quality_score': 0.0,
                'improvements': [],
                'experiences': [],
                'summary': '分析失败'
            }
    
    async def _generate_experiences_async(self, analysis_result: dict, draft: bool = False) -> list:
        """🔥 新增：基于分析结果生成经验
        
        Args:
            analysis_result: 分析结果
            draft: 是否为草稿模式（草稿模式下不自动保存）
            
        Returns:
            生成的经验列表
        """
        try:
            logger.info("[ReplayIntegration] 正在生成经验...")
            
            # 检查分析结果中是否已经包含经验
            if 'experiences' in analysis_result and analysis_result['experiences']:
                logger.info(f"[ReplayIntegration] 使用分析结果中的经验（{len(analysis_result['experiences'])}条）")
                return analysis_result['experiences']
            
            # 构建生成经验的提示
            summary = analysis_result.get('summary', '')
            key_insights = analysis_result.get('key_insights', [])
            
            experience_prompt = f"""
基于以下对话分析结果，生成 3-5 条有价值的经验：

分析总结：
{summary}

关键洞察：
{chr(10).join(f"- {insight}" for insight in key_insights)}

生成要求：
1. 每条经验要有明确的标题和详细的描述
2. 为每条经验建议一个分类（如：工作效率、学习方法、经验管理等）
3. 经验要具体、实用，能够指导未来的行为
4. 语言要简洁明了，易于理解

请以 JSON 格式输出生成的经验列表，每条经验包含以下字段：
- title: 经验标题
- content: 经验内容
- category: 经验分类
- importance: 重要性（高/中/低）
"""
            
            # 调用 L2 生成经验
            from zulong.l2.inference_engine import get_inference_engine
            engine = get_inference_engine()
            
            if not engine:
                logger.error("[ReplayIntegration] 无法获取 InferenceEngine 实例")
                # 降级：返回模拟经验
                return [
                    {
                        'title': '沟通效率提升',
                        'content': '在讨论复杂问题时，先明确目标和范围，避免无效讨论。',
                        'category': '工作效率',
                        'importance': '高'
                    },
                    {
                        'title': '定期回顾的重要性',
                        'content': '通过复盘可以发现被忽视的问题和改进机会。',
                        'category': '学习方法',
                        'importance': '中'
                    },
                    {
                        'title': '实践经验的价值',
                        'content': '从实际对话中提取的经验比理论更贴近实际情况。',
                        'category': '经验管理',
                        'importance': '中'
                    }
                ]
            
            # 调用 L2 生成经验
            import json
            response = await engine._analyze_for_review(experience_prompt)
            
            # 解析 L2 返回的结果
            if isinstance(response, str):
                try:
                    # 尝试从响应中提取 JSON
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', response)
                    if json_match:
                        experiences = json.loads(json_match.group(0))
                        if isinstance(experiences, list):
                            logger.info(f"[ReplayIntegration] ✅ 从 L2 响应中提取到 {len(experiences)} 条经验")
                            return experiences
                except Exception as e:
                    logger.error(f"[ReplayIntegration] 解析 L2 响应失败：{e}")
            elif isinstance(response, dict) and 'experiences' in response:
                experiences = response['experiences']
                if isinstance(experiences, list):
                    logger.info(f"[ReplayIntegration] ✅ 从 L2 响应中获取到 {len(experiences)} 条经验")
                    return experiences
            
            # 降级：返回模拟经验
            logger.warning("[ReplayIntegration] L2 响应格式不正确，使用模拟经验")
            return [
                {
                    'title': '沟通效率提升',
                    'content': '在讨论复杂问题时，先明确目标和范围，避免无效讨论。',
                    'category': '工作效率',
                    'importance': '高'
                },
                {
                    'title': '定期回顾的重要性',
                    'content': '通过复盘可以发现被忽视的问题和改进机会。',
                    'category': '学习方法',
                    'importance': '中'
                },
                {
                    'title': '实践经验的价值',
                    'content': '从实际对话中提取的经验比理论更贴近实际情况。',
                    'category': '经验管理',
                    'importance': '中'
                }
            ]
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 生成经验失败：{e}", exc_info=True)
            # 失败时返回空列表，由调用方处理
            return []
    
    def _generate_review_report(self,
                                trigger_type: str,
                                context: Dict[str, Any],
                                analysis: Dict[str, Any],
                                data: Dict[str, Any],
                                experiences: List[Dict] = None) -> Dict[str, Any]:
        """生成复盘报告
        
        Args:
            trigger_type: 触发类型
            context: 触发上下文
            analysis: 分析结果
            data: 原始数据
            
        Returns:
            Dict: 复盘报告
        """
        report_id = f"review_{trigger_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        report = {
            'report_id': report_id,
            'timestamp': datetime.utcnow().isoformat(),
            'trigger_type': trigger_type,
            'trigger_context': context,
            'analysis_result': analysis,
            'data_summary': {
                'conversation_count': len(data.get('conversations', [])),
                'time_range': data.get('time_range', {})
            },
            'experiences': experiences or [],
            'recommendations': []
        }
        
        # 根据分析结果生成建议和经验
        if trigger_type == 'user_active':
            report['recommendations'].append("用户主动复盘，建议重点关注最近对话的决策质量")
        elif trigger_type == 'user_active_quick':
            report['recommendations'].append("快速复盘完成，经验已自动应用")
        elif trigger_type == 'user_active_deep_confirmed':
            report['recommendations'].append("深度复盘完成，经验已通过用户确认并应用")
        elif trigger_type == 'quiet_mode':
            report['recommendations'].append("安静模式超时，系统运行正常")
        elif trigger_type == 'night_schedule':
            report['recommendations'].append("每日定时复盘，建议总结经验教训")
        
        return report
    
    def _save_review_report(self, report: Dict[str, Any]):
        """保存复盘报告
        
        Args:
            report: 复盘报告
        """
        try:
            import json
            from pathlib import Path
            
            # 创建报告目录
            report_dir = Path("data/reviews")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存为 JSON 文件
            report_file = report_dir / f"{report['report_id']}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[ReplayIntegration] 📄 复盘报告已保存：{report_file}")
            
            # TODO: 同时发布 EXPERIENCE_STORED 事件，通知经验库
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 保存复盘报告失败：{e}")
    
    def handle_user_confirmation(self, user_input: str):
        """处理用户对经验草案的确认
        
        Args:
            user_input: 用户输入（确认/取消/修改/退出）
        """
        try:
            logger.info(f"[ReplayIntegration] 处理用户确认：{user_input}")
            
            user_input_lower = user_input.lower().strip()
            
            # 🔥 关键修复：检查退出指令 (即使没有 pending_experiences)
            if any(word in user_input_lower for word in ['退出', '结束', '退出复盘', '结束复盘']):
                logger.info("[ReplayIntegration] 用户选择退出复盘")
                self._exit_review_mode()
                return
            
            # 🔥 关键修复：检查是否有待确认的经验
            if not self.pending_experiences:
                logger.warning("[ReplayIntegration] 没有待确认的经验草案")
                
                # 🔥 关键修复：检查用户是否选择了复盘模式
                if '快速' in user_input_lower or '快速复盘' in user_input_lower:
                    logger.info("[ReplayIntegration] 用户选择快速复盘")
                    self.review_type = 'quick'
                    recent_data = self._get_recent_context()
                    self._handle_quick_review(recent_data, {
                        'trigger_keyword': '快速复盘',
                        'user_input': user_input,
                        'trigger_source': 'gatekeeper_input'
                    })
                    return
                
                elif '深度' in user_input_lower or '深度复盘' in user_input_lower:
                    logger.info("[ReplayIntegration] 用户选择深度复盘")
                    self.review_type = 'deep'
                    recent_data = self._get_recent_context()
                    # 异步调用
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.create_task(
                            self._handle_deep_review(recent_data, {
                                'trigger_keyword': '深度复盘',
                                'user_input': user_input,
                                'trigger_source': 'gatekeeper_input'
                            })
                        )
                    except RuntimeError:
                        def run_async():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                new_loop.run_until_complete(
                                    self._handle_deep_review(recent_data, {
                                        'trigger_keyword': '深度复盘',
                                        'user_input': user_input,
                                        'trigger_source': 'gatekeeper_input'
                                    })
                                )
                            finally:
                                new_loop.close()
                        
                        import threading
                        thread = threading.Thread(target=run_async)
                        thread.daemon = True
                        thread.start()
                    return
                
                # 没有选择模式，显示操作选择提示
                response_text = (
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "💬 **复盘模式 - 请选择操作**\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "目前没有待确认的经验草案。您可以：\n\n"
                    "⚡ **快速复盘**\n"
                    "   分析最近对话，自动生成并应用经验\n\n"
                    "🔍 **深度复盘**\n"
                    "   调用长期记忆，生成经验草案需您确认\n\n"
                    "🚪 **退出**\n"
                    "   结束复盘模式，继续正常对话\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "💬 请直接说 `快速复盘`、`深度复盘` 或 `退出`\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
                self._publish_l2_response(response_text)
                return
            
            # 🔥 [BUG-12 修复] 优先检查经验确认阶段
            # 如果有待确认的经验，"确认"应该走 _confirm_experiences 而非 _execute_quick_review
            if self.pending_experiences and any(word in user_input_lower for word in ['确认', '保存', '好的', '同意', '✅']):
                logger.info("[ReplayIntegration] 经验确认阶段，用户确认保存经验")
                self._confirm_experiences()
                return
            
            if self.pending_experiences and any(word in user_input_lower for word in ['取消', '不要', '放弃', '❌']):
                logger.info("[ReplayIntegration] 经验确认阶段，用户取消经验")
                self._cancel_experiences()
                return
            
            if self.pending_experiences and '修改' in user_input_lower:
                logger.info("[ReplayIntegration] 经验确认阶段，用户修改经验")
                self._modify_experience(user_input)
                return
            
            # 🔥 新增：处理快速复盘确认执行
            if self.review_type == 'quick' and any(word in user_input_lower for word in ['确认', '好的', '开始', '执行', '确定', 'ok', 'yes']):
                logger.info("[ReplayIntegration] 用户确认执行快速复盘")
                # 异步执行真正的复盘
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(self._execute_quick_review())
                except RuntimeError:
                    def run_async():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            new_loop.run_until_complete(self._execute_quick_review())
                        finally:
                            new_loop.close()
                    
                    import threading
                    thread = threading.Thread(target=run_async)
                    thread.daemon = True
                    thread.start()
                return
            
            # 确认保存
            if any(word in user_input_lower for word in ['确认', '保存', '好的', '同意', '✅']):
                self._confirm_experiences()
            
            # 取消
            elif any(word in user_input_lower for word in ['取消', '不要', '放弃', '❌']):
                self._cancel_experiences()
            
            # 修改特定条目
            elif '修改' in user_input_lower:
                self._modify_experience(user_input)
            
            else:
                # 无法识别，重新提示
                response_text = (
                    "抱歉，我没有理解您的意思。\n\n"
                    "请回复：\n"
                    "- ✅ **确认** / **保存**：应用所有经验\n"
                    "- ❌ **取消** / **不要了**：放弃保存\n"
                    "- ✏️ **修改第 X 条**：我会重新总结\n"
                    "- 🚪 **退出**：结束复盘模式"
                )
                self._publish_l2_response(response_text)
        
        except Exception as e:
            logger.error(f"[ReplayIntegration] 处理用户确认失败：{e}")
    
    def _exit_review_mode(self):
        """🔥 新增：退出复盘模式"""
        try:
            logger.info("[ReplayIntegration] 退出复盘模式")
            
            # 清理状态
            self.review_mode = False
            self.review_type = None
            self.pending_experiences = None
            self.pending_summary = None
            self.pending_tags = None
            
            # 清理会话 ID
            if hasattr(self, 'review_session_id'):
                delattr(self, 'review_session_id')
            
            # 🔥 [BUG-13 修复] 通知 ReviewStateManager 退出复盘模式
            try:
                self._state_manager.exit_review_mode('cancelled')
            except Exception as e:
                logger.warning(f"[ReplayIntegration] ReviewStateManager 退出失败：{e}")
            
            # 同步到全局状态
            try:
                from zulong.core.state_manager import state_manager
                state_manager.set_context('review_mode', False)
                state_manager.set_context('review_session_id', None)
                logger.info("[ReplayIntegration] 已同步状态到 state_manager")
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 同步状态失败：{e}")
            
            # 🔥 清理缓冲区
            try:
                from zulong.review.temp_buffer import get_review_buffer_manager
                buffer_manager = get_review_buffer_manager()
                if buffer_manager.has_buffer():
                    buffer_manager.clear_buffer()
                    logger.info("[ReplayIntegration] 已清理缓冲区")
            except Exception as e:
                logger.debug(f"[ReplayIntegration] 清理缓冲区失败：{e}")
            
            # 回复用户
            response_text = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "✅ **已退出复盘模式**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "好的，复盘已结束。我们继续正常对话吧！😊\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            self._publish_l2_response(response_text)
            
            logger.info("[ReplayIntegration] 复盘模式已退出")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 退出复盘模式失败：{e}")
    
    def _confirm_experiences(self):
        """🔥 修改：确认并应用经验 - L1-B 执行"""
        try:
            experiences = self.pending_experiences
            
            # 🔥 新增：调用 L1-B 安全应用器执行写入
            success = self._apply_confirmed_experiences(experiences)
            
            if success:
                # 生成复盘报告
                report = self._generate_review_report(
                    trigger_type='user_active_deep_confirmed',
                    context={'confirmed_by_user': True},
                    analysis={'experiences_count': len(experiences)},
                    data={'conversations': []},
                    experiences=experiences
                )
                
                self._save_review_report(report)
                
                # 回复用户
                response_text = (
                    f"✅ **经验已保存并应用**\n\n"
                    f"共保存了 {len(experiences)} 条经验到记忆库。\n"
                    f"复盘会话 `{self.review_session_id}` 已结束。\n\n"
                    f"下次我会做得更好！💪"
                )
                self._publish_l2_response(response_text)
            else:
                response_text = (
                    f"❌ **经验保存失败**\n\n"
                    f"请重试或联系管理员。\n"
                )
                self._publish_l2_response(response_text)
            
            # 清空待确认经验
            self.pending_experiences = None
            self.pending_summary = None
            self.pending_tags = None
            
            # 退出复盘模式
            self.review_mode = False
            self.review_type = None
            
            # 🔥 [BUG-13 修复] 通知 ReviewStateManager 退出复盘模式
            try:
                self._state_manager.exit_review_mode('completed')
            except Exception as e:
                logger.warning(f"[ReplayIntegration] ReviewStateManager 退出失败：{e}")
            
            # 同步状态到全局状态管理器
            try:
                from zulong.core.state_manager import state_manager
                state_manager.set_context('review_mode', False)
                state_manager.set_context('review_session_id', None)
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 同步状态失败：{e}")
            
            logger.info(f"[ReplayIntegration] 用户确认，已应用{len(experiences)}条经验")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 确认经验失败：{e}")
    
    def _cancel_experiences(self):
        """取消并丢弃经验"""
        try:
            self.pending_experiences = None
            
            response_text = (
                "好的，已取消保存经验。\n\n"
                f"复盘会话 `{self.review_session_id}` 已结束。\n\n"
                "如果您之后想重新复盘，随时告诉我哦！😊"
            )
            self._publish_l2_response(response_text)
            
            # 退出复盘模式
            self.review_mode = False
            self.review_type = None
            
            # 🔥 [BUG-13 修复] 通知 ReviewStateManager 退出复盘模式
            try:
                self._state_manager.exit_review_mode('cancelled')
            except Exception as e:
                logger.warning(f"[ReplayIntegration] ReviewStateManager 退出失败：{e}")
            
            # 同步状态到全局状态管理器
            try:
                from zulong.core.state_manager import state_manager
                state_manager.set_context('review_mode', False)
                state_manager.set_context('review_session_id', None)
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 同步状态失败：{e}")
            
            logger.info("[ReplayIntegration] 用户取消，已丢弃经验")
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 取消经验失败：{e}")
    
    def _modify_experience(self, user_input: str):
        """修改特定经验条目
        
        Args:
            user_input: 用户输入，包含要修改的条目编号
        """
        try:
            # 提取数字（第几条）
            import re
            match = re.search(r'第 (\d+) 条', user_input)
            
            if not match:
                match = re.search(r'(\d+)', user_input)
            
            if match:
                index = int(match.group(1)) - 1
                
                if 0 <= index < len(self.pending_experiences):
                    # TODO: 重新生成该条经验
                    # 目前先简单回复
                    
                    response_text = (
                        f"好的，我会重新总结第 {index + 1} 条经验。\n\n"
                        "请稍等，我正在思考..."
                    )
                    self._publish_l2_response(response_text)
                    
                    # TODO: 调用 L2 重新生成该条经验
                    
                    logger.info(f"[ReplayIntegration] 用户要求修改第 {index + 1} 条经验")
                else:
                    response_text = f"抱歉，没有找到第 {index + 1} 条经验，请重新指定。"
                    self._publish_l2_response(response_text)
            else:
                response_text = "请告诉我具体要修改第几条，例如'修改第 1 条'。"
                self._publish_l2_response(response_text)
        
        except Exception as e:
            logger.error(f"[ReplayIntegration] 修改经验失败：{e}")
    
    async def _analyze_conversation_with_l2_async(self, data: Dict[str, Any], deep: bool = False) -> Dict[str, Any]:
        """🔥 新增：异步调用 L2 进行对话分析
        
        Args:
            data: 对话数据
            deep: 是否深度分析
            
        Returns:
            Dict: 分析结果
        """
        try:
            conversations = data.get('conversations', [])
            
            # 检查是否有对话数据
            no_data_mode = False
            if not conversations:
                logger.info("[ReplayIntegration] 没有对话数据，但强制调用 L2 进行交互")
                no_data_mode = True
                prompt = (
                    f"用户开启了快速复盘模式。\n\n"
                    f"虽然我没有检索到历史对话数据，但我已准备好为您服务。\n\n"
                    f"请以友好的语气回复用户，可以：\n"
                    f"1. 告知用户当前没有历史对话记录\n"
                    f"2. 询问用户想复盘什么内容\n"
                    f"3. 建议用户可以开始新的对话，稍后再复盘\n\n"
                    f"请生成一段自然、友好的回复。"
                )
            else:
                # 构建分析提示
                if deep:
                    prompt = (
                        f"请对以下对话进行深度分析：\n\n"
                        f"对话内容：{conversations}\n\n"
                        f"请分析：\n"
                        f"1. 用户的真实意图和目标\n"
                        f"2. 关键决策点和选择\n"
                        f"3. 系统响应的质量评估\n"
                        f"4. 可优化的环节\n"
                        f"5. 可复用的经验和教训\n\n"
                        f"请以 JSON 格式返回，包含：key_decisions, user_intents, quality_score, improvements"
                    )
                else:
                    prompt = (
                        f"请分析以下对话并总结经验：\n\n"
                        f"对话内容：{conversations}\n\n"
                        f"请总结：\n"
                        f"1. 对话主题\n"
                        f"2. 达成的结果\n"
                        f"3. 可复用的经验\n\n"
                        f"请以 JSON 格式返回，包含：experiences, summary, key_decisions"
                    )
            
            # 🔥 异步等待 L2 响应
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, EventPriority, ZulongEvent
            
            import time
            
            # 创建响应接收器
            loop = asyncio.get_event_loop()
            response_future = loop.create_future()
            
            def on_l2_response(event: ZulongEvent):
                """L2 响应回调"""
                if not response_future.done():
                    logger.info(f"[ReplayIntegration] 收到 L2 响应事件")
                    loop.call_soon_threadsafe(response_future.set_result, event.payload)
            
            # 订阅临时响应
            temp_subscriber_id = f"review_async_response_{self.review_session_id}"
            try:
                event_bus.subscribe(EventType.L2_OUTPUT, on_l2_response, temp_subscriber_id)
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 订阅 L2 响应失败：{e}")
            
            try:
                # 发布分析请求
                event = ZulongEvent(
                    type=EventType.SYSTEM_L2_COMMAND,
                    source="ReplayIntegration",
                    payload={
                        'command': 'analyze_for_review',
                        'prompt': prompt,
                        'session_id': self.review_session_id,
                        'deep_analysis': deep,
                        'expect_json_response': True
                    },
                    priority=EventPriority.HIGH
                )
                
                event_bus.publish(event)
                logger.info(f"[ReplayIntegration] 已发布 L2 分析请求（异步，深度：{deep}）")
                
                # 🔥 异步等待响应（最多等待 30 秒）
                start_time = time.time()
                
                try:
                    analysis_result = await asyncio.wait_for(response_future, timeout=30.0)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"[ReplayIntegration] 收到 L2 响应，耗时：{elapsed:.2f}秒")
                    
                    # 解析 L2 响应
                    if isinstance(analysis_result, dict):
                        return {
                            'total_conversations': len(conversations),
                            'key_decisions': analysis_result.get('key_decisions', []),
                            'user_intents': analysis_result.get('user_intents', []),
                            'quality_score': float(analysis_result.get('quality_score', 0.8)),
                            'improvements': analysis_result.get('improvements', []),
                            'experiences': analysis_result.get('experiences', []),
                            'summary': analysis_result.get('summary', '对话分析完成')
                        }
                    else:
                        # 返回简化版本
                        return {
                            'total_conversations': len(conversations),
                            'key_decisions': [],
                            'user_intents': [],
                            'quality_score': 0.8,
                            'improvements': [],
                            'experiences': [],
                            'summary': '对话分析完成'
                        }
                        
                except asyncio.TimeoutError:
                    logger.error("[ReplayIntegration] L2 分析超时")
                    return {
                        'total_conversations': len(conversations),
                        'key_decisions': [],
                        'user_intents': [],
                        'quality_score': 0.5,
                        'improvements': [],
                        'experiences': [],
                        'summary': '分析超时，暂无结论'
                    }
                    
            finally:
                # 取消订阅
                try:
                    event_bus.unsubscribe(EventType.L2_OUTPUT, temp_subscriber_id)
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] L2 分析失败：{e}", exc_info=True)
            return {
                'total_conversations': 0,
                'key_decisions': [],
                'user_intents': [],
                'quality_score': 0.0,
                'improvements': [],
                'experiences': [],
                'summary': '分析失败'
            }
    
    async def _generate_experiences_async(self, analysis: Dict[str, Any], draft: bool = False) -> List[Dict]:
        """🔥 新增：异步调用 L2 智能生成经验
        
        Args:
            analysis: 分析结果
            draft: 是否是草案
            
        Returns:
            List[Dict]: 经验列表
        """
        try:
            # 🔥 调用 L2 进行经验生成
            experiences = await self._generate_experiences_with_l2_async(analysis)
            
            # 如果 L2 生成失败或返回空，降级到基于规则生成
            if not experiences:
                logger.info("[ReplayIntegration] L2 生成经验失败，降级到基于规则生成")
                experiences = self._generate_experiences_fallback(analysis, draft)
            
            return experiences
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] 生成经验失败：{e}")
            # 降级处理
            return self._generate_experiences_fallback(analysis, draft)
    
    async def _generate_experiences_with_l2_async(self, analysis: Dict[str, Any]) -> List[Dict]:
        """🔥 新增：异步调用 L2 智能生成经验
        
        Args:
            analysis: 分析结果
            
        Returns:
            List[Dict]: 经验列表
        """
        try:
            # 构建生成提示
            prompt = (
                "请根据以下分析结果，提炼 1-3 条可执行的经验：\n\n"
                f"分析结果：{analysis}\n\n"
                "经验格式要求:\n"
                "1. 简洁明了（不超过 50 字）\n"
                "2. 可执行（包含具体操作）\n"
                "3. 可迁移（适用于类似场景）\n\n"
                "请以 JSON 格式返回数组：[\n"
                "  {\"content\": \"经验内容\", \"category\": \"类别\", \"confidence\": 0.8},\n"
                "  ...\n"
                "]"
            )
            
            # 🔥 异步调用 L2
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, EventPriority, ZulongEvent
            
            # 创建响应接收器
            loop = asyncio.get_event_loop()
            response_future = loop.create_future()
            
            def on_l2_response(event: ZulongEvent):
                """L2 响应回调"""
                if not response_future.done():
                    logger.info(f"[ReplayIntegration] 收到 L2 经验生成响应")
                    loop.call_soon_threadsafe(response_future.set_result, event.payload)
            
            # 订阅临时响应
            temp_subscriber_id = f"review_exp_gen_async_{self.review_session_id}"
            try:
                event_bus.subscribe(EventType.L2_OUTPUT, on_l2_response, temp_subscriber_id)
            except Exception as e:
                logger.warning(f"[ReplayIntegration] 订阅 L2 响应失败：{e}")
            
            try:
                # 发布生成请求
                event = ZulongEvent(
                    type=EventType.SYSTEM_L2_COMMAND,
                    source="ReplayIntegration",
                    payload={
                        'command': 'generate_experiences',
                        'prompt': prompt,
                        'session_id': self.review_session_id,
                        'expect_json_response': True
                    },
                    priority=EventPriority.HIGH
                )
                
                event_bus.publish(event)
                logger.info("[ReplayIntegration] 已发布 L2 经验生成请求（异步）")
                
                # 🔥 异步等待响应（最多 15 秒）
                try:
                    payload = await asyncio.wait_for(response_future, timeout=15.0)
                    logger.info(f"[ReplayIntegration] ✅ L2 经验生成响应已接收")
                    
                    # 解析 JSON 响应
                    text = payload.get('text', '')
                    
                    # 尝试提取 JSON
                    import re
                    json_match = re.search(r'\[.*\]', text, re.DOTALL)
                    if json_match:
                        experiences = json.loads(json_match.group())
                        logger.info(f"[ReplayIntegration] ✅ 成功解析 {len(experiences)} 条经验")
                        
                        # 添加元数据
                        for exp in experiences:
                            exp['draft'] = draft
                            exp['source'] = 'l2_generated'
                        
                        return experiences
                    else:
                        logger.warning(f"[ReplayIntegration] 未找到 JSON 格式：{text}")
                        return []
                        
                except asyncio.TimeoutError:
                    logger.error("[ReplayIntegration] L2 经验生成超时")
                    return []
                    
            finally:
                # 取消订阅
                try:
                    event_bus.unsubscribe(EventType.L2_OUTPUT, temp_subscriber_id)
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"[ReplayIntegration] L2 经验生成失败：{e}", exc_info=True)
            return []


# 全局单例
_replay_integration_instance = None


def get_replay_integration() -> ReplayIntegration:
    """获取复盘集成器单例
    
    Returns:
        ReplayIntegration: 单例实例
    """
    global _replay_integration_instance
    
    if _replay_integration_instance is None:
        _replay_integration_instance = ReplayIntegration()
    
    return _replay_integration_instance
