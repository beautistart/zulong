# File: zulong/l1b/scheduler_gatekeeper.py
# L1-B 调度器与守门员 - 第五阶段总装
# 对应 TSD v1.7: 调度、守门与电源管理

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent, PowerState, L2Status
from zulong.core.state_manager import state_manager
from zulong.core.power_manager import power_manager
from zulong.l2.task_state_manager import task_state_manager
from zulong.review.state_manager import get_review_state_manager, ReviewMode, ReviewStage

from typing import Any, Dict, List, Optional

import time
import threading
import logging
import json

logger = logging.getLogger(__name__)


class Gatekeeper:
    """L1-B 调度器与守门员"""
    
    def __init__(self):
        """初始化守门员"""
        self._last_command_time = 0
        self._cooldown_time = 2.0
        
        # 空闲挂起配置：未完成任务超过此时间（秒）无新指令则自动挂起
        # IDE模式需要更长超时（前端执行命令可能耗时数分钟）
        self._idle_suspend_timeout = 900  # 15 分钟（从300秒延长）
        self._idle_check_timer = None
        self._idle_check_lock = threading.Lock()
        
        self._register_event_handlers()
        
        self._speech_collect_active = False
        self._speech_collect_start_time = 0.0
        self._speech_collect_window = 1.5
        self._collected_audio_frames = []
        
        # 自主注意力循环：焦点漂移防抖时间戳
        self._last_focus_drift_time = 0.0
        
        # 🔥 关键修复：使用 ReviewStateManager 统一管理状态（不再需要本地锁）
        self._review_state_manager = get_review_state_manager()

        # 🔥 初始化 L1-B 意图分类器 (ALBERT)
        self._intent_filter = None
        self._init_intent_filter()

        # 🔥 新增：初始化语音意图分类器（语义模型替代硬编码关键词）
        self._voice_intent_classifier = None
        self._init_voice_intent_classifier()

        # 🔥 模型自主路由：根据模型参数量自动启用/关闭 Hint
        self._hint_enabled = False
        try:
            self._init_hint_mode()
        except Exception as e:
            logger.warning(f"[Gatekeeper] Hint 模式初始化失败: {e}")

        logger.info("L1-B Gatekeeper initialized and subscribed to user events")

    def _register_event_handlers(self):
        """注册事件处理器 - 所有事件都经过 L1-B 路由到 L2"""
        event_bus.subscribe(EventType.USER_SPEECH, self.on_user_voice, "L1-B")
        event_bus.subscribe(EventType.USER_VOICE, self.on_user_voice, "L1-B")
        event_bus.subscribe(EventType.USER_TEXT, self.on_user_text, "L1-B")  # 新增：Web 文本输入
        event_bus.subscribe(EventType.USER_COMMAND, self.on_user_command, "L1-B")
        
        event_bus.subscribe(EventType.INTERACTION_TRIGGER, self.on_interaction_trigger, "L1-B")
        event_bus.subscribe(EventType.DIRECT_WAKEUP, self.on_direct_wakeup, "L1-B")
        event_bus.subscribe(EventType.SENSOR_MOTION, self.on_visual_attention, "L1-B")
        
        event_bus.subscribe(EventType.SENSOR_SOUND, self.on_audio_event, "L1-B")
        
        event_bus.subscribe(EventType.SENSOR_FALL, self.on_sensor_fall, "L1-B")
        event_bus.subscribe(EventType.SENSOR_OBSTACLE, self.on_sensor_obstacle, "L1-B")
        event_bus.subscribe(EventType.SENSOR_VISION, self.on_sensor_vision, "L1-B")
        
        event_bus.subscribe(EventType.VISION_DATA_READY, self.on_vision_data_ready, "L1-B")
        
        # 自主注意力循环：L2 每次回复后触发 BFS 扩散 + 赫布学习
        event_bus.subscribe(EventType.L2_OUTPUT, self._on_l2_output_attention, "AttentionLoop")
        
        logger.info("L1-B Gatekeeper: 已订阅所有事件类型，统一路由到 L2")

    def _retrieve_bfs_context(self, conversation_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """TSD 23.11.3: 由 L1-B 使用 session/conversation anchor 执行 BFS 自恢复。"""
        if not conversation_id:
            return None
        try:
            from zulong.memory.memory_graph import get_memory_graph

            mg = get_memory_graph()
            if mg is None:
                return None

            if hasattr(mg, "get_session_node_id_for_conversation"):
                session_node_id = mg.get_session_node_id_for_conversation(conversation_id)
            else:
                safe_id = "".join(
                    ch if ch.isalnum() or ch in "-_" else "_"
                    for ch in str(conversation_id or "")
                )
                session_node_id = f"dialogue:session_{safe_id}"

            if not session_node_id or not mg.has_node(session_node_id):
                return None

            activations = mg.compute_activations(
                seed_node_ids=[session_node_id],
                max_depth=3,
                decay=0.5,
                min_activation=0.01,
            )

            recent_rounds: List[Dict[str, Any]] = []
            active_tasks: List[Dict[str, Any]] = []
            task_graph_id = ""
            for node_id, activation in sorted(
                activations.items(), key=lambda item: item[1], reverse=True
            ):
                node = mg.get_node(node_id)
                if node is None:
                    continue
                meta = node.metadata or {}
                ntype = node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)
                label = node.label or ""
                if ntype == "dialogue" and meta.get("sub_type") == "round":
                    recent_rounds.append({
                        "node_id": node_id,
                        "label": label,
                        "user_text": meta.get("user_text", ""),
                        "bot_text": meta.get("bot_text", ""),
                        "activation": round(float(activation), 3),
                    })
                elif ntype == "task":
                    active_tasks.append({
                        "node_id": node_id,
                        "label": label,
                        "status": meta.get("status", ""),
                        "desc": meta.get("goal", ""),
                        "activation": round(float(activation), 3),
                    })
                    if meta.get("task_graph_id") and not task_graph_id:
                        task_graph_id = meta["task_graph_id"]

            session_node = mg.get_node(session_node_id)
            logger.info(
                "[Gatekeeper][BFSContext] session=%s, activated_nodes=%s, rounds=%s, tasks=%s",
                session_node_id,
                len(activations),
                len(recent_rounds),
                len(active_tasks),
            )
            return {
                "session_node_id": session_node_id,
                "session_label": session_node.label if session_node else "",
                "recent_rounds": recent_rounds[:5],
                "active_tasks": active_tasks[:3],
                "task_graph_id": task_graph_id,
            }
        except Exception as exc:
            logger.warning(f"[Gatekeeper][BFSContext] 检索失败: {exc}")
            return None

    def _format_bfs_context_sections(self, bfs_context: Optional[Dict[str, Any]]) -> List[str]:
        """将 BFS 自恢复结果压入 L2 可用的记忆上下文文本。"""
        if not bfs_context:
            return []
        sections: List[str] = []
        for round_ctx in bfs_context.get("recent_rounds", []) or []:
            if not isinstance(round_ctx, dict):
                continue
            user_text = str(round_ctx.get("user_text") or "").strip()
            bot_text = str(round_ctx.get("bot_text") or "").strip()
            if user_text or bot_text:
                sections.append(
                    "【会话BFS历史】"
                    + (f"用户：{user_text[:160]}" if user_text else "")
                    + (f"\n回答：{bot_text[:180]}" if bot_text else "")
                )
        for task_ctx in bfs_context.get("active_tasks", []) or []:
            if not isinstance(task_ctx, dict):
                continue
            label = str(task_ctx.get("label") or task_ctx.get("node_id") or "").strip()
            status = str(task_ctx.get("status") or "").strip()
            desc = str(task_ctx.get("desc") or "").strip()
            if label or desc:
                sections.append(
                    "【会话BFS任务】"
                    + (label[:120] if label else "")
                    + (f"（状态：{status}）" if status else "")
                    + (f"\n{desc[:180]}" if desc else "")
                )
        return sections

    def _extract_bfs_task_graph_id(self, bfs_context: Optional[Dict[str, Any]]) -> str:
        """从 BFS 自恢复上下文提取当前任务图兜底锚点。"""
        if not bfs_context:
            return ""
        graph_id = str(bfs_context.get("task_graph_id") or "").strip()
        if graph_id:
            return graph_id
        for task_ctx in bfs_context.get("active_tasks", []) or []:
            if not isinstance(task_ctx, dict):
                continue
            node_id = str(task_ctx.get("node_id") or "").strip()
            if node_id.startswith("task:"):
                return node_id.split("task:", 1)[1].strip()
        return ""

    def _compact_context_value(self, value: Any, limit: int = 1200) -> str:
        """压缩 L1-B 已采集上下文，避免任务包提示过长。"""
        if value is None:
            return ""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                text = str(value)
        text = " ".join(text.split())
        return text[:limit]

    def _build_l1b_context_pack(
        self,
        *,
        local_context: Any,
        shared_context_snapshot: Any,
        pre_retrieved_memory: Optional[str],
        bfs_context: Optional[Dict[str, Any]],
        tool_prediction: Optional[Dict[str, Any]],
        voice_mode: str,
    ) -> str:
        """统一构建交给 L2 消费的 L1-B 上下文包。"""
        lines: List[str] = ["【L1-B 打包上下文】"]
        local_text = self._compact_context_value(local_context, 800)
        if local_text:
            lines.append(f"- 局部上下文: {local_text}")
        shared_text = self._compact_context_value(shared_context_snapshot, 1000)
        if shared_text:
            lines.append(f"- 共享上下文快照: {shared_text}")
        if bfs_context:
            rounds = len(bfs_context.get("recent_rounds", []) or [])
            tasks = len(bfs_context.get("active_tasks", []) or [])
            lines.append(f"- 会话 BFS 自恢复: recent_rounds={rounds}, active_tasks={tasks}")
        if tool_prediction:
            context_bundle = tool_prediction.get("context_bundle", {}) or {}
            lines.append(
                "- 工具预判: "
                f"tools={tool_prediction.get('predicted_tools', [])}, "
                f"policy={tool_prediction.get('task_graph_policy', 'none')}, "
                f"context={context_bundle}"
            )
        lines.append(f"- 输出模态辅助: voice_mode={voice_mode}")
        if pre_retrieved_memory:
            lines.append("\n【L1-B 预检索记忆】\n" + pre_retrieved_memory)
        return "\n".join(lines)

    def _init_intent_filter(self):
        """初始化 L1-B 意图分类器 (ALBERT-tiny Chinese)"""
        try:
            from zulong.config.config_manager import get_config
            from zulong.l1b.intent_filter import IntentFilter

            intent_config = get_config("intent_classification", {})
            self._intent_filter = IntentFilter(config=intent_config)
            logger.info("[Gatekeeper] L1-B 意图分类器已初始化")
        except Exception as e:
            logger.warning(f"[Gatekeeper] L1-B 意图分类器初始化失败: {e}")
            self._intent_filter = None
    
    def _init_voice_intent_classifier(self):
        """初始化语音意图分类器（使用 ALBERT 语义模型）"""
        try:
            from zulong.config.config_manager import get_config
            from zulong.l1b.voice_intent_classifier import VoiceIntentClassifier
            
            config = get_config("voice_intent_classification", {})
            if not config:
                logger.info("[Gatekeeper] 未配置语音意图分类，使用事件类型检测")
                return
            
            if not config.get("enabled", False):
                logger.info("[Gatekeeper] 语音意图分类未启用，使用事件类型检测")
                return
            
            albert_config = config.get("albert", {})
            model_path = albert_config.get("model_path", "./models/albert-tiny-chinese")
            device = albert_config.get("device", "cpu")
            max_length = albert_config.get("max_length", 128)
            head_weights_path = albert_config.get("head_weights_path", "voice_intent_head.pt")
            
            self._voice_intent_classifier = VoiceIntentClassifier(
                model_path=model_path,
                device=device,
                max_length=max_length,
                head_weights_path=head_weights_path,
            )
            
            if self._voice_intent_classifier.load():
                self._voice_intent_classifier.warmup()
                logger.info(
                    "[Gatekeeper] 语音意图分类器已启用 "
                    "(3类: TEXT_ONLY/AUTO_TTS/FORCED_TTS)"
                )
            else:
                logger.warning(
                    "[Gatekeeper] 语音意图分类器加载失败，使用事件类型检测"
                )
                self._voice_intent_classifier = None
        
        except ImportError as e:
            logger.warning(f"[Gatekeeper] 缺少语音意图分类器模块: {e}")
        except Exception as e:
            logger.warning(f"[Gatekeeper] 语音意图分类器初始化异常: {e}")
    
    def _on_l2_output_attention(self, event):
        """L2 回复后自主注意力循环：BFS 扩散 + 赫布学习 + 焦点漂移
        
        L2_OUTPUT 由 EventBus 串行分发，因此这里只调度后台任务；
        实际 BFS/赫布不阻塞 Web Router 接收最终回复。
        """
        try:
            worker = threading.Thread(
                target=self._run_l2_output_attention,
                args=(event,),
                name="zulong-l1b-attention-loop",
                daemon=True,
            )
            worker.start()
        except Exception as e:
            logger.warning(f"[AttentionLoop] 后台调度异常: {e}")

    def _run_l2_output_attention(self, event):
        """后台执行 L2 回复后的注意力扩散。"""
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg is None:
                return

            # === A. 提取 BFS 种子（当前焦点路径） ===
            ctx = mg.get_last_focus_context()
            seeds = (ctx or {}).get("focus_path") or []
            if not seeds:
                logger.info("[AttentionLoop] 无焦点路径，跳过注意力循环")
                return

            # === B. BFS 扩散激活 ===
            activations = mg.compute_activations(
                seed_node_ids=seeds, max_depth=2, decay=0.5
            )

            # === C. 赫布学习：强化共激活边 ===
            mg.hebbian_strengthen()
            logger.info(
                f"[AttentionLoop] 扩散激活完成: "
                f"seeds={len(seeds)}, activated={len(activations)}"
            )

            # === D. 焦点自动漂移 ===
            self._maybe_drift_focus(mg, activations, seeds)

        except Exception as e:
            logger.warning(f"[AttentionLoop] 注意力循环异常: {e}")

    def _maybe_drift_focus(self, mg, activations, current_seeds):
        """基于激活值的焦点自动漂移（确定性规则，无 LLM）
        
        从 compute_activations 的结果中找到激活值最高的非焦点路径节点，
        如果激活值超过阈值且类型为 TASK/DIALOGUE，则自动漂移焦点。
        """
        import time as _time
        now = _time.time()

        # 防抖：两次自动漂移间隔至少 10 秒
        if now - self._last_focus_drift_time < 10.0:
            return

        # 排除当前焦点路径上的节点
        seeds_set = set(current_seeds)
        candidates = [
            (nid, score) for nid, score in activations.items()
            if nid not in seeds_set and score > 0.6
        ]
        if not candidates:
            return

        # 按激活值降序，取最高激活节点
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_nid, top_score = candidates[0]

        # 仅对 TASK/DIALOGUE 类型节点漂移（排除 KNOWLEDGE/CONCEPT 等辅助节点）
        node = mg.get_node(top_nid)
        if not node:
            return
        from zulong.memory.memory_graph import NodeType
        if node.node_type not in (NodeType.TASK, NodeType.DIALOGUE):
            return

        # 执行漂移
        if mg.update_focus_to_node(top_nid):
            self._last_focus_drift_time = now
            logger.info(
                f"[FocusDrift] 焦点自动漂移: → {node.label[:30]} "
                f"(activation={top_score:.2f}, type={node.node_type.value})"
            )
    
    def on_user_voice(self, event: ZulongEvent):
        """处理用户语音事件"""
        # 🔥 [关键调试] 添加详细日志
        logger.info(f"\n{'='*80}")
        logger.info(f"🔔 [L1-B] on_user_voice 被调用！")
        logger.info(f"🔔 [L1-B] 事件类型：{event.type}")
        logger.info(f"🔔 [L1-B] 事件优先级：{event.priority}")
        logger.info(f"🔔 [L1-B] 事件来源：{event.source}")
        # 排除音频数据（太大），仅记录元数据
        payload_for_log = {k: v for k, v in event.payload.items() if k != 'audio_data'}
        if 'audio_data' in event.payload:
            payload_for_log['audio_data_size'] = len(event.payload['audio_data'])
        logger.info(f"🔔 [L1-B] Payload: {payload_for_log}")
        logger.info(f"{'='*80}\n")
        
        text = event.payload.get("text", "").lower()
        confidence = event.payload.get("confidence", 0.0)
        
        logger.info(f"🔔 [L1-B] 提取文本：'{text}'")
        logger.info(f"🔔 [L1-B] 置信度：{confidence}")
        
        # 🔥 关键修复：将语音输入写入共享池（供复盘功能使用）
        if confidence >= 0.8:
            self._write_user_input_to_pool(text, event)
        
        # 检查置信度
        if confidence < 0.8:
            logger.debug(f"Voice command rejected (low confidence: {confidence})")
            return
        
        # 🔥 关键修复：全局复盘关键词检测 (无视 L2 状态) - 强制拦截器
        text_lower_check = text.lower()
        
        # 🔥 强制拦截：所有包含"复盘"的指令
        if '复盘' in text_lower_check:
            logger.info(f"[Gatekeeper] 🔥 检测到复盘相关指令：'{text}'")
            
            # 1. 启动复盘/开始复盘
            if '启动复盘' in text_lower_check or '开始复盘' in text_lower_check:
                logger.info(f"[Gatekeeper] 检测到启动复盘关键词，立即触发 ReviewTrigger")
                self._handle_review_keyword(text, event.priority)
                return
            
            # 2. 快速复盘/深度复盘 - 直接进入对应流程（跳过向导）
            elif '快速复盘' in text_lower_check or '深度复盘' in text_lower_check:
                logger.info(f"[Gatekeeper] 检测到模式选择指令，直接进入对应流程")
                try:
                    # 🔥 关键修复：用户已经明确选择了模式，直接进入对应流程，不显示向导
                    # 设置全局状态：进入复盘模式
                    state_manager.set_context('review_mode', True)
                    state_manager.set_context('review_session_id', f'review_{id(event)}')
                    logger.info(f"[Gatekeeper] ✅ 已设置 review_mode=True, session_id=review_{id(event)}")
                    
                    # 设置复盘类型
                    if '快速复盘' in text_lower_check:
                        state_manager.set_context('review_type', 'quick')
                        logger.info("[Gatekeeper] ✅ 已设置复盘类型：快速复盘")
                    else:
                        state_manager.set_context('review_type', 'deep')
                        logger.info("[Gatekeeper] ✅ 已设置复盘类型：深度复盘")
                    
                    # 🔥 关键修复：直接转发到 ReplayIntegration，不经过 ReviewTrigger
                    # 因为用户已经明确选择了模式，不需要再触发"选择模式"的流程
                    self._forward_to_replay_integration_immediate(text, event.priority)
                    return
                    
                except Exception as e:
                    logger.error(f"[Gatekeeper] 检查复盘模式状态失败：{e}")
            
            # 3. 结束复盘/退出复盘
            elif '结束复盘' in text_lower_check or '退出复盘' in text_lower_check:
                logger.info(f"[Gatekeeper] 检测到退出复盘指令")
                try:
                    review_mode = state_manager.get_context('review_mode', False)
                    if review_mode:
                        logger.info(f"[Gatekeeper] 复盘模式已激活，转发到 _handle_review_mode_input")
                        self._handle_review_mode_input(text, event.priority)
                        return
                    else:
                        logger.warning(f"[Gatekeeper] 复盘模式未激活，忽略退出指令")
                        # 发布提示：当前不在复盘模式
                        response_text = "💬 当前未处于复盘模式。请先说'启动复盘'开始复盘。"
                        event = ZulongEvent(
                            type=EventType.L2_OUTPUT,
                            source="Gatekeeper",
                            payload={
                                'text': response_text,
                                'session_id': None,
                                'review_mode': False
                            },
                            priority=EventPriority.NORMAL
                        )
                        event_bus.publish(event)
                        return
                except Exception as e:
                    logger.error(f"[Gatekeeper] 检查复盘模式状态失败：{e}")
            
            # 4. 其他复盘相关指令 (确认、取消等)
            else:
                logger.info(f"[Gatekeeper] 检测到其他复盘指令，检查复盘模式状态")
                try:
                    review_mode = state_manager.get_context('review_mode', False)
                    if review_mode:
                        logger.info(f"[Gatekeeper] 复盘模式已激活，转发到 _handle_review_mode_input")
                        self._handle_review_mode_input(text, event.priority)
                        return
                    else:
                        logger.warning(f"[Gatekeeper] 复盘模式未激活，忽略指令")
                        # 发布提示
                        response_text = "💬 当前未处于复盘模式。请先说'启动复盘'开始复盘。"
                        event = ZulongEvent(
                            type=EventType.L2_OUTPUT,
                            source="Gatekeeper",
                            payload={
                                'text': response_text,
                                'session_id': None,
                                'review_mode': False
                            },
                            priority=EventPriority.NORMAL
                        )
                        event_bus.publish(event)
                        return
                except Exception as e:
                    logger.error(f"[Gatekeeper] 检查复盘模式状态失败：{e}")
        
        # 🔥 关键修复：检查是否处于复盘模式，如果是，根据阶段路由
        try:
            review_mode = state_manager.get_context('review_mode', False)
            if review_mode:
                logger.info(f"[Gatekeeper] 复盘模式下收到用户输入：{text}")
                
                # 🔥 新增：获取当前复盘阶段，决定是否放行给 L2
                review_stage = state_manager.get_context('review_stage', 'mode_selecting')
                logger.info(f"[Gatekeeper] 当前复盘阶段：{review_stage}")
                
                if review_stage == 'review_active':
                    # 🔥 核心修复：对话进行阶段 - 检测特殊指令或放行给 L2
                    logger.info(f"[Gatekeeper] 对话进行阶段，检查输入：{text}")
                    
                    text_stripped = text.strip()
                    
                    # 🔥 检测结束复盘指令（最高优先级）
                    if text_stripped in ['结束复盘', '完成复盘', '退出复盘', '退出', '结束', '完成']:
                        logger.info(f"[Gatekeeper] 🔍 在对话阶段检测到结束指令：{text_stripped}")
                        
                        # 🔥 直接触发 ReplayIntegration 的 handle_end_review
                        try:
                            from zulong.review.integration import get_replay_integration
                            replay_integration = get_replay_integration()
                            
                            if replay_integration and hasattr(replay_integration, 'handle_end_review'):
                                import asyncio
                                
                                try:
                                    # 尝试在现有事件循环中创建任务
                                    loop = asyncio.get_running_loop()
                                    asyncio.create_task(replay_integration.handle_end_review())
                                    logger.info("[Gatekeeper] ✅ 已异步触发 handle_end_review()")
                                except RuntimeError:
                                    # 没有运行中的事件循环
                                    def run_end_review():
                                        new_loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(new_loop)
                                        try:
                                            new_loop.run_until_complete(
                                                replay_integration.handle_end_review()
                                            )
                                        finally:
                                            new_loop.close()
                                    
                                    thread = threading.Thread(target=run_end_review, daemon=True)
                                    thread.start()
                                    logger.info("[Gatekeeper] ✅ 已在新线程中触发 handle_end_review()")
                                    
                            else:
                                logger.warning("[Gatekeeper] ReplayIntegration 不可用或缺少 handle_end_review 方法")
                                # 降级：转发到 _handle_review_mode_input 处理
                                self._handle_review_mode_input(text, event.priority)
                                
                        except Exception as e:
                            logger.error(f"[Gatekeeper] 触发结束复盘失败：{e}", exc_info=True)
                        
                        return
                    
                    # 🔥 其他输入：记录到缓冲区并放行给 L2
                    logger.info(f"[Gatekeeper] ✅ 对话进行阶段，放行输入给 L2 对话")
                    
                    try:
                        from zulong.review.temp_buffer import get_review_buffer_manager
                        buffer_manager = get_review_buffer_manager()
                        
                        if buffer_manager.has_buffer():
                            buffer_data = {
                                'role': 'user',
                                'content': text,
                                'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
                                'source': 'review_conversation'
                            }
                            buffer_manager.add_user_input(buffer_data)
                            logger.debug(f"[Gatekeeper] 已记录对话到缓冲区")
                    except Exception as e:
                        logger.warning(f"[Gatekeeper] 记录到缓冲区失败：{e}")
                    
                    # 🔥 关键：不调用 _handle_review_mode_input()，直接放行！
                    # 让后续的 ReviewTriggerNode 和 L2 节点正常处理
                    # 不 return，继续执行后续逻辑
                    
                elif review_stage == 'experience_confirming':
                    # 🔥 [BUG-12 修复] 经验确认阶段：路由到正确的处理器
                    # 之前错误地路由到 _handle_review_mode_input，导致"确认"无法到达 ReplayIntegration
                    logger.info(f"[Gatekeeper] 经验确认阶段，转发到经验确认处理器")
                    self._handle_experience_confirm_stage(text, event.priority)
                    return
                    
                else:
                    # mode_selecting 阶段或其他：按原逻辑处理
                    logger.info(f"[Gatekeeper] {review_stage} 阶段，按原逻辑处理")
                    self._handle_review_mode_input(text, event.priority)
                    return
                    
        except Exception as e:
            logger.debug(f"[Gatekeeper] 检查复盘模式状态失败：{e}")
        
        # L1-A 条件反射指令（最高优先级）
        if "安静" in text or "睡觉" in text:
            self._handle_silent_mode()
            return
        elif "救命" in text:
            self._handle_wakeup(event.priority, event)
            return

        # 系统未激活时，任何输入都先唤醒
        current_power = state_manager.get_power_state()
        if current_power != PowerState.ACTIVE:
            self._handle_wakeup(event.priority, event)
            return

        # 所有其他消息统一路由（模型自主判断意图）
        self._handle_normal_command(text, event.priority, event.type)
    
    def on_user_text(self, event: ZulongEvent):
        """🔥 新增：三阶段状态机流量控制
        
        核心逻辑：
        1. 模式选择阶段 (MODE_SELECTING): 拦截一切，只听"快速"、"深度"、"退出"
        2. 对话进行阶段 (REVIEW_ACTIVE): L2 正常对话，后台监听"结束复盘"
        3. 经验确认阶段 (EXPERIENCE_CONFIRMING): 拦截一切，只听"确认"、"修改"、"退出"
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📝 [L1-B] on_user_text 被调用！")
        logger.info(f"📝 [L1-B] 事件类型：{event.type}")
        logger.info(f"📝 [L1-B] 事件来源：{event.source}")
        logger.info(f"📝 [L1-B] Payload: {event.payload}")
        logger.info(f"{'='*80}\n")
        
        text = event.payload.get("text", "")
        logger.info(f"📝 [L1-B] 提取文本：'{text}'")
        
        # 停止生成信号拦截
        if event.payload.get("action") == "stop_generation":
            logger.info("[Gatekeeper] 收到停止生成请求，设置中断标志")
            try:
                from zulong.l2.inference_engine import InferenceEngine
                engine = InferenceEngine()
                with engine._lock:
                    engine._interrupt_flag = True
                logger.info("[Gatekeeper] 中断标志已设置")
            except Exception as e:
                logger.error(f"[Gatekeeper] 设置中断标志失败: {e}")
            return
        
        # 🔥 关键修复：将用户输入写入共享池（供复盘功能使用）
        self._write_user_input_to_pool(text, event)
        
        # 🔥 关键修复：全局复盘关键词检测
        text_lower = text.lower()
        
        # 1️⃣ 检测是否处于复盘模式（优先检查全局状态）
        try:
            review_mode = state_manager.get_context('review_mode', False)
            if review_mode:
                logger.info(f"[Gatekeeper] 复盘模式下收到用户输入：{text}")
                
                # 尝试获取阶段信息
                review_stage = state_manager.get_context('review_stage', 'mode_selecting')
                logger.info(f"[Gatekeeper] 当前复盘阶段：{review_stage}")
                
                if review_stage == 'review_active':
                    # 🔥 核心修复：对话进行阶段 - 检测特殊指令或放行给 L2
                    logger.info(f"[Gatekeeper] 对话进行阶段，检查输入：{text}")
                    
                    text_stripped = text.strip()
                    
                    # 🔥 检测结束复盘指令（最高优先级）
                    if text_stripped in ['结束复盘', '完成复盘', '退出复盘', '退出', '结束', '完成']:
                        logger.info(f"[Gatekeeper] 🔍 在对话阶段检测到结束指令：{text_stripped}")
                        
                        # 🔥 直接触发 ReplayIntegration 的 handle_end_review
                        try:
                            from zulong.review.integration import get_replay_integration
                            replay_integration = get_replay_integration()
                            
                            if replay_integration and hasattr(replay_integration, 'handle_end_review'):
                                import asyncio
                                
                                try:
                                    # 尝试在现有事件循环中创建任务
                                    loop = asyncio.get_running_loop()
                                    asyncio.create_task(replay_integration.handle_end_review())
                                    logger.info("[Gatekeeper] ✅ 已异步触发 handle_end_review()")
                                except RuntimeError:
                                    # 没有运行中的事件循环
                                    def run_end_review():
                                        new_loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(new_loop)
                                        try:
                                            new_loop.run_until_complete(
                                                replay_integration.handle_end_review()
                                            )
                                        finally:
                                            new_loop.close()
                                    
                                    thread = threading.Thread(target=run_end_review, daemon=True)
                                    thread.start()
                                    logger.info("[Gatekeeper] ✅ 已在新线程中触发 handle_end_review()")
                                    
                            else:
                                logger.warning("[Gatekeeper] ReplayIntegration 不可用或缺少 handle_end_review 方法")
                                # 降级：转发到 _handle_review_mode_input 处理
                                self._handle_review_mode_input(text, event.priority)
                                
                        except Exception as e:
                            logger.error(f"[Gatekeeper] 触发结束复盘失败：{e}", exc_info=True)
                        
                        return
                    
                    # 🔥 其他输入：记录到缓冲区并放行给 L2
                    logger.info(f"[Gatekeeper] ✅ 对话进行阶段，放行输入给 L2 对话")
                    
                    try:
                        from zulong.review.temp_buffer import get_review_buffer_manager
                        buffer_manager = get_review_buffer_manager()
                        
                        if buffer_manager.has_buffer():
                            buffer_data = {
                                'role': 'user',
                                'content': text,
                                'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
                                'source': 'review_conversation'
                            }
                            buffer_manager.add_user_input(buffer_data)
                            logger.debug(f"[Gatekeeper] 已记录对话到缓冲区")
                    except Exception as e:
                        logger.warning(f"[Gatekeeper] 记录到缓冲区失败：{e}")
                    
                    # 🔥 关键：不调用 _handle_review_mode_input()，直接放行！
                    # 让后续的 ReviewTriggerNode 和 L2 节点正常处理
                    # 不 return，继续执行后续逻辑
                    
                elif review_stage == 'experience_confirming':
                    # 🔥 [BUG-12 修复] 经验确认阶段：路由到正确的处理器
                    # 之前错误地路由到 _handle_review_mode_input，导致"确认"无法到达 ReplayIntegration
                    logger.info(f"[Gatekeeper] 经验确认阶段，转发到经验确认处理器")
                    self._handle_experience_confirm_stage(text, event.priority)
                    return
                    
                else:
                    # mode_selecting 阶段或其他：按原逻辑处理
                    logger.info(f"[Gatekeeper] {review_stage} 阶段，按原逻辑处理")
                    self._handle_review_mode_input(text, event.priority)
                    return
                    
        except Exception as e:
            logger.debug(f"[Gatekeeper] 检查复盘模式状态失败：{e}")
        
        # 2️⃣ 检测是否处于复盘模式（使用 ReviewStateManager）
        if self._review_state_manager.is_active():
            logger.info(f"[Gatekeeper] 🔥 系统处于复盘模式")
            
            # 获取当前阶段
            current_stage = self._review_state_manager.get_stage()
            logger.info(f"[Gatekeeper] 📊 当前复盘阶段：{current_stage.value if current_stage else 'None'}")
            
            # 2️⃣ 模式选择阶段：拦截一切，只听模式选择
            if self._review_state_manager.is_mode_selecting():
                logger.info(f"[Gatekeeper] 🔒 阶段：模式选择 - 拦截所有输入")
                self._handle_mode_select_stage(text, event.priority)
                return
            
            # 3️⃣ 对话进行阶段：L2 正常对话，后台监听"结束复盘"
            elif self._review_state_manager.is_review_active():
                logger.info(f"[Gatekeeper] 💬 阶段：对话进行 - 正常对话 + 监听结束指令")
                self._handle_review_active_stage(text, event.priority)
                return
            
            # 4️⃣ 经验确认阶段：拦截一切，只听确认指令
            elif self._review_state_manager.is_experience_confirming():
                logger.info(f"[Gatekeeper] 🔒 阶段：经验确认 - 拦截所有输入")
                self._handle_experience_confirm_stage(text, event.priority)
                return
            
            # 5️⃣ 其他阶段：转发到复盘集成器
            else:
                logger.info(f"[Gatekeeper] 📊 其他复盘阶段，转发到 ReplayIntegration")
                self._forward_to_replay_integration(text, 'review_input')
                return
        
        # 3️⃣ 非复盘模式：正常流程
        logger.info(f"[Gatekeeper] ✅ 非复盘模式，执行正常流程")
        
        # 🔥 强制拦截：所有包含"复盘"的指令
        if '复盘' in text_lower:
            logger.info(f"[Gatekeeper] 🔥 检测到复盘相关指令：'{text}'")
            
            # 1. 启动复盘/开始复盘
            if '启动复盘' in text_lower or '开始复盘' in text_lower:
                logger.info(f"[Gatekeeper] 检测到启动复盘关键词，立即触发 ReviewTrigger")
                self._handle_review_keyword(text, event.priority)
                return
            
            # 2. 快速复盘/深度复盘 - 直接启动对应模式的复盘
            elif '快速复盘' in text_lower or '深度复盘' in text_lower:
                logger.info(f"[Gatekeeper] 🔍 检测到快速/深度复盘指令：'{text}'")
                # 🔥 修改：直接触发对应模式的复盘，不经过模式选择
                self._handle_direct_review_start(text, event.priority)
                return
            
            # 3. 结束复盘/退出复盘
            elif '结束复盘' in text_lower or '退出复盘' in text_lower:
                logger.info(f"[Gatekeeper] 检测到退出复盘指令，但当前不在复盘模式")
                # 发布提示：当前不在复盘模式
                self._publish_l2_response("⚠️ 当前不在复盘模式，无需退出")
                return
        
        # L1-A 条件反射
        if "安静" in text_lower or "睡觉" in text_lower:
            self._handle_silent_mode()
            return
        elif "救命" in text_lower:
            self._handle_wakeup(event.priority, event)
            return

        # 系统未激活时先唤醒
        current_power = state_manager.get_power_state()
        if current_power != PowerState.ACTIVE:
            self._handle_wakeup(event.priority, event)
            return

        # 所有其他消息统一路由
        session_id = event.payload.get("session_id")
        request_id = event.payload.get("request_id")
        logger.info(f"🏷️ [Gatekeeper] on_user_text 收到 session_id: {session_id}, request_id: {request_id}")
        worker = threading.Thread(
            target=self._handle_normal_command,
            args=(text, event.priority, EventType.USER_TEXT, session_id, request_id),
            kwargs={"source_payload": dict(event.payload or {})},
            name="zulong-l1b-user-text",
            daemon=True,
        )
        worker.start()
    
    def on_user_command(self, event: ZulongEvent):
        """处理用户命令事件"""
        command = event.payload.get("command", "").lower()
        text = event.payload.get("text", "")
        
        # 优先处理特殊命令
        if "silent" in command:
            self._handle_silent_mode()
            return
        elif "wake" in command:
            self._handle_wakeup(event.priority, event)
            return
        
        # 普通用户文本输入 (包含语音关键词或普通对话)
        if text:
            logger.info(f"📝 处理用户命令文本：'{text[:50]}...' " if len(text) > 50 else f"📝 处理用户命令文本：'{text}'")
            self._handle_normal_user_text(text, event.type)
    
    def _handle_silent_mode(self):
        """处理安静模式指令"""
        logger.info("Silent Mode Entered")
        state_manager.set_power_state(PowerState.SILENT)
        power_manager.unload_to_cpu()
    
    def _handle_wakeup(self, priority: EventPriority, event: ZulongEvent = None):
        """处理唤醒指令
        
        Args:
            priority: 事件优先级
            event: 原始事件（可选，用于在唤醒后继续处理）
        """
        logger.info("Emergency WakeUp!")
        state_manager.set_power_state(PowerState.ACTIVE)
        power_manager.load_to_gpu()
        
        # 如果有原始事件，等待 L2 加载完成后处理
        # 这里简化处理：直接发布事件到队列，让 L2 加载完成后处理
        if event:
            logger.info(f"WakeUp triggered by event, will process after L2 loads: {event.payload.get('text', '')}")
            # 发布到事件队列，L2 加载完成后可通过 SYSTEM_L2_READY 事件触发处理
            # 但这里我们简化：直接发布到 L2
            l2_event = ZulongEvent(
                type=EventType.SYSTEM_L2_COMMAND,
                priority=priority,
                source="Gatekeeper",
                payload={"text": event.payload.get("text", ""), "wakeup_command": True}
            )
            logger.info(f"Routing wakeup command to L2: {event.payload.get('text', '')}")
            event_bus.publish(l2_event)
    
    def _handle_review_keyword(self, text: str, priority: EventPriority):
        """处理复盘关键词，触发 ReviewTrigger - 🔥 修复：增加防重入锁
        
        Args:
            text: 用户输入文本
            priority: 事件优先级
        """
        logger.info(f"[Gatekeeper] _handle_review_keyword 被调用")
        
        # 🔥 关键修复：使用 ReviewStateManager 的检查锁机制
        if not self._review_state_manager.acquire_processing_lock():
            logger.debug("[Gatekeeper] 复盘正在处理中，忽略重复指令")
            return
        
        try:
            # 🔥 关键修复：检查是否已设置 review_mode，避免重复设置
            review_mode = state_manager.get_context('review_mode', False)
            if not review_mode:
                state_manager.set_context('review_mode', True)
                logger.info(f"[Gatekeeper] ✅ 已设置 review_mode=True")
            else:
                logger.info(f"[Gatekeeper] ✅ review_mode 已为 True，跳过设置")
            
            # 🔥 v3.0 修改：设置 L2 状态为 REVIEW_WAITING
            state_manager.set_l2_status(L2Status.REVIEW_WAITING, task_id=f"review_{id(text)}")
            logger.info(f"[Gatekeeper] ✅ L2 状态已设置为 REVIEW_WAITING")
            
            from zulong.review.trigger import get_review_trigger
            review_trigger = get_review_trigger()
            
            if review_trigger:
                # 🔥 关键修复：使用 run_coroutine_threadsafe 在无事件循环的线程中运行异步任务
                import asyncio
                
                # 获取或创建事件循环
                try:
                    loop = asyncio.get_running_loop()
                    # 如果有运行中的循环，直接创建任务
                    asyncio.create_task(
                        review_trigger.trigger_user_active(
                            context={
                                'trigger_keyword': '启动复盘',
                                'user_input': text,
                                'trigger_source': 'gatekeeper'
                            }
                        )
                    )
                    logger.info("[Gatekeeper] 已异步触发 ReviewTrigger (使用当前循环)")
                except RuntimeError:
                    # 没有运行中的循环，使用新线程启动循环
                    logger.info("[Gatekeeper] 检测到无运行循环，创建新事件循环")
                    
                    def run_async_task():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            new_loop.run_until_complete(
                                review_trigger.trigger_user_active(
                                    context={
                                        'trigger_keyword': '启动复盘',
                                        'user_input': text,
                                        'trigger_source': 'gatekeeper'
                                    }
                                )
                            )
                        finally:
                            new_loop.close()
                    
                    # 在后台线程运行
                    import threading
                    thread = threading.Thread(target=run_async_task)
                    thread.daemon = True
                    thread.start()
                    logger.info("[Gatekeeper] 已在新线程中启动 ReviewTrigger")
                
                # 🔥 v3.0 修改：简化回复，由 L2 主导后续流程
                response_text = (
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "🎯 **复盘向导已启动**\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "检测到您想进行复盘。请选择模式：\n\n"
                    "⚡ **快速复盘**\n"
                    "   • 基于关键词和短时记忆，生成摘要\n"
                    "   • 自动分析并应用经验\n\n"
                    "🔍 **深度复盘**\n"
                    "   • 调用长期记忆库，进行多维分析\n"
                    "   • 生成经验草案，需您确认\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "💬 请直接说 `快速复盘` 或 `深度复盘`\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
                
                event = ZulongEvent(
                    type=EventType.L2_OUTPUT,
                    source="Gatekeeper",
                    payload={
                        'text': response_text,
                        'session_id': None,
                        'review_mode': True,
                        'l2_status': 'REVIEW_WAITING'  # 🔥 v3.0 新增：标记状态
                    },
                    priority=EventPriority.HIGH
                )
                
                event_bus.publish(event)
                logger.info("[Gatekeeper] 已发布复盘向导响应事件")
            else:
                logger.warning("[Gatekeeper] ReviewTrigger 未初始化")
                # 🔥 v3.0 修改：即使 Trigger 未初始化，也重置状态
                state_manager.set_l2_status(L2Status.IDLE)
        except Exception as e:
            logger.error(f"[Gatekeeper] 触发 ReviewTrigger 失败：{e}", exc_info=True)
            # 🔥 v3.0 修改：失败时也重置状态
            state_manager.set_l2_status(L2Status.IDLE)
        finally:
            # 🔥 关键修复：确保释放处理锁
            try:
                self._review_state_manager.release_processing_lock()
                logger.info("[Gatekeeper] ✅ 已释放处理锁")
            except Exception as e:
                logger.error(f"[Gatekeeper] 释放处理锁失败：{e}")
    
    def _handle_review_mode_input(self, text: str, priority: EventPriority):
        """处理复盘模式下的用户输入
        
        Args:
            text: 用户输入文本
            priority: 事件优先级
        """
        logger.info(f"[Gatekeeper] _handle_review_mode_input 被调用：{text}")
        
        try:
            text_stripped = text.strip()
            text_lower = text_stripped.lower()
            
            # 🔥 1. 退出指令检测 (最高优先级)
            if text_stripped in ['退出复盘', '结束复盘', '退出', '结束', '退出复盘模式', '结束复盘模式']:
                logger.info("[Gatekeeper] 检测到退出复盘指令")
                
                # 检查当前阶段
                try:
                    review_stage = state_manager.get_context('review_stage', 'mode_selecting')
                    logger.info(f"[Gatekeeper] 当前复盘阶段：{review_stage}")
                    
                    if review_stage == 'review_active':
                        # 在对话进行阶段，应该触发经验提取而不是直接退出
                        logger.info("[Gatekeeper] 在对话进行阶段，触发经验提取")
                        
                        # 直接触发 ReplayIntegration 的 handle_end_review
                        try:
                            from zulong.review.integration import get_replay_integration
                            replay_integration = get_replay_integration()
                            
                            if replay_integration and hasattr(replay_integration, 'handle_end_review'):
                                import asyncio
                                
                                try:
                                    # 尝试在现有事件循环中创建任务
                                    loop = asyncio.get_running_loop()
                                    asyncio.create_task(replay_integration.handle_end_review())
                                    logger.info("[Gatekeeper] ✅ 已异步触发 handle_end_review()")
                                except RuntimeError:
                                    # 没有运行中的事件循环
                                    def run_end_review():
                                        new_loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(new_loop)
                                        try:
                                            new_loop.run_until_complete(
                                                replay_integration.handle_end_review()
                                            )
                                        finally:
                                            new_loop.close()
                                    
                                    import threading
                                    thread = threading.Thread(target=run_end_review, daemon=True)
                                    thread.start()
                                    logger.info("[Gatekeeper] ✅ 已在新线程中触发 handle_end_review()")
                                    
                            else:
                                logger.warning("[Gatekeeper] ReplayIntegration 不可用或缺少 handle_end_review 方法")
                                
                        except Exception as e:
                            logger.error(f"[Gatekeeper] 触发结束复盘失败：{e}", exc_info=True)
                        
                        return
                    else:
                        # 在其他阶段，直接退出
                        logger.info("[Gatekeeper] 在非对话阶段，直接退出复盘")
                        
                        # 清理全局状态
                        state_manager.set_context('review_mode', False)
                        state_manager.set_context('review_session_id', None)
                        
                        # 调用 ReplayIntegration 清理状态
                        from zulong.review.integration import get_replay_integration
                        replay_integration = get_replay_integration()
                        
                        if replay_integration:
                            replay_integration._exit_review_mode()
                            logger.info("[Gatekeeper] 已通知 ReplayIntegration 清理状态")
                        
                        # 发布退出响应
                        response_text = (
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            "✅ **已退出复盘模式**\n"
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                            "好的，已退出复盘模式。我们继续正常对话吧！😊\n\n"
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                        )
                        
                        event = ZulongEvent(
                            type=EventType.L2_OUTPUT,
                            source="Gatekeeper",
                            payload={
                                'text': response_text,
                                'session_id': None,
                                'review_mode': False
                            },
                            priority=EventPriority.HIGH
                        )
                        
                        event_bus.publish(event)
                        logger.info("[Gatekeeper] 已发布退出复盘响应")
                        
                        # 🔥 关键修复：使用 ReviewStateManager 释放锁
                        self._review_state_manager.release_processing_lock()
                        logger.info("[Gatekeeper] 🔓 已释放复盘处理锁")
                        return
                        
                except Exception as e:
                    logger.error(f"[Gatekeeper] 检查复盘阶段失败：{e}")
                    
                    # 降级：直接退出
                    # 清理全局状态
                    state_manager.set_context('review_mode', False)
                    state_manager.set_context('review_session_id', None)
                    
                    # 调用 ReplayIntegration 清理状态
                    from zulong.review.integration import get_replay_integration
                    replay_integration = get_replay_integration()
                    
                    if replay_integration:
                        replay_integration._exit_review_mode()
                        logger.info("[Gatekeeper] 已通知 ReplayIntegration 清理状态")
                    
                    # 发布退出响应
                    response_text = (
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "✅ **已退出复盘模式**\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        "好的，已退出复盘模式。我们继续正常对话吧！😊\n\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )
                    
                    event = ZulongEvent(
                        type=EventType.L2_OUTPUT,
                        source="Gatekeeper",
                        payload={
                            'text': response_text,
                            'session_id': None,
                            'review_mode': False
                        },
                        priority=EventPriority.HIGH
                    )
                    
                    event_bus.publish(event)
                    logger.info("[Gatekeeper] 已发布退出复盘响应")
                    
                    # 🔥 关键修复：使用 ReviewStateManager 释放锁
                    self._review_state_manager.release_processing_lock()
                    logger.info("[Gatekeeper] 🔓 已释放复盘处理锁")
                    return
            
            # 🔥 2. 模式选择指令检测
            if '快速' in text_lower or '快速复盘' in text_lower:
                logger.info("[Gatekeeper] 检测到快速复盘指令")
                # 🔥 关键修复：使用 ReviewStateManager 设置模式
                self._review_state_manager.enter_review_mode(ReviewMode.QUICK, state_manager.get_context('review_session_id', 'unknown'))
                # 转发到 ReplayIntegration 处理
                self._forward_to_replay_integration(text, 'quick_review')
                return
            
            if '深度' in text_lower or '深度复盘' in text_lower:
                logger.info("[Gatekeeper] 检测到深度复盘指令")
                # 🔥 关键修复：使用 ReviewStateManager 设置模式
                self._review_state_manager.enter_review_mode(ReviewMode.DEEP, state_manager.get_context('review_session_id', 'unknown'))
                # 转发到 ReplayIntegration 处理
                self._forward_to_replay_integration(text, 'deep_review')
                return
            
            # 🔥 3. 确认/取消指令检测 (复盘模式下)
            review_type = state_manager.get_context('review_type', None)
            
            if review_type == 'quick':
                # 快速复盘待确认状态
                if any(word in text_lower for word in ['确认', '好的', '开始', '执行', '确定', 'ok', 'yes']):
                    logger.info("[Gatekeeper] 检测到快速复盘确认指令")
                    # 🔥 关键修复：清理 review_type 状态，避免重复处理
                    state_manager.set_context('review_type', None)
                    logger.info(f"[Gatekeeper] ✅ 已清理 review_type 状态")
                    # 转发到 ReplayIntegration 执行真正的复盘
                    self._forward_to_replay_integration_immediate(text, EventPriority.HIGH)
                    return
                
                if any(word in text_lower for word in ['取消', '不要', '放弃', '不了', 'no', 'cancel']):
                    logger.info("[Gatekeeper] 检测到快速复盘取消指令")
                    # 清理状态
                    state_manager.set_context('review_type', None)
                    state_manager.set_l2_status(L2Status.IDLE)
                    
                    response_text = (
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "❌ **已取消快速复盘**\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        "好的，已取消快速复盘。我们继续正常对话吧！😊\n\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )
                    
                    event = ZulongEvent(
                        type=EventType.L2_OUTPUT,
                        source="Gatekeeper",
                        payload={
                            'text': response_text,
                            'session_id': None,
                            'review_mode': False
                        },
                        priority=EventPriority.HIGH
                    )
                    
                    event_bus.publish(event)
                    logger.info("[Gatekeeper] 已发布取消快速复盘响应")
                    
                    # 🔥 关键修复：使用 ReviewStateManager 释放锁
                    self._review_state_manager.release_processing_lock()
                    logger.info("[Gatekeeper] 🔓 已释放复盘处理锁")
                    return
            
            # 🔥 4. 其他输入 - 显示提示
            logger.info("[Gatekeeper] 复盘模式下收到其他输入，显示提示")
            
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
            
            event = ZulongEvent(
                type=EventType.L2_OUTPUT,
                source="Gatekeeper",
                payload={
                    'text': response_text,
                    'session_id': state_manager.get_context('review_session_id'),
                    'review_mode': True
                },
                priority=EventPriority.NORMAL
            )
            
            event_bus.publish(event)
            
        except Exception as e:
            logger.error(f"[Gatekeeper] 处理复盘模式输入失败：{e}", exc_info=True)
    
    def _forward_to_replay_integration_immediate(self, text: str, priority: EventPriority):
        """🔥 修复：异步转发用户输入到 ReplayIntegration
        
        Args:
            text: 用户输入文本
            priority: 事件优先级
        """
        try:
            from zulong.review.integration import get_replay_integration
            replay_integration = get_replay_integration()
            
            if replay_integration:
                # 🔥 关键修复：异步调用 ReplayIntegration，避免死锁
                review_type = state_manager.get_context('review_type', 'quick')
                
                logger.info(f"[Gatekeeper] 立即转发到 ReplayIntegration，类型：{review_type}")
                
                # 🔥 修复：统一异步调用
                import asyncio
                
                try:
                    loop = asyncio.get_running_loop()
                    # 在已有事件循环中，创建异步任务
                    logger.info(f"[Gatekeeper] 检测到运行中的事件循环：{loop}")
                    asyncio.create_task(
                        replay_integration._handle_user_active_review({
                            'user_input': text,
                            'trigger_source': 'gatekeeper_immediate',
                            'review_type': review_type
                        })
                    )
                    logger.info(f"[Gatekeeper] ✅ 已创建异步任务到 ReplayIntegration")
                    
                    # 🔥 关键修复：异步任务已创建，但不在这里释放锁
                    # 锁将在 ReplayIntegration 完成处理后由 _handle_review_mode_input 的退出分支释放
                except RuntimeError as e:
                    # 没有运行中的事件循环，创建新线程执行
                    logger.info(f"[Gatekeeper] 无事件循环 ({e})，创建新线程执行")
                    
                    def run_async_task():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            logger.info(f"[Gatekeeper][线程] 新事件循环已创建：{new_loop}")
                            logger.info(f"[Gatekeeper][线程] 开始执行异步任务...")
                            new_loop.run_until_complete(
                                replay_integration._handle_user_active_review({
                                    'user_input': text,
                                    'trigger_source': 'gatekeeper_immediate',
                                    'review_type': review_type
                                })
                            )
                            logger.info(f"[Gatekeeper][线程] ✅ 异步任务执行完成")
                        except Exception as inner_e:
                            logger.error(f"[Gatekeeper][线程] ❌ 异步任务执行失败：{inner_e}", exc_info=True)
                        finally:
                            logger.info(f"[Gatekeeper][线程] 关闭事件循环")
                            new_loop.close()
                    
                    import threading
                    thread = threading.Thread(target=run_async_task, daemon=True)
                    thread.start()
                    logger.info(f"[Gatekeeper] ✅ 已创建线程执行异步任务 (线程 ID: {thread.ident})")
                    # 🔥 关键修复：等待线程启动
                    import time
                    time.sleep(0.1)
                    logger.info(f"[Gatekeeper] 线程已启动，等待执行...")
            else:
                logger.warning("[Gatekeeper] ReplayIntegration 未初始化")
        except Exception as e:
            logger.error(f"[Gatekeeper] 立即转发到 ReplayIntegration 失败：{e}", exc_info=True)
    
    async def _handle_deep_review_async(self, replay_integration, user_input: str):
        """异步处理深度复盘输入
        
        Args:
            replay_integration: ReplayIntegration 实例
            user_input: 用户输入
        """
        try:
            replay_integration._handle_user_active_review({
                'user_input': user_input,
                'trigger_source': 'gatekeeper_immediate',
                'review_type': 'deep'
            })
            logger.info(f"[Gatekeeper] 深度复盘处理完成：{user_input}")
        except Exception as e:
            logger.error(f"[Gatekeeper] 异步处理深度复盘失败：{e}", exc_info=True)
    
    def _forward_to_replay_integration(self, text: str, command_type: str):
        """转发用户输入到 ReplayIntegration（保留向后兼容）
        
        Args:
            text: 用户输入文本
            command_type: 命令类型 ('quick_review', 'deep_review', 'confirmation')
        """
        # 调用新方法
        from zulong.core.types import EventPriority
        self._forward_to_replay_integration_immediate(text, EventPriority.NORMAL)
    
    async def _handle_review_input_async(self, replay_integration, user_input: str):
        """异步处理复盘输入
        
        Args:
            replay_integration: ReplayIntegration 实例
            user_input: 用户输入
        """
        try:
            # 调用 ReplayIntegration 的处理方法
            if hasattr(replay_integration, 'handle_user_confirmation'):
                replay_integration.handle_user_confirmation(user_input)
            else:
                # 降级处理：直接调用处理用户主动复盘
                replay_integration._handle_user_active_review({
                    'user_input': user_input,
                    'trigger_source': 'gatekeeper_input'
                })
            logger.info(f"[Gatekeeper] 用户复盘输入处理完成：{user_input}")
        except Exception as e:
            logger.error(f"[Gatekeeper] 异步处理复盘输入失败：{e}", exc_info=True)
    
    def on_visual_attention(self, event: ZulongEvent):
        """
        处理 L1-C 视觉注意力事件 (TSD v1.8)
        
        当 L1-C 检测到交互意图（挥手、注视、靠近）时触发。
        L1-B 负责收集上下文并路由到 L2。
        
        TSD v1.8 对应:
        - 4.2.1 L1-B 注意力控制器
        - L1-B (中断/重组): 接收事件，若 L2 繁忙则打断旧任务，
          收集上下文，将关键帧序列 + 文本引导组装成多模态 Prompt。
        
        Args:
            event: ZulongEvent (包含 attention_event 信息)
        """
        payload = event.payload
        intent_type = payload.get("intent_type", "unknown")
        intent_confidence = payload.get("intent_confidence", 0.0)
        person_distance = payload.get("person_distance", float('inf'))
        
        logger.info(
            f"👁️ [Gatekeeper] 收到视觉注意力事件："
            f"intent={intent_type}, confidence={intent_confidence:.2f}, distance={person_distance:.2f}m"
        )
        
        # 1. 获取 L2 状态
        effective_state = state_manager.get_effective_status()
        l2_status = state_manager.get_l2_status()
        
        # 2. 构建 Prompt
        prompt = self._build_visual_attention_prompt(payload)
        
        # 3. 根据状态决定处理方式
        if effective_state == "ACTIVE_TASK":
            if l2_status == L2Status.WAITING:
                # WAITING 状态：可以直接插入新指令
                logger.info("👁️ [Gatekeeper] L2 在 WAITING 状态，插入视觉注意力任务")
                self._handle_visual_attention_interrupt(prompt, payload)
            else:
                # BUSY 状态：排队或抢占
                logger.info("👁️ [Gatekeeper] L2 在 BUSY 状态，排队视觉注意力任务")
                self._queue_visual_attention_task(prompt, payload)
        else:
            # IDLE 状态：直接路由
            logger.info("👁️ [Gatekeeper] L2 空闲，直接路由视觉注意力任务")
            self._route_visual_attention_to_l2(prompt, payload)
    
    def on_interaction_trigger(self, event: ZulongEvent):
        """
        处理 INTERACTION_TRIGGER 事件 (动态路由架构)
        
        L2 忙碌时触发，走 L1-B 标准中断流程：
        1. 冻结当前任务快照
        2. 重组多模态 Prompt
        3. 强制打断 L2
        4. 注入新任务
        
        Args:
            event: ZulongEvent
        """
        payload = event.payload
        intent_type = payload.get("intent_type", "unknown")
        route_mode = payload.get("route_mode", "interrupt")
        
        logger.info(
            f"🔄 [Gatekeeper] 收到 INTERACTION_TRIGGER 事件："
            f"intent={intent_type}, route_mode={route_mode}"
        )
        
        # 构建中断模式的 Prompt（包含历史上下文）
        prompt = self._build_interrupt_mode_prompt(payload)
        
        # 执行标准中断流程
        self._handle_visual_attention_interrupt(prompt, payload)
    
    def on_direct_wakeup(self, event: ZulongEvent):
        """
        处理 DIRECT_WAKEUP 事件 (动态路由架构)
        
        L2 空闲时直连，L1-B 仅做简单透传：
        - 不执行冻结/快照操作
        - 构建最简 Prompt（无需历史上下文）
        - 直接路由到 L2
        
        Args:
            event: ZulongEvent
        """
        payload = event.payload
        intent_type = payload.get("intent_type", "unknown")
        route_mode = payload.get("route_mode", "direct")
        
        logger.info(
            f"⚡ [Gatekeeper] 收到 DIRECT_WAKEUP 事件："
            f"intent={intent_type}, route_mode={route_mode}"
        )
        
        # 构建简单 Prompt（空闲模式，无需历史上下文）
        prompt = self._build_simple_wakeup_prompt(payload)
        
        # 直接路由到 L2
        self._route_visual_attention_to_l2(prompt, payload)
    
    def _build_interrupt_mode_prompt(self, payload: dict) -> str:
        """
        构建中断模式 Prompt (包含历史上下文)
        
        Args:
            payload: 事件负载
            
        Returns:
            str: 构建的 Prompt
        """
        intent_type = payload.get("intent_type", "unknown")
        person_distance = payload.get("person_distance", float('inf'))
        
        # 根据意图类型构建引导文本
        intent_prompts = {
            "WAVING": "用户正在向你挥手，可能想要引起你的注意或打招呼。",
            "GAZING": "用户正在注视着你，可能想要与你交流。",
            "APPROACHING": f"用户正在靠近你（距离约 {person_distance:.1f} 米），可能想要与你互动。"
        }
        
        intent_desc = intent_prompts.get(intent_type, f"检测到用户的交互意图：{intent_type}")
        
        # 中断模式：包含"暂停当前任务"提示
        prompt = (
            f"[视觉注意力事件 - 中断模式]\n"
            f"当前任务已暂停。\n"
            f"{intent_desc}\n"
            f"置信度：{payload.get('intent_confidence', 0):.0%}\n"
            f"距离：{person_distance:.2f} 米\n"
            f"请简短地回应用户的交互意图，之后可以询问是否继续之前的任务。"
        )
        
        return prompt
    
    def _build_simple_wakeup_prompt(self, payload: dict) -> str:
        """
        构建简单唤醒 Prompt (空闲模式，无需历史上下文)
        
        Args:
            payload: 事件负载
            
        Returns:
            str: 构建的 Prompt
        """
        intent_type = payload.get("intent_type", "unknown")
        person_distance = payload.get("person_distance", float('inf'))
        
        # 根据意图类型构建引导文本
        intent_prompts = {
            "WAVING": "用户正在向你挥手，可能想要引起你的注意或打招呼。",
            "GAZING": "用户正在注视着你，可能想要与你交流。",
            "APPROACHING": f"用户正在靠近你（距离约 {person_distance:.1f} 米），可能想要与你互动。"
        }
        
        intent_desc = intent_prompts.get(intent_type, f"检测到用户的交互意图：{intent_type}")
        
        # 简单模式：快速响应，不加载历史
        prompt = (
            f"[视觉注意力事件]\n"
            f"{intent_desc}\n"
            f"置信度：{payload.get('intent_confidence', 0):.0%}\n"
            f"距离：{person_distance:.2f} 米\n"
            f"请简短、自然地回应用户的交互意图。"
        )
        
        return prompt
    
    def _build_visual_attention_prompt(self, payload: dict) -> str:
        """
        构建视觉注意力 Prompt (TSD v1.8)
        
        Args:
            payload: 事件负载
            
        Returns:
            str: 构建的 Prompt
        """
        intent_type = payload.get("intent_type", "unknown")
        person_distance = payload.get("person_distance", float('inf'))
        
        # 根据意图类型构建不同的引导文本
        intent_prompts = {
            "WAVING": "用户正在向你挥手，可能想要引起你的注意或打招呼。",
            "GAZING": "用户正在注视着你，可能想要与你交流。",
            "APPROACHING": f"用户正在靠近你（距离约 {person_distance:.1f} 米），可能想要与你互动。"
        }
        
        intent_desc = intent_prompts.get(intent_type, f"检测到用户的交互意图：{intent_type}")
        
        prompt = (
            f"[视觉注意力事件]\n"
            f"{intent_desc}\n"
            f"置信度：{payload.get('intent_confidence', 0):.0%}\n"
            f"距离：{person_distance:.2f} 米\n"
            f"请简短地回应用户的交互意图。"
        )
        
        return prompt
    
    def _handle_visual_attention_interrupt(self, prompt: str, payload: dict):
        """处理视觉注意力中断（WAITING 状态）"""
        # 构建事件发送到 L2
        l2_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=EventPriority.HIGH,
            source="Gatekeeper_VisualAttention",
            payload={
                "text": prompt,
                "visual_attention": True,
                "intent_type": payload.get("intent_type"),
                "intent_confidence": payload.get("intent_confidence"),
                "person_distance": payload.get("person_distance"),
                "keyframe_b64": payload.get("keyframe_b64"),  # 关键帧截图 (Base64)
                "crop_b64": payload.get("crop_b64"),  # 裁剪人物区域 (Base64)
                "cascade_stats": payload.get("cascade_stats")
            }
        )
        event_bus.publish(l2_event)
        logger.info(
            f"👁️ [Gatekeeper] 已发送视觉注意力事件到 L2（中断模式）"
            f"(keyframe={payload.get('keyframe_b64') is not None}, crop={payload.get('crop_b64') is not None})"
        )
    
    def _queue_visual_attention_task(self, prompt: str, payload: dict):
        """排队视觉注意力任务（BUSY 状态）"""
        # 简化处理：直接发送，让 L2 的队列机制处理
        l2_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=EventPriority.NORMAL,
            source="Gatekeeper_VisualAttention",
            payload={
                "text": prompt,
                "visual_attention": True,
                "queued": True,
                "intent_type": payload.get("intent_type")
            }
        )
        event_bus.publish(l2_event)
        logger.info("👁️ [Gatekeeper] 已排队视觉注意力事件")
    
    def _route_visual_attention_to_l2(self, prompt: str, payload: dict):
        """直接路由视觉注意力到 L2（IDLE 状态）"""
        l2_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=EventPriority.HIGH,
            source="Gatekeeper_VisualAttention",
            payload={
                "text": prompt,
                "visual_attention": True,
                "intent_type": payload.get("intent_type"),
                "intent_confidence": payload.get("intent_confidence"),
                "person_distance": payload.get("person_distance"),
                "keyframe_b64": payload.get("keyframe_b64"),  # 关键帧截图 (Base64)
                "crop_b64": payload.get("crop_b64"),  # 裁剪人物区域 (Base64)
                "cascade_stats": payload.get("cascade_stats")
            }
        )
        event_bus.publish(l2_event)
        logger.info(
            f"👁️ [Gatekeeper] 已路由视觉注意力事件到 L2（直接模式）"
            f"(keyframe={payload.get('keyframe_b64') is not None}, crop={payload.get('crop_b64') is not None})"
        )
    
    def _handle_normal_user_text(self, text: str, event_type: str):
        """处理普通用户文本输入
        
        Args:
            text: 用户输入文本
            event_type: 事件类型
        """
        logger.info(f"📝 [Gatekeeper] 处理用户文本：'{text[:50]}...' " if len(text) > 50 else f"📝 [Gatekeeper] 处理用户文本：'{text}'")
        
        # 1. L1-B 局部理解：整理新任务的上下文
        local_context = self._build_local_context(text)
        logger.info(f"L1-B built local context: {local_context}")
        
        # 2. L1-B 搜索共享上下文
        shared_context_snapshot = self._search_shared_context(text)
        logger.info(f"L1-B retrieved shared context: {shared_context_snapshot}")
        
        # 🎯 3. 检测语音模式 (TSD v1.7 规范)
        voice_mode = self._detect_voice_mode(text, event_type or EventType.USER_COMMAND)
        logger.info(f"🎙️ Voice Mode Detected: {voice_mode}")
        
        # 4. 打包新任务：用户原文 + 局部上下文 + 共享上下文快照 + 语音模式
        packaged_task = {
            "text": text,
            "local_context": local_context,
            "shared_context_snapshot": shared_context_snapshot,
            "voice_mode": voice_mode,
        }
        
        # 🔥 新增：添加 session_id（如果有）
        if session_id:
            packaged_task["session_id"] = session_id
            logger.info(f"🏷️ [Gatekeeper] Session ID 已添加到任务包：{session_id}")
        
        logger.info(f"L1-B packaged normal task with shared context: {packaged_task}")
        
        # 5. 注入 L2
        l2_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=EventPriority.NORMAL,
            source="Gatekeeper",
            payload=packaged_task
        )
        
        logger.info(f"Publishing event to L2...")
        event_bus.publish(l2_event)
        logger.info("Event published successfully!")
    
    def _detect_voice_mode(self, text: str, event_type: str) -> str:
        """
        检测语音模式（使用语义分类器）
        
        Args:
            text: 用户输入文本
            event_type: 事件类型 (USER_SPEECH / USER_TEXT / USER_COMMAND)
            
        Returns:
            str: 语音模式 ("TEXT_ONLY", "AUTO_TTS", "FORCED_TTS")
        """
        from zulong.config.output_routing_config import OutputMode
        
        # 1. 优先使用语音意图分类器（语义模型）
        if self._voice_intent_classifier and self._voice_intent_classifier.is_available():
            try:
                predicted_mode, confidence, _ = self._voice_intent_classifier.predict(text)
                logger.info(
                    f"🎤 [语音意图分类] {predicted_mode} "
                    f"(置信度: {confidence:.3f}, 模型: albert)"
                )
                
                # 置信度足够，直接返回
                if confidence >= 0.6:
                    return predicted_mode
                else:
                    logger.debug(
                        f"[语音意图分类] 置信度不足 ({confidence:.3f})，使用事件类型检测"
                    )
            except Exception as e:
                logger.warning(f"[语音意图分类] 推理失败: {e}")
        
        # 2. Fallback：基于事件类型检测
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        if event_type_str in ["USER_SPEECH", "USER_VOICE"]:
            logger.info(f"🎙️ Voice Mode: AUTO_TTS (语音输入事件)")
            return "AUTO_TTS"
        else:
            logger.info(f"🎙️ Voice Mode: TEXT_ONLY (input_type={event_type_str})")
            return "TEXT_ONLY"
    
    def _build_local_context(self, text: str) -> dict:
        """L1-B 局部上下文（基础元数据，不做意图识别）"""
        return {
            "event_type": "USER_INPUT",
            "timestamp": time.time(),
        }
    
    def _search_shared_context(self, text: str) -> dict:
        """L1-B 从过去一段时间的视听信息流中搜索上下文
        
        🔥 关键修复：按需加载视觉上下文（解决上下文污染问题）
        
        关键点：
        - 用户说第二句话时，可能已经不在原场景
        - 需要回溯过去 N 秒的视听记录
        - 视觉信息是用户指着桌子时捕捉的快照
        - 音频信息包含之前的对话历史
        - **也要关注当前的上下文状态**（用于对比变化）
        - 🔥 **只在需要时加载视觉上下文**（避免上下文污染）
        
        Args:
            text: 用户输入文本
            
        Returns:
            dict: 相关上下文快照（包含历史 + 当前）
        """
        # 🔥 关键修复 1: 检测是否需要视觉上下文
        # ✅ 修复：移除"什么"、"谁"等通用疑问词，只保留空间疑问词
        visual_keywords = [
            # 视觉动作
            "看", "看见", "观察", "查看", "注视", "瞅",
            # 视觉属性
            "颜色", "形状", "大小", "外观", "样子", "模样",
            # 视觉对象
            "图片", "图像", "照片", "视频", "画面", "屏幕",
            # 空间位置（需要视觉）
            "哪里", "哪儿", "什么地方", "哪个位置",
        ]
        
        need_vision = any(kw in text for kw in visual_keywords)
        logger.info(f"🔍 [L1-B] 视觉上下文检测：need_vision={need_vision}, text='{text[:50]}...'")
        
        # 🔥 关键修复 2: 只在需要时加载视觉上下文
        if not need_vision:
            logger.info("🔍 [L1-B] 不需要视觉上下文，返回空上下文（纯文本对话）")
            return {
                "scene": None,
                "objects": [],
                "audio_history": [],
                "current": None,
                "has_user_moved": False,
                "time_window": "last_30_seconds",
                "vision_skipped": True  # 标记：跳过了视觉上下文加载
            }
        
        # 从视听信息流管理器获取最近 N 秒的记录
        # 这里简化实现，使用模拟数据
        # 实际应该调用：audiovisual_stream_manager.search(time_window=30)
        
        # 模拟过去 30 秒的视听信息流（滚动缓冲区）
        audiovisual_stream = [
            {
                "timestamp": time.time() - 5,  # 5 秒前
                "modality": "VISION",
                "content": {
                    "scene": "桌子",
                    "objects": [
                        {"name": "苹果", "color": "红色", "location": "桌子", "status": "stable"},
                        {"name": "梨", "color": "黄色", "location": "桌子", "status": "stable"}
                    ],
                    "user_location": "桌子前"
                }
            },
            {
                "timestamp": time.time() - 5,
                "modality": "AUDIO",
                "content": {
                    "text": "把苹果拿给我",
                    "speaker": "user"
                }
            },
            {
                "timestamp": time.time() - 3,  # 3 秒前
                "modality": "VISION",
                "content": {
                    "scene": "走廊",  # 用户转身走开了
                    "objects": [],
                    "user_location": "走廊"
                }
            }
        ]
        
        logger.info(f"L1-B searching audiovisual stream (last 30 seconds, {len(audiovisual_stream)} records)")
        
        # 根据用户输入中的关键词，回溯搜索相关记录
        if "黄色" in text:
            logger.info("L1-B searching for keyword '黄色' in audiovisual stream")
        
        if "苹果" in text:
            logger.info("L1-B searching for keyword '苹果' in audiovisual stream")
        
        # 提取相关上下文（从历史视听记录中）
        historical_context = self._extract_relevant_context(audiovisual_stream, text)
        
        # 获取当前上下文（用于后续对比）
        current_context = self._get_current_context()
        
        # 合并历史和当前上下文
        full_context = {
            **historical_context,
            "current": current_context,  # 当前状态快照
            "has_user_moved": current_context.get("user_location") != historical_context.get("user_location"),
            "time_window": "last_30_seconds"
        }
        
        logger.info(f"L1-B retrieved context from audiovisual stream (historical + current)")
        return full_context
    
    def _get_current_context(self) -> dict:
        """获取当前上下文状态（实时感知）
        
        Returns:
            dict: 当前上下文快照
        """
        # 简化实现，模拟当前状态
        # 实际应该从实时视觉系统获取
        current_context = {
            "scene": "走廊",  # 当前场景
            "objects": [
                {"name": "苹果", "color": "红色", "location": "桌子", "status": "stable"},
                {"name": "梨", "color": "黄色", "location": "地上", "status": "fallen"}  # 梨掉地上了
            ],
            "user_location": "走廊",  # 用户已走开
            "timestamp": time.time()
        }
        
        logger.info(f"L1-B retrieved current context: {current_context}")
        return current_context
    
    def _extract_relevant_context(self, stream: list, text: str) -> dict:
        """从视听信息流中提取相关上下文
        
        Args:
            stream: 视听信息流（滚动缓冲区）
            text: 用户输入文本
            
        Returns:
            dict: 相关上下文快照
        """
        # 简化实现：遍历流，找到包含关键词的记录
        relevant_objects = []
        last_scene = None
        audio_history = []
        
        for record in stream:
            if record["modality"] == "VISION":
                content = record["content"]
                last_scene = content.get("scene", "unknown")
                
                # 检查物体是否匹配关键词
                for obj in content.get("objects", []):
                    # 简单匹配：颜色或名称
                    if any(keyword in obj.get("color", "") or keyword in obj.get("name", "") 
                           for keyword in ["黄色", "红色", "苹果", "梨"]):
                        if obj not in relevant_objects:
                            relevant_objects.append(obj)
            
            elif record["modality"] == "AUDIO":
                audio_text = record["content"].get("text", "")
                audio_history.append(audio_text)
        
        # 构建上下文快照
        context_snapshot = {
            "scene": last_scene or "unknown",
            "objects": relevant_objects,
            "audio_history": audio_history,
            "time_window": "last_30_seconds",
            "timestamp": time.time()
        }
        
        return context_snapshot
    
    # ==================== 空闲挂起 ====================

    def touch_idle_timer(self):
        """刷新空闲挂起计时器（FC循环工具执行/进度汇报时调用，防止误判空闲）"""
        with self._idle_check_lock:
            self._last_command_time = time.time()
            self.start_idle_suspend_timer()

    def start_idle_suspend_timer(self):
        """启动空闲挂起定时器
        
        当有活跃任务但用户长时间无新指令时，自动挂起任务。
        应在每次用户指令处理完成后调用。
        """
        import threading
        
        # 取消已有的定时器
        if self._idle_check_timer is not None:
            self._idle_check_timer.cancel()
            self._idle_check_timer = None
        
        # 只在有活跃任务时启动
        active_task = task_state_manager.get_active_task()
        if not active_task:
            return
        
        def _check_and_suspend():
            """定时器回调：检查是否需要自动挂起"""
            elapsed = time.time() - self._last_command_time
            if elapsed >= self._idle_suspend_timeout:
                active = task_state_manager.get_active_task()
                if active:
                    # 🔥 修复：检查 FC 循环是否仍在运行，已完成/停止的任务不应被挂起
                    try:
                        from zulong.core.state_manager import state_manager
                        if not state_manager.is_fc_loop_running():
                            logger.info(
                                f"[ZULONG] 任务 {active} FC 循环已停止，"
                                f"跳过空闲超时挂起（任务可能已完成）")
                            return
                    except Exception:
                        pass

                    # 🔥 修复：检查任务归档文件是否存在（双重保险）
                    try:
                        import os as _os
                        _archive_file = _os.path.join(
                            "./data/completed_tasks", f"{active}.json")
                        if _os.path.exists(_archive_file):
                            logger.info(
                                f"[ZULONG] 任务 {active} 已归档完成，跳过空闲超时挂起")
                            return
                    except Exception:
                        pass

                    logger.info(
                        f"[ZULONG] 任务 {active} 空闲超时 ({elapsed:.0f}s >= {self._idle_suspend_timeout}s)，自动挂起。"
                    )
                    task_state_manager.freeze_current()
                    
                    idle_event = ZulongEvent(
                        type=EventType.SYSTEM_INTERRUPT,
                        priority=EventPriority.LOW,
                        source="Gatekeeper",
                        payload={
                            "task_id": active,
                            "reason": "idle_timeout",
                            "idle_seconds": elapsed,
                        }
                    )
                    event_bus.publish(idle_event)
                    logger.info(f"[ZULONG] 任务 {active} 已因空闲超时自动挂起。说「继续」可恢复。")
        
        self._idle_check_timer = threading.Timer(self._idle_suspend_timeout, _check_and_suspend)
        self._idle_check_timer.daemon = True  # 守护线程，主程序退出时自动终止
        self._idle_check_timer.start()
    
    def _handle_normal_command(
        self,
        text: str,
        priority: EventPriority,
        event_type: str = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source_payload: Optional[Dict[str, Any]] = None,
    ):
        """处理普通命令
        
        Args:
            text: 命令文本
            priority: 事件优先级
            event_type: 事件类型 (用于语音模式检测)
            session_id: 会话 ID（来自 Web 测试 API）
            request_id: 请求 ID（用于思考过程和流式输出关联）
            source_payload: 原始 USER_TEXT 载荷（包含 conversation/workspace/task anchors）
        """
        source_payload = dict(source_payload or {})
        conversation_id = (
            source_payload.get("conversation_id")
            or source_payload.get("session_id")
            or session_id
            or ""
        )
        # 🔥 关键修复：移除复盘模式处理逻辑，统一由 _handle_review_mode_input 处理
        # review_mode = state_manager.get_context('review_mode', False)
        # if review_mode:
        #     # 移除整个 if 块，避免逻辑重复
        
        # 🔥 新增：正常模式下，检测"启动复盘"关键词
        text_lower = text.strip().lower()
        if '启动复盘' in text_lower or '开始复盘' in text_lower:
            logger.info(f"[Gatekeeper] 检测到复盘关键词：'{text}'，触发 ReviewTrigger")
            
            # 调用 ReviewTrigger 触发复盘
            try:
                from zulong.review.trigger import get_review_trigger
                review_trigger = get_review_trigger()
                
                if review_trigger:
                    # 异步触发复盘
                    import asyncio
                    asyncio.create_task(
                        review_trigger.trigger_user_active(
                            context={
                                'trigger_keyword': '启动复盘',
                                'user_input': text,
                                'trigger_source': 'gatekeeper'
                            }
                        )
                    )
                    logger.info("[Gatekeeper] 已异步触发 ReviewTrigger")
                    
                    # 立即回复用户，进入复盘模式
                    response_text = (
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "🎯 **复盘向导已启动**\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        "检测到您想进行复盘。请选择模式：\n\n"
                        "⚡ **快速复盘**\n"
                        "   • 基于关键词和短时记忆，生成摘要\n"
                        "   • 自动分析并应用经验\n\n"
                        "🔍 **深度复盘**\n"
                        "   • 调用长期记忆库，进行多维分析\n"
                        "   • 生成经验草案，需您确认\n\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "💬 请直接说 `快速复盘` 或 `深度复盘`\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )
                    
                    event = ZulongEvent(
                        type=EventType.L2_OUTPUT,
                        source="Gatekeeper",
                        payload={
                            'text': response_text,
                            'session_id': None,
                            'review_mode': True
                        },
                        priority=EventPriority.HIGH
                    )
                    
                    event_bus.publish(event)
                    return
                else:
                    logger.warning("[Gatekeeper] ReviewTrigger 未初始化")
            except Exception as e:
                logger.error(f"[Gatekeeper] 触发 ReviewTrigger 失败：{e}", exc_info=True)
        
        # 检查冷却
        current_time = time.time()
        if current_time - self._last_command_time < self._cooldown_time:
            if priority != EventPriority.HIGH:
                logger.debug("Command skipped due to cooldown")
                return
        
        # 更新最后命令时间
        self._last_command_time = current_time
        
        # 刷新空闲挂起定时器
        self.start_idle_suspend_timer()
        
        # Phase A3: 中断决策 — 如果 L2 正在 BUSY，设置中断标志
        l2_status = state_manager.get_l2_status()
        if l2_status == L2Status.BUSY:
            logger.info(f"[Gatekeeper] L2 BUSY，设置中断标志以打断当前任务")
            try:
                from zulong.l2.inference_engine import InferenceEngine
                engine = InferenceEngine()
                with engine._lock:
                    engine._interrupt_flag = True
            except Exception as e:
                logger.error(f"[Gatekeeper] 设置中断标志失败: {e}")
        
        logger.info("[Gatekeeper] 步骤 1-4: L1-B 并行执行上下文恢复、输出模态、ALBERT 与工具预判")

        def _run_async_bridge(coro, timeout: int = 15):
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                import concurrent.futures as _cf_inner
                with _cf_inner.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result(timeout=timeout)
            return asyncio.run(coro)

        def _context_branch():
            logger.info("🔍 [Gatekeeper] 并行分支/context: 局部上下文 + 共享上下文 + BFS + MemoryGraph")
            local = self._build_local_context(text)
            shared = self._search_shared_context(text)
            bfs = source_payload.get("bfs_context") if isinstance(source_payload.get("bfs_context"), dict) else None
            if not bfs and conversation_id:
                logger.info("🔍 [Gatekeeper] L1-B BFS 自恢复上下文...")
                bfs = self._retrieve_bfs_context(conversation_id)
                if bfs:
                    logger.info(
                        "✅ [Gatekeeper] BFS 自恢复完成：rounds=%s, tasks=%s",
                        len(bfs.get("recent_rounds", []) or []),
                        len(bfs.get("active_tasks", []) or []),
                    )
                else:
                    logger.info("[Gatekeeper] BFS 自恢复无结果，按全新上下文继续")

            memory_sections = self._format_bfs_context_sections(bfs)
            try:
                from zulong.memory.memory_graph import get_memory_graph
                _mg = get_memory_graph()
                if _mg and hasattr(_mg, "retrieve_context"):
                    logger.info("🔍 [Gatekeeper] L1-B 预检索 MemoryGraph 记忆（热/冷双路）...")
                    mg_results = _run_async_bridge(
                        _mg.retrieve_context(
                            text,
                            top_k=5,
                            session_id=conversation_id or session_id or "",
                        )
                    )
                    if mg_results:
                        for r in mg_results:
                            ntype = r.get("node_type", "")
                            content = r.get("content", "")
                            label = r.get("label", "")
                            if not content:
                                continue
                            if ntype == "experience":
                                continue
                            if ntype == "dialogue":
                                memory_sections.append(f"【历史对话】{content[:200]}")
                            elif ntype == "task":
                                status = r.get("metadata", {}).get("status", "")
                                memory_sections.append(
                                    f"【相关任务】{label}" + (f"（状态：{status}）" if status else "")
                                )
                            elif ntype == "knowledge":
                                memory_sections.append(f"【知识参考】{content[:300]}")
                            elif ntype == "episode":
                                memory_sections.append(f"【历史摘要】{content[:200]}")
                            elif ntype in ("person", "concept"):
                                memory_sections.append(f"【知识参考】{label}: {content[:200]}")
                            else:
                                memory_sections.append(f"【参考】{content[:200]}")
                        logger.info("✅ [Gatekeeper] MemoryGraph 预检索完成：%s 条候选", len(mg_results))
                    else:
                        logger.debug("[Gatekeeper] MemoryGraph 检索结果为空")
                else:
                    logger.debug("[Gatekeeper] MemoryGraph 未初始化，跳过预检索")
            except Exception as e:
                logger.warning("[Gatekeeper] MemoryGraph 预检索失败: %s", e)

            pre_memory = "\n".join(memory_sections) if memory_sections else None
            if pre_memory:
                logger.info("✅ [Gatekeeper] 上下文记忆打包完成：%s 段", len(memory_sections))
            return local, shared, bfs, pre_memory

        def _voice_branch():
            logger.info("🔍 [Gatekeeper] 并行分支/voice: 输出模态辅助判断")
            mode = self._detect_voice_mode(text, event_type or EventType.USER_TEXT)
            logger.info("✅ [Gatekeeper] 语音模式检测完成：%s", mode)
            return mode

        def _intent_and_tool_branch():
            logger.info("🔍 [Gatekeeper] 并行分支/tool: ALBERT 辅助分类 + 工具预判")
            intent = None
            if self._intent_filter:
                try:
                    intent = self._intent_filter.analyze(text)
                    logger.debug(
                        "✅ [Gatekeeper] 细粒度分类（辅助信息）: %s (置信度: %.3f, 模型: %s)",
                        intent.get("intent", "unknown"),
                        intent.get("confidence", 0),
                        intent.get("model", "keyword"),
                    )
                except Exception as e:
                    logger.debug("[Gatekeeper] 细粒度分类失败（不影响工具预判）: %s", e)
            else:
                logger.debug("[Gatekeeper] 细粒度分类器未初始化，跳过")

            try:
                from zulong.l2.inference_engine import InferenceEngine
                from zulong.tools.task_tools import get_active_task_graph
                from zulong.tools.tool_bag import predict_tools_for_turn

                _predict_engine = InferenceEngine().tool_engine
                _active_tg = get_active_task_graph()
                prediction = predict_tools_for_turn(
                    text,
                    registry=_predict_engine.registry,
                    intent_result=intent,
                    referenced_nodes=source_payload.get("referenced_nodes"),
                    has_task_graph=_active_tg is not None,
                ).to_dict()
                logger.info(
                    "[Gatekeeper] 工具预判完成: tools=%s, policy=%s",
                    prediction.get("predicted_tools", []),
                    prediction.get("task_graph_policy", "none"),
                )
                return intent, prediction
            except Exception as e:
                logger.warning("[Gatekeeper] 工具预判失败: %s", e)
                return intent, {
                    "predicted_tools": ["request_tool_supplement"],
                    "context_bundle": {},
                    "reasons": ["工具预判失败，保留工具补充入口"],
                    "risk_notes": [],
                    "task_graph_policy": "none",
                }

        local_context = {}
        shared_context_snapshot = {}
        bfs_context = None
        pre_retrieved_memory = None
        voice_mode = "TEXT_ONLY"
        intent_result = None
        tool_prediction = None
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="l1b-pack") as pool:
            context_future = pool.submit(_context_branch)
            voice_future = pool.submit(_voice_branch)
            tool_future = pool.submit(_intent_and_tool_branch)
            try:
                local_context, shared_context_snapshot, bfs_context, pre_retrieved_memory = context_future.result()
            except Exception as exc:
                logger.warning("[Gatekeeper] 上下文并行分支失败，使用空上下文继续: %s", exc)
            try:
                voice_mode = voice_future.result()
            except Exception as exc:
                logger.warning("[Gatekeeper] 输出模态并行分支失败，回退 TEXT_ONLY: %s", exc)
            try:
                intent_result, tool_prediction = tool_future.result()
            except Exception as exc:
                logger.warning("[Gatekeeper] 工具预判并行分支失败，保留工具补充入口: %s", exc)
                tool_prediction = {
                    "predicted_tools": ["request_tool_supplement"],
                    "context_bundle": {},
                    "reasons": ["工具预判失败，保留工具补充入口"],
                    "risk_notes": [],
                    "task_graph_policy": "none",
                }

        l1b_context_pack = self._build_l1b_context_pack(
            local_context=local_context,
            shared_context_snapshot=shared_context_snapshot,
            pre_retrieved_memory=pre_retrieved_memory,
            bfs_context=bfs_context,
            tool_prediction=tool_prediction,
            voice_mode=voice_mode,
        )
        logger.info("✅ [Gatekeeper] L1-B 统一上下文打包完成：%s 字符", len(l1b_context_pack))
        self._publish_tool_prediction_event(
            {**(tool_prediction or {}), "user_text": text},
            request_id=request_id,
            session_id=session_id,
        )
        
        # 6. 打包新任务：用户原文 + 局部上下文 + 共享上下文快照 + 语音模式 + 工具预判 + 预检索记忆
        packaged_task = {
            "text": text,
            "local_context": local_context,
            "shared_context_snapshot": shared_context_snapshot,
            "voice_mode": voice_mode,
            "intent_result": intent_result,  # L1-B 细粒度分类结果，仅作为工具预判辅助
            "pre_retrieved_memory": pre_retrieved_memory,  # L1-B 预检索的记忆上下文
            "l1b_context_pack": l1b_context_pack,
            "bfs_context": bfs_context,
            "tool_prediction": tool_prediction,
            "tool_bundle": tool_prediction.get("predicted_tools", []) if tool_prediction else [],
            "context_bundle": tool_prediction.get("context_bundle", {}) if tool_prediction else {},
            "task_graph_policy": tool_prediction.get("task_graph_policy", "none") if tool_prediction else "none",
        }
        _context_bundle = packaged_task.get("context_bundle") or {}
        _simple_dialogue_turn = (
            _context_bundle.get("turn_shape") in {"simple_social", "presentation_only"}
            and packaged_task.get("task_graph_policy") == "none"
        )
        for key in (
            "conversation_id",
            "turn_id",
            "workspace_path",
            "cwd",
            "project_id",
            "task_graph_id",
            "referenced_task_graph_id",
            "task_graph_reference_mode",
            "referenced_nodes",
            "source",
        ):
            if _simple_dialogue_turn and key in {
                "workspace_path",
                "cwd",
                "project_id",
                "task_graph_id",
                "referenced_task_graph_id",
                "task_graph_reference_mode",
                "referenced_nodes",
            }:
                continue
            value = source_payload.get(key)
            if value:
                packaged_task[key] = value
        recovered_task_graph_id = self._extract_bfs_task_graph_id(bfs_context)
        _task_policy_allows_recovery = packaged_task.get("task_graph_policy") in {
            "reuse",
            "inspect",
            "continue",
            "inspect_or_create",
        }
        if (
            recovered_task_graph_id
            and _task_policy_allows_recovery
            and not packaged_task.get("task_graph_id")
        ):
            packaged_task["task_graph_id"] = recovered_task_graph_id
        
        # 🔥 新增：添加 session_id（如果有）
        if session_id:
            packaged_task["session_id"] = session_id
            logger.info(f"🏷️ [Gatekeeper] Session ID 已添加到任务包：{session_id}")
        
        # 添加 request_id（用于思考过程和流式输出关联）
        if request_id:
            packaged_task["request_id"] = request_id
        
        logger.info(f"📦 [Gatekeeper] 步骤 5: 打包任务完成")
        
        # ── 直接发布 SYSTEM_L2_COMMAND 事件，由 L2 FC 循环自主处理 ──
        
        # ── 创建 MemoryGraph 对话节点 ──
        dialogue_round_id = self._ensure_dialogue_node(text, packaged_task)
        if dialogue_round_id:
            packaged_task["dialogue_round_id"] = dialogue_round_id

        # 直接发布 SYSTEM_L2_COMMAND 事件到 L2
        logger.info(f"🚀 [Gatekeeper] 直接路由到 L2，发布 SYSTEM_L2_COMMAND 事件")
        
        l2_command_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=priority,
            source="Gatekeeper",
            payload=packaged_task
        )
        event_bus.publish(l2_command_event)
    
    def _publish_tool_prediction_event(self, tool_prediction, request_id: Optional[str] = None, session_id: Optional[str] = None):
        """向 Web 前端发布 L1-B 工具预判启动说明。"""
        if not tool_prediction:
            return
        try:
            tools = tool_prediction.get("predicted_tools", []) or []
            reasons = tool_prediction.get("reasons", []) or []
            policy = tool_prediction.get("task_graph_policy", "none")
            context_bundle = tool_prediction.get("context_bundle", {}) or {}
            task_text = str(tool_prediction.get("user_text") or tool_prediction.get("text") or "").strip()
            if len(task_text) > 76:
                task_text = task_text[:73].rstrip() + "..."
            steps = []
            if context_bundle.get("needs_memory") or any("memory" in str(t) for t in tools):
                steps.append("先取相关记忆和经验")
            if context_bundle.get("needs_project_context") or any(str(t) in {"read_file", "search_code_symbols", "index_project", "analyze_module"} for t in tools):
                steps.append("读取或检索项目代码")
            if policy in {"reuse", "inspect", "inspect_or_create", "continue"} or any(str(t).startswith("task_") for t in tools):
                steps.append("同步任务图，确认当前进度")
            if tools:
                steps.append("把候选工具交给 L2 自主选择")
            if not steps:
                steps.append("判断是否需要工具，不需要时直接回复")
            title = f"先看目标: {task_text}" if task_text else "先看目标和上下文"
            detail_lines = [
                "我会先确认这件事需要哪些上下文，再让 L2 决定是否真的调用工具。"
            ]
            if tools:
                detail_lines.append("候选能力: " + ", ".join(str(t) for t in tools[:6]))
            if policy and policy != "none":
                detail_lines.append("任务图: " + str(policy))
            if reasons:
                detail_lines.append("判断依据: " + "；".join(str(r) for r in reasons[:2]))
            interaction = {
                "pair_id": f"l1b-tool-prediction-{request_id or int(time.time() * 1000)}",
                "kind": "plan",
                "status": "running",
                "title": title,
                "detail": "\n".join(detail_lines),
                "progress": 0,
                "plan_steps": steps[:5],
                "next_step": "接下来由 L2 根据上下文做真实决策",
            }
            payload = {
                "request_id": request_id or f"req_{int(time.time() * 1000)}",
                "session_id": session_id or "",
                "step_type": "pipeline.agent_start",
                "iteration": 0,
                "timestamp": time.time(),
                "data": {
                    "tools": tools,
                    "predicted_tools": tools,
                    "reasons": reasons,
                    "task_graph_policy": policy,
                    "context_bundle": context_bundle,
                    "tool_count": len(tools),
                    "progress": {
                        "total": 0,
                        "completed": 0,
                        "running": 0,
                        "blocked": 0,
                        "percent": 0,
                    },
                    "interaction": interaction,
                },
            }
            event_bus.publish(ZulongEvent(
                type=EventType.L2_THINKING_STEP,
                priority=EventPriority.NORMAL,
                source="Gatekeeper",
                payload=payload,
            ))
        except Exception as e:
            logger.debug(f"[Gatekeeper] 工具预判前端事件发布失败: {e}")

    # ── MemoryGraph 对话节点管理 ──

    def _ensure_dialogue_node(self, text, packaged_task):
        """为用户消息创建 MemoryGraph 对话 round 骨架（不做 session 分配）
        
        流程:
        1. 创建无 session 的 round 骨架节点（仅记录用户输入和时间戳）
        2. Session 分配由 L2 InferenceEngine._update_memory() 在有完整上下文后执行
           （使用 Embedding 相似度做智能话题归属）
        3. 复杂任务后续由 L2 FC 循环通过 task_tools 自主管理 TaskGraph
        
        Returns:
            str | None: 创建的 dialogue_round_id，失败返回 None
        """
        try:
            from zulong.memory.memory_graph import get_memory_graph, NodeType
            from zulong.memory.graph_adapters import DialogueAdapter
            mg = get_memory_graph()
            if mg is None:
                return None

            request_id = packaged_task.get("request_id", "")
            if not request_id:
                request_id = f"gk_{int(time.time() * 1000)}"

            adapter = DialogueAdapter()

            # 不做 session 分配 — session 归属由 L2 在有完整上下文后决定
            # （L2 使用 Embedding 相似度做智能话题检测，比 GK 硬编码关键词更准确）

            # 查找最近的 round（用于 TEMPORAL 边连接）
            prev_round_id = None
            dialogue_nodes = mg.get_nodes_by_type(NodeType.DIALOGUE)
            rounds = [
                n for n in dialogue_nodes
                if n.metadata.get("sub_type") == "round"
            ]
            if rounds:
                rounds.sort(key=lambda n: n.created_at, reverse=True)
                prev_round_id = rounds[0].node_id

            # 创建无 session 的 round 骨架（session_id=None，L2 稍后分配）
            round_id = adapter.add_round(
                mg, request_id, text, prev_round_id,
                session_id=None,
            )

            # 更新思维焦点到新创建的对话轮次
            try:
                mg.update_focus_to_node(round_id)
            except Exception:
                pass  # 焦点更新失败不影响主流程

            logger.info(
                f"[Gatekeeper] Round 骨架已创建: {round_id} (session 待 L2 分配)"
            )
            # 发布图谱更新事件到前端
            self._publish_memory_graph_event_async(mg)
            return round_id
        except Exception as e:
            logger.debug(f"[Gatekeeper] 对话节点创建跳过: {e}")
            return None

    def _publish_memory_graph_event_async(self, mg):
        """后台发布 MemoryGraph 前端更新，不阻塞 L1-B → L2 主链路。"""
        try:
            worker = threading.Thread(
                target=self._publish_memory_graph_event,
                args=(mg,),
                name="zulong-l1b-memory-graph-update",
                daemon=True,
            )
            worker.start()
        except Exception as exc:
            logger.debug(f"[Gatekeeper] MemoryGraph 异步更新调度跳过: {exc}")

    def _publish_memory_graph_event(self, mg):
        """发布 MemoryGraph 更新事件到前端"""
        try:
            from zulong.core.types import EventType as ZulongEventType
            event = ZulongEvent(
                type=ZulongEventType.MEMORY_GRAPH_UPDATED,
                priority=EventPriority.LOW,
                source="Gatekeeper",
                payload=mg.to_frontend_dict(depth=0),
            )
            event_bus.publish(event)
        except Exception:
            pass

    # ── 模型自主路由：Hint 机制（替代硬编码复杂度分类） ──

    def _init_hint_mode(self):
        """根据模型参数量自动决定是否启用 Hint

        小参数模型 (<=8B): 启用 Hint，辅助模型判断是否使用规划工具
        大参数模型 (>8B): 关闭 Hint，完全信任模型自主判断
        """
        try:
            from zulong.models.container import LLM_MODEL_ID
            model_id = (LLM_MODEL_ID or "").lower()
        except ImportError:
            model_id = ""

        small_model_patterns = [
            "0.5b", "1b", "1.5b", "2b", "3b", "3.5b", "4b", "7b", "8b",
        ]
        self._hint_enabled = any(p in model_id for p in small_model_patterns)
        logger.info(
            f"[Gatekeeper] Hint 模式: {'启用' if self._hint_enabled else '关闭'} "
            f"(model={model_id})"
        )

    def on_audio_event(self, event: ZulongEvent):
        """
        处理 L1-D 音频事件 (TSD v1.8 三层注意力)
        
        当 L1-D 检测到有效语音交互时触发。
        L1-B 负责收集语音窗口并路由到 L2。
        
        Args:
            event: ZulongEvent (包含音频事件信息)
        """
        payload = event.payload
        event_subtype = payload.get("event_subtype", "")
        
        if event_subtype == "AUDIO_SPEECH_START":
            self._handle_audio_speech_start(payload)
        elif event_subtype == "AUDIO_SPEECH_END":
            self._handle_audio_speech_end(payload)
        else:
            logger.debug(f"🔊 [Gatekeeper] 收到音频事件：{event_subtype}")
    
    def _handle_audio_speech_start(self, payload: dict):
        """
        处理语音开始事件
        
        启动语音收集窗口，等待完整的语音片段
        
        Args:
            payload: 事件负载
        """
        current_time = time.time()
        
        self._speech_collect_active = True
        self._speech_collect_start_time = current_time
        self._collected_audio_frames = []
        
        logger.info(
            f"🎙️ [Gatekeeper] 语音收集窗口启动 "
            f"(窗口时长={self._speech_collect_window}s)"
        )
    
    def _handle_audio_speech_end(self, payload: dict):
        """
        处理语音结束事件
        
        收集完整的语音片段，打包发送给 L2
        
        Args:
            payload: 事件负载
        """
        current_time = time.time()
        speech_duration = payload.get("duration", 0.0)
        buffer_frames = payload.get("buffer_frames", 0)
        
        logger.info(
            f"🎙️ [Gatekeeper] 语音收集完成 "
            f"(时长={speech_duration:.2f}s, 帧数={buffer_frames})"
        )
        
        self._speech_collect_active = False
        
        if speech_duration < 0.3:
            logger.info("🎙️ [Gatekeeper] 语音时长过短，忽略")
            return
        
        effective_state = state_manager.get_effective_status()
        l2_status = state_manager.get_l2_status()
        
        prompt = self._build_audio_attention_prompt(payload)
        
        if effective_state == "ACTIVE_TASK":
            if l2_status == L2Status.WAITING:
                logger.info("🎙️ [Gatekeeper] L2 在 WAITING 状态，插入音频注意力任务")
                self._route_audio_to_l2(prompt, payload, interrupt=True)
            else:
                logger.info("🎙️ [Gatekeeper] L2 在 BUSY 状态，排队音频注意力任务")
                self._route_audio_to_l2(prompt, payload, interrupt=False)
        else:
            logger.info("🎙️ [Gatekeeper] L2 空闲，直接路由音频注意力任务")
            self._route_audio_to_l2(prompt, payload, interrupt=False)
    
    def _build_audio_attention_prompt(self, payload: dict) -> str:
        """
        构建音频注意力 Prompt
        
        Args:
            payload: 事件负载
            
        Returns:
            str: 构建的 Prompt
        """
        speech_duration = payload.get("duration", 0.0)
        
        prompt = (
            f"[音频注意力事件]\n"
            f"检测到用户的语音输入。\n"
            f"语音时长：{speech_duration:.2f} 秒\n"
            f"请等待语音识别结果，或主动询问用户的意图。"
        )
        
        return prompt
    
    def _route_audio_to_l2(self, prompt: str, payload: dict, interrupt: bool = False):
        """
        路由音频事件到 L2
        
        Args:
            prompt: 构建的 Prompt
            payload: 事件负载
            interrupt: 是否中断模式
        """
        priority = EventPriority.HIGH if interrupt else EventPriority.NORMAL
        
        l2_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=priority,
            source="Gatekeeper_AudioAttention",
            payload={
                "text": prompt,
                "audio_attention": True,
                "interrupt": interrupt,
                "speech_duration": payload.get("duration", 0.0),
                "buffer_frames": payload.get("buffer_frames", 0)
            }
        )
        
        event_bus.publish(l2_event)
        logger.info(
            f"🎙️ [Gatekeeper] 已路由音频注意力事件到 L2 "
            f"(中断模式={interrupt})"
        )
    
    def on_sensor_fall(self, event: ZulongEvent):
        """
        处理摔倒传感器事件 - 经过 L1-B 路由到 L2
        
        Args:
            event: ZulongEvent
        """
        logger.warning("🚨 [Gatekeeper] 收到摔倒传感器事件")
        
        effective_state = state_manager.get_effective_status()
        
        if effective_state == "ACTIVE_TASK":
            task_state_manager.freeze_current()
        
        l2_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=EventPriority.CRITICAL,
            source="Gatekeeper_SensorFall",
            payload={
                "text": "检测到摔倒事件！用户可能需要帮助。",
                "sensor_type": "FALL",
                "emergency": True,
                "original_event": event.payload
            }
        )
        event_bus.publish(l2_event)
        logger.info("🚨 [Gatekeeper] 已路由摔倒事件到 L2")
    
    def on_sensor_obstacle(self, event: ZulongEvent):
        """
        处理障碍传感器事件 - 经过 L1-B 路由到 L2
        
        Args:
            event: ZulongEvent
        """
        distance = event.payload.get("distance", 0)
        logger.info(f"🚧 [Gatekeeper] 收到障碍传感器事件: 距离={distance}cm")
        
        l2_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=EventPriority.HIGH,
            source="Gatekeeper_SensorObstacle",
            payload={
                "text": f"检测到前方障碍物，距离约 {distance} 厘米。",
                "sensor_type": "OBSTACLE",
                "distance": distance,
                "original_event": event.payload
            }
        )
        event_bus.publish(l2_event)
        logger.info("🚧 [Gatekeeper] 已路由障碍事件到 L2")
    
    def on_sensor_vision(self, event: ZulongEvent):
        """
        处理视觉传感器事件 - 经过 L1-B 路由到 L2
        
        Args:
            event: ZulongEvent
        """
        vision_data = event.payload.get("vision_data", {})
        objects = vision_data.get("objects", [])
        logger.info(f"👁️ [Gatekeeper] 收到视觉传感器事件: 检测到 {len(objects)} 个对象")
        
        l2_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=EventPriority.NORMAL,
            source="Gatekeeper_SensorVision",
            payload={
                "text": f"视觉检测到 {len(objects)} 个对象。",
                "sensor_type": "VISION",
                "vision_data": vision_data,
                "original_event": event.payload
            }
        )
        event_bus.publish(l2_event)
        logger.info("👁️ [Gatekeeper] 已路由视觉事件到 L2")
    
    def on_vision_data_ready(self, event: ZulongEvent):
        """
        处理视觉数据就绪事件 - 经过 L1-B 路由到 L2
        
        Args:
            event: ZulongEvent
        """
        video_path = event.payload.get("video_path", "")
        logger.info(f"📹 [Gatekeeper] 收到视觉数据就绪事件: {video_path}")
        
        l2_event = ZulongEvent(
            type=EventType.SYSTEM_L2_COMMAND,
            priority=EventPriority.HIGH,
            source="Gatekeeper_VisionDataReady",
            payload={
                "text": "视觉数据已就绪，可以进行详细分析。",
                "vision_ready": True,
                "video_path": video_path,
                "original_event": event.payload
            }
        )
        event_bus.publish(l2_event)
        logger.info("📹 [Gatekeeper] 已路由视觉数据就绪事件到 L2")
    
    def _write_user_input_to_pool(self, text: str, event: ZulongEvent):
        """将用户输入异步写入共享池（供复盘功能使用）
        
        Args:
            text: 用户输入文本
            event: 原始事件
        """
        import asyncio
        import time
        import uuid
        
        async def write_async():
            try:
                from zulong.infrastructure.shared_memory_pool import SharedMemoryPool, DataType
                
                pool = await SharedMemoryPool.get_instance()
                
                trace_id = await pool.write_text(
                    key=f"user_{uuid.uuid4().hex[:8]}",
                    data={
                        'text': text,
                        'source': event.source,
                    },
                    metadata={
                        'event_type': event.type.value,
                        'priority': event.priority.value,
                        'timestamp': time.time()
                    }
                )
                
                logger.debug(f"[Gatekeeper] 已记录用户输入到共享池：{trace_id}")
            except Exception as e:
                logger.error(f"[Gatekeeper] 写入共享池失败：{e}", exc_info=True)
        
        # 异步执行，不阻塞主流程
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(write_async())
        except RuntimeError:
            # 没有运行中的事件循环，创建新线程
            import threading
            thread = threading.Thread(target=lambda: asyncio.run(write_async()))
            thread.daemon = True
            thread.start()
    
    # ========== 🔥 新增：三阶段状态机核心方法 ==========
    
    def _handle_mode_select_stage(self, text: str, priority: EventPriority):
        """🔥 新增：处理模式选择阶段的输入
        
        Args:
            text: 用户输入文本
            priority: 事件优先级
        """
        logger.info(f"[Gatekeeper] 🔒 模式选择阶段处理：{text}")
        
        # 转发到 ReplayIntegration 处理
        self._forward_to_replay_integration(text, 'mode_select')
    
    def _handle_review_active_stage(self, text: str, priority: EventPriority):
        """🔥 新增/修复：处理对话进行阶段的输入
        
        Args:
            text: 用户输入文本
            priority: 事件优先级
        """
        logger.info(f"[Gatekeeper] 💬 对话进行阶段处理：{text}")
        
        # 检测结束复盘关键词
        if '结束复盘' in text.lower() or '完成复盘' in text.lower():
            logger.info(f"[Gatekeeper] 🔍 检测到结束复盘指令")
            
            # 🔥 直接触发 ReplayIntegration 的 handle_end_review
            try:
                from zulong.review.integration import get_replay_integration
                replay_integration = get_replay_integration()
                
                if replay_integration and hasattr(replay_integration, 'handle_end_review'):
                    import asyncio
                    
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.create_task(replay_integration.handle_end_review())
                        logger.info("[Gatekeeper] ✅ 已异步触发 handle_end_review()")
                    except RuntimeError:
                        def run_end_review():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                new_loop.run_until_complete(
                                    replay_integration.handle_end_review()
                                )
                            finally:
                                new_loop.close()
                        
                        import threading
                        thread = threading.Thread(target=run_end_review, daemon=True)
                        thread.start()
                        logger.info("[Gatekeeper] ✅ 已在新线程中触发 handle_end_review()")
                        
                else:
                    logger.warning("[Gatekeeper] ReplayIntegration 不可用或缺少 handle_end_review 方法")
                    
            except Exception as e:
                logger.error(f"[Gatekeeper] 触发结束复盘失败：{e}", exc_info=True)
            
            return
        
        # 🔥 核心修复：其他输入 - 真正放行给 L2 对话！
        logger.info(f"[Gatekeeper] ✅ 放行到 L2 对话（使用 _handle_normal_command）")
        
        # 🔥 关键修复：使用 _handle_normal_command() 而不是不存在的 _forward_to_l2()
        # _handle_normal_command 会正确地打包上下文并发布 SYSTEM_L2_COMMAND 事件给 L2
        self._handle_normal_command(text, priority, EventType.USER_TEXT)
    
    def _handle_experience_confirm_stage(self, text: str, priority: EventPriority):
        """🔥 新增：处理经验确认阶段的输入
        
        Args:
            text: 用户输入文本
            priority: 事件优先级
        """
        logger.info(f"[Gatekeeper] 🔒 经验确认阶段处理：{text}")
        
        # 转发到 ReplayIntegration 处理
        self._forward_to_replay_integration(text, 'experience_confirm')
    
    def _handle_direct_review_start(self, text: str, priority: EventPriority):
        """🔥 新增：直接启动复盘（不经过模式选择）
        
        Args:
            text: 用户输入文本
            priority: 事件优先级
        """
        logger.info(f"[Gatekeeper] ⚡ 直接启动复盘：{text}")
        
        try:
            if not self._review_state_manager.acquire_processing_lock():
                logger.warning("[Gatekeeper] 复盘正在处理中，忽略重复请求")
                return
            
            # 判断复盘类型
            review_type = 'quick' if '快速复盘' in text.lower() else 'deep'
            logger.info(f"[Gatekeeper] 检测到复盘类型：{review_type}")
            
            # 直接触发对应类型的复盘
            from zulong.review.trigger import get_review_trigger
            review_trigger = get_review_trigger()
            
            if review_trigger:
                import asyncio
                
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(
                        review_trigger.trigger_user_active(
                            context={
                                'trigger_keyword': text,
                                'user_input': text,
                                'trigger_source': 'gatekeeper',
                                'review_type': review_type  # 🔥 直接指定类型
                            }
                        )
                    )
                    logger.info("[Gatekeeper] 已异步触发 ReviewTrigger")
                except RuntimeError:
                    def run_async_task():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            new_loop.run_until_complete(
                                review_trigger.trigger_user_active(
                                    context={
                                        'trigger_keyword': text,
                                        'user_input': text,
                                        'trigger_source': 'gatekeeper',
                                        'review_type': review_type
                                    }
                                )
                            )
                        finally:
                            new_loop.close()
                    
                    import threading
                    thread = threading.Thread(target=run_async_task)
                    thread.daemon = True
                    thread.start()
                    logger.info("[Gatekeeper] 已异步触发 ReviewTrigger (新线程)")
                    
                    # 🔥 关键修复 1：手动设置复盘状态为 REVIEW_ACTIVE（在异步线程设置之前）
                    self._review_state_manager._is_active = True
                    self._review_state_manager._stage = ReviewStage.REVIEW_ACTIVE
                    self._review_state_manager._mode = ReviewMode.QUICK if review_type == 'quick' else ReviewMode.DEEP
                    self._review_state_manager._notify_state_change("direct_start_review")
                    logger.info(f"[Gatekeeper] ✅ 已手动设置复盘状态为 REVIEW_ACTIVE")
                    
                    # 🔥 关键修复 2：发布 L2 响应，告诉用户复盘已启动
                    review_type_text = "快速复盘" if review_type == 'quick' else "深度复盘"
                    response_text = (
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"🎯 **{review_type_text} 已启动**\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"正在分析最近的对话和记忆...\n\n"
                        f"⏳ 请稍候，马上回来~\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )
                    
                    event = ZulongEvent(
                        type=EventType.L2_OUTPUT,
                        source="Gatekeeper",
                        payload={
                            'text': response_text,
                            'session_id': None,
                            'review_mode': True,
                            'l2_status': 'REVIEW_ANALYZING'  # 🔥 新增：标记状态
                        },
                        priority=EventPriority.HIGH
                    )
                    
                    event_bus.publish(event)
                    logger.info(f"[Gatekeeper] 已发布{review_type_text}启动响应事件")
            else:
                logger.error("[Gatekeeper] ReviewTrigger 未初始化")
        
        except Exception as e:
            logger.error(f"直接启动复盘失败：{e}")
        finally:
            try:
                self._review_state_manager.release_processing_lock()
            except Exception as e:
                logger.error(f"释放处理锁失败：{e}")


gatekeeper = Gatekeeper()
