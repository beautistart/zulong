"""
LLM自主注意力模式选择 - AttentionWindowManager集成扩展

本文件提供AttentionWindowManager的集成扩展代码片段，
需要在attention_window.py中手动集成这些代码。

集成步骤：
1. 在AttentionWindowManager.__init__()中添加新组件初始化
2. 在apply_window()开头添加LLM选择检查
3. 实现LLM选择相关的辅助方法
"""

# ============================================================================
# 步骤1: 在AttentionWindowManager.__init__()中添加导入
# ============================================================================

"""
在文件顶部添加导入:
from .attention_types import (
    PressureMetrics, DecisionRequest, DecisionResponse,
    SwitchRecord, OscillationState, TriggerType,
    PressureTrend, OscillationLevel,
)
from .attention_config import AttentionConfig
from .pressure_detector import PressureDetector
from .attention_mode_selector import AttentionModeSelector
from .mode_switch_controller import (
    ModeSwitchController, CooldownManager, OscillationDetector
)
"""

# ============================================================================
# 步骤2: 在__init__()方法末尾添加新组件初始化
# ============================================================================

INIT_EXTENSION = """
        # ── LLM自主注意力选择组件初始化 ──
        try:
            self._llm_config = AttentionConfig.load_from_yaml()
            self._llm_config.validate()
            
            if self._llm_config.enabled:
                self._pressure_detector = PressureDetector(self, self._llm_config)
                self._mode_selector = AttentionModeSelector(self._llm_config)
                
                self._cooldown_manager = CooldownManager(self._llm_config)
                self._oscillation_detector = OscillationDetector(self._llm_config)
                self._mode_controller = ModeSwitchController(
                    self._llm_config,
                    self._cooldown_manager,
                    self._oscillation_detector
                )
                
                self._llm_client = None  # 后续通过set_llm_client()设置
                self._last_llm_selection_time = None
                
                logger.info(
                    f"[AttentionWindow] LLM自主注意力选择已启用: "
                    f"pressure_threshold={self._llm_config.pressure_threshold_high}"
                )
            else:
                self._llm_config = None
                logger.info("[AttentionWindow] LLM自主注意力选择已禁用")
                
        except Exception as e:
            logger.error(f"[AttentionWindow] LLM注意力选择初始化失败: {e}")
            self._llm_config = None
"""

# ============================================================================
# 步骤3: 在apply_window()方法开头添加LLM选择检查
# ============================================================================

APPLY_WINDOW_EXTENSION = """
    def apply_window(self) -> List[Dict]:
        # ── 新增：LLM自主注意力选择检查 ──
        if self._should_try_llm_selection():
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._try_llm_mode_selection_async())
                else:
                    loop.run_until_complete(self._try_llm_mode_selection_async())
            except Exception as e:
                logger.warning(f"[AttentionWindow] LLM注意力选择执行失败: {e}")
        
        # ── 原有裁剪逻辑继续执行 ──
        # ... (保持原有代码不变)
"""

# ============================================================================
# 步骤4: 实现LLM选择相关的辅助方法
# ============================================================================

HELPER_METHODS = """
    def set_llm_client(self, llm_client):
        """设置LLM客户端实例
        
        Args:
            llm_client: LLM客户端实例（需支持chat或generate方法）
        """
        if self._llm_config and self._llm_config.enabled:
            self._llm_client = llm_client
            self._mode_selector.set_llm_client(llm_client)
            logger.info("[AttentionWindow] LLM客户端已设置")
    
    def _should_try_llm_selection(self) -> bool:
        """检查是否应尝试LLM自主选择
        
        Returns:
            是否应尝试
        """
        if not self._llm_config or not self._llm_config.enabled:
            return False
        
        if not self._llm_client:
            return False
        
        cooldown_result = self._mode_controller.can_switch()
        if not cooldown_result.is_allowed:
            logger.debug(f"[AttentionWindow] 冷却期未过: {cooldown_result.message}")
            return False
        
        return True
    
    async def _try_llm_mode_selection_async(self):
        """异步执行LLM自主注意力模式选择
        
        完整流程：
        1. 计算当前压力
        2. 检查是否超过阈值
        3. 构建决策请求
        4. 调用LLM决策
        5. 应用决策结果
        """
        try:
            metrics = self._pressure_detector.calculate_pressure()
            
            threshold_result = self._pressure_detector.check_threshold(metrics)
            if not threshold_result.should_trigger:
                logger.debug(f"[AttentionWindow] 压力未超阈值: {threshold_result.message}")
                return
            
            logger.info(
                f"[AttentionWindow] 触发LLM注意力选择: {threshold_result.trigger_type} "
                f"(压力={metrics.current_pressure:.3f})"
            )
            
            task_context = self._get_task_context_summary()
            current_mode = self.mode.name
            
            switch_history = self._mode_controller.get_switch_history(5)
            
            request = self._mode_selector.build_decision_request(
                pressure_metrics=metrics,
                current_mode=current_mode,
                task_context=task_context,
                switch_history=switch_history,
            )
            
            decision = await self._mode_selector.call_llm_decision(request)
            
            logger.info(
                f"[AttentionWindow] LLM决策结果: mode={decision.mode}, "
                f"confidence={decision.confidence:.2f}, fallback={decision.is_fallback}"
            )
            
            if decision.mode != current_mode:
                result = self._mode_controller.apply_llm_decision(
                    current_mode=current_mode,
                    decision=decision,
                    pressure_value=metrics.current_pressure,
                )
                
                if result.switched:
                    try:
                        new_mode = AttentionMode[decision.mode]
                        self.mode = new_mode
                        logger.info(
                            f"[AttentionWindow] 注意力模式已切换: "
                            f"{current_mode} → {decision.mode}"
                        )
                    except KeyError:
                        logger.error(f"[AttentionWindow] 无效的模式名称: {decision.mode}")
            else:
                logger.info(f"[AttentionWindow] LLM选择保持当前模式: {current_mode}")
                
        except Exception as e:
            logger.error(f"[AttentionWindow] LLM注意力选择流程异常: {e}")
    
    def _get_task_context_summary(self) -> str:
        """获取任务上下文摘要（供LLM决策参考）
        
        Returns:
            任务上下文摘要文本（≤500字符）
        """
        summary_parts = []
        
        if self.task_graph:
            try:
                node_count = self.task_graph.node_count if hasattr(self.task_graph, 'node_count') else 0
                summary_parts.append(f"任务图节点数: {node_count}")
                
                if self._current_node_id:
                    summary_parts.append(f"当前节点: {self._current_node_id}")
            except Exception:
                pass
        
        message_count = len(self.envelopes)
        total_tokens = sum(e.tokens for e in self.envelopes)
        summary_parts.append(f"消息数量: {message_count}")
        summary_parts.append(f"总tokens: {total_tokens}")
        
        recent_tools = []
        for env in self.envelopes[-5:]:
            if env.tool_name:
                recent_tools.append(env.tool_name)
        if recent_tools:
            summary_parts.append(f"近期工具: {', '.join(recent_tools)}")
        
        summary = " | ".join(summary_parts)
        return summary[:500]
    
    def get_llm_selection_stats(self) -> Dict:
        """获取LLM注意力选择的统计信息
        
        Returns:
            统计信息字典
        """
        if not self._llm_config:
            return {"enabled": False}
        
        history = self._mode_controller.get_switch_history(10)
        llm_switches = [r for r in history if r.trigger_type == TriggerType.LLM_AUTONOMOUS]
        fallback_switches = [r for r in history if r.trigger_type == TriggerType.FALLBACK]
        
        return {
            "enabled": self._llm_config.enabled,
            "current_mode": self.mode.name,
            "total_switches": len(history),
            "llm_autonomous_switches": len(llm_switches),
            "fallback_switches": len(fallback_switches),
            "cooldown_seconds": self._cooldown_manager.get_current_cooldown_seconds(),
            "switch_history": [r.to_dict() for r in history[-5:]],
        }
"""

# ============================================================================
# 配置文件示例 (config/zulong_config.yaml)
# ============================================================================

CONFIG_EXAMPLE = """
# 在config/zulong_config.yaml中添加以下配置段:

attention_selection:
  # 功能开关
  enabled: true
  
  # 压力阈值配置
  pressure_threshold_high: 0.9      # 高压阈值 (推荐0.8-1.0)
  pressure_threshold_medium: 0.75   # 中压阈值 (推荐0.6-0.8)
  
  # 冷却时间配置
  cooldown_base_seconds: 30.0       # 基础冷却时间(秒)
  
  # Fallback配置
  fallback_mode: "FOCUS"            # Fallback默认模式
  
  # 性能配置
  decision_timeout_ms: 500          # LLM决策超时(毫秒)
  
  # 震荡检测配置
  oscillation_detection_window: 10  # 震荡检测窗口大小
  max_switch_history: 50            # 最大切换历史记录数
  min_confidence_threshold: 0.3     # 最低置信度阈值
"""

# ============================================================================
# 使用示例
# ============================================================================

USAGE_EXAMPLE = """
# 在InferenceEngine中使用:

class InferenceEngine:
    def __init__(self, ...):
        # 初始化AttentionWindowManager
        self._attn_window = AttentionWindowManager(
            context_window_size=128000,
            task_graph=self.task_graph,
            memory_graph=self.memory_graph,
        )
        
        # 设置LLM客户端（使能LLM自主选择）
        self._attn_window.set_llm_client(self.llm_client)
    
    async def run_fc_loop(self, ...):
        # FC循环中会自动调用apply_window()
        # apply_window()内部会自动检查压力并触发LLM选择
        
        # 获取LLM选择统计信息
        stats = self._attn_window.get_llm_selection_stats()
        logger.info(f"LLM注意力选择统计: {stats}")
"""

if __name__ == "__main__":
    print("=" * 80)
    print("LLM自主注意力模式选择 - AttentionWindowManager集成扩展")
    print("=" * 80)
    print("\n【步骤1】在文件顶部添加导入语句")
    print("-" * 80)
    
    print("\n【步骤2】在__init__()方法末尾添加新组件初始化")
    print("-" * 80)
    print(INIT_EXTENSION)
    
    print("\n【步骤3】在apply_window()方法开头添加LLM选择检查")
    print("-" * 80)
    print(APPLY_WINDOW_EXTENSION)
    
    print("\n【步骤4】添加辅助方法")
    print("-" * 80)
    print(HELPER_METHODS)
    
    print("\n【配置文件示例】")
    print("-" * 80)
    print(CONFIG_EXAMPLE)
    
    print("\n【使用示例】")
    print("-" * 80)
    print(USAGE_EXAMPLE)
