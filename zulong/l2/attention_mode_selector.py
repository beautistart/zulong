"""
LLM自主注意力模式选择 - LLM决策模块
"""
from typing import Optional, Dict, Any, List
import json
import re
import logging
import asyncio
from datetime import datetime

from .attention_types import (
    PressureMetrics,
    DecisionRequest,
    DecisionResponse,
    SwitchRecord,
    PressureTrend,
)
from .attention_config import AttentionConfig

logger = logging.getLogger(__name__)


DECISION_PROMPT_TEMPLATE = """你是一个注意力模式选择助手。当前系统上下文压力较高，需要选择最合适的注意力模式来优化推理效率。

## 当前压力指标
- **压力值**: {pressure_value:.3f}
- **压力趋势**: {pressure_trend}
- **预算使用率**: {budget_usage:.1%}
- **消息数量**: {message_count}

## 当前状态
- **当前注意力模式**: {current_mode}
- **任务上下文**: {task_context}

## 可选注意力模式

1. **GLOBAL** (全局注意)
   - 适用场景: 概览任务、全局规划、多任务协调
   - 特点: 大纲和概览权重高，深层节点递减
   - 保留策略: 优先保留overview、task节点信息

2. **FOCUS** (聚焦模式)
   - 适用场景: 深度推理、单节点分析、问题诊断
   - 特点: 当前节点权重最高(3.0)，祖先链次之(2.0)
   - 保留策略: 聚焦当前节点及其依赖链

3. **SINGLE_CHAIN** (单链推理)
   - 适用场景: 顺序任务、文件操作链、单路径执行
   - 特点: 当前链节点高权重(2.5)，其他链降权(0.5)
   - 保留策略: 保留当前推理链完整上下文

## 近期切换历史
{switch_history}

## 决策要求

请根据当前压力状态和任务上下文，选择最合适的注意力模式。

**返回格式** (必须是合法的JSON):
```json
{{
  "mode": "GLOBAL",
  "reason": "选择理由(简短说明为什么选择这个模式)",
  "confidence": 0.8
}}
```

**注意**:
- mode必须是 GLOBAL、FOCUS 或 SINGLE_CHAIN 之一
- confidence是置信度，范围[0.0, 1.0]，推荐值0.5-0.9
- 请直接返回JSON，不要包含其他文字
"""


class AttentionModeSelector:
    """LLM注意力模式选择决策器
    
    构建决策Prompt，调用LLM，解析决策结果
    """
    
    def __init__(self, config: AttentionConfig, llm_client: Optional[Any] = None):
        """初始化决策器
        
        Args:
            config: 配置对象
            llm_client: LLM客户端实例(可选，后续可设置)
        """
        self._config = config
        self._llm_client = llm_client
        self._mode_mapping = {
            "GLOBAL": "GLOBAL",
            "FOCUS": "FOCUS", 
            "SINGLE_CHAIN": "SINGLE_CHAIN",
            "global": "GLOBAL",
            "focus": "FOCUS",
            "single_chain": "SINGLE_CHAIN",
        }
    
    def set_llm_client(self, llm_client: Any):
        """设置LLM客户端
        
        Args:
            llm_client: LLM客户端实例
        """
        self._llm_client = llm_client
    
    def build_decision_request(
        self,
        pressure_metrics: PressureMetrics,
        current_mode: str,
        task_context: str,
        switch_history: List[SwitchRecord] = None,
    ) -> DecisionRequest:
        """构建LLM决策请求
        
        Args:
            pressure_metrics: 压力指标
            current_mode: 当前注意力模式
            task_context: 任务上下文摘要
            switch_history: 切换历史记录
            
        Returns:
            DecisionRequest决策请求对象
        """
        trend_map = {
            PressureTrend.RISING: "上升中 ↑",
            PressureTrend.STABLE: "稳定 →",
            PressureTrend.FALLING: "下降中 ↓",
        }
        
        history_str = self._format_switch_history(switch_history or [])
        
        prompt = DECISION_PROMPT_TEMPLATE.format(
            pressure_value=pressure_metrics.current_pressure,
            pressure_trend=trend_map.get(pressure_metrics.pressure_trend, "未知"),
            budget_usage=pressure_metrics.budget_usage,
            message_count=pressure_metrics.message_count,
            current_mode=current_mode,
            task_context=task_context[:500] if task_context else "无任务上下文",
            switch_history=history_str,
        )
        
        mode_options = ["GLOBAL", "FOCUS", "SINGLE_CHAIN"]
        
        history_dicts = [r.to_dict() for r in (switch_history or [])[-5:]]
        
        return DecisionRequest(
            pressure_metrics=pressure_metrics,
            current_mode=current_mode,
            task_context=task_context[:500] if task_context else "",
            mode_options=mode_options,
            switch_history=history_dicts,
            prompt=prompt,
        )
    
    def _format_switch_history(self, history: List[SwitchRecord]) -> str:
        """格式化切换历史为文本
        
        Args:
            history: 切换历史列表
            
        Returns:
            格式化的历史文本
        """
        if not history:
            return "无近期切换记录"
        
        lines = []
        for record in history[-5:]:
            time_str = record.timestamp.strftime("%H:%M:%S")
            lines.append(
                f"- [{time_str}] {record.old_mode} → {record.new_mode} "
                f"(原因: {record.reason[:30] if record.reason else 'N/A'})"
            )
        
        return "\n".join(lines)
    
    async def call_llm_decision(
        self,
        request: DecisionRequest,
        timeout_ms: Optional[int] = None,
    ) -> DecisionResponse:
        """调用LLM进行决策
        
        Args:
            request: 决策请求
            timeout_ms: 超时时间(毫秒)，默认使用配置值
            
        Returns:
            DecisionResponse决策响应
        """
        if timeout_ms is None:
            timeout_ms = self._config.decision_timeout_ms
        
        if self._llm_client is None:
            logger.warning("[AttentionModeSelector] LLM客户端未设置，返回Fallback响应")
            return DecisionResponse(
                mode=self._config.fallback_mode,
                reason="LLM客户端未初始化，使用Fallback模式",
                confidence=0.5,
                is_fallback=True,
            )
        
        try:
            timeout_sec = timeout_ms / 1000.0
            
            response = await asyncio.wait_for(
                self._call_llm_internal(request.prompt),
                timeout=timeout_sec,
            )
            
            parsed = self.parse_decision_response(response)
            
            if parsed.mode not in ["GLOBAL", "FOCUS", "SINGLE_CHAIN"]:
                logger.warning(f"[AttentionModeSelector] LLM返回无效模式: {parsed.mode}，使用Fallback")
                return DecisionResponse(
                    mode=self._config.fallback_mode,
                    reason=f"LLM返回无效模式({parsed.mode})，Fallback",
                    confidence=0.5,
                    is_fallback=True,
                )
            
            return parsed
            
        except asyncio.TimeoutError:
            logger.warning(f"[AttentionModeSelector] LLM决策超时({timeout_ms}ms)，返回Fallback")
            return DecisionResponse(
                mode=self._config.fallback_mode,
                reason=f"LLM决策超时({timeout_ms}ms)",
                confidence=0.5,
                is_fallback=True,
            )
        except Exception as e:
            logger.error(f"[AttentionModeSelector] LLM调用失败: {e}")
            return DecisionResponse(
                mode=self._config.fallback_mode,
                reason=f"LLM调用失败: {str(e)[:50]}",
                confidence=0.5,
                is_fallback=True,
            )
    
    async def _call_llm_internal(self, prompt: str) -> str:
        """内部LLM调用方法
        
        Args:
            prompt: 完整Prompt
            
        Returns:
            LLM响应文本
        """
        if hasattr(self._llm_client, 'chat'):
            messages = [{"role": "user", "content": prompt}]
            response = await self._llm_client.chat(messages)
            return response
        elif hasattr(self._llm_client, 'generate'):
            response = await self._llm_client.generate(prompt)
            return response
        else:
            raise ValueError("LLM客户端不支持chat或generate方法")
    
    def parse_decision_response(self, response_text: str) -> DecisionResponse:
        """解析LLM决策响应
        
        Args:
            response_text: LLM返回的文本
            
        Returns:
            DecisionResponse决策响应对象
        """
        response_text = response_text.strip()
        
        try:
            json_match = re.search(r'\{[^{}]*"mode"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                mode = self._mode_mapping.get(data.get("mode", ""), "FOCUS")
                reason = data.get("reason", "LLM决策")
                confidence = float(data.get("confidence", 0.5))
                
                return DecisionResponse(
                    mode=mode,
                    reason=reason,
                    confidence=confidence,
                    is_fallback=False,
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[AttentionModeSelector] JSON解析失败: {e}")
        
        mode_patterns = [
            (r'\bGLOBAL\b', 'GLOBAL'),
            (r'\bFOCUS\b', 'FOCUS'),
            (r'\bSINGLE.?CHAIN\b', 'SINGLE_CHAIN'),
        ]
        
        for pattern, mode in mode_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                logger.info(f"[AttentionModeSelector] 通过正则匹配到模式: {mode}")
                return DecisionResponse(
                    mode=mode,
                    reason="通过正则匹配提取",
                    confidence=0.6,
                    is_fallback=False,
                )
        
        logger.warning("[AttentionModeSelector] 无法解析LLM响应，返回默认模式")
        return DecisionResponse(
            mode=self._config.fallback_mode,
            reason="无法解析LLM响应",
            confidence=0.5,
            is_fallback=True,
        )
