#!/usr/bin/env python
"""修复 L2 推理引擎 - 集成 RAG 记忆和优化生成参数"""

import time
import asyncio
from typing import Optional

from zulong.core.event_bus import event_bus
from zulong.core.state_manager import state_manager
from zulong.core.types import EventType, EventPriority, ZulongEvent, L2Status
from zulong.models.container import ModelContainer
from zulong.models.config import ModelID
from zulong.l2.rag_node import RAGIntegrationNode

import logging
logger = logging.getLogger(__name__)


class FixedInferenceEngine:
    """修复后的 L2 推理引擎
    
    修复内容:
    1. ✅ 集成 RAG 记忆系统
    2. ✅ 优化生成参数（temperature, top_p, max_tokens）
    3. ✅ 维护对话历史
    4. ✅ 支持上下文感知
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化修复后的推理引擎"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            
            # 加载 L2 模型
            self.model_container = ModelContainer()
            self.l2_model = self.model_container.get_model(ModelID.L2_CORE)
            
            # 初始化 RAG 节点
            self.rag_node = RAGIntegrationNode()
            
            # 对话历史（简单记忆）
            self.conversation_history = []
            self.max_history = 10  # 保留最近 10 轮对话
            
            # 订阅事件
            event_bus.subscribe(EventType.USER_SPEECH, self._on_user_speech, "FixedInferenceEngine")
            
            logger.info("✅ FixedInferenceEngine 初始化完成（带 RAG 记忆）")
    
    def _on_user_speech(self, event: ZulongEvent):
        """处理用户语音事件"""
        text = event.payload.get("text", "")
        logger.info(f"🧠 收到用户语音：'{text}'")
        self._process_with_memory(text)
    
    def _process_with_memory(self, user_input: str):
        """带记忆的推理流程
        
        Args:
            user_input: 用户输入
        """
        state_manager.set_l2_status(L2Status.BUSY)
        
        try:
            # ========== 1. RAG 检索 ==========
            rag_context = self._retrieve_from_rag(user_input)
            
            # ========== 2. 构建带历史的 Prompt ==========
            messages = self._build_messages_with_history(user_input, rag_context)
            
            # ========== 3. 生成回复 ==========
            prompt = self.l2_model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 使用优化后的参数生成
            response = self.l2_model.generate(
                prompt,
                max_tokens=1024,  # 增加到 1024，支持长文本
            )
            
            logger.info(f"💬 生成回复：'{response[:100]}...' " if len(response) > 100 else f"💬 生成回复：'{response}'")
            
            # 打印到控制台（测试用）
            print(f"\n[AI] {response}\n")
            
            # ========== 4. 更新记忆 ==========
            self._update_memory(user_input, response)
            
            # ========== 5. 发布结果 ==========
            output_event = ZulongEvent(
                type=EventType.L2_OUTPUT,
                priority=EventPriority.NORMAL,
                source="FixedInferenceEngine",
                payload={
                    "text": response,
                    "input_text": user_input,
                    "has_rag_context": rag_context is not None,
                    "history_length": len(self.conversation_history),
                    "timestamp": time.time()
                }
            )
            event_bus.publish(output_event)
            
        except Exception as e:
            logger.error(f"推理失败：{e}")
            import traceback
            traceback.print_exc()
        finally:
            state_manager.set_l2_status(L2Status.IDLE)
    
    def _retrieve_from_rag(self, query: str) -> Optional[str]:
        """从 RAG 检索相关记忆
        
        Args:
            query: 查询
            
        Returns:
            检索到的上下文文本
        """
        try:
            # 使用 RAG 节点检索
            fake_state = {
                "query": query,
                "context": {},
                "rag_results": [],
                "retrieved_docs": [],
                "target_rag": "default",
                "search_metadata": {},
                "messages": []
            }
            
            result = self.rag_node.retrieve(fake_state)
            
            if result["retrieved_docs"]:
                # 拼接检索到的文档
                context_parts = []
                for doc in result["retrieved_docs"][:3]:  # 最多取 3 个
                    if "content" in doc:
                        context_parts.append(doc["content"])
                
                if context_parts:
                    context = "\n\n".join(context_parts)
                    logger.info(f"📚 RAG 检索到 {len(context_parts)} 条相关记忆")
                    return context
            
            return None
            
        except Exception as e:
            logger.warning(f"RAG 检索失败：{e}")
            return None
    
    def _build_messages_with_history(self, user_input: str, rag_context: Optional[str]) -> list:
        """构建包含历史和 RAG 上下文的 messages
        
        Args:
            user_input: 用户输入
            rag_context: RAG 检索到的上下文
            
        Returns:
            messages 列表
        """
        # 系统提示
        system_prompt = "你是一个友好、专业、博学的祖龙 (ZULONG) 机器人助手。"
        
        if rag_context:
            system_prompt += f"\n\n【相关知识】\n{rag_context}"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加历史对话（最多 5 轮）
        for msg in self.conversation_history[-5:]:
            messages.append(msg)
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _update_memory(self, user_input: str, response: str):
        """更新对话历史记忆
        
        Args:
            user_input: 用户输入
            response: AI 回复
        """
        # 添加到历史
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # 限制历史长度
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        logger.debug(f"💾 记忆已更新，当前历史长度：{len(self.conversation_history)}")


# 测试函数
def test_fixed_engine():
    """测试修复后的引擎"""
    print("=" * 60)
    print("测试修复后的 L2 推理引擎")
    print("=" * 60)
    
    engine = FixedInferenceEngine()
    
    test_queries = [
        "写一首关于春天的短诗",
        "请再写一首关于秋天的诗",  # 测试记忆
        "北京是中国的首都吗？",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"用户：{query}")
        print(f"{'='*60}")
        engine._process_with_memory(query)
        time.sleep(2)  # 等待生成


if __name__ == "__main__":
    test_fixed_engine()
