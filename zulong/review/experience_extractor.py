# File: zulong/review/experience_extractor.py
# 经验提取器 - L2 生成结构化数据

from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperienceExtractor:
    """经验提取器
    
    负责：
    - 调用 L2 进行结构化分析
    - 解析 L2 返回的 JSON
    - 验证数据格式
    - 提供给 L1-B 进行安全写入
    """
    
    def __init__(self):
        """初始化提取器"""
        logger.info("[ExperienceExtractor] 初始化完成")
    
    async def extract_from_buffer(self, buffer_data: Dict[str, Any], deep: bool = False) -> Dict[str, Any]:
        """从缓冲区数据中提取经验
        
        Args:
            buffer_data: 缓冲区导出的数据
            deep: 是否深度分析
            
        Returns:
            Dict: 结构化的经验数据
        """
        try:
            # 1. 构建 L2 的系统级指令
            prompt = self._build_system_prompt(buffer_data, deep)
            
            # 2. 调用 L2 进行分析
            l2_response = await self._call_l2_for_analysis(prompt)
            
            # 3. 解析 L2 返回的 JSON
            structured_data = self._parse_l2_response(l2_response)
            
            # 4. 验证数据格式
            self._validate_structure(structured_data)
            
            logger.info(f"[ExperienceExtractor] 成功提取 {len(structured_data.get('experiences', []))} 条经验")
            return structured_data
            
        except Exception as e:
            logger.error(f"[ExperienceExtractor] 提取经验失败：{e}", exc_info=True)
            # 返回空结构，避免系统崩溃
            return self._empty_structure()
    
    def _build_system_prompt(self, buffer_data: Dict[str, Any], deep: bool = False) -> str:
        """构建给 L2 的系统级指令
        
        Args:
            buffer_data: 缓冲区数据
            deep: 是否深度分析
            
        Returns:
            str: 系统指令
        """
        conversations = buffer_data.get('conversations', [])
        duration = buffer_data.get('duration_seconds', 0)
        
        if deep:
            prompt = (
                f"📋 **复盘分析任务**\n\n"
                f"请对以下对话进行深度分析，并严格按照 JSON 格式输出：\n\n"
                f"对话时长：{duration:.1f}秒\n"
                f"对话轮数：{len(conversations)}轮\n\n"
                f"对话内容：\n"
            )
            
            for i, conv in enumerate(conversations, 1):
                prompt += f"\n--- 第{i}轮 ---\n"
                prompt += f"用户：{conv.get('user', '')}\n"
                prompt += f"系统：{conv.get('system', '')}\n"
            
            prompt += (
                f"\n\n请分析并生成以下 JSON 格式：\n"
                f"```json\n"
                f"{{\n"
                f'  "summary": "对话整体总结（100 字以内）",\n'
                f'  "experiences": [\n'
                f'    {{\n'
                f'      "type": "decision|improvement|lesson|best_practice",\n'
                f'      "content": "经验内容（简洁明确）",\n'
                f'      "confidence": 0.0-1.0,\n'
                f'      "tags": ["标签 1", "标签 2"],\n'
                f'      "evidence": "支撑该经验的对话原文引用"\n'
                f'    }}\n'
                f'  ],\n'
                f'  "key_points": ["关键知识点 1", "关键知识点 2"],\n'
                f'  "suggested_tags": ["建议标签 1", "建议标签 2"],\n'
                f'  "action_items": ["后续行动建议 1", "后续行动建议 2"]\n'
                f'}}\n'
                f"```\n\n"
                f"要求：\n"
                f"1. 经验必须是可复用的、通用的原则\n"
                f"2. 避免空洞的套话，要具体明确\n"
                f"3. 置信度基于证据强度评估\n"
                f"4. 标签要简洁，便于检索\n"
            )
        else:
            prompt = (
                f"📝 **快速复盘任务**\n\n"
                f"请快速总结以下对话并提取经验：\n\n"
            )
            
            for conv in conversations[-3:]:  # 只看最近 3 轮
                prompt += f"用户：{conv.get('user', '')}\n"
                prompt += f"系统：{conv.get('system', '')}\n\n"
            
            prompt += (
                f"请生成简化的 JSON：\n"
                f"```json\n"
                f"{{\n"
                f'  "summary": "一句话总结",\n'
                f'  "experiences": [\n'
                f'    {{\n'
                f'      "type": "general",\n'
                f'      "content": "经验内容",\n'
                f'      "confidence": 0.8,\n'
                f'      "tags": ["标签"]\n'
                f'    }}\n'
                f'  ],\n'
                f'  "suggested_tags": ["标签"]\n'
                f'}}\n'
                f"```\n"
            )
        
        return prompt
    
    async def _call_l2_for_analysis(self, prompt: str) -> str:
        """调用 L2 进行分析
        
        通过 vLLM OpenAI API 调用真实 L2 模型进行经验提取分析。
        
        Args:
            prompt: 提示词
            
        Returns:
            str: L2 的响应文本
        """
        logger.info(f"[ExperienceExtractor] 调用 L2 进行分析（真实模型）...")
        
        try:
            # 复用系统已有的 vLLM 配置
            import os
            vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
            vllm_model_id = "/mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-0.8B-AWQ"
            
            try:
                from openai import OpenAI
            except ImportError:
                logger.error("[ExperienceExtractor] openai SDK 未安装，无法调用 L2")
                return json.dumps(self._empty_structure(), ensure_ascii=False)
            
            client = OpenAI(base_url=vllm_base_url, api_key="EMPTY")
            
            messages = [
                {"role": "system", "content": "你是一个专业的经验分析助手。请严格按照用户要求的 JSON 格式输出分析结果，不要输出其他内容。"},
                {"role": "user", "content": prompt}
            ]
            
            import concurrent.futures
            
            def call_vllm():
                return client.chat.completions.create(
                    model=vllm_model_id,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.3,
                    top_p=0.85,
                    extra_body={"repetition_penalty": 1.2},
                    stream=False
                )
            
            # 使用线程池避免阻塞事件循环，30秒超时
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(call_vllm)
                try:
                    response = future.result(timeout=30)
                except concurrent.futures.TimeoutError:
                    logger.error("[ExperienceExtractor] L2 调用超时 (>30秒)")
                    return json.dumps(self._empty_structure(), ensure_ascii=False)
            
            response_text = response.choices[0].message.content
            logger.info(f"[ExperienceExtractor] L2 返回 {len(response_text)} 字符")
            return response_text
            
        except Exception as e:
            logger.error(f"[ExperienceExtractor] L2 调用失败：{e}", exc_info=True)
            return json.dumps(self._empty_structure(), ensure_ascii=False)
    
    def _parse_l2_response(self, response: str) -> Dict[str, Any]:
        """解析 L2 返回的 JSON
        
        Args:
            response: L2 响应文本
            
        Returns:
            Dict: 解析后的结构化数据
        """
        try:
            # 尝试直接解析 JSON
            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            # 如果包含 Markdown 代码块，提取 JSON 部分
            import re
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            else:
                # 尝试直接解析整个响应
                return json.loads(response.strip())
    
    def _validate_structure(self, data: Dict[str, Any]) -> bool:
        """验证数据结构
        
        Args:
            data: 待验证的数据
            
        Returns:
            bool: 是否有效
            
        Raises:
            ValueError: 结构无效时抛出
        """
        # 必需字段
        required_fields = ['summary', 'experiences']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"缺少必需字段：{field}")
        
        # 验证 experiences 数组
        experiences = data.get('experiences', [])
        if not isinstance(experiences, list):
            raise ValueError("experiences 必须是数组")
        
        for exp in experiences:
            if not isinstance(exp, dict):
                raise ValueError("每条经验必须是对象")
            
            if 'content' not in exp:
                raise ValueError("经验必须包含 content 字段")
            
            if 'confidence' in exp:
                confidence = exp['confidence']
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                    raise ValueError("confidence 必须是 0-1 之间的数值")
        
        return True
    
    def _empty_structure(self) -> Dict[str, Any]:
        """返回空结构
        
        Returns:
            Dict: 空结构
        """
        return {
            "summary": "暂无总结",
            "experiences": [],
            "key_points": [],
            "suggested_tags": [],
            "action_items": []
        }


# 全局单例
experience_extractor = ExperienceExtractor()


def get_experience_extractor() -> ExperienceExtractor:
    """获取经验提取器实例
    
    Returns:
        ExperienceExtractor: 实例
    """
    return experience_extractor
