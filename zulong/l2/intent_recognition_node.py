# File: zulong/l2/intent_recognition_node.py
# 意图识别节点，利用 L2 模型将自然语言转换为结构化指令标签

from zulong.models.container import ModelContainer
from zulong.models.config import ModelID
from zulong.l2.intent_schema import IntentResult, SUPPORTED_INTENTS
import json
import re


class IntentRecognitionNode:
    """意图识别节点"""
    
    def __init__(self):
        """初始化意图识别节点"""
        self.model_container = ModelContainer()
        self.l2_model = self.model_container.get_model(ModelID.L2_GATEKEEPER)
    
    def recognize_intent(self, text: str) -> IntentResult:
        """识别意图
        
        Args:
            text: 自然语言文本
            
        Returns:
            IntentResult: 意图识别结果
        """
        try:
            # 构建 prompt，要求模型输出严格的 JSON 格式
            prompt = f'''
你是一个机器人意图分类器。
分析用户输入并只输出一个有效的 JSON 对象，包含以下键："intent", "confidence", "parameters"。
不要输出任何解释、Markdown、思考过程或额外内容。
不要输出任何 <think>、</think> 或类似的标签。
不要输出任何 "请继续" 或类似的提示。
只输出一个完整的 JSON 对象，不要重复输出。
支持的意图：MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP, QUERY_STATUS, UNKNOWN。

用户输入："{text}"
JSON 输出：
'''
            
            # 生成回复
            response = self.l2_model.generate(prompt, max_tokens=200)
            
            # 解析 JSON 输出
            try:
                # 打印原始输出以便调试
                print(f"[L2] [Model] Raw response: {response}")
                
                # 清理响应文本
                cleaned_response = response.strip()
                
                # 移除 Markdown 标记
                cleaned_response = re.sub(r'```json|```', '', cleaned_response)
                
                # 移除思考内容
                cleaned_response = re.sub(r'<think>[\s\S]*?</think>', '', cleaned_response)
                
                # 分割可能的多个 JSON 对象
                json_candidates = []
                brace_count = 0
                start_idx = 0
                
                for i, char in enumerate(cleaned_response):
                    if char == '{':
                        if brace_count == 0:
                            start_idx = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and i > start_idx:
                            json_candidates.append(cleaned_response[start_idx:i+1])
                
                # 尝试解析每个候选 JSON
                for candidate in json_candidates:
                    try:
                        # 清理 JSON 字符串
                        json_str = candidate.strip().replace('\n', '').replace('\r', '')
                        result = json.loads(json_str)
                        
                        # 处理参数类型
                        parameters = result.get('parameters', {})
                        if not isinstance(parameters, dict):
                            # 将非字典类型转换为字典
                            parameters = {"value": parameters}
                        
                        # 验证结果
                        if result.get('intent') in SUPPORTED_INTENTS:
                            return IntentResult(
                                intent=result['intent'],
                                confidence=result.get('confidence', 0.0),
                                parameters=parameters,
                                original_text=text
                            )
                    except Exception:
                        continue
            except Exception as e:
                print(f"[IntentRecognitionNode] JSON 解析失败: {e}")
            
            # 如果解析失败，返回 UNKNOWN 意图
            return IntentResult(
                intent="UNKNOWN",
                confidence=0.0,
                parameters={},
                original_text=text
            )
        except Exception as e:
            print(f"[IntentRecognitionNode] 意图识别失败: {e}")
            # 返回 UNKNOWN 意图
            return IntentResult(
                intent="UNKNOWN",
                confidence=0.0,
                parameters={},
                original_text=text
            )
