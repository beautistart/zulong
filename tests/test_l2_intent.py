# File: tests/test_l2_intent.py
# 意图识别集成测试

import pytest
from zulong.l2.intent_recognition_node import IntentRecognitionNode


@pytest.mark.slow
class TestL2IntentRecognition:
    """测试 L2 意图识别"""
    
    def setup_method(self):
        """设置测试环境"""
        self.node = IntentRecognitionNode()
    
    def test_explicit_command(self):
        """场景 1: 明确指令"""
        print("\n--- 场景 1: 明确指令 ---")
        
        # 输入
        text = "向前走两米"
        print(f"Input: {text}")
        
        # 识别意图
        result = self.node.recognize_intent(text)
        print(f"[Model] Raw Output: {result}")
        print(f"[Parsed] Intent: {result.intent}, Params: {result.parameters}")
        
        # 验证结果
        assert result.intent == "MOVE_FORWARD"
        assert result.confidence > 0.8
        assert "distance" in result.parameters
        assert result.parameters.get("unit") == "meter"
        print("✅ 场景 1 通过")
    
    def test_emergency_stop(self):
        """场景 2: 紧急停止"""
        print("\n--- 场景 2: 紧急停止 ---")
        
        # 测试中文
        text = "停下！"
        print(f"Input: {text}")
        result = self.node.recognize_intent(text)
        print(f"[Model] Raw Output: {result}")
        assert result.intent == "STOP"
        assert result.confidence > 0.9
        
        # 测试英文
        text = "Stop"
        print(f"Input: {text}")
        result = self.node.recognize_intent(text)
        print(f"[Model] Raw Output: {result}")
        assert result.intent == "STOP"
        assert result.confidence > 0.9
        
        print("✅ 场景 2 通过")
    
    def test_fuzzy_chat(self):
        """场景 3: 模糊/闲聊"""
        print("\n--- 场景 3: 模糊闲聊 ---")
        
        text = "今天天气不错"
        print(f"Input: {text}")
        
        result = self.node.recognize_intent(text)
        print(f"[Model] Raw Output: {result}")
        
        # 应该识别为 UNKNOWN 或 QUERY_STATUS
        assert result.intent in ["UNKNOWN", "QUERY_STATUS"]
        print("✅ 场景 3 通过 (正确归类为未知或闲聊)")
    
    def test_robustness(self):
        """场景 4: 鲁棒性测试"""
        print("\n--- 场景 4: 鲁棒性测试 ---")
        
        text = "能不能麻烦你往左边转一下？"
        print(f"Input: {text}")
        
        result = self.node.recognize_intent(text)
        print(f"[Model] Raw Output: {result}")
        
        # 应该识别为 TURN_LEFT
        assert result.intent == "TURN_LEFT"
        print("✅ 场景 4 通过 (正确处理长句)")


if __name__ == "__main__":
    """直接运行测试"""
    print("=== 测试 L2 意图识别 (真实模型) ===")
    
    test = TestL2IntentRecognition()
    test.setup_method()
    
    try:
        test.test_explicit_command()
        test.test_emergency_stop()
        test.test_fuzzy_chat()
        test.test_robustness()
        print("\n==================================================")
        print("🎉 L2 意图识别节点验证完成！")
        print("模型能正确输出结构化 JSON，路由逻辑准备就绪。")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
