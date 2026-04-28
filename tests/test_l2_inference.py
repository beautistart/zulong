# File: tests/test_l2_inference.py
"""
测试 L2 模型推理功能
"""
import sys
import os
import time

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zulong.models.container import ModelContainer
from zulong.models.config import ModelID

def test_l2_inference():
    """测试 L2 模型推理"""
    print("\n" + "=" * 60)
    print("测试 L2 模型推理功能")
    print("=" * 60)
    
    try:
        # 1. 加载模型容器
        print("\n1. 加载模型容器...")
        container = ModelContainer()
        
        # 2. 获取 L2 模型
        print("\n2. 获取 L2_CORE 模型...")
        if ModelID.L2_CORE not in container.resident_models:
            print("❌ L2_CORE 未加载")
            return
        
        l2_model = container.resident_models[ModelID.L2_CORE]
        print(f"✅ L2_CORE 已加载：{type(l2_model)}")
        
        # 3. 测试推理
        print("\n3. 测试推理...")
        test_input = "你好，你是谁？"
        print(f"用户输入：{test_input}")
        
        # 构建 prompt
        messages = [
            {"role": "user", "content": test_input}
        ]
        
        prompt = l2_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print(f"\nPrompt 长度：{len(prompt)} 字符")
        print(f"Prompt 前 100 字符：{prompt[:100]}...")
        
        # 生成回复
        print(f"\n开始生成回复...")
        start_time = time.time()
        
        response = l2_model.generate(
            prompt,
            max_tokens=100,  # 限制生成长度
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # 提取回复
        if "</think>" in response:
            parts = response.split("</think>")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        print(f"\n✅ 生成完成！耗时：{generation_time:.2f}秒")
        print(f"\n回复内容：{response[:200]}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_l2_inference()
