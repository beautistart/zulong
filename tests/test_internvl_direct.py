#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接测试 InternVL 官方调用方式

参考：https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat_llama.md
"""

import sys
from pathlib import Path
import torch
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModel, AutoTokenizer, AutoProcessor


def test_internvl_official_way():
    """使用 InternVL 官方推荐的方式"""
    print("=" * 80)
    print("🔬 测试 InternVL 官方调用方式")
    print("=" * 80)
    
    model_path = "./models/OpenGVLab/InternVL2_5-1B"
    
    print(f"\n📂 加载模型：{model_path}")
    
    # 加载模型
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 创建测试图像
    image = Image.new('RGB', (448, 448), color='red')
    
    # 测试问题
    question = "请描述这张图片"
    
    print(f"\n🖼️  图像：红色方块 (448x448)")
    print(f"❓ 问题：{question}")
    print("\n🤖 使用 model.chat() 方法...")
    
    try:
        # 使用 InternVL 的 chat 方法
        response = model.chat(tokenizer, question, image)
        print(f"\n💬 回答:\n{response}")
        
        if response and len(response) > 0:
            print("\n✅ 成功!")
            return True
        else:
            print("\n❌ 回答为空")
            return False
            
    except Exception as e:
        print(f"\n❌ 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_internvl_with_generation_config():
    """使用 InternVL 的 generate 方法 + generation_config"""
    print("\n" + "=" * 80)
    print("🔬 测试 InternVL generate 方法")
    print("=" * 80)
    
    model_path = "./models/OpenGVLab/InternVL2_5-1B"
    
    print(f"\n📂 加载模型：{model_path}")
    
    # 加载模型
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 创建测试图像
    image = Image.new('RGB', (448, 448), color='blue')
    
    # 测试问题
    question = "这是什么颜色？"
    
    print(f"\n🖼️  图像：蓝色方块 (448x448)")
    print(f"❓ 问题：{question}")
    print("\n🤖 使用 processor + generate...")
    
    try:
        # 使用 processor 处理
        inputs = processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(model.device)
        
        print(f"Inputs 键：{list(inputs.keys())}")
        
        # 生成
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100
            )
        
        # 解码
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        print(f"\n💬 回答:\n{response}")
        
        if response and len(response) > 0:
            print("\n✅ 成功!")
            return True
        else:
            print("\n❌ 回答为空")
            return False
            
    except Exception as e:
        print(f"\n❌ 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("InternVL 官方调用方式测试")
    print("=" * 80)
    
    # 测试 1: chat 方法
    result1 = test_internvl_official_way()
    
    # 测试 2: generate 方法
    result2 = test_internvl_with_generation_config()
    
    print("\n" + "=" * 80)
    print("📊 测试摘要")
    print("=" * 80)
    print(f"chat 方法：{'✅' if result1 else '❌'}")
    print(f"generate 方法：{'✅' if result2 else '❌'}")
    print("=" * 80)
