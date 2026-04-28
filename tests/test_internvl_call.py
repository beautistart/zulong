#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 InternVL 模型的正确调用方式

目的：
- 验证图像格式 (PIL Image vs Tensor)
- 验证图像尺寸 (是否需要 resize)
- 验证 prompt 格式
- 验证 processor 用法
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from zulong.models.container import ModelContainer
from zulong.models.config import ModelID


def test_internvl_basic():
    """测试 1: 基础调用 (使用纯色图像)"""
    print("=" * 80)
    print("🧪 测试 1: InternVL 基础调用")
    print("=" * 80)
    
    try:
        # 创建模型容器
        container = ModelContainer()
        
        # 获取 InternVL 模型
        model = container.get_model(ModelID.L1_PERCEPTION)
        
        if model is None:
            print("❌ 模型未加载")
            return False
        
        print("✅ 模型已加载")
        
        # 创建测试图像 (红色方块)
        image = Image.new('RGB', (224, 224), color='red')
        
        # 测试 prompt
        prompt = "请描述这张图片的内容。"
        
        print(f"📷 图像尺寸：{image.size}")
        print(f"📝 Prompt: {prompt}")
        print("🤖 开始生成...")
        
        # 调用生成
        response = model.generate(prompt=prompt, image=image, max_tokens=100)
        
        print(f"\n💬 回答:\n{response}")
        
        # 验证回答是否包含有效内容
        if "失败" in response or "error" in response.lower():
            print("\n❌ 生成失败")
            return False
        else:
            print("\n✅ 生成成功")
            return True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_internvl_with_cv2_image():
    """测试 2: 使用 OpenCV 图像转换"""
    print("\n" + "=" * 80)
    print("🧪 测试 2: OpenCV 图像转换")
    print("=" * 80)
    
    try:
        container = ModelContainer()
        model = container.get_model(ModelID.L1_PERCEPTION)
        
        if model is None:
            print("❌ 模型未加载")
            return False
        
        # 创建 OpenCV 图像 (BGR)
        cv2_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2_image[:, :] = [255, 0, 0]  # 蓝色背景 (BGR)
        
        # 添加一些内容 (白色矩形)
        cv2.rectangle(cv2_image, (200, 150), (440, 330), (255, 255, 255), -1)
        
        # 转换为 RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        print(f"📷 原始尺寸：{cv2_image.shape}")
        print(f"📷 PIL 尺寸：{pil_image.size}")
        
        prompt = "这张图片里有什么？请描述颜色和形状。"
        print(f"📝 Prompt: {prompt}")
        print("🤖 开始生成...")
        
        response = model.generate(prompt=prompt, image=pil_image, max_tokens=100)
        
        print(f"\n💬 回答:\n{response}")
        
        if "失败" in response or "error" in response.lower():
            print("\n❌ 生成失败")
            return False
        else:
            print("\n✅ 生成成功")
            return True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_internvl_with_resize():
    """测试 3: 调整图像尺寸到 448x448 (InternVL 推荐尺寸)"""
    print("\n" + "=" * 80)
    print("🧪 测试 3: 调整图像尺寸到 448x448")
    print("=" * 80)
    
    try:
        container = ModelContainer()
        model = container.get_model(ModelID.L1_PERCEPTION)
        
        if model is None:
            print("❌ 模型未加载")
            return False
        
        # 创建测试图像
        cv2_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2_image[:, :] = [0, 255, 0]  # 绿色背景 (BGR)
        
        # 添加圆形
        cv2.circle(cv2_image, (320, 240), 100, (255, 255, 0), -1)
        
        # 转换为 RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸到 448x448
        resized = cv2.resize(rgb_image, (448, 448))
        pil_image = Image.fromarray(resized)
        
        print(f"📷 原始尺寸：{rgb_image.shape}")
        print(f"📷 调整后尺寸：{pil_image.size}")
        
        prompt = "请描述这张图片。"
        print(f"📝 Prompt: {prompt}")
        print("🤖 开始生成...")
        
        response = model.generate(prompt=prompt, image=pil_image, max_tokens=100)
        
        print(f"\n💬 回答:\n{response}")
        
        if "失败" in response or "error" in response.lower():
            print("\n❌ 生成失败")
            return False
        else:
            print("\n✅ 生成成功")
            return True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_internvl_with_office_scene():
    """测试 4: 模拟办公室场景"""
    print("\n" + "=" * 80)
    print("🧪 测试 4: 办公室场景模拟")
    print("=" * 80)
    
    try:
        container = ModelContainer()
        model = container.get_model(ModelID.L1_PERCEPTION)
        
        if model is None:
            print("❌ 模型未加载")
            return False
        
        # 创建办公室场景
        height, width = 448, 448
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 墙壁 (浅灰色)
        image[:, :] = [200, 200, 200]
        
        # 桌子 (棕色，底部)
        table_y = int(height * 0.6)
        image[table_y:, :] = [101, 67, 33]
        
        # 电脑屏幕 (黑色矩形)
        screen_x = width // 4
        screen_y = int(height * 0.3)
        screen_w = int(width * 0.3)
        screen_h = int(height * 0.25)
        image[screen_y:screen_y+screen_h, screen_x:screen_x+screen_w] = [30, 30, 30]
        
        # 键盘 (灰色小矩形)
        keyboard_y = int(height * 0.55)
        keyboard_x = width // 4
        keyboard_w = int(width * 0.25)
        keyboard_h = int(height * 0.05)
        image[keyboard_y:keyboard_y+keyboard_h, keyboard_x:keyboard_x+keyboard_w] = [50, 50, 50]
        
        # 转换为 PIL
        pil_image = Image.fromarray(image)
        
        print("📷 场景：办公室 (桌子 + 电脑 + 键盘)")
        
        prompt = "请描述这个场景，有哪些物体？"
        print(f"📝 Prompt: {prompt}")
        print("🤖 开始生成...")
        
        response = model.generate(prompt=prompt, image=pil_image, max_tokens=150)
        
        print(f"\n💬 回答:\n{response}")
        
        # 验证是否识别出物体
        keywords = ["桌子", "电脑", "屏幕", "键盘", "办公桌"]
        has_object = any(kw in response for kw in keywords)
        
        if has_object:
            print("\n✅ 成功识别物体")
            return True
        elif "失败" in response or "error" in response.lower():
            print("\n❌ 生成失败")
            return False
        else:
            print("\n⚠️  生成成功但未识别出物体")
            return False
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("🔬 InternVL 调用方式测试")
    print("=" * 80)
    
    test_results = []
    
    # 测试 1: 基础调用
    test_results.append(("基础调用 (红色方块)", test_internvl_basic()))
    
    # 测试 2: OpenCV 转换
    test_results.append(("OpenCV 图像转换", test_internvl_with_cv2_image()))
    
    # 测试 3: 调整尺寸
    test_results.append(("调整到 448x448", test_internvl_with_resize()))
    
    # 测试 4: 办公室场景
    test_results.append(("办公室场景", test_internvl_with_office_scene()))
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("📊 测试摘要")
    print("=" * 80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计：{passed}/{total} 测试通过")
    
    if passed > 0:
        print(f"\n✅ 至少有一个测试通过，InternVL 可以正常工作")
    else:
        print(f"\n❌ 所有测试失败，需要检查 InternVL 配置")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
