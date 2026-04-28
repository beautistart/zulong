# File: tests/debug_layer3_confidence.py
"""
Layer 3 置信度低问题排查工具

根据提供的排查清单，系统性检查以下环节：
1. 输出解码错误（量化模型归一化）
2. 模型能力与任务匹配度
3. 数据质量问题
4. 后处理策略问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("L3_Diagnosis")

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def check_model_output_normalization():
    """
    检查 1: 模型输出归一化
    
    MobileNetV3 如果是量化模型 (uint8)，需要除以 255.0 得到 0-1 的置信度
    """
    print_section("检查 1: 模型输出归一化")
    
    from zulong.l1c.action_classifier import MobileNetV4_TSM
    
    # 创建分类器
    config = {
        'slow_fps': 8,
        'fast_fps': 30,
        'slow_frame_interval': 4,
        'num_frames_slow': 8,
        'num_frames_fast': 16,
        'intent_threshold': 0.25,
        'interact_threshold': 0.05,
    }
    
    classifier = MobileNetV4_TSM(config)
    classifier.load_model()
    
    if classifier._model is None:
        print("❌ 模型未加载，无法检查")
        return False
    
    # 检查模型类型
    print("\n[模型信息]")
    print(f"  模型类型：{type(classifier._model).__name__}")
    print(f"  设备：{classifier._device}")
    
    # 创建测试输入
    test_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    test_tensor = torch.from_numpy(test_frame).permute(2, 0, 1).unsqueeze(0).float()
    test_tensor = test_tensor / 255.0  # 归一化到 [0, 1]
    test_tensor = test_tensor.to(classifier._device)
    
    # 执行推理
    with torch.no_grad():
        output = classifier._model(test_tensor)
    
    print(f"\n[输出张量信息]")
    print(f"  输出类型：{type(output)}")
    print(f"  输出 dtype: {output.dtype}")
    print(f"  输出形状：{output.shape}")
    print(f"  输出范围：[{output.min():.4f}, {output.max():.4f}]")
    print(f"  输出均值：{output.mean():.4f}")
    print(f"  输出标准差：{output.std():.4f}")
    
    # 检查是否需要归一化
    if output.dtype == torch.uint8:
        print("\n⚠️  警告：模型输出是 uint8 类型 (0-255)")
        print("   需要除以 255.0 进行归一化")
        
        # 检查当前代码是否处理了归一化
        print("\n[检查代码]")
        print("   请查看 action_classifier.py 中的 _slow_pathway 和 _fast_pathway 方法")
        print("   确认是否对输出进行了归一化处理")
        
        return False
    elif output.dtype in [torch.float32, torch.float64]:
        print("\n✅ 模型输出是浮点类型，无需额外归一化")
        print(f"   输出值范围：[{output.min():.4f}, {output.max():.4f}]")
        
        # 检查输出值是否合理
        if output.max() > 10.0:
            print("⚠️  警告：输出值较大，可能需要 Softmax 或 Sigmoid 激活")
        elif output.max() < 0.1:
            print("⚠️  警告：输出值非常小，可能是量化模型未正确解码")
        
        return True
    else:
        print(f"\n⚠️  未知输出类型：{output.dtype}")
        return False

def check_model_architecture():
    """
    检查 2: 模型架构与任务匹配度
    
    MobileNetV3 是轻量级模型，可能不适合复杂动作识别
    """
    print_section("检查 2: 模型架构与任务匹配度")
    
    from zulong.l1c.action_classifier import MobileNetV4_TSM
    
    config = {
        'slow_fps': 8,
        'fast_fps': 30,
        'slow_frame_interval': 4,
        'num_frames_slow': 8,
        'num_frames_fast': 16,
        'intent_threshold': 0.25,
        'interact_threshold': 0.05,
    }
    
    classifier = MobileNetV4_TSM(config)
    classifier.load_model()
    
    if classifier._model is None:
        print("❌ 模型未加载")
        return
    
    # 统计模型参数
    total_params = sum(p.numel() for p in classifier._model.parameters())
    trainable_params = sum(p.numel() for p in classifier._model.parameters() if p.requires_grad)
    
    print(f"\n[模型参数]")
    print(f"  总参数：{total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数：{trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 分析
    print(f"\n[架构分析]")
    print(f"  MobileNetV3-Large 参数量：~5.4M")
    print(f"  当前模型参数量：{total_params/1e6:.2f}M")
    
    if total_params < 10e6:
        print("  ✅ 轻量级模型，适合实时推理")
        print("  ⚠️  但可能不足以捕捉复杂动作特征")
    else:
        print("  ✅ 中等规模模型，适合动作识别")
    
    # 检查特征维度
    print(f"\n[特征维度]")
    print(f"  MobileNetV3 输出特征：1280 维")
    print(f"  融合后特征：2560 维 (Slow 1280 + Fast 1280)")
    
    # 建议
    print(f"\n[建议]")
    print(f"  如果置信度持续偏低，可以考虑:")
    print(f"    1. 使用更强大的 backbone (如 ResNet-18/34)")
    print(f"    2. 增加训练数据量")
    print(f"    3. 优化后处理策略")

def check_feature_statistics():
    """
    检查 3: 特征统计分析
    
    分析 Slow 和 Fast 流的特征分布
    """
    print_section("检查 3: 特征统计分析")
    
    from zulong.l1c.action_classifier import MobileNetV4_TSM
    
    config = {
        'slow_fps': 8,
        'fast_fps': 30,
        'slow_frame_interval': 4,
        'num_frames_slow': 8,
        'num_frames_fast': 16,
        'intent_threshold': 0.25,
        'interact_threshold': 0.05,
    }
    
    classifier = MobileNetV4_TSM(config)
    classifier.load_model()
    
    if classifier._model is None:
        print("❌ 模型未加载")
        return
    
    # 创建测试视频序列
    frames = []
    for i in range(16):
        # 模拟简单运动：从左到右的移动
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        x = int(i * 10) % 224
        cv2.circle(frame, (x, 112), 30, (0, 255, 0), -1)
        frames.append(frame)
    
    # 添加到缓冲区
    for i, frame in enumerate(frames):
        timestamp = i / 30.0
        classifier.add_frame(frame, timestamp)
    
    # 执行分类
    score, intent_type, details = classifier.classify_action()
    
    print(f"\n[测试结果]")
    print(f"  意图分数：{score:.4f}")
    print(f"  意图类型：{intent_type}")
    print(f"  详细特征：{details}")
    
    # 分析
    print(f"\n[分析]")
    if score < 0.3:
        print(f"  ⚠️  置信度较低 (< 0.3)")
        print(f"     可能原因:")
        print(f"       1. 特征能量 (feature_energy) 较低")
        print(f"       2. 特征差异 (feature_diff) 不明显")
        print(f"       3. 规则分类阈值设置不合理")
    elif score < 0.6:
        print(f"  ✅ 置信度中等 (0.3-0.6)")
        print(f"     模型工作正常，但可能不够自信")
    else:
        print(f"  ✅ 置信度良好 (> 0.6)")
    
    # 检查规则分类阈值
    print(f"\n[规则分类阈值检查]")
    print(f"  当前阈值设置:")
    print(f"    high_energy_threshold: 0.05 (挥手)")
    print(f"    low_energy_threshold: 0.02 (靠近/注视)")
    print(f"    high_diff_threshold: 0.015 (挥手)")
    print(f"    low_diff_threshold: 0.008 (注视)")
    
    if details:
        feature_energy = details.get('feature_energy', 0)
        feature_diff = details.get('feature_diff', 0)
        
        print(f"\n  当前特征:")
        print(f"    feature_energy: {feature_energy:.6f}")
        print(f"    feature_diff: {feature_diff:.6f}")
        
        # 判断是否落入某个类别
        if feature_energy > 0.05 and feature_diff > 0.015:
            print(f"    → 分类为：WAVING (挥手)")
        elif feature_energy > 0.02 and feature_diff <= 0.015:
            print(f"    → 分类为：APPROACHING (靠近)")
        elif feature_energy <= 0.02 and feature_diff > 0.008:
            print(f"    → 分类为：GAZING (注视)")
        else:
            print(f"    → 分类为：STILL (静止)")

def check_data_quality():
    """
    检查 4: 数据质量
    
    检查输入数据的分布和质量
    """
    print_section("检查 4: 输入数据质量")
    
    # 创建测试数据
    test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    print(f"\n[测试帧信息]")
    print(f"  形状：{test_frame.shape}")
    print(f"  数据类型：{test_frame.dtype}")
    print(f"  像素范围：[{test_frame.min()}, {test_frame.max()}]")
    print(f"  均值：{test_frame.mean():.2f}")
    print(f"  标准差：{test_frame.std():.2f}")
    
    # 检查预处理
    print(f"\n[预处理检查]")
    resized = cv2.resize(test_frame, (224, 224))
    normalized = resized.astype(np.float32) / 255.0
    
    print(f"  Resize 后：{resized.shape}, 范围：[{resized.min()}, {resized.max()}]")
    print(f"  Normalize 后：{normalized.shape}, 范围：[{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # 检查
    if normalized.min() >= 0.0 and normalized.max() <= 1.0:
        print(f"  ✅ 归一化正确")
    else:
        print(f"  ❌ 归一化错误！范围应该是 [0, 1]")
    
    # 建议
    print(f"\n[数据质量建议]")
    print(f"  1. 确保输入图像清晰，无模糊")
    print(f"  2. 确保光线充足，对比度适中")
    print(f"  3. 确保动作幅度足够大，便于检测")
    print(f"  4. 避免背景过于复杂或杂乱")

def check_post_processing():
    """
    检查 5: 后处理策略
    
    检查从模型输出到最终结果的每一步处理
    """
    print_section("检查 5: 后处理策略")
    
    print(f"\n[后处理流程检查]")
    print(f"  1. 模型输出 (float32)")
    print(f"  2. 特征提取 (Slow + Fast)")
    print(f"  3. 特征融合 (concatenate)")
    print(f"  4. 特征统计 (energy, diff)")
    print(f"  5. 规则分类 (if-else)")
    print(f"  6. 分数计算 (基于特征)")
    
    print(f"\n[潜在问题点]")
    print(f"  1. ❓ 特征能量计算是否正确？")
    print(f"     当前：feature_energy = np.mean(np.abs(features))")
    print(f"  2. ❓ 特征差异计算是否合理？")
    print(f"     当前：feature_diff = np.mean(np.abs(fast_feat - slow_feat))")
    print(f"  3. ❓ 规则分类阈值是否合适？")
    print(f"     需要根据实测数据调整")
    print(f"  4. ❓ 分数计算公式是否合理？")
    print(f"     当前：score = min(1.0, 0.75 + (energy + diff) * 0.25)")
    
    print(f"\n[建议]")
    print(f"  1. 打印中间变量，检查数值范围")
    print(f"  2. 使用实际测试数据，统计特征分布")
    print(f"  3. 根据分布调整阈值和分数计算公式")

def main():
    """主诊断流程"""
    print("=" * 80)
    print(" Layer 3 置信度低问题排查工具")
    print("=" * 80)
    
    # 执行所有检查
    check_model_output_normalization()
    check_model_architecture()
    check_feature_statistics()
    check_data_quality()
    check_post_processing()
    
    # 总结
    print_section("诊断总结")
    
    print("\n[排查清单]")
    print("  ✅ 1. 检查模型输出归一化")
    print("  ✅ 2. 检查模型架构与任务匹配度")
    print("  ✅ 3. 检查特征统计分析")
    print("  ✅ 4. 检查输入数据质量")
    print("  ✅ 5. 检查后处理策略")
    
    print("\n[下一步建议]")
    print("  1. 运行实际测试，收集 L3 置信度数据")
    print("  2. 分析特征分布，调整规则分类阈值")
    print("  3. 如果置信度仍然偏低，考虑:")
    print("     - 使用更强大的 backbone (ResNet)")
    print("     - 增加训练数据")
    print("     - 优化特征融合策略")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
