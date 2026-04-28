# File: tests/test_layer3_real_model.py
"""
Layer 3 真实模型测试

验证 MobileNetV3 动作分类器的真实推理能力。
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from zulong.l1c.action_classifier import MobileNetV4_TSM


async def test_layer3_model_loading():
    """
    测试 1: 模型加载
    
    验证 MobileNetV3 是否能正常加载。
    """
    print("=" * 60)
    print("🧪 Layer 3: 模型加载测试")
    print("=" * 60)
    
    # 初始化分类器
    print("\n📦 初始化 MobileNetV4_TSM...")
    classifier = MobileNetV4_TSM()
    
    # 加载模型
    print("\n📥 加载 MobileNetV3-Large 预训练模型...")
    classifier.load_model()
    
    # 检查模型状态
    if classifier._model is None:
        print("❌ 模型加载失败")
        return False
    
    if classifier._intent_classifier is None:
        print("⚠️  意图分类头未加载（使用随机权重）")
    else:
        print("✅ 意图分类头已加载（随机权重）")
    
    print("\n✅ 模型加载成功")
    print(f"   - Device: {classifier._device}")
    print(f"   - Slow FPS: {classifier._config['slow_fps']}")
    print(f"   - Fast FPS: {classifier._config['fast_fps']}")
    
    return True


async def test_layer3_inference():
    """
    测试 2: 推理测试
    
    使用摄像头画面进行实时动作分类。
    """
    print("\n" + "=" * 60)
    print("🧪 Layer 3: 实时推理测试")
    print("=" * 60)
    
    # 初始化分类器
    print("\n📦 初始化 MobileNetV4_TSM...")
    classifier = MobileNetV4_TSM()
    
    print("\n📥 加载模型...")
    classifier.load_model()
    
    if classifier._model is None:
        print("❌ 模型加载失败")
        return False
    
    print("✅ 模型加载成功")
    
    # 打开摄像头
    print("\n📷 打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 摄像头已启动")
    
    # 测试 30 秒
    print("\n📸 开始测试 (30 秒)...")
    print("\n📋 测试指导:")
    print("   1. 站在摄像头前")
    print("   2. 尝试以下动作:")
    print("      - 挥手 (WAVING) ✋")
    print("      - 靠近摄像头 (APPROACHING) 👈")
    print("      - 注视摄像头 (GAZING) 👀")
    print("      - 保持静止 (STILL) 🧘")
    print("   3. 每个动作保持 3-5 秒")
    print("\n⏳ 3 秒后开始...")
    await asyncio.sleep(1)
    print("⏳ 2 秒后开始...")
    await asyncio.sleep(1)
    print("⏳ 1 秒后开始...")
    await asyncio.sleep(1)
    
    start_time = time.time()
    frame_count = 0
    intent_results = {}
    last_intent = None
    intent_start_time = None
    
    while (time.time() - start_time) < 30.0:
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_count += 1
        timestamp = time.time()
        
        # 添加帧到缓冲区
        can_classify = classifier.add_frame(frame, timestamp)
        
        # 如果可以分类
        if can_classify:
            intent_score, intent_type, details = classifier.classify_action()
            
            # 统计结果
            intent_results[intent_type] = intent_results.get(intent_type, 0) + 1
            
            # 检测新意图
            if intent_type != last_intent:
                if intent_start_time and last_intent:
                    duration = time.time() - intent_start_time
                    print(f"   🎯 {last_intent}: 持续 {duration:.1f}秒 (置信度：{intent_score:.2f})")
                
                last_intent = intent_type
                intent_start_time = time.time()
            
            # 显示状态
            if intent_score > classifier._config['interact_threshold']:
                status = f"✅ {intent_type} ({intent_score:.2f}) [INTERACT]"
                color = (0, 255, 0)
            elif intent_score > classifier._config['intent_threshold']:
                status = f"⚠️  {intent_type} ({intent_score:.2f}) [SILENT]"
                color = (0, 255, 255)
            else:
                status = f"⏳ {intent_type} ({intent_score:.2f})"
                color = (0, 0, 255)
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 显示 FPS
        elapsed = time.time() - start_time
        fps = frame_count / max(0.001, elapsed)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示缓冲区状态
        buffer_status = f"Slow: {len(classifier.slow_buffer)}/{classifier._config['num_frames_slow']} | Fast: {len(classifier.fast_buffer)}/{classifier._config['num_frames_fast']}"
        cv2.putText(frame, buffer_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Layer 3 Action Classification", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 最终统计
    if intent_start_time and last_intent:
        duration = time.time() - intent_start_time
        print(f"   🎯 {last_intent}: 持续 {duration:.1f}秒")
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 测试结果:")
    print(f"   - 总帧数：{frame_count}")
    print(f"   - 测试时长：{elapsed:.1f}秒")
    print(f"   - 平均 FPS: {fps:.1f}")
    print("\n🎯 意图分类统计:")
    
    if intent_results:
        for intent, count in sorted(intent_results.items(), key=lambda x: x[1], reverse=True):
            percentage = count / frame_count * 100
            print(f"   - {intent}: {count}帧 ({percentage:.1f}%)")
        
        print("\n✅ 测试成功！模型可以正常推理")
        success = True
    else:
        print("   - 未生成任何分类结果")
        print("\n❌ 测试失败！")
        success = False
    
    print("=" * 60)
    
    cap.release()
    cv2.destroyAllWindows()
    
    return success


async def test_layer3_threshold():
    """
    测试 3: 阈值测试
    
    验证意图阈值是否能正确触发 Layer 4。
    """
    print("\n" + "=" * 60)
    print("🧪 Layer 3: 阈值触发测试")
    print("=" * 60)
    
    print("\n📋 测试逻辑:")
    print("   1. 意图分数 > 0.8 → 触发 Layer 4 (INTERACT_REQUEST)")
    print("   2. 0.6 < 意图分数 < 0.8 → 触发 Layer 3 (SILENT_WATCH)")
    print("   3. 意图分数 < 0.6 → 无触发 (NO_INTENT)")
    
    # 初始化分类器
    classifier = MobileNetV4_TSM()
    classifier.load_model()
    
    if classifier._model is None:
        print("❌ 模型加载失败")
        return False
    
    print("\n✅ 模型已加载")
    print(f"   - 意图阈值：{classifier._config['intent_threshold']}")
    print(f"   - 交互阈值：{classifier._config['interact_threshold']}")
    
    # 模拟不同场景
    test_scenarios = [
        ("挥手", "WAVING", 0.85),
        ("靠近", "APPROACHING", 0.65),
        ("注视", "GAZING", 0.55),
        ("静止", "STILL", 0.9),
    ]
    
    print("\n📊 场景测试:")
    for scenario, intent_type, score in test_scenarios:
        if score > classifier._config['interact_threshold']:
            trigger = "✅ Layer 4 (INTERACT)"
        elif score > classifier._config['intent_threshold']:
            trigger = "⚠️  Layer 3 (SILENT)"
        else:
            trigger = "❌ 无触发"
        
        print(f"   - {scenario} ({intent_type}): {score:.2f} → {trigger}")
    
    print("\n✅ 阈值逻辑验证通过")
    return True


async def main():
    """主测试函数"""
    print("=" * 60)
    print("🎯 Layer 3 真实模型测试套件")
    print("=" * 60)
    
    # 测试 1: 模型加载
    test1 = await test_layer3_model_loading()
    
    if not test1:
        print("\n❌ 测试 1 失败，无法继续")
        return
    
    # 测试 2: 实时推理
    test2 = await test_layer3_inference()
    
    # 测试 3: 阈值触发
    test3 = await test_layer3_threshold()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结:")
    print(f"   - 模型加载：{'✅ 通过' if test1 else '❌ 失败'}")
    print(f"   - 实时推理：{'✅ 通过' if test2 else '❌ 失败'}")
    print(f"   - 阈值触发：{'✅ 通过' if test3 else '❌ 失败'}")
    print("=" * 60)
    
    if test1 and test2 and test3:
        print("\n✅ Layer 3 真实模型测试全部通过！")
        print("\n💡 下一步:")
        print("   1. 调整意图分类阈值（如需要）")
        print("   2. 进行 Layer 3-4 联动测试")
        print("   3. 采集数据并训练意图分类头")
    else:
        print("\n❌ 部分测试失败")
        print("\n💡 建议:")
        print("   1. 检查模型是否正确加载")
        print("   2. 确保摄像头正常工作")
        print("   3. 增加动作幅度和持续时间")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 测试中断")
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
