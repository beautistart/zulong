# File: tests/analyze_thresholds.py
"""
阈值分析工具

根据 Layer 3 的测试结果，分析并建议最优的阈值配置
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_layer3_results():
    """分析 Layer 3 测试结果，给出阈值建议"""
    
    print("\n" + "=" * 80)
    print(" 阈值分析工具")
    print("=" * 80)
    
    print("\n请运行 Layer 3 置信度统计测试:")
    print("  python tests/test_layer3_confidence_stats.py")
    print("\n然后输入测试结果中的关键数据:\n")
    
    # 获取用户输入
    try:
        max_score = float(input("请输入最高分数 (max_score): ").strip())
        avg_score = float(input("请输入平均分数 (avg_score): ").strip())
        total_frames = int(input("请输入总帧数: ").strip())
        
        print("\n各意图的平均分数 (输入 0 跳过):")
        intent_scores = {}
        intents = ["LYING", "STANDING", "SITTING", "WAVING", "INTERACTING"]
        
        for intent in intents:
            score = input(f"  {intent}: ").strip()
            if score and float(score) > 0:
                intent_scores[intent] = float(score)
        
    except ValueError:
        print("❌ 输入无效，使用示例数据")
        max_score = 0.347
        avg_score = 0.15
        total_frames = 1000
        intent_scores = {
            "WAVING": 0.347,
            "STANDING": 0.070,
            "SITTING": 0.184,
            "LYING": 0.153
        }
    
    print("\n" + "=" * 80)
    print(" 分析结果")
    print("=" * 80)
    
    # 分析
    print(f"\n数据概览:")
    print(f"  最高分数：{max_score:.3f}")
    print(f"  平均分数：{avg_score:.3f}")
    print(f"  总帧数：{total_frames}")
    
    # 阈值建议
    print("\n" + "-" * 80)
    print(" 阈值配置建议")
    print("-" * 80)
    
    # Layer 3 阈值
    print("\n【Layer 3 阈值】")
    
    # 根据最高分数建议
    if max_score < 0.3:
        intent_threshold = 0.2
        interact_threshold = 0.5
        print(f"⚠️  动作分数较低，建议使用宽松阈值")
    elif max_score < 0.5:
        intent_threshold = 0.3
        interact_threshold = 0.6
        print(f"✅ 动作分数中等，建议使用标准阈值")
    elif max_score < 0.7:
        intent_threshold = 0.4
        interact_threshold = 0.7
        print(f"✅ 动作分数良好，建议使用标准阈值")
    else:
        intent_threshold = 0.5
        interact_threshold = 0.8
        print(f"✅ 动作分数优秀，建议使用严格阈值")
    
    print(f"\n建议配置:")
    print(f"  intent_threshold: {intent_threshold} (意图检测阈值)")
    print(f"  interact_threshold: {interact_threshold} (交互触发阈值)")
    
    # Layer 4 阈值
    print("\n【Layer 4 阈值】")
    print(f"Layer 4 触发条件：L3 分数 >= interact_threshold ({interact_threshold})")
    
    # 计算有多少帧能触发 Layer 4
    # 假设分数分布均匀，估算超过阈值的比例
    if max_score > interact_threshold:
        trigger_ratio = (max_score - interact_threshold) / (max_score - avg_score) if max_score > avg_score else 0
        trigger_ratio = min(1.0, max(0.0, trigger_ratio))
        trigger_frames = int(total_frames * trigger_ratio * 0.1)  # 假设 10% 的时间在交互
        
        print(f"\n预估触发情况:")
        print(f"  预计触发帧数：~{trigger_frames}帧")
        print(f"  触发频率：{'高' if trigger_frames > 100 else '中' if trigger_frames > 20 else '低'}")
        
        if trigger_frames < 10:
            print(f"\n⚠️  Layer 4 触发机会较少")
            print(f"建议:")
            print(f"  - 降低 interact_threshold 到 {interact_threshold - 0.1:.1f}")
            print(f"  - 或降低 eagle_eye_cooldown 到 0.3 秒")
        else:
            print(f"\n✅ Layer 4 触发频率合适")
    else:
        print(f"\n⚠️  最高分数 ({max_score:.3f}) 未达到交互阈值 ({interact_threshold})")
        print(f"Layer 4 将不会被触发！")
        print(f"\n建议:")
        print(f"  - 必须降低 interact_threshold 到 {max_score * 0.9:.2f} 以下")
        interact_threshold = max_score * 0.9
        print(f"  - 建议值：{interact_threshold:.2f}")
    
    # 综合配置建议
    print("\n" + "=" * 80)
    print(" 推荐配置文件")
    print("=" * 80)
    
    print(f"""
# OptimizedVisionProcessor 配置建议

config = {{
    # Layer 3 配置
    'intent_threshold': {intent_threshold},      # 意图检测阈值
    'interact_threshold': {interact_threshold},  # 交互触发阈值
    
    # Layer 4 配置
    'eagle_eye_cooldown': 0.5,  # 鹰眼冷却时间 (秒)
    'digital_zoom_factor': 5.0, # 数字变焦倍数
    
    # 其他配置
    'yolo_inference_frequency': 3,  # YOLO 推理频率
}}
""")
    
    print("\n" + "=" * 80)
    print(" 下一步操作")
    print("=" * 80)
    print("""
1. 应用建议配置到 OptimizedVisionProcessor
    
2. 运行 Layer 4 测试:
   python tests/test_layer4_manual.py

3. 如果 Layer 4 仍然不触发，检查:
   - MediaPipe 是否正常加载
   - 手势是否清晰
   - 光线是否充足

4. 联合测试:
   python tests/test_layer3_4_quick.py
""")
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_layer3_results()
