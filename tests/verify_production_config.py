# File: tests/verify_production_config.py
"""
生产配置验证工具

验证所有配置参数是否已正确应用到生产文件
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_file_config(file_path, config_name, expected_value, pattern):
    """检查文件中的配置值"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找匹配
        match = re.search(pattern, content)
        if match:
            actual_value = match.group(1)
            if actual_value == expected_value:
                print(f"✅ {config_name}: {actual_value}")
                return True
            else:
                print(f"❌ {config_name}: 期望 {expected_value}, 实际 {actual_value}")
                return False
        else:
            print(f"❌ {config_name}: 未找到配置")
            return False
    except Exception as e:
        print(f"❌ {config_name}: 读取失败 - {e}")
        return False

def main():
    print("\n" + "=" * 80)
    print(" 祖龙视觉系统生产配置验证 (v4.0)")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 配置文件路径
    vision_processor_file = os.path.join(base_dir, 'zulong', 'l1c', 'optimized_vision_processor.py')
    mediapipe_file = os.path.join(base_dir, 'zulong', 'l1c', 'mediapipe_gesture_recognizer.py')
    
    print("\n[1/6] 检查 Layer 3 配置...")
    print("-" * 80)
    
    checks = [
        # Layer 3 配置
        (vision_processor_file, "intent_threshold", "0.25", 
         r"'intent_threshold':\s*([\d.]+)"),
        (vision_processor_file, "interact_threshold", "0.05",
         r"'interact_threshold':\s*([\d.]+)"),
        
        # Layer 4 配置
        (vision_processor_file, "digital_zoom_factor", "5.0",
         r"'digital_zoom_factor':\s*([\d.]+)"),
        (vision_processor_file, "gesture_conf_threshold", "0.25",
         r"'gesture_conf_threshold':\s*([\d.]+)"),
        (vision_processor_file, "eagle_eye_cooldown", "0.3",
         r"'eagle_eye_cooldown':\s*([\d.]+)"),
        
        # MediaPipe 配置
        (mediapipe_file, "min_hand_detection_confidence", "0.3",
         r"min_hand_detection_confidence=([\d.]+)"),
        (mediapipe_file, "min_hand_presence_confidence", "0.3",
         r"min_hand_presence_confidence=([\d.]+)"),
        (mediapipe_file, "min_tracking_confidence", "0.3",
         r"min_tracking_confidence=([\d.]+)"),
    ]
    
    results = []
    for file_path, config_name, expected_value, pattern in checks:
        result = check_file_config(file_path, config_name, expected_value, pattern)
        results.append(result)
    
    # 统计结果
    print("\n" + "=" * 80)
    print(" 验证结果")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n通过：{passed}/{total}")
    
    if passed == total:
        print("\n✅ 所有配置已正确应用！")
        print("\n📋 配置汇总:")
        print("-" * 80)
        print(" Layer 3:")
        print("   - intent_threshold: 0.25")
        print("   - interact_threshold: 0.05 (L3 最低置信度 0.033 之上)")
        print("\n Layer 4:")
        print("   - digital_zoom_factor: 5.0")
        print("   - gesture_conf_threshold: 0.25")
        print("   - eagle_eye_cooldown: 0.3")
        print("\n MediaPipe:")
        print("   - min_hand_detection_confidence: 0.3")
        print("   - min_hand_presence_confidence: 0.3")
        print("   - min_tracking_confidence: 0.3")
        print("\n" + "=" * 80)
        print(" 下一步:")
        print("=" * 80)
        print(" 1. 运行功能测试:")
        print("    python tests/test_layer3_4_quick.py")
        print("\n 2. OK 手势专项测试:")
        print("    python tests/test_ok_gesture.py")
        print("\n 3. MediaPipe 诊断:")
        print("    python tests/test_mediapipe_diagnosis.py")
        print("\n" + "=" * 80)
        return 0
    else:
        print(f"\n❌ {total - passed} 个配置未正确应用")
        print("\n请检查:")
        print("  1. 文件是否正确保存")
        print("  2. 配置值是否正确")
        print("  3. 是否需要重启系统")
        print("\n" + "=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
