# File: tests/optimize_gesture_params.py
"""
手势识别参数优化

根据测试结果调整关键参数，提高识别率

问题分析:
- 当前平均置信度：0.19
- 当前最高置信度：0.35 (thumbs_up)
- 识别阈值：0.7 (过高)

优化方案:
1. 降低手势识别阈值：0.7 -> 0.3
2. 增加 Digital Zoom 倍数：3.0 -> 5.0
3. 降低 YOLO 检测阈值：0.5 -> 0.4
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def optimize_parameters():
    """优化手势识别参数"""
    
    config_file = Path("zulong/l1c/optimized_vision_processor.py")
    
    if not config_file.exists():
        print(f"❌ 文件不存在：{config_file}")
        return False
    
    # 读取文件
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 保存原始内容用于对比
    original_content = content
    
    print("="*60)
    print("🔧 优化手势识别参数")
    print("="*60)
    
    # 优化 1: 降低手势识别阈值
    print("\n📊 优化 1: 手势识别阈值")
    old_line = "'gesture_conf_threshold': 0.7,"
    new_line = "'gesture_conf_threshold': 0.3,  # 降低阈值 (0.7->0.3)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print(f"   ✅ {old_line}")
        print(f"   👉 {new_line}")
    else:
        print(f"   ⚠️ 未找到：{old_line}")
    
    # 优化 2: 增加 Digital Zoom 倍数
    print("\n📊 优化 2: Digital Zoom 倍数")
    old_line = "'digital_zoom_factor': 3.0,"
    new_line = "'digital_zoom_factor': 5.0,  # 增加放大倍数 (3.0->5.0)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print(f"   ✅ {old_line}")
        print(f"   👉 {new_line}")
    else:
        print(f"   ⚠️ 未找到：{old_line}")
    
    # 优化 3: 降低 YOLO 检测阈值
    print("\n📊 优化 3: YOLO 检测阈值")
    old_line = "'yolo_conf_threshold': 0.5,"
    new_line = "'yolo_conf_threshold': 0.4,  # 降低检测阈值 (0.5->0.4)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print(f"   ✅ {old_line}")
        print(f"   👉 {new_line}")
    else:
        print(f"   ⚠️ 未找到：{old_line}")
    
    # 优化 4: 降低 ROI 运动阈值
    print("\n📊 优化 4: ROI 运动阈值")
    old_line = "'roi_motion_threshold': 200,"
    new_line = "'roi_motion_threshold': 100,  # 降低阈值提高敏感度 (200->100)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print(f"   ✅ {old_line}")
        print(f"   👉 {new_line}")
    else:
        print(f"   ⚠️ 未找到：{old_line}")
    
    # 写入文件
    if content != original_content:
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n" + "="*60)
        print("✅ 参数优化完成！")
        print("="*60)
        
        print("\n📋 优化总结:")
        print("   1. 手势识别阈值：0.7 -> 0.3 (提高敏感度)")
        print("   2. Digital Zoom: 3.0 -> 5.0 (放大手势区域)")
        print("   3. YOLO 阈值：0.5 -> 0.4 (更容易检测人体)")
        print("   4. ROI 阈值：200 -> 100 (提高运动敏感度)")
        
        print("\n🎯 预期效果:")
        print("   - 识别率提升：0% -> 30-50%")
        print("   - 置信度提升：0.19 -> 0.4-0.6")
        
        print("\n⚠️ 注意事项:")
        print("   - 降低阈值可能增加误识别")
        print("   - 建议在真实环境中重新测试")
        
        print("\n📋 下一步:")
        print("   运行：python tests/test_real_gesture_auto.py")
        print("   查看优化后的识别效果")
        
        return True
    else:
        print("\n❌ 没有进行任何修改")
        return False


if __name__ == "__main__":
    optimize_parameters()
