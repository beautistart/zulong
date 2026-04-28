# File: scripts/cleanup_old_vision_arch.py
"""
清理旧视觉处理架构代码

旧架构文件列表：
- zulong/l1c/vision_processor.py (旧版，890 行)
- zulong/l1c/vision_short_term_memory.py (旧版 VSTM)
- zulong/l1c/frame_buffer.py (旧版帧缓冲)
- zulong/l1c/gesture_classifier.py (旧版手势分类器)
- zulong/l1c/vision_model_loader.py (旧版模型加载器)

新架构文件：
- zulong/l1c/optimized_vision_processor.py (新版，四层架构)
- zulong/l1c/action_classifier.py (MobileNetV4-TSM 动作分类)
- zulong/l1c/mediapipe_gesture_recognizer.py (MediaPipe 手势识别)
"""

import os
import shutil
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 旧架构文件列表
OLD_FILES = [
    "zulong/l1c/vision_processor.py",
    "zulong/l1c/vision_short_term_memory.py",
    "zulong/l1c/frame_buffer.py",
    "zulong/l1c/gesture_classifier.py",
    "zulong/l1c/vision_model_loader.py",
]

# 备份目录
BACKUP_DIR = BASE_DIR / "backup_old_vision_arch"

def backup_old_files():
    """备份旧文件到 backup 目录"""
    print("\n📦 备份旧架构文件...")
    
    BACKUP_DIR.mkdir(exist_ok=True)
    
    for file_path in OLD_FILES:
        src = BASE_DIR / file_path
        if src.exists():
            dst = BACKUP_DIR / Path(file_path).name
            shutil.copy2(src, dst)
            print(f"  ✅ 备份：{file_path} -> {dst}")
        else:
            print(f"  ⚠️  文件不存在：{file_path}")

def delete_old_files():
    """删除旧架构文件"""
    print("\n🗑️  删除旧架构文件...")
    
    deleted_count = 0
    for file_path in OLD_FILES:
        full_path = BASE_DIR / file_path
        if full_path.exists():
            full_path.unlink()
            deleted_count += 1
            print(f"  ✅ 删除：{file_path}")
        else:
            print(f"  ⚠️  文件不存在：{file_path}")
    
    print(f"\n✅ 共删除 {deleted_count}/{len(OLD_FILES)} 个文件")

def verify_cleanup():
    """验证清理结果"""
    print("\n🔍 验证清理结果...")
    
    remaining_files = []
    for file_path in OLD_FILES:
        full_path = BASE_DIR / file_path
        if full_path.exists():
            remaining_files.append(file_path)
    
    if remaining_files:
        print(f"\n⚠️  以下文件未被删除:")
        for file_path in remaining_files:
            print(f"  - {file_path}")
        return False
    else:
        print("\n✅ 所有旧文件已成功删除")
        return True

def verify_new_arch():
    """验证新架构文件存在"""
    print("\n🔍 验证新架构文件...")
    
    NEW_FILES = [
        "zulong/l1c/optimized_vision_processor.py",
        "zulong/l1c/action_classifier.py",
        "zulong/l1c/mediapipe_gesture_recognizer.py",
    ]
    
    missing_files = []
    for file_path in NEW_FILES:
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"  ❌ 缺失：{file_path}")
        else:
            print(f"  ✅ 存在：{file_path}")
    
    if missing_files:
        print(f"\n❌ 新架构文件缺失：{missing_files}")
        return False
    else:
        print(f"\n✅ 新架构文件完整")
        return True

def main():
    """主函数"""
    print("=" * 80)
    print(" 清理旧视觉处理架构")
    print("=" * 80)
    
    # 1. 备份旧文件
    backup_old_files()
    
    # 2. 删除旧文件
    delete_old_files()
    
    # 3. 验证清理结果
    if not verify_cleanup():
        print("\n⚠️  清理未完成，请手动检查")
        return 1
    
    # 4. 验证新架构
    if not verify_new_arch():
        print("\n⚠️  新架构文件缺失，请检查")
        return 1
    
    # 5. 更新 bootstrap.py (已完成)
    print("\n✅ Bootstrap.py 已更新使用新架构")
    
    print("\n" + "=" * 80)
    print(" 清理完成！")
    print("=" * 80)
    print("\n备份位置:", BACKUP_DIR)
    print("\n下一步:")
    print("  1. 测试系统启动：python -m zulong.bootstrap")
    print("  2. 测试视觉模块：python tests/test_layer3_4_quick.py")
    print("  3. 如有问题，可从备份恢复旧文件")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    exit(main())
