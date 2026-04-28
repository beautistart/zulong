# File: scripts/install_mediapipe.py
"""
MediaPipe 安装脚本

安装 MediaPipe 库和 Gesture Recognizer 模型。
"""

import subprocess
import sys
import os
from pathlib import Path


def install_package(package_name):
    """安装 Python 包"""
    print(f"📦 正在安装 {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    print(f"✅ {package_name} 安装完成")


def download_model():
    """下载 MediaPipe Gesture Recognizer 模型"""
    print("\n📥 下载 MediaPipe Gesture Recognizer 模型...")
    
    model_url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    project_root = Path(__file__).parent.parent
    model_path = project_root / "gesture_recognizer.task"
    
    try:
        import urllib.request
        
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r   下载进度：{percent:.1f}%", end='')
        
        urllib.request.urlretrieve(model_url, model_path, download_progress)
        print(f"\n✅ 模型下载完成：{model_path}")
        print(f"   文件大小：{model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"\n❌ 模型下载失败：{e}")
        print("\n💡 请手动下载:")
        print(f"   {model_url}")
        print(f"   并保存到：{model_path}")
        return False
    
    return True


def verify_installation():
    """验证安装"""
    print("\n🔍 验证安装...")
    
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe 版本：{mp.__version__}")
        
        # 检查模型文件
        project_root = Path(__file__).parent.parent
        model_path = project_root / "gesture_recognizer.task"
        
        if model_path.exists():
            print(f"✅ 模型文件存在：{model_path}")
            print(f"   大小：{model_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"⚠️  模型文件不存在：{model_path}")
            print("   请手动下载模型文件")
            return False
        
        return True
        
    except ImportError:
        print("❌ MediaPipe 未正确安装")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 MediaPipe 安装脚本")
    print("=" * 60)
    
    # 1. 安装 MediaPipe
    print("\n📦 步骤 1: 安装 MediaPipe 库")
    install_package("mediapipe>=0.10.0")
    
    # 2. 下载模型
    print("\n📦 步骤 2: 下载 Gesture Recognizer 模型")
    if not download_model():
        print("\n⚠️  模型下载失败，但 MediaPipe 已安装")
        print("   可以稍后手动下载模型")
    
    # 3. 验证安装
    print("\n📦 步骤 3: 验证安装")
    if verify_installation():
        print("\n✅ MediaPipe 安装成功！")
        print("\n💡 下一步:")
        print("   运行测试：python tests/test_mediapipe_gesture.py")
    else:
        print("\n❌ 安装失败，请检查错误信息")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 安装失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
