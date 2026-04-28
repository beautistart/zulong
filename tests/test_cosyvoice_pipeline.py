# File: tests/test_cosyvoice_pipeline.py
"""
测试 ModelScope CosyVoice Pipeline
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cosyvoice2_pipeline():
    """使用 ModelScope Pipeline 加载 CosyVoice2"""
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    
    print("\n" + "="*70)
    print("测试 ModelScope CosyVoice2 Pipeline")
    print("="*70)
    
    try:
        print("\n加载 CosyVoice2-0.5B...")
        tts_pipeline = pipeline(
            task=Tasks.text_to_speech,
            model="iic/CosyVoice2-0.5B",
        )
        print("  ✓ Pipeline 加载成功")
        
        print("\n测试语音合成...")
        result = tts_pipeline("你好，我是祖龙机器人。")
        print(f"  结果类型: {type(result)}")
        
        if isinstance(result, dict):
            print(f"  结果键: {result.keys()}")
            if 'output_wav' in result:
                output_path = result['output_wav']
                print(f"  输出文件: {output_path}")
                
                if os.path.exists(output_path):
                    print(f"  文件大小: {os.path.getsize(output_path)} bytes")
                    print("  ✓ 语音合成成功!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cosyvoice1_pipeline():
    """使用 ModelScope Pipeline 加载 CosyVoice1"""
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    
    print("\n" + "="*70)
    print("测试 ModelScope CosyVoice1 Pipeline")
    print("="*70)
    
    try:
        print("\n加载 CosyVoice-300M...")
        tts_pipeline = pipeline(
            task=Tasks.text_to_speech,
            model="iic/CosyVoice-300M",
        )
        print("  ✓ Pipeline 加载成功")
        
        print("\n测试语音合成...")
        result = tts_pipeline("你好，我是祖龙机器人。")
        print(f"  结果类型: {type(result)}")
        
        if isinstance(result, dict):
            print(f"  结果键: {result.keys()}")
            if 'output_wav' in result:
                output_path = result['output_wav']
                print(f"  输出文件: {output_path}")
                
                if os.path.exists(output_path):
                    print(f"  文件大小: {os.path.getsize(output_path)} bytes")
                    print("  ✓ 语音合成成功!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_cosyvoice2_pipeline()
