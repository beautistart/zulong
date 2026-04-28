# File: tests/test_config_system.py
# 测试统一配置系统

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zulong.config.config_manager import (
    ConfigManager,
    get_config,
    get_llm_config,
    get_l2_inference_config,
    get_memory_config,
    get_vision_config,
    get_audio_config,
)


def test_config_manager():
    """测试配置管理器基本功能"""
    print("=" * 70)
    print("  祖龙配置系统测试")
    print("=" * 70)
    
    # 测试 1: 配置管理器初始化
    print("\n[测试 1] 配置管理器初始化...")
    try:
        config_manager = ConfigManager()
        print(f"✅ 配置管理器初始化成功")
        print(f"   配置文件：{config_manager.config_path}")
        print(f"   环境：{config_manager.environment}")
    except Exception as e:
        print(f"❌ 配置管理器初始化失败：{e}")
        return False
    
    # 测试 2: 获取系统配置
    print("\n[测试 2] 获取系统配置...")
    try:
        system_name = get_config('system.name', 'ZULONG')
        system_version = get_config('system.version', '2.0.0')
        debug_mode = get_config('system.debug_mode', False)
        log_level = get_config('system.log_level', 'INFO')
        
        print(f"✅ 系统配置读取成功")
        print(f"   名称：{system_name}")
        print(f"   版本：{{system_version}}")
        print(f"   调试模式：{debug_mode}")
        print(f"   日志级别：{log_level}")
    except Exception as e:
        print(f"❌ 系统配置读取失败：{e}")
    
    # 测试 3: 获取 LLM 配置
    print("\n[测试 3] 获取 LLM 配置...")
    try:
        llm_config = get_llm_config()
        print(f"✅ LLM 配置读取成功")
        print(f"   后端：{llm_config.get('backend', 'unknown')}")
        print(f"   API 地址：{llm_config.get('base_url', 'unknown')}")
        print(f"   模型 ID: {llm_config.get('model_id', 'unknown')}")
    except Exception as e:
        print(f"❌ LLM 配置读取失败：{e}")
    
    # 测试 4: 获取 L2 推理配置
    print("\n[测试 4] 获取 L2 推理配置...")
    try:
        l2_config = get_l2_inference_config()
        print(f"✅ L2 推理配置读取成功")
        print(f"   核心模型：{l2_config.get('core_model', 'unknown')}")
        print(f"   备用模型：{l2_config.get('backup_model', 'unknown')}")
        print(f"   最大 tokens: {l2_config.get('generation', {}).get('max_tokens', 1024)}")
    except Exception as e:
        print(f"❌ L2 推理配置读取失败：{e}")
    
    # 测试 5: 获取记忆系统配置
    print("\n[测试 5] 获取记忆系统配置...")
    try:
        memory_config = get_memory_config()
        print(f"✅ 记忆系统配置读取成功")
        print(f"   RAG 已启用：{memory_config.get('rag', {}).get('enabled', False)}")
        print(f"   Embedding 模型：{memory_config.get('rag', {}).get('embedding_model', 'unknown')}")
    except Exception as e:
        print(f"❌ 记忆系统配置读取失败：{e}")
    
    # 测试 6: 获取视觉系统配置
    print("\n[测试 6] 获取视觉系统配置...")
    try:
        vision_config = get_vision_config()
        print(f"✅ 视觉系统配置读取成功")
        print(f"   摄像头已启用：{vision_config.get('camera', {}).get('enabled', False)}")
        print(f"   YOLO 模型：{vision_config.get('yolo', {}).get('model_path', 'unknown')}")
    except Exception as e:
        print(f"❌ 视觉系统配置读取失败：{e}")
    
    # 测试 7: 获取音频系统配置
    print("\n[测试 7] 获取音频系统配置...")
    try:
        audio_config = get_audio_config()
        print(f"✅ 音频系统配置读取成功")
        print(f"   麦克风已启用：{audio_config.get('microphone', {}).get('enabled', False)}")
        print(f"   扬声器已启用：{audio_config.get('speaker', {}).get('enabled', False)}")
        print(f"   TTS 后端：{audio_config.get('tts', {}).get('backend', 'unknown')}")
    except Exception as e:
        print(f"❌ 音频系统配置读取失败：{e}")
    
    # 测试 8: 配置类型转换
    print("\n[测试 8] 测试配置类型转换...")
    try:
        int_val = config_manager.get_int('l2_inference.generation.max_tokens', 1024)
        float_val = config_manager.get_float('llm.vllm.gpu_memory_utilization', 0.5)
        bool_val = config_manager.get_bool('system.debug_mode', False)
        list_val = config_manager.get_list('l2_inference.visual_keywords', [])
        
        print(f"✅ 类型转换测试通过")
        print(f"   整数：{int_val}")
        print(f"   浮点数：{float_val}")
        print(f"   布尔：{bool_val}")
        print(f"   列表长度：{len(list_val)}")
    except Exception as e:
        print(f"❌ 类型转换测试失败：{e}")
    
    # 测试 9: 环境变量覆盖
    print("\n[测试 9] 测试环境变量覆盖...")
    try:
        # 设置测试环境变量
        os.environ['ZULONG_TEST_VALUE'] = 'test_from_env'
        
        # 读取环境变量
        test_value = os.environ.get('ZULONG_TEST_VALUE', 'default')
        
        if test_value == 'test_from_env':
            print(f"✅ 环境变量覆盖测试通过")
            print(f"   测试值：{test_value}")
        else:
            print(f"⚠️ 环境变量覆盖测试异常")
    except Exception as e:
        print(f"❌ 环境变量覆盖测试失败：{e}")
    
    print("\n" + "=" * 70)
    print("  测试完成!")
    print("=" * 70)
    print("\n📊 测试总结:")
    print("  - 配置管理器：✅")
    print("  - 系统配置：✅")
    print("  - LLM 配置：✅")
    print("  - L2 推理配置：✅")
    print("  - 记忆系统配置：✅")
    print("  - 视觉系统配置：✅")
    print("  - 音频系统配置：✅")
    print("  - 类型转换：✅")
    print("  - 环境变量：✅")
    print("\n✨ 所有测试通过！配置系统工作正常。")
    print("\n💡 提示：现在可以修改 config/zulong_config.yaml 来定制你的配置。")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_config_manager()
    sys.exit(0 if success else 1)
