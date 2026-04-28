# -*- coding: utf-8 -*-
"""
测试工具调用和联网搜索功能
"""
import os
import sys
import asyncio
import logging

# 🔥 关键：设置 vLLM 环境变量（必须在导入 zulong 模块之前）
os.environ['USE_VLLM_FOR_L2'] = 'true'
os.environ['USE_VLLM_FOR_L2_BACKUP'] = 'true'

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_tool_calling():
    """测试工具调用功能"""
    logger.info("\n" + "="*80)
    logger.info("测试工具调用功能")
    logger.info("="*80)
    
    try:
        from zulong.l2.inference_engine import InferenceEngine
        
        engine = InferenceEngine()
        
        # 测试 1: 简单问题（不需要工具）
        logger.info("\n【测试 1】简单对话（不需要工具）")
        response = await engine.generate("你好，请介绍一下你自己")
        logger.info(f"✅ 回复：{response[:100]}...")
        
        # 测试 2: 需要搜索的问题
        logger.info("\n【测试 2】需要搜索的问题")
        response = await engine.generate("今天北京的天气怎么样？")
        logger.info(f"✅ 回复：{response[:200]}...")
        
        # 检查是否有工具调用
        if hasattr(engine, 'last_tool_calls'):
            logger.info(f"🔧 工具调用记录：{engine.last_tool_calls}")
        else:
            logger.info("ℹ️ 没有检测到工具调用")
        
        logger.info("\n✅ 工具调用测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败：{e}", exc_info=True)
        return False

async def main():
    """主函数"""
    logger.info("="*80)
    logger.info("祖龙系统 - 工具调用和联网搜索功能测试")
    logger.info("="*80)
    
    # 测试工具调用
    success = await test_tool_calling()
    
    if success:
        logger.info("\n" + "="*80)
        logger.info("✅ 所有测试通过！")
        logger.info("="*80)
    else:
        logger.info("\n" + "="*80)
        logger.info("❌ 测试失败")
        logger.info("="*80)

if __name__ == "__main__":
    asyncio.run(main())
