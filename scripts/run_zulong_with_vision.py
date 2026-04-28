# File: scripts/run_zulong_with_vision.py
"""
祖龙系统启动脚本（集成优化视觉系统）

启动完整的祖龙系统，包括：
- L0: 执行器、摄像头、麦克风
- L1-A: 反射控制器
- L1-B: 调度器/看门人
- L1-C: 优化视觉处理器（四层视觉检测）
- L2: 推理引擎、中断控制器
- L3: 专家技能

视觉系统配置：
- YOLO 置信度：0.25（高灵敏度）
- ROI 运动阈值：100px
- 手势置信度：0.25
- 鹰眼冷却：0.5 秒
"""

import sys
import os
import time
import signal
import asyncio
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RunZULONG")


def print_banner():
    """打印启动横幅"""
    print("=" * 80)
    print(" 祖龙 (ZULONG) 机器人系统")
    print(" 版本：v1.8 (集成优化视觉系统)")
    print("=" * 80)
    print()
    print("视觉系统配置:")
    print("  - Layer 1: YOLO 人体检测 (置信度阈值：0.25)")
    print("  - Layer 2: ROI 运动检测 (阈值：100px)")
    print("  - Layer 3: 动作分类 (MobileNetV4-TSM)")
    print("  - Layer 4: 手势识别 (MediaPipe, 置信度：0.25)")
    print()
    print("系统模块:")
    print("  - L0: 执行器、摄像头、麦克风")
    print("  - L1-A: 反射控制器")
    print("  - L1-B: 调度器/看门人")
    print("  - L1-C: 优化视觉处理器")
    print("  - L2: 推理引擎、中断控制器")
    print("  - L3: 专家技能")
    print()
    print("按 Ctrl+C 退出系统")
    print("=" * 80)
    print()


async def run_system():
    """运行祖龙系统"""
    
    print_banner()
    
    try:
        # ========== 1. 导入核心模块 ==========
        logger.info("导入核心模块...")
        
        from zulong.core.event_bus import event_bus
        from zulong.core.state_manager import state_manager
        
        # ========== 2. 导入 L1 模块 ==========
        logger.info("导入 L1 模块...")
        
        from zulong.l1a.reflex_controller import reflex_controller
        from zulong.l1b.scheduler_gatekeeper import gatekeeper
        from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
        
        # ========== 3. 导入 L2 模块 ==========
        logger.info("导入 L2 模块...")
        
        from zulong.l2.task_state_manager import task_state_manager
        from zulong.l2.interrupt_controller import interrupt_controller
        from zulong.l2.inference_engine import inference_engine
        
        # ========== 4. 初始化视觉系统 ==========
        logger.info("初始化优化视觉处理器...")
        
        vision_processor = OptimizedVisionProcessor()
        await vision_processor.initialize(load_models=True)
        
        logger.info("✅ 视觉系统初始化完成")
        logger.info("   - YOLO 置信度：0.25")
        logger.info("   - ROI 运动阈值：100px")
        logger.info("   - 手势置信度：0.25")
        logger.info("   - 鹰眼冷却：0.5 秒")
        
        # ========== 5. 启动摄像头 ==========
        logger.info("启动摄像头设备...")
        
        from zulong.l0.devices.camera_device import CameraDevice
        
        camera = CameraDevice()
        await camera.initialize()
        camera.start_streaming()
        
        logger.info("✅ 摄像头已启动")
        
        # ========== 6. 启动麦克风 ==========
        logger.info("启动麦克风设备...")
        
        from zulong.l0.devices.microphone_device import MicrophoneDevice
        
        microphone = MicrophoneDevice()
        await microphone.initialize()
        
        logger.info("✅ 麦克风已启动")
        
        # ========== 7. 注入推理引擎 ==========
        logger.info("注入推理引擎到中断控制器...")
        
        interrupt_controller.set_inference_engine(inference_engine)
        
        logger.info("✅ 推理引擎已注入")
        
        # ========== 8. 启动调试控制台（可选） ==========
        logger.info("启动调试控制台...")
        
        from zulong.tools.debug_console import debug_console
        
        # 在后台启动调试控制台
        asyncio.create_task(debug_console.start())
        
        logger.info("✅ 调试控制台已启动")
        
        # ========== 9. 系统就绪 ==========
        print("\n" + "=" * 80)
        print(" ✅ 祖龙系统已就绪！")
        print("=" * 80)
        print()
        print("视觉系统正在运行:")
        print("  - Layer 1: 人体检测 ✅")
        print("  - Layer 2: 运动检测 ✅")
        print("  - Layer 3: 意图识别 ✅")
        print("  - Layer 4: 手势识别 ✅")
        print()
        print("请执行以下动作测试系统:")
        print("  1. 站在摄像头前（测试 Layer 1）")
        print("  2. 挥动手臂（测试 Layer 2）")
        print("  3. 持续挥手 2-3 秒（测试 Layer 3）")
        print("  4. 比出 V 字、OK 等手势（测试 Layer 4）")
        print()
        print("查看实时监控:")
        print("  - 运行：python tests/test_simple_vision_debug.py")
        print()
        print("按 Ctrl+C 退出系统")
        print("=" * 80)
        print()
        
        # ========== 10. 主循环 ==========
        logger.info("进入主循环...")
        
        running = True
        
        def signal_handler(sig, frame):
            nonlocal running
            logger.info("收到退出信号，正在关闭系统...")
            running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 保持运行，等待事件
        while running:
            await asyncio.sleep(1)
            
            # 每秒打印一次系统状态
            if running:
                state = state_manager.get_state()
                print(f"\r[System] 状态：{state.get('power_state', 'UNKNOWN')} | "
                      f"L2: {state.get('l2_status', 'UNKNOWN')} | "
                      f"事件：{len(event_bus._EventBus__subscribers)} 订阅者",
                      end='', flush=True)
        
        # ========== 11. 清理资源 ==========
        logger.info("关闭系统资源...")
        
        camera.stop_streaming()
        await camera.cleanup()
        await microphone.cleanup()
        vision_processor.stop()
        
        logger.info("✅ 系统已安全关闭")
        
    except Exception as e:
        logger.error(f"❌ 系统运行失败：{e}", exc_info=True)
        raise


def main():
    """主函数"""
    try:
        asyncio.run(run_system())
    except KeyboardInterrupt:
        print("\n\n系统已中断")
    except Exception as e:
        logger.error(f"系统异常：{e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
