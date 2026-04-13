# File: zulong/bootstrap.py
# 系统引导器 - 生产环境版本
# 对应 TSD v1.7: 一键启动整个祖龙系统

import sys
import os

# 🔥 设置 vLLM 环境变量（必须在导入 zulong 模块之前）
os.environ['USE_VLLM_FOR_L2'] = 'true'
os.environ['USE_VLLM_FOR_L2_BACKUP'] = 'true'

import threading
import time
import signal
import asyncio
import traceback

# 导入性能分析工具
from zulong.utils.performance import PerformanceTimer

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心模块
from zulong.core.event_bus import event_bus
from zulong.core.state_manager import state_manager
from zulong.core.types import EventType, EventPriority, ZulongEvent
from zulong.core.websocket_server import start_websocket_server

# 导入 L1 模块
from zulong.l1a.reflex_controller import reflex_controller

# 导入 L1-C 视觉静默注意层 (TSD v1.8 三层注意力机制)
# 使用优化版本的视觉处理器
from zulong.l1c.optimized_vision_processor import init_vision_processor, get_vision_processor

# 导入 L0 设备模块
from zulong.l0.devices.camera_device import CameraDevice
from zulong.l0.devices.microphone_device import MicrophoneDevice

# 导入摄像头配置
try:
    from zulong.config.camera_config import CAMERA_DEVICE_INDEX, ENABLE_CAMERA
except ImportError:
    # 默认使用 USB 摄像头 (索引 1)
    CAMERA_DEVICE_INDEX = 1
    ENABLE_CAMERA = True

# 导入 L1-B 模块
from zulong.l1b.scheduler_gatekeeper import gatekeeper
from zulong.core.power_manager import power_manager

# 🔥 关键修复：先配置日志
from zulong.utils.monitor import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

# 🔥 关键修复：在导入任何依赖共享池的模块之前，先初始化共享池单例
# 这样确保 ShortTermMemory、DataIngestion 等都能使用同一个实例
logger.info("🧠 [BOOTSTRAP] 提前初始化 SharedMemoryPool 单例...")
import asyncio
from zulong.infrastructure.shared_memory_pool import SharedMemoryPool

# 在同步上下文中异步初始化
try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    pool = loop.run_until_complete(SharedMemoryPool.get_instance())
    loop.close()
    logger.info(f"✅ [BOOTSTRAP] SharedMemoryPool 单例已创建：{id(pool)}")
    logger.info(f"   Memory Zone: {len(pool._memory_zone)} 条")
    logger.info(f"   Raw Zone: {len(pool._raw_zone)} 条")
except Exception as e:
    logger.error(f"❌ [BOOTSTRAP] SharedMemoryPool 初始化失败：{e}", exc_info=True)
    raise

# 导入 L2 模块（现在可以安全导入，因为共享池已初始化）
from zulong.l2.task_state_manager import task_state_manager
from zulong.l2.interrupt_controller import interrupt_controller
from zulong.l2.inference_engine import inference_engine  # 使用真实推理引擎

# 导入 L3 专家模块 (TTS 专家由 SpeakerDevice 按需调用)
# 不再需要 TTS 事件处理器 - TSD v1.7 架构修正

# 导入调试控制台
from zulong.tools.debug_console import debug_console

# 🔥 技能包运行时
from zulong.skill_packs.runtime import SkillPackRuntime
from zulong.skill_packs.loader import SkillPackLoader


class SystemBootstrap:
    """系统引导类"""
    
    def __init__(self):
        self._running = False
        self._camera_device = None  # 摄像头设备
        self._microphone_device = None  # 麦克风设备
        self._threads = []
        self._camera_loop = None  # 摄像头事件循环
        self._shutdown_requested = False
    
    def initialize(self):
        """初始化系统"""
        logger.info("=== ZULONG System Initialization ===")
        
        # 1. 初始化核心单例
        logger.info("Initializing core singletons...")
        
        # 🔥 关键修复：共享池已在模块导入时初始化，无需重复初始化
        logger.info("✅ SharedMemoryPool 已在模块导入时初始化")
        
        # StateManager 和 EventBus 会在首次导入时自动初始化
        
        # 2. 注册 L1-A 反射控制器
        logger.info("Registering L1-A Reflex Controller...")
        # reflex_controller 会在导入时自动注册到 EventBus
        
        # 3. 注册 L1-B 调度器
        logger.info("Registering L1-B Gatekeeper...")
        # gatekeeper 会在导入时自动注册到 EventBus
        
        # 4. 注册 L2 中断控制器
        logger.info("Registering L2 Interrupt Controller...")
        # interrupt_controller 会在导入时自动注册到 EventBus
        
        # 5. 注入推理引擎
        logger.info("Injecting Inference Engine...")
        interrupt_controller.set_inference_engine(inference_engine)
        
        # 🔥 新增：初始化复盘集成器
        logger.info("🔄 Initializing Replay Integration...")
        try:
            from zulong.review.integration import get_replay_integration
            self.replay_integration = get_replay_integration()
            logger.info("✅ ReplayIntegration initialized")
        except Exception as e:
            logger.error(f"❌ [BOOTSTRAP] ReplayIntegration 初始化失败：{e}", exc_info=True)
            raise
        
        # 🔥 新增：初始化技能包运行时
        logger.info("📦 Initializing SkillPackRuntime...")
        try:
            from zulong.memory.experience_generator import ExperienceGenerator
            from zulong.memory.hot_update_engine import get_hot_update_engine
            
            self.skill_pack_runtime = SkillPackRuntime(
                tool_engine=inference_engine.tool_engine,
                experience_store=None,
                hot_update_engine=get_hot_update_engine()
            )
            self.skill_pack_loader = SkillPackLoader(self.skill_pack_runtime)
            
            # 从配置加载技能包
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'skill_packs.yaml'
            )
            if os.path.exists(config_path):
                loaded_count = self.skill_pack_loader.load_from_config(config_path)
                logger.info(f"✅ SkillPackRuntime initialized, loaded {loaded_count} packs")
            else:
                logger.warning(f"⚠️ 技能包配置文件不存在: {config_path}")
        except Exception as e:
            logger.warning(f"⚠️ [BOOTSTRAP] SkillPackRuntime 初始化失败（非致命）: {e}")
            self.skill_pack_runtime = None
            self.skill_pack_loader = None
        
        # 🔥 v3.0 新增：初始化并启动复盘触发器 (解决断层 2)
        logger.info("⏰ Initializing Review Trigger...")
        try:
            from zulong.review.trigger import ReviewTrigger, TriggerType
            self.review_trigger = ReviewTrigger()
            
            # 注册回调：复盘触发时调用 ReplayIntegration
            replay_integration = get_replay_integration()
            self.review_trigger.register_callback(
                TriggerType.USER_ACTIVE,  # 🔥 v3.0 修复：使用枚举类型
                replay_integration.on_replay_triggered
            )
            
            logger.info("✅ ReviewTrigger initialized (will start in async mode)")
        except Exception as e:
            logger.error(f"❌ [BOOTSTRAP] ReviewTrigger 初始化失败：{e}", exc_info=True)
            raise
        
        logger.info("=== Initialization Complete ===")
    
    async def _start_camera(self):
        """异步启动摄像头设备"""
        try:
            # 🔥 v3.0 新增：启动 ReviewTrigger 后台监控
            if self.review_trigger:
                logger.info("⏰ Starting ReviewTrigger background monitoring...")
                asyncio.create_task(self.review_trigger.start())
                logger.info("✅ ReviewTrigger started")
            
            # 检查是否启用摄像头
            if not ENABLE_CAMERA:
                logger.info("[BOOTSTRAP] 摄像头已禁用（配置 ENABLE_CAMERA=False）")
                print("[BOOTSTRAP] 摄像头已禁用")
                return
            
            logger.info(f"[BOOTSTRAP] Initializing camera device (index={CAMERA_DEVICE_INDEX})...")
            
            # 🎯 TSD v1.8 关键修正：先初始化 VisionProcessor，再初始化 Camera
            # 这样 Camera 在 warmup 时就可以推送帧到 VP
            logger.info("[BOOTSTRAP] Initializing VisionProcessor (before Camera)...")
            print("[BOOTSTRAP] Initializing VisionProcessor (before Camera)...")
            try:
                vp = await init_vision_processor()  # 异步调用
                logger.info(f"VisionProcessor singleton created: {vp}")
                logger.info(f"VP 对象地址：{id(vp)}")
                print(f"[BOOTSTRAP] VP singleton: {vp}, id={id(vp)}, is_running={vp.is_running}")
                logger.info(f"VP status: is_running={vp.is_running}, is_initialized={vp.is_initialized}")
                print(f"[BOOTSTRAP] VP status: is_running={vp.is_running}, is_initialized={vp.is_initialized}")
            except Exception as vp_error:
                logger.error(f"VisionProcessor initialization failed: {vp_error}", exc_info=True)
                print(f"[BOOTSTRAP] VisionProcessor initialization failed: {vp_error}")
                import traceback
                traceback.print_exc()
            
            # 初始化摄像头
            logger.info("[Bootstrap] Calling camera.initialize()...")
            self._camera_device = CameraDevice(device_index=CAMERA_DEVICE_INDEX)
            success = await self._camera_device.initialize()
            logger.info(f"[Bootstrap] camera.initialize() returned: {success}")
            
            if success:
                logger.info("Camera device initialized")
                
                # 启动视频采集循环
                await self._camera_device.start()
                logger.info("Camera streaming started")
            else:
                logger.error("Failed to initialize camera device")
                logger.error(f"提示：请检查摄像头 #{CAMERA_DEVICE_INDEX} 是否可用")
                logger.error(f"提示：运行 'python test_list_cameras.py' 检测可用摄像头")
                
        except Exception as e:
            logger.error(f"❌ Camera startup error: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self, interactive=True):
        """启动系统
        
        Args:
            interactive: 是否启动交互式调试控制台
        """
        logger.info("=== ZULONG System Startup ===")
        
        self._running = True
        
        # 1. 启动摄像头设备 (真实模式)
        logger.info("[BOOTSTRAP] Starting Camera Device...")
        # 使用独立线程启动摄像头，避免事件循环问题
        camera_thread = threading.Thread(target=self._run_camera_async, daemon=False)  # 改为非守护线程
        camera_thread.start()
        self._threads.append(camera_thread)
        
        # 等待摄像头线程启动完成（最多等待 5 秒）
        logger.info("[BOOTSTRAP] Waiting for camera thread to start...")
        time.sleep(2)  # 给摄像头线程 2 秒时间启动
        
        # 2. 启动麦克风设备
        logger.info("[BOOTSTRAP] Starting Microphone Device...")
        self._microphone_device = MicrophoneDevice()
        mic_thread = threading.Thread(target=self._run_microphone_async, daemon=True)
        mic_thread.start()
        self._threads.append(mic_thread)
        
        # 3. 启动交互式调试控制台（交互模式下）
        if interactive:
            logger.info("[BOOTSTRAP] Starting Debug Console...")
            logger.info(f"[BOOTSTRAP] debug_console object: {debug_console}")
            debug_console.start()
            logger.info("[BOOTSTRAP] debug_console.start() returned")
            self._threads.append(debug_console)
            logger.info(f"[BOOTSTRAP] debug_console appended to _threads (len={len(self._threads)})")
            logger.info("[BOOTSTRAP] Debug Console started, continuing...")
        
        # 🎯 4. TTS 集成说明 (TSD v1.7 架构修正)
        # TTS 专家现在由 SpeakerDevice 在收到 ACTION_SPEAK 事件时按需调用
        # 不再需要独立的 TTS 事件处理器监听 L2_OUTPUT
        logger.info("[BOOTSTRAP] TTS Integration: SpeakerDevice will call TTS expert on ACTION_SPEAK events")
        
        # 🎯 5. 启动 WebSocket 服务器 (OpenClaw Bridge 连接)
        logger.info("[BOOTSTRAP] Starting WebSocket Server for OpenClaw Bridge...")
        
        async def start_ws():
            await start_websocket_server()
        
        # 在后台线程中启动 WebSocket 服务器（非守护线程，确保服务器持续运行）
        ws_thread = threading.Thread(target=lambda: asyncio.run(start_ws()), daemon=False)
        ws_thread.start()
        
        # 等待 WebSocket Server 启动（最多等待 5 秒）
        logger.info("[BOOTSTRAP] Waiting for WebSocket Server to start...")
        time.sleep(2)  # 给 WebSocket 服务器 2 秒时间启动
        
        logger.info("[BOOTSTRAP] WebSocket Server started on ws://localhost:5555")
        
        logger.info("[BOOTSTRAP] ZULONG System Online")
        if interactive:
            logger.info("Debug Console is ready. Type your message or '/help' for commands.")
        else:
            logger.info("System is ready to process events")
    
    def _run_camera_async(self):
        """在线程中运行摄像头异步启动"""
        try:
            logger.info("[Camera Thread] Starting new event loop...")
            print("[Camera Thread] Starting new event loop...")
            # 创建新的事件循环
            self._camera_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._camera_loop)
            
            logger.info("[Camera Thread] Running _start_camera()...")
            print("[Camera Thread] Running _start_camera()...")
            try:
                # 运行异步启动函数
                self._camera_loop.run_until_complete(self._start_camera())
                logger.info("[Camera Thread] _start_camera() completed successfully")
                print("[Camera Thread] _start_camera() completed successfully")
            except Exception as inner_e:
                logger.error(f"[Camera Thread] _start_camera() failed: {inner_e}", exc_info=True)
                print(f"[Camera Thread] _start_camera() failed: {inner_e}")
                import traceback
                traceback.print_exc()
                raise
            
            # 检查是否请求关闭
            if self._shutdown_requested:
                logger.info("[Camera Thread] Shutdown requested, exiting...")
                return
            
            logger.info("[Camera Thread] Keeping loop alive...")
            print("[Camera Thread] Keeping loop alive...")
            # 保持事件循环运行
            self._camera_loop.run_forever()
        except Exception as e:
            logger.error(f"[Camera Thread] Error: {e}", exc_info=True)
            print(f"[Camera Thread] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("[Camera Thread] Closing event loop...")
            print("[Camera Thread] Closing event loop...")
            if self._camera_loop:
                try:
                    # 取消所有待处理的任务
                    pending = asyncio.all_tasks(self._camera_loop)
                    for task in pending:
                        task.cancel()
                    self._camera_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    self._camera_loop.close()
                except Exception as close_error:
                    logger.error(f"[Camera Thread] Error closing camera loop: {close_error}")
    
    def _run_microphone_async(self):
        """在线程中运行麦克风设备"""
        try:
            logger.info("🎤 [Microphone Thread] Starting...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            loop.run_until_complete(self._microphone_device.start())
            logger.info("🎤 [Microphone Thread] Device started")
            
            loop.run_forever()
        except Exception as e:
            logger.error(f"❌ Microphone thread error: {e}", exc_info=True)
        finally:
            if loop:
                loop.close()
    
    def stop(self):
        """停止系统"""
        logger.info("=== ZULONG System Shutdown ===")
        
        self._running = False
        self._shutdown_requested = True
        
        # 🔥 关键修复：在关闭前保存所有持久化数据
        logger.info("💾 保存持久化数据...")
        try:
            import asyncio
            from zulong.infrastructure.shared_memory_pool import SharedMemoryPool
            from zulong.memory.short_term_memory import ShortTermMemory
            
            # 检查实例是否存在
            pool_exists = SharedMemoryPool._instance is not None
            stm_exists = ShortTermMemory._instance is not None
            
            logger.info(f"   SharedMemoryPool 实例存在：{pool_exists}")
            logger.info(f"   ShortTermMemory 实例存在：{stm_exists}")
            
            if pool_exists or stm_exists:
                loop = asyncio.new_event_loop()
                try:
                    # 保存共享池快照
                    if pool_exists:
                        logger.info("   保存共享池快照...")
                        loop.run_until_complete(SharedMemoryPool._instance.save_snapshot_now())
                        logger.info(f"   ✅ 共享池快照已保存 (Memory Zone: {len(SharedMemoryPool._instance._memory_zone)} 条)")
                    
                    # 保存短期记忆索引
                    if stm_exists:
                        logger.info("   保存短期记忆索引...")
                        ShortTermMemory._instance._save_index()
                        logger.info(f"   ✅ 短期记忆索引已保存 (索引数量：{len(ShortTermMemory._instance._turn_index)})")
                except Exception as e:
                    logger.error(f"   ❌ 保存失败：{e}", exc_info=True)
                finally:
                    loop.close()
            else:
                logger.info("   ⚠️ 没有需要保存的实例")
        except Exception as e:
            logger.error(f"❌ 保存持久化数据失败：{e}", exc_info=True)
        
        # 1. 停止调试控制台
        logger.info("Stopping Debug Console...")
        debug_console.stop()
        
        # 2. 停止摄像头事件循环
        if self._camera_loop and self._camera_loop.is_running():
            logger.info("📷 Stopping Camera event loop...")
            self._camera_loop.call_soon_threadsafe(self._camera_loop.stop)
        
        # 3. 停止摄像头设备
        if self._camera_device:
            logger.info("📷 Stopping Camera Device...")
            try:
                if hasattr(self._camera_device, 'stop'):
                    import asyncio
                    if self._camera_loop:
                        future = asyncio.run_coroutine_threadsafe(
                            self._camera_device.stop(), 
                            self._camera_loop
                        )
                        future.result(timeout=5)
            except Exception as e:
                logger.warning(f"⚠️ Camera stop error: {e}")
        
        # 4. 停止麦克风设备
        if self._microphone_device:
            logger.info("🎤 Stopping Microphone Device...")
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._microphone_device.stop())
                loop.close()
            except Exception as e:
                logger.warning(f"⚠️ Microphone stop error: {e}")
        
        # 5. 等待所有线程结束
        for thread in self._threads:
            if hasattr(thread, 'join'):
                thread.join(timeout=2)
        
        logger.info("ZULONG System Offline")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='祖龙 (ZULONG) 系统启动器')
    parser.add_argument('--no-interactive', action='store_true', 
                        help='禁用交互式调试控制台')
    args = parser.parse_args()
    
    bootstrap = SystemBootstrap()
    
    def signal_handler(signum, frame):
        """信号处理器"""
        logger.info(f"Received signal {signum}, shutting down...")
        bootstrap.stop()
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 初始化系统
        bootstrap.initialize()
        
        # 启动系统
        interactive = not args.no_interactive
        bootstrap.start(interactive=interactive)
        
        if interactive:
            # 交互模式：主线程等待调试控制台结束
            logger.info("Interactive mode active. Type '/quit' or press Ctrl+C to stop.")
            try:
                while bootstrap._running and debug_console._running:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                logger.info("Interactive mode interrupted by user")
        else:
            # 非交互模式：等待键盘中断
            logger.info("Press Ctrl+C to stop the system")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Non-interactive mode interrupted by user")
            
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止系统
        bootstrap.stop()


if __name__ == "__main__":
    main()
