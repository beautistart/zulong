"""
Full 模式模块组 — 完整祖龙系统独有的模块

包含硬件接入、L1/L2 控制器、技能包、记忆进化等，
仅在 Full 模式启动时加载。
"""

import asyncio
import logging
import threading
from typing import List, Set

from zulong.launcher.module_base import Module, ModuleState

logger = logging.getLogger(__name__)


# ── 1. ModelPreloadModule ────────────────────────────

class ModelPreloadModule(Module):
    name = "model_preload"
    display_name = "模型预加载"
    dependencies = ["config"]
    mode_tags: Set[str] = {"full"}

    async def start(self) -> None:
        self.progress_message = "正在后台预加载模型..."
        try:
            from zulong.utils.model_preloader import preload_model_from_config
            config_mgr = self._context.get("config_manager")
            preload_model_from_config(config_mgr)
            logger.info("[ModelPreloadModule] 模型预加载已在后台启动")
        except Exception as e:
            logger.warning(f"[ModelPreloadModule] 模型预加载失败（非致命）: {e}")
        self.state = ModuleState.RUNNING


# ── 2. L1AReflexModule ──────────────────────────────

class L1AReflexModule(Module):
    name = "l1a_reflex"
    display_name = "L1-A 反射控制器"
    dependencies = ["config"]
    mode_tags: Set[str] = {"full"}

    async def start(self) -> None:
        self.progress_message = "正在注册 L1-A 反射控制器..."
        # 导入即注册到 EventBus
        from zulong.l1a.reflex_controller import reflex_controller  # noqa: F401
        self._context["reflex_controller"] = reflex_controller
        self.state = ModuleState.RUNNING
        logger.info("[L1AReflexModule] L1-A 反射控制器已注册")


# ── 3. L1BGatekeeperModule ──────────────────────────

class L1BGatekeeperModule(Module):
    name = "l1b_gatekeeper"
    display_name = "L1-B 调度网关"
    dependencies = ["config"]
    mode_tags: Set[str] = {"full"}

    async def start(self) -> None:
        self.progress_message = "正在注册 L1-B 调度网关..."
        from zulong.l1b.scheduler_gatekeeper import gatekeeper  # noqa: F401
        from zulong.core.power_manager import power_manager  # noqa: F401
        self._context["gatekeeper"] = gatekeeper
        self.state = ModuleState.RUNNING
        logger.info("[L1BGatekeeperModule] L1-B Gatekeeper 已注册")


# ── 4. L2InterruptModule ────────────────────────────

class L2InterruptModule(Module):
    name = "l2_interrupt"
    display_name = "L2 中断控制器"
    dependencies = ["inference_engine"]
    mode_tags: Set[str] = {"full"}

    async def start(self) -> None:
        self.progress_message = "正在初始化 L2 中断控制器..."
        from zulong.l2.interrupt_controller import interrupt_controller
        engine = self._context.get("inference_engine")
        if engine:
            interrupt_controller.set_inference_engine(engine)
        self._context["interrupt_controller"] = interrupt_controller
        self.state = ModuleState.RUNNING
        logger.info("[L2InterruptModule] L2 中断控制器已就绪")


# ── 5. ReplayIntegrationModule ──────────────────────

class ReplayIntegrationModule(Module):
    name = "replay_integration"
    display_name = "复盘集成器"
    dependencies = ["inference_engine"]
    mode_tags: Set[str] = {"full"}

    async def start(self) -> None:
        self.progress_message = "正在初始化复盘集成器..."
        from zulong.review.integration import get_replay_integration
        replay = get_replay_integration()
        self._context["replay_integration"] = replay
        self.state = ModuleState.RUNNING
        logger.info("[ReplayIntegrationModule] ReplayIntegration 已初始化")


# ── 6. ReviewTriggerModule ──────────────────────────

class ReviewTriggerModule(Module):
    name = "review_trigger"
    display_name = "复盘触发器"
    dependencies = ["replay_integration"]
    mode_tags: Set[str] = {"full"}

    async def start(self) -> None:
        self.progress_message = "正在初始化复盘触发器..."
        from zulong.review.trigger import ReviewTrigger, TriggerType

        trigger = ReviewTrigger()
        replay = self._context.get("replay_integration")
        if replay:
            trigger.register_callback(TriggerType.USER_ACTIVE, replay.on_replay_triggered)
            trigger.register_callback(TriggerType.QUIET_MODE, replay.on_replay_triggered)
            trigger.register_callback(TriggerType.NIGHT_SCHEDULE, replay.on_replay_triggered)

        # 启动后台监控
        asyncio.create_task(trigger.start())

        self._context["review_trigger"] = trigger
        self.state = ModuleState.RUNNING
        logger.info("[ReviewTriggerModule] ReviewTrigger 已启动")

    async def stop(self) -> None:
        trigger = self._context.get("review_trigger")
        if trigger and hasattr(trigger, "stop"):
            try:
                await trigger.stop()
            except Exception as e:
                logger.warning(f"[ReviewTriggerModule] 停止失败: {e}")
        self.state = ModuleState.STOPPED


# ── 7. SkillPackModule ──────────────────────────────

class SkillPackModule(Module):
    name = "skill_packs"
    display_name = "技能包运行时"
    dependencies = ["inference_engine"]
    mode_tags: Set[str] = {"full"}

    async def start(self) -> None:
        self.progress_message = "正在加载技能包..."
        import os
        from zulong.skill_packs.runtime import SkillPackRuntime
        from zulong.skill_packs.loader import SkillPackLoader

        engine = self._context.get("inference_engine")
        if not engine:
            logger.warning("[SkillPackModule] 无 InferenceEngine，跳过")
            self.state = ModuleState.RUNNING
            return

        # ExperienceStore
        try:
            from zulong.memory.enhanced_experience_store import EnhancedExperienceStore
            experience_store = EnhancedExperienceStore()
        except Exception:
            experience_store = None

        # HotUpdateEngine
        try:
            from zulong.memory.hot_update_engine import get_hot_update_engine
            hot_update = get_hot_update_engine()
        except Exception:
            hot_update = None

        # CoreToolManager
        core_tool_manager = None
        try:
            from zulong.tools.core_tool_manager import CoreToolManager
            tool_rag = None
            rag_manager = getattr(engine, "rag_manager", None)
            if rag_manager and hasattr(rag_manager, "get_tool_rag"):
                tool_rag = rag_manager.get_tool_rag()
            core_tool_manager = CoreToolManager(tool_rag=tool_rag)

            search_tools_tool = engine.tool_engine.registry.get("search_tools")
            if search_tools_tool and tool_rag:
                search_tools_tool.set_tool_rag(tool_rag)
        except Exception as e:
            logger.warning(f"[SkillPackModule] CoreToolManager 初始化失败: {e}")

        runtime = SkillPackRuntime(
            tool_engine=engine.tool_engine,
            experience_store=experience_store,
            hot_update_engine=hot_update,
            core_tool_manager=core_tool_manager,
        )
        runtime._vllm_client = getattr(engine, "vllm_client", None)
        try:
            from zulong.models.container import LLM_MODEL_ID
            runtime._vllm_model_id = LLM_MODEL_ID
        except ImportError:
            runtime._vllm_model_id = None

        loader = SkillPackLoader(runtime)
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "skill_packs.yaml",
        )
        if os.path.exists(config_path):
            loaded = loader.load_from_config(config_path)
            logger.info(f"[SkillPackModule] 已加载 {loaded} 个技能包")

        self._context["skill_pack_runtime"] = runtime
        self.state = ModuleState.RUNNING
        logger.info("[SkillPackModule] SkillPackRuntime 已就绪")


# ── 8. MemoryEvolutionModule ────────────────────────

class MemoryEvolutionModule(Module):
    name = "memory_evolution"
    display_name = "记忆进化引擎"
    dependencies = ["inference_engine", "memory_graph"]
    mode_tags: Set[str] = {"full"}

    async def start(self) -> None:
        self.progress_message = "正在启动记忆进化引擎..."

        # MemoryGraph 修剪循环
        mg = self._context.get("memory_graph")
        if mg:
            try:
                asyncio.create_task(mg.start_prune_loop())
                logger.info("[MemoryEvolutionModule] MemoryGraph 修剪循环已启动")
            except Exception as e:
                logger.warning(f"[MemoryEvolutionModule] 修剪循环启动失败: {e}")

        # MemoryEvolutionEngine 进化循环
        try:
            from zulong.memory.memory_evolution import (
                get_evolution_engine, set_evolution_engine, MemoryEvolutionEngine,
            )
            engine = self._context.get("inference_engine")
            evo = get_evolution_engine()
            if evo is None:
                rag = getattr(engine, "rag_manager", None) if engine else None
                if rag is not None:
                    evo = MemoryEvolutionEngine(rag)
                    set_evolution_engine(evo)
            if evo:
                asyncio.create_task(evo.start_evolution_loop())
                logger.info("[MemoryEvolutionModule] 进化循环已启动")
        except Exception as e:
            logger.warning(f"[MemoryEvolutionModule] 进化循环启动失败: {e}")

        self.state = ModuleState.RUNNING


# ── 9. CameraModule ─────────────────────────────────

class CameraModule(Module):
    name = "camera"
    display_name = "摄像头设备"
    dependencies = ["config"]
    mode_tags: Set[str] = {"full"}

    def __init__(self):
        super().__init__()
        self._camera = None
        self._loop = None
        self._thread = None

    async def start(self) -> None:
        self.progress_message = "正在初始化摄像头..."
        try:
            from zulong.config.camera_config import CAMERA_DEVICE_INDEX, ENABLE_CAMERA
        except ImportError:
            CAMERA_DEVICE_INDEX = 1
            ENABLE_CAMERA = True

        if not ENABLE_CAMERA:
            logger.info("[CameraModule] 摄像头已禁用")
            self.state = ModuleState.RUNNING
            return

        def _run():
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop.run_until_complete(self._start_camera(CAMERA_DEVICE_INDEX))
                self._loop.run_forever()
            except Exception as e:
                logger.error(f"[CameraModule] 线程异常: {e}", exc_info=True)
            finally:
                if self._loop:
                    self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=False, name="CameraDevice")
        self._thread.start()
        await asyncio.sleep(2.0)
        self.state = ModuleState.RUNNING
        logger.info(f"[CameraModule] 摄像头启动中 (index={CAMERA_DEVICE_INDEX})")

    async def _start_camera(self, device_index: int) -> None:
        # 先初始化 VisionProcessor
        try:
            from zulong.l1c.optimized_vision_processor import init_vision_processor
            await init_vision_processor()
        except Exception as e:
            logger.error(f"[CameraModule] VisionProcessor 初始化失败: {e}")

        from zulong.l0.devices.camera_device import CameraDevice
        self._camera = CameraDevice(device_index=device_index)
        success = await self._camera.initialize()
        if success:
            await self._camera.start()
            logger.info("[CameraModule] 摄像头已启动")
        else:
            logger.error(f"[CameraModule] 摄像头初始化失败 (index={device_index})")

    async def stop(self) -> None:
        if self._camera and self._loop and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._camera.stop(), self._loop)
                future.result(timeout=5)
            except Exception as e:
                logger.warning(f"[CameraModule] 停止摄像头异常: {e}")
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3)
        self.state = ModuleState.STOPPED


# ── 10. MicrophoneModule ────────────────────────────

class MicrophoneModule(Module):
    name = "microphone"
    display_name = "麦克风设备"
    dependencies = ["config"]
    mode_tags: Set[str] = {"full"}

    def __init__(self):
        super().__init__()
        self._mic = None
        self._thread = None

    async def start(self) -> None:
        self.progress_message = "正在初始化麦克风..."
        from zulong.l0.devices.microphone_device import MicrophoneDevice
        self._mic = MicrophoneDevice()

        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._mic.start())
                loop.run_forever()
            except Exception as e:
                logger.error(f"[MicrophoneModule] 线程异常: {e}", exc_info=True)

        self._thread = threading.Thread(target=_run, daemon=True, name="MicrophoneDevice")
        self._thread.start()
        await asyncio.sleep(1.0)
        self.state = ModuleState.RUNNING
        logger.info("[MicrophoneModule] 麦克风已启动")

    async def stop(self) -> None:
        if self._mic and hasattr(self._mic, "stop"):
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._mic.stop())
                loop.close()
            except Exception as e:
                logger.warning(f"[MicrophoneModule] 停止异常: {e}")
        self.state = ModuleState.STOPPED


# ── 10b. SpeakerModule ──────────────────────────────

class SpeakerModule(Module):
    name = "speaker"
    display_name = "扬声器设备"
    dependencies = ["config"]
    mode_tags: Set[str] = {"full"}

    def __init__(self):
        super().__init__()
        self._speaker = None
        self._thread = None

    async def start(self) -> None:
        self.progress_message = "正在初始化扬声器..."
        from zulong.l0.devices.speaker_device import SpeakerDevice
        
        try:
            # 创建 SpeakerDevice 实例（不再自动启动）
            self._speaker = SpeakerDevice()
            
            # 在独立线程中启动扬声器，避免事件循环冲突
            def _run_speaker():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._speaker.start())
                    loop.run_forever()
                except Exception as e:
                    logger.error(f"[SpeakerModule] 线程异常: {e}", exc_info=True)
            
            self._thread = threading.Thread(target=_run_speaker, daemon=True, name="SpeakerDevice")
            self._thread.start()
            
            # 等待扬声器启动
            await asyncio.sleep(1.0)
            
            self.state = ModuleState.RUNNING
            logger.info("[SpeakerModule] 扬声器已启动")
        except Exception as e:
            logger.error(f"[SpeakerModule] 启动失败: {e}", exc_info=True)
            self.state = ModuleState.RUNNING  # 标记为运行但记录错误

    async def stop(self) -> None:
        if self._speaker and hasattr(self._speaker, "stop"):
            try:
                await self._speaker.stop()
            except Exception as e:
                logger.warning(f"[SpeakerModule] 停止异常: {e}")
        self.state = ModuleState.STOPPED


# ── 11. RecoveryNotifierModule ──────────────────────

class RecoveryNotifierModule(Module):
    name = "recovery_notifier"
    display_name = "任务恢复检查"
    dependencies = ["inference_engine"]
    mode_tags: Set[str] = {"full"}

    async def start(self) -> None:
        self.progress_message = "正在检查可恢复任务..."
        try:
            from zulong.l2.recovery_notifier import RecoveryNotifier
            RecoveryNotifier.check_and_notify()
        except Exception as e:
            logger.warning(f"[RecoveryNotifierModule] 检查失败: {e}")
        self.state = ModuleState.RUNNING


# ── 导出函数 ────────────────────────────────────────

def get_full_modules() -> List[Module]:
    """返回 Full 模式的所有模块实例（供 LauncherApp 注册）"""
    return [
        ModelPreloadModule(),
        L1AReflexModule(),
        L1BGatekeeperModule(),
        L2InterruptModule(),
        ReplayIntegrationModule(),
        ReviewTriggerModule(),
        SkillPackModule(),
        MemoryEvolutionModule(),
        CameraModule(),
        MicrophoneModule(),
        SpeakerModule(),  # 🔊 新增：扬声器设备模块
        RecoveryNotifierModule(),
    ]
