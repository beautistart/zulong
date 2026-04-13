# File: openclaw_bridge\bootstrap.py
"""
OpenClaw Bridge 启动脚本

功能：
1. 初始化 EventBusClient（连接到祖龙系统）
2. 初始化 OpenClawMicAdapter（语音输入）
3. 初始化 OpenClawVisionReporter（视觉同步）
4. 初始化 ExecuteListener（执行指令监听）
5. 初始化 SpeakListener（语音播报监听）
6. 启动所有异步任务

对应 TSD v2.2 第 5 阶段（总装）
架构说明：所有模块通过 EventBus 与祖龙 L1-B 通信
"""

import asyncio
import signal
import logging
import time
from typing import Optional
from dataclasses import dataclass

from openclaw_bridge.event_bus_client import EventBusClient, EventBusConfig
from openclaw_bridge.adapters.mic_adapter import OpenClawMicAdapter, MicConfig
from openclaw_bridge.adapters.vision_reporter import OpenClawVisionReporter, VisionConfig
from openclaw_bridge.adapters.web_adapter import OpenClawWebAdapter, WebConfig
from openclaw_bridge.api_server import OpenClawAPIServer
from openclaw_bridge.listeners.execute_listener import ExecuteListener, ExecuteConfig
from openclaw_bridge.listeners.speak_listener import SpeakListener, SpeakConfig
from openclaw_bridge.listeners.web_response_listener import WebResponseListener

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 🔥 [架构修正] OpenClaw Bridge 只作为事件桥接器，不加载任何模型
# 所有智能都在祖龙主系统中，Bridge 只负责事件传递
# 导入 L1-B 事件路由逻辑（不加载模型，只注册事件处理器）
from zulong.l1b.scheduler_gatekeeper import gatekeeper

logger.info("[OpenClawBridge] ✅ 祖龙 L1-B Gatekeeper 事件路由已注册")


@dataclass
class BridgeConfig:
    """OpenClaw Bridge 配置"""
    # EventBus 配置
    event_bus_host: str = "localhost"
    event_bus_port: int = 5555
    client_name: str = "OpenClaw_Bridge"
    
    # Web 配置
    web_enabled: bool = True
    web_host: str = "localhost"
    web_port: int = 8080
    
    # API 服务器配置
    api_enabled: bool = True
    api_host: str = "localhost"
    api_port: int = 3000
    
    # 适配器配置
    mic_mock_mode: bool = False  # 🔥 关闭 Mock 模式，使用真实麦克风
    vision_mock_mode: bool = False  # 🔥 关闭 Mock 模式，使用真实视觉
    vision_report_rate: float = 10.0  # Hz
    
    # 监听器配置
    executor_mock_mode: bool = False  # 🔥 关闭 Mock 模式，执行真实指令
    speak_mock_mode: bool = False  # 🔥 关闭 Mock 模式，使用真实语音播报
    
    # 系统配置
    enable_mic: bool = True
    enable_vision: bool = True
    enable_executor: bool = True
    enable_speak: bool = True


class OpenClawBridge:
    """
    OpenClaw Bridge 主控制器
    
    工作流程：
    1. 连接祖龙系统 EventBus
    2. 启动所有适配器（输入）
    3. 启动所有监听器（输出）
    4. 维持运行直到用户中断
    """
    
    def __init__(self, config: Optional[BridgeConfig] = None):
        """
        初始化 OpenClaw Bridge
        
        Args:
            config: Bridge 配置
        """
        self.config = config or BridgeConfig()
        self._running = False
        
        logger.info("=" * 80)
        logger.info("🦾 OpenClaw Bridge 初始化")
        logger.info("=" * 80)
        
        # 1. 初始化 EventBus 客户端
        logger.info("\n[1/5] 初始化 EventBus 客户端...")
        event_bus_config = EventBusConfig(
            host=self.config.event_bus_host,
            port=self.config.event_bus_port,
            client_name=self.config.client_name
        )
        self.event_bus = EventBusClient(event_bus_config)
        logger.info("✅ EventBus 客户端已初始化")
        
        # 2. 初始化麦克风适配器
        if self.config.enable_mic:
            logger.info("\n[2/5] 初始化麦克风适配器...")
            mic_config = MicConfig(mock_mode=self.config.mic_mock_mode)
            self.mic_adapter = OpenClawMicAdapter(self.event_bus, mic_config)
            logger.info("✅ 麦克风适配器已初始化")
        else:
            self.mic_adapter = None
            logger.info("\n[2/5] 麦克风适配器已禁用")
        
        # 3. 初始化视觉报告器
        if self.config.enable_vision:
            logger.info("\n[3/5] 初始化视觉报告器...")
            vision_config = VisionConfig(
                mock_mode=self.config.vision_mock_mode,
                report_rate=self.config.vision_report_rate
            )
            self.vision_reporter = OpenClawVisionReporter(self.event_bus, vision_config)
            logger.info("✅ 视觉报告器已初始化")
        else:
            self.vision_reporter = None
            logger.info("\n[3/5] 视觉报告器已禁用")
        
        # 4. 初始化执行监听器
        if self.config.enable_executor:
            logger.info("\n[4/5] 初始化执行监听器...")
            executor_config = ExecuteConfig(mock_mode=self.config.executor_mock_mode)
            self.execute_listener = ExecuteListener(self.event_bus, executor_config)
            
            # 订阅 TASK_EXECUTE 事件
            from openclaw_bridge.openclaw_types import OpenClawEventType
            self.event_bus.subscribe(
                OpenClawEventType.TASK_EXECUTE,
                self.execute_listener.on_execute_event,
                "ExecuteListener"
            )
            logger.info("✅ 执行监听器已初始化（已订阅 TASK_EXECUTE）")
        else:
            self.execute_listener = None
            logger.info("\n[4/5] 执行监听器已禁用")
        
        # 5. 初始化语音播报监听器
        if self.config.enable_speak:
            logger.info("\n[5/5] 初始化语音播报监听器...")
            speak_config = SpeakConfig(mock_mode=self.config.speak_mock_mode)
            self.speak_listener = SpeakListener(self.event_bus, speak_config)
            
            # 订阅 ACTION_SPEAK 事件
            from openclaw_bridge.openclaw_types import OpenClawEventType
            self.event_bus.subscribe(
                OpenClawEventType.ACTION_SPEAK,
                self.speak_listener.on_speak_event,
                "SpeakListener"
            )
            logger.info("✅ 语音播报监听器已初始化（已订阅 ACTION_SPEAK）")
        else:
            self.speak_listener = None
            logger.info("\n[5/5] 语音播报监听器已禁用")
        
        # 6. 初始化 Web 适配器（如果启用）
        if self.config.web_enabled:
            logger.info("\n[6/7] 初始化 Web 适配器...")
            web_config = WebConfig(
                host=self.config.web_host,
                port=self.config.web_port
            )
            self.web_adapter = OpenClawWebAdapter(self.event_bus, web_config)
            
            # 初始化 Web 响应监听器
            self.web_response_listener = WebResponseListener(
                self.event_bus,
                self.web_adapter
            )
            
            # 订阅 ACTION_SPEAK 事件（用于语音播报推送到前端）
            from openclaw_bridge.openclaw_types import OpenClawEventType
            self.event_bus.subscribe(
                OpenClawEventType.ACTION_SPEAK,
                self.web_response_listener.on_speak_event,
                "WebResponseListener"
            )
            
            # 🔥 [关键修复] 订阅 L2_OUTPUT 事件（用于文本模式推送到前端）
            # 当用户通过 Web 文本输入时，L2 使用 TEXT_ONLY 模式，只发布 L2_OUTPUT
            self.event_bus.subscribe(
                OpenClawEventType.L2_OUTPUT,
                self.web_response_listener.on_l2_output_event,
                "WebResponseListener"
            )
            
            logger.info("✅ Web 适配器已初始化（已订阅 ACTION_SPEAK + L2_OUTPUT）")
        else:
            self.web_adapter = None
            self.web_response_listener = None
            logger.info("\n[6/7] Web 适配器已禁用")
        
        # 7. 初始化 API 服务器（如果启用）
        if self.config.api_enabled:
            logger.info("\n[7/7] 初始化 API 服务器...")
            self.api_server = OpenClawAPIServer(
                host=self.config.api_host,
                port=self.config.api_port
            )
            logger.info(f"✅ API 服务器已初始化于 http://{self.config.api_host}:{self.config.api_port}")
        else:
            self.api_server = None
            logger.info("\n[7/7] API 服务器已禁用")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ OpenClaw Bridge 初始化完成")
        logger.info("=" * 80)
    
    async def start(self):
        """启动 Bridge"""
        logger.info("\n🚀 启动 OpenClaw Bridge...")
        print("\n[START] 开始启动 Bridge...")
        self._running = True
        
        try:
            # 🔥 新增：检测祖龙系统是否已启动
            print(f"[START] 步骤 0: 检测祖龙系统启动状态...")
            logger.info("\n[0/4] 检测祖龙系统是否已启动...")
            
            zulong_started = await self._wait_for_zulong(timeout=60)
            if not zulong_started:
                logger.error("❌ 祖龙系统未在 60 秒内启动，Bridge 无法启动")
                return
            
            logger.info("✅ 祖龙系统已启动")
            
            # 1. 连接 EventBus
            print(f"[START] 步骤 1: 连接 EventBus...")
            logger.info("\n[1/4] 连接到祖龙系统 EventBus...")
            connected = await self.event_bus.connect()
            
            print(f"[START] EventBus 连接结果：{connected}")
            if not connected:
                logger.error("❌ 无法连接到 EventBus，启动失败")
                return
            
            logger.info("✅ EventBus 连接成功")
            
            # 2. 启动麦克风适配器
            if self.mic_adapter:
                logger.info("\n[2/4] 启动麦克风适配器...")
                await self.mic_adapter.start()
                logger.info("✅ 麦克风适配器已启动")
            
            # 3. 启动视觉报告器
            if self.vision_reporter:
                logger.info("\n[3/4] 启动视觉报告器...")
                await self.vision_reporter.start()
                logger.info("✅ 视觉报告器已启动")
            
            # 3.5. 启动 Web 适配器
            if self.web_adapter:
                logger.info("\n[3.5/5] 启动 Web 适配器...")
                await self.web_adapter.start()
                logger.info(f"✅ Web 适配器已启动于 http://{self.config.web_host}:{self.config.web_port}")
            
            # 3.6. 启动 API 服务器
            if self.api_server:
                logger.info("\n[3.6/5] 启动 API 服务器...")
                await self.api_server.start()
                logger.info(f"✅ API 服务器已启动于 http://{self.config.api_host}:{self.config.api_port}")
            
            # 4. 注册事件处理器
            logger.info("\n[4/5] 注册事件处理器...")
            if self.execute_listener:
                logger.info("  ✅ 执行事件处理器已注册")
            if self.speak_listener:
                logger.info("  ✅ 语音播报事件处理器已注册")
            
            logger.info("\n" + "=" * 80)
            logger.info("🎉 OpenClaw Bridge 启动成功！")
            logger.info("=" * 80)
            logger.info("\n💡 系统运行中，按 Ctrl+C 停止...")
            
            # 维持运行
            await self._run_loop()
            
        except KeyboardInterrupt:
            logger.info("\n\n👋 用户中断，正在停止...")
        except Exception as e:
            logger.error(f"\n❌ 运行错误：{e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            await self.stop()
    
    async def _wait_for_zulong(self, timeout: int = 60) -> bool:
        """
        等待祖龙系统启动
        
        Args:
            timeout: 超时时间（秒），默认 60 秒
            
        Returns:
            bool: 祖龙系统是否在超时时间内启动
        """
        import aiohttp
        
        elapsed = 0
        while elapsed < timeout:
            try:
                # 尝试连接 EventBus WebSocket
                async with aiohttp.ClientSession() as session:
                    ws_url = f"ws://{self.config.event_bus_host}:{self.config.event_bus_port}/eventbus"
                    async with session.ws_connect(ws_url, timeout=aiohttp.ClientTimeout(total=2)) as ws:
                        # 能连接上说明 EventBus 已启动
                        await ws.close()
                        logger.info(f"✅ 祖龙系统 EventBus 已启动（耗时：{elapsed}秒）")
                        return True
                        
            except Exception:
                elapsed += 1
                if elapsed % 5 == 0:  # 每 5 秒提示一次
                    logger.info(f"⏳ 等待祖龙系统启动...（已等待 {elapsed}秒 / 最多 {timeout}秒）")
                await asyncio.sleep(1.0)
        
        logger.warning(f"⚠️ 祖龙系统在 {timeout}秒内未启动")
        return False
    
    async def _run_loop(self):
        """主运行循环"""
        while self._running:
            await asyncio.sleep(1.0)
            
            # 检查连接状态
            if not self.event_bus.is_connected:
                logger.warning("⚠️ EventBus 连接断开，尝试重连...")
                await self.event_bus.connect()
    
    async def stop(self):
        """停止 Bridge"""
        logger.info("\n🛑 停止 OpenClaw Bridge...")
        self._running = False
        
        # 1. 停止适配器
        if self.mic_adapter:
            logger.info("  - 停止麦克风适配器...")
            await self.mic_adapter.stop()
        
        if self.vision_reporter:
            logger.info("  - 停止视觉报告器...")
            await self.vision_reporter.stop()
        
        if self.web_adapter:
            logger.info("  - 停止 Web 适配器...")
            await self.web_adapter.stop()
        
        # 3. 停止 API 服务器
        if self.api_server:
            logger.info("  - 停止 API 服务器...")
            await self.api_server.stop()
        
        # 4. 断开 EventBus
        logger.info("  - 断开 EventBus 连接...")
        await self.event_bus.disconnect()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ OpenClaw Bridge 已安全停止")
        logger.info("=" * 80)


async def main():
    """主函数"""
    # 创建配置
    config = BridgeConfig(
        # 使用 Mock 模式（开发测试）
        mic_mock_mode=True,
        vision_mock_mode=True,
        executor_mock_mode=True,
        speak_mock_mode=True,
        
        # 启用 Web 功能
        web_enabled=True,
        web_host="localhost",
        web_port=8080,
        
        # 启用所有模块
        enable_mic=True,
        enable_vision=True,
        enable_executor=True,
        enable_speak=True
    )
    
    # 创建并启动 Bridge
    bridge = OpenClawBridge(config)
    await bridge.start()


def run():
    """运行入口函数"""
    print("\n" + "=" * 80)
    print("OpenClaw Bridge - 祖龙系统集成".center(80))
    print("=" * 80)
    print(f"启动时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        logger.error(f"启动失败：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run()
