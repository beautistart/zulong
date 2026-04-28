# File: openclaw_bridge\listeners\execute_listener.py
"""
OpenClaw 执行指令监听器

功能：
1. 监听 TASK_EXECUTE 事件（来自 L1-B）
2. 解析 payload.name 和 payload.arguments
3. 调用 OpenClaw SDK 对应的函数（如 claw.grasp(object_id)）
4. 执行完成后，生成 ACTION_RESULT 事件回传给祖龙 L2

对应 TSD v2.2 第 3.1 节（事件模型）、第 4.3 节（L1-B 下行链路）
关键逻辑：执行完成后必须回传 ACTION_RESULT，否则 L2 会卡在等待状态
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from openclaw_bridge.openclaw_types import (
    ZulongEvent,
    OpenClawEventType,
    OpenClawEventPriority,
    create_action_result_event
)
from openclaw_bridge.event_bus_client import EventBusClient

logger = logging.getLogger(__name__)


@dataclass
class ExecuteConfig:
    """执行器配置"""
    mock_mode: bool = True  # Mock 模式
    execution_timeout: float = 10.0  # 执行超时（秒）
    enable_feedback: bool = True  # 启用执行反馈


class OpenClawExecutor:
    """
    OpenClaw 执行器（Mock 实现）
    
    模拟 OpenClaw SDK 的功能
    TODO: 替换为真实的 OpenClaw SDK
    """
    
    def __init__(self, mock_mode: bool = True):
        self.mock_mode = mock_mode
        logger.info(f"[OpenClawExecutor] 初始化完成，Mock 模式：{mock_mode}")
    
    async def grasp_object(self, object_id: str) -> Dict[str, Any]:
        """
        抓取物体
        
        Args:
            object_id: 物体 ID
        
        Returns:
            Dict: 执行结果
        """
        logger.info(f"[OpenClawExecutor] 执行抓取：{object_id}")
        
        if self.mock_mode:
            # Mock 模式：模拟执行
            await asyncio.sleep(1.0)  # 模拟执行时间
            
            # 模拟成功
            return {
                "success": True,
                "message": f"成功抓取物体 {object_id}",
                "object_id": object_id,
                "timestamp": time.time()
            }
        else:
            # TODO: 真实 OpenClaw SDK 调用
            # return await claw.grasp(object_id)
            raise NotImplementedError("真实模式尚未实现")
    
    async def move_arm(self, x: float, y: float, z: float) -> Dict[str, Any]:
        """
        移动机械臂
        
        Args:
            x, y, z: 目标坐标
        
        Returns:
            Dict: 执行结果
        """
        logger.info(f"[OpenClawExecutor] 移动机械臂到 ({x}, {y}, {z})")
        
        if self.mock_mode:
            await asyncio.sleep(1.5)
            return {
                "success": True,
                "message": f"机械臂已移动到 ({x}, {y}, {z})",
                "position": {"x": x, "y": y, "z": z},
                "timestamp": time.time()
            }
        else:
            # TODO: 真实 OpenClaw SDK 调用
            raise NotImplementedError("真实模式尚未实现")
    
    async def place_object(self, object_id: str, x: float, y: float, z: float) -> Dict[str, Any]:
        """
        放置物体
        
        Args:
            object_id: 物体 ID
            x, y, z: 放置位置
        
        Returns:
            Dict: 执行结果
        """
        logger.info(f"[OpenClawExecutor] 放置物体 {object_id} 到 ({x}, {y}, {z})")
        
        if self.mock_mode:
            await asyncio.sleep(1.2)
            return {
                "success": True,
                "message": f"物体 {object_id} 已放置到 ({x}, {y}, {z})",
                "object_id": object_id,
                "position": {"x": x, "y": y, "z": z},
                "timestamp": time.time()
            }
        else:
            raise NotImplementedError("真实模式尚未实现")


class ExecuteListener:
    """
    执行指令监听器
    
    工作流程：
    1. 订阅 TASK_EXECUTE 事件
    2. 解析事件 payload
    3. 调用对应的执行函数
    4. 执行完成后，发送 ACTION_RESULT 事件回传
    """
    
    def __init__(
        self,
        event_bus: EventBusClient,
        config: Optional[ExecuteConfig] = None
    ):
        """
        初始化执行监听器
        
        Args:
            event_bus: EventBus 客户端
            config: 执行器配置
        """
        self.event_bus = event_bus
        self.config = config or ExecuteConfig()
        self.executor = OpenClawExecutor(mock_mode=self.config.mock_mode)
        
        # 注册动作映射
        self._action_handlers = {
            "grasp": self._handle_grasp,
            "grasp_object": self._handle_grasp,
            "move_arm": self._handle_move_arm,
            "place": self._handle_place,
            "place_object": self._handle_place
        }
        
        logger.info("[ExecuteListener] 初始化完成")
    
    async def on_execute_event(self, event: ZulongEvent):
        """
        处理执行指令事件
        
        Args:
            event: 执行指令事件
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 [ExecuteListener] 收到执行指令")
        logger.info(f"🤖 [ExecuteListener] 动作名称：{event.payload.get('name')}")
        logger.info(f"🤖 [ExecuteListener] 动作参数：{event.payload.get('arguments')}")
        logger.info(f"{'='*80}\n")
        
        action_name = event.payload.get("name")
        arguments = event.payload.get("arguments", {})
        
        if not action_name:
            logger.error("[ExecuteListener] ❌ 动作名称为空")
            return
        
        # 查找处理函数
        handler = self._action_handlers.get(action_name)
        
        if not handler:
            logger.error(f"[ExecuteListener] ❌ 未知动作：{action_name}")
            await self._send_result(action_name, False, {"error": "Unknown action"})
            return
        
        try:
            # 执行动作
            result = await handler(arguments)
            
            # 发送执行结果
            await self._send_result(action_name, True, result)
            
        except asyncio.TimeoutError:
            logger.error(f"[ExecuteListener] ❌ 执行超时：{action_name}")
            await self._send_result(
                action_name,
                False,
                {"error": "Execution timeout"}
            )
        
        except Exception as e:
            logger.error(f"[ExecuteListener] ❌ 执行失败：{e}")
            import traceback
            logger.error(traceback.format_exc())
            await self._send_result(action_name, False, {"error": str(e)})
    
    async def _handle_grasp(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理抓取动作
        
        Args:
            arguments: 动作参数（包含 object_id）
        
        Returns:
            Dict: 执行结果
        """
        object_id = arguments.get("object_id") or arguments.get("object_name")
        
        if not object_id:
            raise ValueError("缺少参数：object_id")
        
        return await self.executor.grasp_object(object_id)
    
    async def _handle_move_arm(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理移动机械臂动作
        
        Args:
            arguments: 动作参数（包含 x, y, z）
        
        Returns:
            Dict: 执行结果
        """
        x = arguments.get("x", 0.0)
        y = arguments.get("y", 0.0)
        z = arguments.get("z", 0.0)
        
        return await self.executor.move_arm(x, y, z)
    
    async def _handle_place(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理放置动作
        
        Args:
            arguments: 动作参数（包含 object_id, x, y, z）
        
        Returns:
            Dict: 执行结果
        """
        object_id = arguments.get("object_id") or arguments.get("object_name")
        x = arguments.get("x", 0.0)
        y = arguments.get("y", 0.0)
        z = arguments.get("z", 0.0)
        
        if not object_id:
            raise ValueError("缺少参数：object_id")
        
        return await self.executor.place_object(object_id, x, y, z)
    
    async def _send_result(
        self,
        action_name: str,
        success: bool,
        result: Any = None
    ):
        """
        发送执行结果
        
        Args:
            action_name: 动作名称
            success: 是否成功
            result: 结果数据
        """
        if not self.config.enable_feedback:
            logger.warning("[ExecuteListener] ⚠️ 反馈已禁用，不发送结果")
            return
        
        # 创建动作结果事件
        result_event = create_action_result_event(
            action_name=action_name,
            success=success,
            result=result
        )
        
        # 发布到 EventBus
        self.event_bus.publish(result_event)
        
        logger.info(
            f"[ExecuteListener] ✅ 执行结果已发送：{action_name} "
            f"({'成功' if success else '失败'})"
        )
    
    def register_action(self, name: str, handler: Callable):
        """
        注册自定义动作
        
        Args:
            name: 动作名称
            handler: 处理函数
        """
        self._action_handlers[name] = handler
        logger.info(f"[ExecuteListener] ✅ 已注册动作：{name}")


# 便捷创建函数

def create_execute_listener(
    event_bus: EventBusClient,
    mock_mode: bool = True
) -> ExecuteListener:
    """
    创建执行监听器
    
    Args:
        event_bus: EventBus 客户端
        mock_mode: 是否使用 Mock 模式
    
    Returns:
        ExecuteListener: 执行监听器实例
    """
    config = ExecuteConfig(mock_mode=mock_mode)
    listener = ExecuteListener(event_bus, config)
    return listener
