# File: zulong/tools/debug_console.py
# 交互式调试控制台
# 对应 TSD v1.7: 调试工具模块

import threading
import time
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

from zulong.core.event_bus import event_bus
from zulong.core.state_manager import state_manager
from zulong.core.types import EventType, EventPriority, ZulongEvent
from zulong.l2.task_state_manager import task_state_manager
from zulong.l0.devices.speaker_device import SpeakerDevice

import logging
logger = logging.getLogger(__name__)


def safe_print(text: str) -> None:
    """安全打印（处理 emoji 编码问题）
    
    Args:
        text: 要打印的文本（可能包含 emoji）
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # Windows 控制台无法显示 emoji，替换为 ? 后输出
        safe_text = text.encode('gbk', errors='replace').decode('gbk')
        print(safe_text)


class DebugConsole:
    """交互式调试控制台
    
    提供双向通信能力：
    - 输入: 捕获用户键盘输入 -> 封装为 USER_SPEECH 事件 -> 发布到 EventBus
    - 输出: 订阅 SYSTEM_LOG, L2_OUTPUT, STATE_CHANGE 事件 -> 格式化打印到控制台
    
    支持特殊指令 (以 / 开头):
    - /status: 打印当前系统状态
    - /freeze: 手动冻结当前任务
    - /resume <id>: 手动恢复指定任务
    - /inject <event_type> <payload>: 手动注入任意事件
    - /clear: 清屏
    - /help: 显示帮助
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化调试控制台"""
        if not hasattr(self, '_initialized'):
            self._running = False
            self._input_thread: Optional[threading.Thread] = None
            self._stream_prefix_printed = False  # 流式输出前缀标志
            self._initialized = True
            
            # 🎯 初始化扬声器设备 (自动启动)
            logger.info("初始化扬声器设备...")
            self.speaker = SpeakerDevice()
            logger.info("✅ 扬声器设备初始化完成")
            
            # 订阅系统事件用于输出显示
            self._subscribe_to_events()
            
            logger.info("DebugConsole initialized. Type '/help' for commands.")
    
    def _subscribe_to_events(self):
        """订阅系统事件用于输出显示"""
        # 订阅 L2 输出事件
        event_bus.subscribe(
            EventType.L2_OUTPUT,
            self._on_l2_output,
            "DebugConsole"
        )
        
        # 订阅 L2 流式输出事件
        event_bus.subscribe(
            EventType.L2_OUTPUT_STREAM,
            self._on_l2_output_stream,
            "DebugConsole"
        )
        
        # 也兼容 SYSTEM_L2_READY
        event_bus.subscribe(
            EventType.SYSTEM_L2_READY,
            self._on_l2_output,
            "DebugConsole"
        )
        
        # 订阅状态变更事件 (通过 SYSTEM_INTERRUPT 模拟)
        event_bus.subscribe(
            EventType.SYSTEM_INTERRUPT,
            self._on_state_change,
            "DebugConsole"
        )
        
        # 订阅任务事件
        event_bus.subscribe(
            EventType.TASK_CREATED,
            self._on_task_event,
            "DebugConsole"
        )
        event_bus.subscribe(
            EventType.TASK_FROZEN,
            self._on_task_event,
            "DebugConsole"
        )
        event_bus.subscribe(
            EventType.TASK_RESUMED,
            self._on_task_event,
            "DebugConsole"
        )
        event_bus.subscribe(
            EventType.TASK_COMPLETED,
            self._on_task_event,
            "DebugConsole"
        )
        
        # 订阅系统反射事件
        event_bus.subscribe(
            EventType.SYSTEM_REFLEX,
            self._on_system_event,
            "DebugConsole"
        )
    
    def start(self):
        """启动调试控制台"""
        if self._running:
            return
        
        self._running = True
        
        # 检测 stdin 是否可用（非后台进程）
        if sys.stdin is not None and sys.stdin.readable() and hasattr(sys.stdin, 'isatty') and sys.stdin.isatty():
            # 启动输入线程
            self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
            self._input_thread.start()
            
            logger.info("Debug Console started. Waiting for input...")
            print("\n" + "="*50)
            print("  [ZULONG] Debug Console")
            print("="*50)
            print("Type your message or '/help' for commands.\n")
        else:
            logger.info("Debug Console started in headless mode (no stdin). Use WebSocket or Bridge for interaction.")
    
    def stop(self):
        """停止调试控制台"""
        self._running = False
        
        if self._input_thread:
            self._input_thread.join(timeout=1)
        
        logger.info("Debug Console stopped")
    
    def _input_loop(self):
        """输入循环 - 捕获用户输入并处理（修复版）"""
        # 给系统一点初始化时间
        time.sleep(1.0)
        
        print("\n" + "="*50)
        print("  [MIC] 调试控制台已启动，可以开始输入")
        print("="*50)
        
        logger.info("[DebugConsole] Input loop started")
        
        eof_count = 0  # 连续 EOFError 计数
        max_eof_retries = 5  # 连续 EOFError 超过此数则退出输入循环
        
        while self._running:
            try:
                # 显示提示符（使用 stderr 避免与 stdout 冲突）
                sys.stderr.write("\n[You] ")
                sys.stderr.flush()
                
                try:
                    user_input = input().strip()
                    eof_count = 0  # 成功读取，重置计数
                        
                except EOFError:
                    eof_count += 1
                    if eof_count >= max_eof_retries:
                        logger.warning(f"[DebugConsole] {max_eof_retries} consecutive EOFErrors, stdin unavailable. Stopping input loop.")
                        break
                    logger.warning(f"[DebugConsole] EOFError ({eof_count}/{max_eof_retries}), retrying...")
                    time.sleep(1.0)
                    continue
                except Exception as input_e:
                    # input() 调用失败
                    logger.error(f"[DebugConsole] input() failed: {input_e}")
                    time.sleep(0.5)
                    continue
                
                if not user_input:
                    continue
                
                # 处理特殊指令
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    # 普通文本 -> 封装为 USER_SPEECH 事件
                    self._publish_user_speech(user_input)
                    
            except KeyboardInterrupt:
                # Ctrl+C
                print("\nUse '/quit' or Ctrl+D to exit.")
            except Exception as e:
                logger.error(f"Input loop error: {e}")
                time.sleep(0.5)
    
    def _output_loop(self):
        """输出循环 - 格式化打印系统事件"""
        while self._running:
            try:
                with self._lock:
                    if self._output_queue:
                        message = self._output_queue.pop(0)
                        print(message)
                
                time.sleep(0.05)  # 避免忙等
            except Exception as e:
                logger.error(f"Output loop error: {e}")
    
    def _enqueue_output(self, message: str):
        """将消息加入输出队列"""
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._output_queue.append(f"[{timestamp}] {message}")
    
    def _publish_user_speech(self, text: str):
        """发布用户语音事件
        
        Args:
            text: 用户输入文本
        """
        event = ZulongEvent(
            type=EventType.USER_SPEECH,
            priority=EventPriority.NORMAL,
            source="DebugConsole",
            payload={
                "text": text,
                "confidence": 1.0,
                "timestamp": time.time()
            }
        )
        event_bus.publish(event)
        logger.info(f"Published USER_SPEECH event: '{text[:50]}...' " if len(text) > 50 else f"Published USER_SPEECH event: '{text}'")
    
    def _handle_command(self, cmd: str):
        """处理特殊指令
        
        Args:
            cmd: 指令字符串 (以 / 开头)
        """
        parts = cmd.split()
        command = parts[0].lower()
        args = parts[1:]
        
        if command == '/help':
            self._show_help()
        elif command == '/status':
            self._show_status()
        elif command == '/freeze':
            self._freeze_task()
        elif command == '/resume':
            if len(args) < 1:
                safe_print("❌ Usage: /resume <task_id>")
            else:
                self._resume_task(args[0])
        elif command == '/inject':
            if len(args) < 2:
                safe_print("❌ Usage: /inject <event_type> <payload>")
                safe_print("   Example: /inject SENSOR_OBSTACLE '{\"distance\": 0.5}'")
            else:
                self._inject_event(args[0], ' '.join(args[1:]))
        elif command == '/clear':
            self._clear_screen()
        elif command == '/graph':
            self._show_graph_status(args)
        elif command == '/quit' or command == '/exit':
            safe_print("👋 Exiting debug console...")
            self._running = False
        else:
            safe_print(f"❌ Unknown command: {command}")
            safe_print("Type '/help' for available commands.")
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
╔══════════════════════════════════════════════════════════╗
║              [ZULONG] Debug Console Commands              ║
╠══════════════════════════════════════════════════════════╣
║  /status                  - 显示系统状态                  ║
║  /freeze                  - 冻结当前活跃任务              ║
║  /resume <task_id>        - 恢复指定任务                  ║
║  /inject <type> <payload> - 手动注入事件                  ║
║  /graph [save]            - 查看/保存记忆图谱状态          ║
║  /clear                   - 清屏                          ║
║  /help                    - 显示此帮助                    ║
║  /quit 或 /exit            - 退出控制台                   ║
╠══════════════════════════════════════════════════════════╣
║  普通文本输入将自动封装为 USER_SPEECH 事件                 ║
╚══════════════════════════════════════════════════════════╝
"""
        safe_print(help_text)
    
    def _show_status(self):
        """显示当前系统状态"""
        power_state = state_manager.get_power_state()
        l2_status = state_manager.get_l2_status()
        active_task = task_state_manager.get_active_task()
        task_stack = task_state_manager.get_task_stack()
        
        status_text = f"""
╔══════════════════════════════════════════════════════════╗
║                    📊 System Status                       ║
╠══════════════════════════════════════════════════════════╣
║  Power Mode:    {power_state.name:<20}                    ║
║  L2 Status:     {l2_status.name:<20}                    ║
║  Active Task:   {str(active_task):<20}                    ║
║  Task Stack:    {len(task_stack)} tasks                   ║
╚══════════════════════════════════════════════════════════╝
"""
        safe_print(status_text)
        
        if task_stack:
            safe_print("📚 Task Stack (top to bottom):")
            for i, task_id in enumerate(reversed(task_stack), 1):
                safe_print(f"   {i}. {task_id}")
    
    def _freeze_task(self):
        """手动冻结当前任务"""
        active_task = task_state_manager.get_active_task()
        if active_task is None:
            safe_print("⚠️  No active task to freeze.")
            return
        
        # 发布任务冻结事件
        event = ZulongEvent(
            type=EventType.TASK_FROZEN,
            priority=EventPriority.HIGH,
            source="DebugConsole",
            payload={
                "task_id": active_task,
                "reason": "manual_freeze",
                "timestamp": time.time()
            }
        )
        event_bus.publish(event)
        
        # 调用任务状态管理器冻结
        task_state_manager.freeze_current()
        safe_print(f"🧊 Task '{active_task}' frozen manually.")
    
    def _resume_task(self, task_id: str):
        """手动恢复指定任务
        
        Args:
            task_id: 任务 ID
        """
        task_stack = task_state_manager.get_task_stack()
        
        if task_id not in task_stack:
            safe_print(f"❌ Task '{task_id}' not found in frozen tasks.")
            safe_print(f"   Available tasks: {task_stack}")
            return
        
        # 发布任务恢复事件
        event = ZulongEvent(
            type=EventType.TASK_RESUMED,
            priority=EventPriority.HIGH,
            source="DebugConsole",
            payload={
                "task_id": task_id,
                "timestamp": time.time()
            }
        )
        event_bus.publish(event)
        
        # 调用任务状态管理器恢复
        task_state_manager.resume_task(task_id)
        safe_print(f"▶️  Task '{task_id}' resumed.")
    
    def _inject_event(self, event_type: str, payload_str: str):
        """手动注入任意事件
        
        Args:
            event_type: 事件类型名称
            payload_str: JSON 格式的 payload 字符串
        """
        try:
            # 解析事件类型
            try:
                event_type_enum = EventType[event_type.upper()]
            except KeyError:
                available_types = [et.name for et in EventType]
                safe_print(f"❌ Unknown event type: {event_type}")
                safe_print(f"   Available types: {', '.join(available_types)}")
                return
            
            # 解析 payload
            import json
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                # 如果不是有效 JSON，作为简单字符串处理
                payload = {"message": payload_str}
            
            # 创建并发布事件
            event = ZulongEvent(
                type=event_type_enum,
                priority=EventPriority.HIGH,
                source="DebugConsole.Injection",
                payload=payload
            )
            event_bus.publish(event)
            
            safe_print(f"💉 Injected event: {event_type}")
            safe_print(f"   Payload: {payload}")
            
        except Exception as e:
            safe_print(f"❌ Failed to inject event: {e}")
    
    def _clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
        safe_print("[ZULONG] Debug Console - Screen cleared.\n")

    def _show_graph_status(self, args=None):
        """显示/保存 MemoryGraph 状态"""
        try:
            from zulong.memory.memory_graph import MemoryGraph
            mg = MemoryGraph._instance
            if mg is None:
                safe_print("[MemoryGraph] 未初始化 (降级运行中)")
                return

            stats = mg.stats
            safe_print("\n" + "=" * 56)
            safe_print("  [MemoryGraph] 记忆图谱状态")
            safe_print("=" * 56)
            safe_print(f"  节点总数: {stats['total_nodes']}")
            safe_print(f"  边总数:   {stats['total_edges']}")

            # 节点类型分布
            if stats.get('node_types'):
                safe_print(f"  节点类型: {stats['node_types']}")
            if stats.get('edge_types'):
                safe_print(f"  边类型:   {stats['edge_types']}")

            # 列出前 10 个节点
            nodes = list(mg._nodes.values())
            if nodes:
                safe_print(f"\n  前 {min(10, len(nodes))} 个节点:")
                for n in nodes[:10]:
                    safe_print(f"    [{n.node_type.value}] {n.node_id}: {n.label}")

            # 列出前 10 条边
            edges = list(mg._graph.edges(data=True))
            if edges:
                safe_print(f"\n  前 {min(10, len(edges))} 条边:")
                for src, dst, data in edges[:10]:
                    etype = data.get('edge_type', '?')
                    weight = data.get('weight', 0)
                    prot = " [protected]" if data.get('protected') else ""
                    safe_print(f"    {src} --[{etype} w={weight:.2f}{prot}]--> {dst}")

            # 适配器状态
            adapters = list(mg._adapters.keys())
            if adapters:
                safe_print(f"\n  已注册适配器: {adapters}")

            # 保存命令
            if args and 'save' in args:
                mg.save()
                safe_print(f"\n  >> MemoryGraph 已保存到 {mg._persist_path}")

            safe_print("=" * 56)
        except Exception as e:
            safe_print(f"[MemoryGraph] 查询失败: {e}")
    
    # 事件处理器
    def _on_l2_output(self, event: ZulongEvent):
        """处理 L2 输出事件"""
        payload = event.payload
        # 兼容两种格式：text (L2_OUTPUT) 和 message (SYSTEM_L2_READY)
        text = payload.get("text", payload.get("message", "L2 Output"))
        # 重置流式前缀标志，以便下次流式输出时重新打印前缀
        self._stream_prefix_printed = False
        # L2_OUTPUT 事件打印完整回复（在流式输出后）
        if event.type == EventType.L2_OUTPUT:
            # 先换行，确保在单独的一行显示
            # 🔥 修复：Windows 控制台 emoji 编码错误，使用 errors='replace' 处理无法显示的字符
            try:
                print(f"\n[ZULONG] {text}\n")
            except UnicodeEncodeError:
                # 如果包含无法显示的字符（如 emoji），替换为 ? 后输出
                safe_text = text.encode('gbk', errors='replace').decode('gbk')
                print(f"\n[ZULONG] {safe_text}\n")
        elif event.type == EventType.SYSTEM_L2_READY:
            self._enqueue_output(f"\n[ZULONG] {text}")
    
    def _on_state_change(self, event: ZulongEvent):
        """处理状态变更事件"""
        payload = event.payload
        reason = payload.get("reason", "State changed")
        self._enqueue_output(f"[STATE] {reason}")
    
    def _on_task_event(self, event: ZulongEvent):
        """处理任务事件"""
        event_name = event.type.name
        task_id = event.payload.get("task_id", "unknown")
        self._enqueue_output(f"[TASK:{event_name}] {task_id}")
    
    def _on_system_event(self, event: ZulongEvent):
        """处理系统事件"""
        payload = event.payload
        command = payload.get("command", "unknown")
        self._enqueue_output(f"[SYSTEM] Reflex command: {command}")
    
    def _on_l2_output_stream(self, event: ZulongEvent):
        """处理 L2 流式输出事件"""
        payload = event.payload
        text = payload.get("text", "")
        if text:
            # 第一次流式事件时打印前缀，后续只打印文本
            if not self._stream_prefix_printed:
                # 先换行，确保流式输出在单独的一行
                print("\n[ZULONG] ", end="", flush=True)
                self._stream_prefix_printed = True
            # 只打印文本片段，不添加前缀
            print(text, end="", flush=True)


# 全局调试控制台实例
debug_console = DebugConsole()


if __name__ == "__main__":
    """独立运行模式"""
    # 配置基础日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 启动控制台
    console = DebugConsole()
    console.start()
    
    # 保持主线程运行
    try:
        while console._running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        console.stop()
