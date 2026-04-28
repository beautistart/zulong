# File: tests/test_full_stack_interaction.py
"""
完整系统联动测试（L0-L4）

测试场景：
1. 真实摄像头采集
2. L1: 人体检测
3. L2: 运动检测
4. L3: 意图识别
5. L4: 手势识别
6. EventBus 事件流转
7. L0 执行器响应

测试步骤：
1. 坐在摄像头前静止（测试 Layer 1）
2. 挥动手臂（测试 Layer 2）
3. 继续挥手 2-3 秒（测试 Layer 3）
4. 比出 V 字手势（测试 Layer 4）
5. 观察 EventBus 事件和 L0 响应
"""

import asyncio
import time
import cv2
import logging
from typing import Dict, List, Any
from datetime import datetime

from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("FullStackTest")


class EventMonitor:
    """事件监控器 - 捕获并显示所有相关事件"""
    
    def __init__(self):
        self.events: List[ZulongEvent] = []
        self.event_timeline: List[Dict[str, Any]] = []
        
        # 订阅所有相关事件
        event_bus.subscribe(EventType.SENSOR_VISION_STATE, self._on_vision_state, "TestMonitor")
        event_bus.subscribe(EventType.INTERACTION_TRIGGER, self._on_interaction, "TestMonitor")
        event_bus.subscribe(EventType.DIRECT_WAKEUP, self._on_wakeup, "TestMonitor")
        event_bus.subscribe(EventType.CMD_EMERGENCY_STOP, self._on_emergency, "TestMonitor")
        event_bus.subscribe(EventType.CMD_BRAKE, self._on_brake, "TestMonitor")
        
        logger.info("✅ 事件监控器初始化完成")
    
    def _on_vision_state(self, event: ZulongEvent):
        """处理视觉状态事件"""
        self._record_event("SENSOR_VISION_STATE", event)
        motion_pixels = event.payload.get('motion_pixels', 0)
        logger.info(f"👆 [L2] 检测到运动：{motion_pixels}像素")
    
    def _on_interaction(self, event: ZulongEvent):
        """处理交互事件"""
        self._record_event("INTERACTION_TRIGGER", event)
        
        layer = event.payload.get('layer', '?')
        intent = event.payload.get('intent_type', '?')
        gesture = event.payload.get('gesture', 'N/A')
        route_mode = event.payload.get('route_mode', '?')
        
        logger.info(f"🧠 [L{layer}] 交互请求：intent={intent}, gesture={gesture}, mode={route_mode}")
    
    def _on_wakeup(self, event: ZulongEvent):
        """处理唤醒事件"""
        self._record_event("DIRECT_WAKEUP", event)
        logger.info(f"⚡ [L1-B] 直接唤醒 L2")
    
    def _on_emergency(self, event: ZulongEvent):
        """处理紧急停止事件"""
        self._record_event("CMD_EMERGENCY_STOP", event)
        logger.warning(f"🛑 [L0] 紧急停止！")
    
    def _on_brake(self, event: ZulongEvent):
        """处理刹车事件"""
        self._record_event("CMD_BRAKE", event)
        logger.warning(f"🛑 [L0] 刹车！")
    
    def _record_event(self, event_type: str, event: ZulongEvent):
        """记录事件到时间线"""
        self.events.append(event)
        self.event_timeline.append({
            'timestamp': event.timestamp,
            'time_str': datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S.%f')[:-3],
            'type': event_type,
            'priority': event.priority.value,
            'source': event.source,
            'payload': event.payload
        })
    
    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        stats = {}
        for event in self.events:
            key = f"{event.type.value}"
            stats[key] = stats.get(key, 0) + 1
        return stats
    
    def print_timeline(self):
        """打印事件时间线"""
        print("\n" + "=" * 80)
        print(" 事件时间线")
        print("=" * 80)
        
        if not self.event_timeline:
            print("⚠️ 没有记录到任何事件")
            return
        
        for i, item in enumerate(self.event_timeline, 1):
            print(f"\n[{i}] {item['time_str']} | {item['type']:<25} | P{item['priority']} | {item['source']}")
            
            # 显示关键 payload
            if 'motion_pixels' in item['payload']:
                print(f"    └─ motion_pixels: {item['payload']['motion_pixels']}")
            if 'intent_type' in item['payload']:
                print(f"    └─ intent: {item['payload']['intent_type']} ({item['payload'].get('intent_confidence', 0):.0%})")
            if 'gesture' in item['payload']:
                print(f"    └─ gesture: {item['payload']['gesture']} ({item['payload'].get('gesture_confidence', 0):.0%})")
            if 'route_mode' in item['payload']:
                print(f"    └─ route_mode: {item['payload']['route_mode']}")


async def test_full_stack():
    """完整系统联动测试"""
    
    print("=" * 80)
    print(" 祖龙系统 - 完整联动测试 (L0-L4)")
    print("=" * 80)
    print()
    
    # 初始化事件监控器
    monitor = EventMonitor()
    
    # 初始化视觉处理器
    print("初始化 OptimizedVisionProcessor...")
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    logger.info("✅ 视觉处理器初始化完成")
    
    # 打开摄像头
    print("\n打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("❌ 无法打开摄像头")
        return False
    
    logger.info("✅ 摄像头已打开")
    
    # 等待模型加载
    print("\n等待 3 秒让模型完全加载...")
    await asyncio.sleep(3)
    
    # 开始测试
    print("\n" + "=" * 80)
    print(" 测试说明")
    print("=" * 80)
    print("请按以下步骤操作：")
    print()
    print("1️⃣ 坐在摄像头前静止 3 秒（测试 Layer 1 - 人体检测）")
    print("2️⃣ 挥动手臂 2-3 秒（测试 Layer 2 - 运动检测）")
    print("3️⃣ 继续挥手并比出 V 字手势（测试 Layer 3/4 - 意图 + 手势）")
    print("4️⃣ 观察控制台输出和可视化窗口")
    print()
    print("按 'q' 键随时退出测试")
    print("=" * 80)
    print()
    
    # 等待用户准备
    print("准备倒计时：3...", end='', flush=True)
    await asyncio.sleep(1)
    print("2...", end='', flush=True)
    await asyncio.sleep(1)
    print("1...", end='', flush=True)
    await asyncio.sleep(1)
    print("开始！\n")
    
    # 开始采集
    start_time = time.time()
    frame_count = 0
    test_duration = 60  # 秒
    
    try:
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = time.time()
            
            # 喂食帧到视觉处理器
            processor.feed_frame(frame, timestamp)
            
            # 获取状态
            state_machine = processor.state_machine
            shared_memory = processor.shared_memory
            
            # 绘制可视化信息
            h, w = frame.shape[:2]
            
            # Layer 1
            if shared_memory['human_detected'] and shared_memory['human_bbox']:
                x1, y1, x2, y2 = [int(c) for c in shared_memory['human_bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "L1: DETECTED", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Layer 2
            motion_pixels = shared_memory.get('motion_pixels', 0)
            if motion_pixels > 0:
                cv2.putText(frame, f"L2: {motion_pixels}px", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Layer 3
            action_score = shared_memory.get('action_score', 0.0)
            intent_type = shared_memory.get('intent_type', 'UNKNOWN')
            score_color = (0, 0, 255) if action_score < 0.6 else (0, 255, 255) if action_score < 0.8 else (0, 255, 0)
            cv2.putText(frame, f"L3: {intent_type} ({action_score:.2f})", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
            
            # Layer 4
            gesture_type = shared_memory.get('gesture_type', 'NONE')
            if gesture_type:
                cv2.putText(frame, f"L4: {gesture_type}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # 性能信息
            fps = 1.0 / (time.time() - timestamp) if time.time() - timestamp > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 时间信息
            elapsed = time.time() - start_time
            remaining = test_duration - elapsed
            cv2.putText(frame, f"Time: {remaining:.0f}s", (w-120, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 状态机
            status_text = f"L1:{state_machine['layer1_state']} | L2:{state_machine['layer2_state']} | L3:{state_machine['layer3_state']} | L4:{state_machine['layer4_state']}"
            cv2.putText(frame, status_text, (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Full Stack Test (L0-L4)", frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("👤 用户请求退出")
                break
        
        # 测试完成，打印统计
        print("\n" + "=" * 80)
        print(" 测试结果")
        print("=" * 80)
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n总帧数：{frame_count}")
        print(f"总时间：{total_time:.2f}秒")
        print(f"平均 FPS: {avg_fps:.2f}")
        
        # 事件统计
        print("\n事件统计:")
        stats = monitor.get_statistics()
        for event_type, count in stats.items():
            print(f"  - {event_type}: {count}次")
        
        # 打印事件时间线
        monitor.print_timeline()
        
        # 验证结果
        print("\n" + "=" * 80)
        print(" 验证结果")
        print("=" * 80)
        
        passed = True
        
        if stats.get('SENSOR_VISION_STATE', 0) > 0:
            print("✅ Layer 2: SENSOR_VISION_STATE 事件发布成功")
        else:
            print("❌ Layer 2: SENSOR_VISION_STATE 事件未发布")
            passed = False
        
        if stats.get('INTERACTION_TRIGGER', 0) > 0:
            print("✅ Layer 3/4: INTERACTION_TRIGGER 事件发布成功")
        else:
            print("❌ Layer 3/4: INTERACTION_TRIGGER 事件未发布")
            passed = False
        
        if passed:
            print("\n🎉 所有测试通过！L0-L4 联动成功！")
        else:
            print("\n⚠️ 部分测试未通过，请检查日志")
        
        return passed
        
    except Exception as e:
        logger.error(f"❌ 测试失败：{e}", exc_info=True)
        return False
    
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        logger.info("✅ 资源已释放")


if __name__ == "__main__":
    try:
        result = asyncio.run(test_full_stack())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        exit(0)
