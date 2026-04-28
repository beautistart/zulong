# File: tests/test_vision_eventbus_integration.py
"""
视觉系统 - EventBus 集成测试

测试 OptimizedVisionProcessor 是否正确发布事件到 EventBus：
1. Layer 2: SENSOR_VISION_STATE (运动检测)
2. Layer 3: INTERACTION_TRIGGER (静默注意)
3. Layer 4: INTERACTION_TRIGGER (交互请求)
"""

import asyncio
import time
import cv2
import logging
from typing import List, Dict, Any

from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
from zulong.core.types import EventType, ZulongEvent
from zulong.core.event_bus import event_bus

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("TestVisionEventBus")


class EventCollector:
    """事件收集器 - 用于测试中捕获发布的事件"""
    
    def __init__(self):
        self.events: List[ZulongEvent] = []
        self.event_counts: Dict[str, int] = {}
        
        # 订阅所有相关事件
        event_bus.subscribe(EventType.SENSOR_VISION_STATE, self._on_vision_state, "TestCollector")
        event_bus.subscribe(EventType.INTERACTION_TRIGGER, self._on_interaction, "TestCollector")
    
    def _on_vision_state(self, event: ZulongEvent):
        """处理视觉状态事件"""
        self.events.append(event)
        self.event_counts['SENSOR_VISION_STATE'] = self.event_counts.get('SENSOR_VISION_STATE', 0) + 1
        logger.info(f"📨 收到 SENSOR_VISION_STATE: motion_pixels={event.payload.get('motion_pixels', 0)}")
    
    def _on_interaction(self, event: ZulongEvent):
        """处理交互事件"""
        self.events.append(event)
        self.event_counts['INTERACTION_TRIGGER'] = self.event_counts.get('INTERACTION_TRIGGER', 0) + 1
        
        layer = event.payload.get('layer', 'Unknown')
        intent = event.payload.get('intent_type', 'Unknown')
        gesture = event.payload.get('gesture', 'N/A')
        
        logger.info(f"📨 收到 INTERACTION_TRIGGER (L{layer}): intent={intent}, gesture={gesture}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_events': len(self.events),
            'event_counts': self.event_counts.copy(),
            'event_types': list(set(e.type.value for e in self.events))
        }
    
    def reset(self):
        """重置收集器"""
        self.events.clear()
        self.event_counts.clear()


async def test_eventbus_integration():
    """测试视觉系统与 EventBus 的集成"""
    
    print("=" * 60)
    print(" 视觉系统 - EventBus 集成测试")
    print("=" * 60)
    
    # 初始化事件收集器
    collector = EventCollector()
    logger.info("✅ 事件收集器初始化完成")
    
    # 初始化视觉处理器
    print("\n初始化 OptimizedVisionProcessor...")
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    logger.info("✅ 视觉处理器初始化完成")
    
    # 打开摄像头
    print("\n打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("❌ 无法打开摄像头")
        return
    
    logger.info("✅ 摄像头已打开")
    
    # 等待模型加载
    print("\n等待 3 秒让模型完全加载...")
    await asyncio.sleep(3)
    
    # 开始测试
    print("\n" + "=" * 60)
    print(" 开始测试 (请依次执行以下动作)")
    print("=" * 60)
    print("1. 坐在摄像头前静止 (测试 Layer 1 - 人体检测)")
    print("2. 挥动手臂 (测试 Layer 2 - 运动检测)")
    print("3. 继续挥手 2-3 秒 (测试 Layer 3 - 意图分类)")
    print("4. 比出手势 (测试 Layer 4 - 手势识别)")
    print("\n测试将在 30 秒后自动结束...\n")
    
    start_time = time.time()
    frame_count = 0
    test_duration = 30  # 秒
    
    try:
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = time.time()
            
            # 喂食帧到视觉处理器
            processor.feed_frame(frame, timestamp)
            
            # 显示实时状态
            elapsed = time.time() - start_time
            remaining = test_duration - elapsed
            
            cv2.putText(frame, f"Time: {remaining:.0f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示事件统计
            stats = collector.get_statistics()
            y_offset = 90
            for event_type, count in stats['event_counts'].items():
                cv2.putText(frame, f"{event_type}: {count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 30
            
            cv2.imshow("Vision-EventBus Integration Test", frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await asyncio.sleep(0.01)
        
        # 测试完成，打印统计
        print("\n" + "=" * 60)
        print(" 测试结果")
        print("=" * 60)
        
        stats = collector.get_statistics()
        print(f"\n总帧数：{frame_count}")
        print(f"总事件数：{stats['total_events']}")
        print(f"\n事件统计:")
        for event_type, count in stats['event_counts'].items():
            print(f"  - {event_type}: {count}")
        
        # 验证结果
        print("\n" + "=" * 60)
        print(" 验证结果")
        print("=" * 60)
        
        passed = True
        
        if stats['event_counts'].get('SENSOR_VISION_STATE', 0) > 0:
            print("✅ Layer 2: SENSOR_VISION_STATE 事件发布成功")
        else:
            print("❌ Layer 2: SENSOR_VISION_STATE 事件未发布")
            passed = False
        
        if stats['event_counts'].get('INTERACTION_TRIGGER', 0) > 0:
            print("✅ Layer 3/4: INTERACTION_TRIGGER 事件发布成功")
        else:
            print("❌ Layer 3/4: INTERACTION_TRIGGER 事件未发布")
            passed = False
        
        if passed:
            print("\n🎉 所有测试通过！视觉系统与 EventBus 集成成功！")
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
        result = asyncio.run(test_eventbus_integration())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        exit(0)
