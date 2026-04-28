# File: tests/debug_yolo_detailed.py
"""
YOLO 人体检测详细调试

测试目标:
1. 查看摄像头实际画面内容
2. 检查 YOLO 检测到的所有物体（不只是人体）
3. 分析检测失败原因

TSD v1.7 对应:
- 4.4 感知预处理
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def debug_yolo_detection():
    """详细调试 YOLO 检测"""
    print("\n" + "="*60)
    print("🔍 YOLO 人体检测详细调试")
    print("="*60)
    
    try:
        from zulong.l0.usb_camera import USBCamera
        from zulong.l1c.vision_model_loader import get_vision_model_loader
        
        # 创建摄像头
        camera = USBCamera(device_id=0, width=640, height=480, fps=30)
        
        if not camera.connect():
            print("❌ 摄像头连接失败")
            return False
        
        camera.start()
        print("✅ 摄像头已启动")
        
        # 加载 YOLO 模型
        print("📦 加载 YOLO 模型...")
        model_loader = get_vision_model_loader()
        model_loader.load_all_models()  # 不是异步函数
        
        yolo_model = model_loader.yolo_model
        
        if yolo_model is None:
            print("❌ YOLO 模型加载失败")
            return False
        
        print("✅ YOLO 模型已加载")
        print(f"   - 模型类型：{type(yolo_model)}")
        
        # COCO 数据集类别名称 (前 10 个)
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat'
        ]
        
        # 测试 20 帧
        print("\n📸 测试 20 帧 YOLO 检测...")
        print(f"   COCO 类别：{len(coco_classes)} 类")
        
        all_detections = []
        
        for i in range(20):
            frame, timestamp = camera.get_latest_frame()
            
            if frame is None:
                print(f"\n   帧 {i+1}: ❌ 无法获取帧")
                continue
            
            # 检查帧内容
            mean_brightness = np.mean(frame)
            print(f"\n   帧 {i+1}: 亮度={mean_brightness:.1f}")
            
            # YOLO 推理
            results = yolo_model(frame, verbose=False)
            
            # 解析所有检测结果
            detections = results[0].boxes
            
            if len(detections) == 0:
                print(f"           ❌ 未检测到任何物体")
                all_detections.append([])
                continue
            
            frame_detections = []
            
            for j, det in enumerate(detections):
                cls_id = int(det.cls[0])
                conf = float(det.conf[0])
                bbox = det.xyxy[0].cpu().numpy()
                
                class_name = coco_classes[cls_id] if cls_id < len(coco_classes) else f"class_{cls_id}"
                
                frame_detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': bbox.tolist()
                })
                
                print(f"           ✅ [{j+1}] {class_name} (conf={conf:.2f}, bbox={bbox})")
            
            all_detections.append(frame_detections)
            time.sleep(0.1)
        
        # 统计分析
        print("\n" + "="*60)
        print("📊 YOLO 检测统计")
        print("="*60)
        
        total_frames = len([d for d in all_detections if d is not None])
        frames_with_person = sum(1 for detections in all_detections if any(d['class'] == 'person' for d in detections))
        frames_with_any_object = sum(1 for detections in all_detections if len(detections) > 0)
        
        print(f"\n总帧数：{total_frames}")
        print(f"检测到物体的帧数：{frames_with_any_object} ({frames_with_any_object/total_frames*100:.0f}%)")
        print(f"检测到人体的帧数：{frames_with_person} ({frames_with_person/total_frames*100:.0f}%)")
        
        # 统计所有检测到的类别
        class_counts = {}
        for detections in all_detections:
            for det in detections:
                class_name = det['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n检测到的物体类别统计:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {class_name}: {count} 次")
        
        # 保存一帧带检测框的图片
        print("\n💾 保存检测示例图片...")
        frame, _ = camera.get_latest_frame()
        
        if frame is not None:
            results = yolo_model(frame, verbose=False)
            
            # 绘制检测框
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = coco_classes[cls_id] if cls_id < len(coco_classes) else f"class_{cls_id}"
                        
                        # 绘制矩形框
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # 添加标签
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 保存图片
            save_dir = Path("debug_data")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / f"yolo_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(save_path), frame)
            
            print(f"✅ 已保存到：{save_path}")
        
        camera.stop()
        camera.disconnect()
        
        return frames_with_person > 0
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("="*60)
    print("🔍 YOLO 人体检测详细调试")
    print("="*60)
    print(f"📅 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = await debug_yolo_detection()
    
    if success:
        print("\n" + "="*60)
        print("✅ YOLO 检测到人体!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ YOLO 未检测到人体")
        print("="*60)
        print("\n📋 可能原因:")
        print("   1. 摄像头画面中没有人体")
        print("   2. 人体太小或太远")
        print("   3. 光照条件不佳")
        print("   4. YOLO 阈值过高")
        print("   5. 人体角度问题（背对/侧对）")
    
    return success


if __name__ == "__main__":
    import asyncio
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
