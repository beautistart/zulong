# File: zulong/l1/sensors.py
# 定义传感器数据结构与模拟源

from typing import TypedDict, List, Dict
import time


class SensorData(TypedDict):
    """传感器数据结构"""
    # IMU 状态: 'STABLE', 'TILT', 'FALL'
    imu_status: str
    # 激光雷达障碍物: 距离和角度信息
    lidar_obstacles: List[Dict]
    # 摄像头帧引用 ID (不存真实图像，只存 ID)
    camera_frame_ref: str
    # 来自 L0 的文本输入
    text_input_buffer: str
    # 时间戳
    timestamp: float


class MockSensorSource:
    """模拟传感器源，提供测试数据"""
    
    def __init__(self):
        """初始化模拟传感器源"""
        self.current_scenario = "normal"
    
    def set_scenario(self, scenario: str):
        """设置测试场景
        
        Args:
            scenario: 场景名称，可选值: 'fall', 'person_approaching', 'normal'
        """
        self.current_scenario = scenario
    
    def get_latest_data(self) -> SensorData:
        """获取最新传感器数据
        
        Returns:
            SensorData: 传感器数据
        """
        timestamp = time.time()
        
        if self.current_scenario == "fall":
            # 场景 A: 摔倒
            return SensorData(
                imu_status="FALL",
                lidar_obstacles=[
                    {"distance": 1.5, "angle": 0.0},
                    {"distance": 2.0, "angle": 90.0}
                ],
                camera_frame_ref=f"frame_{int(timestamp)}",
                text_input_buffer="",
                timestamp=timestamp
            )
        elif self.current_scenario == "person_approaching":
            # 场景 B: 有人靠近
            return SensorData(
                imu_status="STABLE",
                lidar_obstacles=[
                    {"distance": 0.8, "angle": 0.0},  # 前方 0.8 米有障碍物
                    {"distance": 3.0, "angle": 180.0}
                ],
                camera_frame_ref=f"frame_{int(timestamp)}",
                text_input_buffer="",
                timestamp=timestamp
            )
        else:
            # 场景 C: 正常行走
            return SensorData(
                imu_status="STABLE",
                lidar_obstacles=[
                    {"distance": 5.0, "angle": 0.0},
                    {"distance": 4.5, "angle": 90.0},
                    {"distance": 4.0, "angle": 180.0}
                ],
                camera_frame_ref=f"frame_{int(timestamp)}",
                text_input_buffer="",
                timestamp=timestamp
            )
