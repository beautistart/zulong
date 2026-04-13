# File: zulong/l3/nav_expert_node.py
# 导航专家节点 - 实现路径规划、避障、定位功能

from typing import Dict, Any, List, Tuple
import logging
import random
import time

from .base_expert_node import BaseExpertNode

logger = logging.getLogger(__name__)


class NavExpertNode(BaseExpertNode):
    """导航专家节点
    
    TSD v1.7 对应规则:
    - 2.2.4 L3: 专家技能池 - 导航专家 (SLAM 建图、路径规划、定位、避障)
    - 4.3 L2: 中枢与上下文管理 - L2 调用 L3 专家
    
    功能:
    - 路径规划：给定起点和终点，返回路径点列表
    - 避障：检测障碍物并重新规划
    - 定位：返回当前位置估计
    """
    
    # 预定义位置坐标（模拟室内地图）
    LOCATIONS = {
        "客厅": (0, 0),
        "厨房": (5, 3),
        "卧室": (3, -2),
        "卫生间": (6, -1),
        "阳台": (2, 4),
        "走廊": (1, 2),
        "餐厅": (4, 1)
    }
    
    def __init__(self):
        """初始化导航专家"""
        super().__init__("EXPERT_NAV")
        self.current_location = "客厅"  # 默认位置
        self.map_loaded = False
        
        # 加载地图
        self._load_map()
    
    def _load_map(self):
        """加载地图数据（模拟）"""
        logger.info("[EXPERT_NAV] 加载地图数据...")
        time.sleep(0.5)  # 模拟加载延迟
        self.map_loaded = True
        logger.info("[EXPERT_NAV] 地图加载完成")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """获取导航专家能力描述"""
        return {
            "expert_type": "EXPERT_NAV",
            "description": "导航专家 - SLAM 建图、路径规划、定位、避障",
            "version": "1.0",
            "capabilities": [
                "path_planning",      # 路径规划
                "obstacle_avoidance", # 避障
                "localization",       # 定位
                "map_management"      # 地图管理
            ],
            "locations": list(self.LOCATIONS.keys())
        }
    
    def validate_payload(self, task_payload: Dict[str, Any]) -> bool:
        """验证任务载荷"""
        if not super().validate_payload(task_payload):
            return False
        
        # 检查任务类型
        task_type = task_payload.get("task_type", "navigate")
        if task_type not in ["navigate", "locate", "map"]:
            logger.error(f"[EXPERT_NAV] 未知任务类型：{task_type}")
            return False
        
        return True
    
    def execute(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行导航任务
        
        Args:
            task_payload: 任务载荷，包含:
                - task_type: 任务类型 (navigate/locate/map)
                - destination: 目的地 (navigate 任务需要)
                - start: 起点 (可选，默认当前位置)
                - avoid_obstacles: 是否避障 (可选，默认 True)
                - obstacles: 障碍物列表 (可选)
                
        Returns:
            Dict[str, Any]: 执行结果，包含:
                - status: success/error
                - path: 路径点列表 (navigate 任务)
                - current_location: 当前位置 (locate 任务)
                - map_data: 地图数据 (map 任务)
                - estimated_time: 预计时间 (秒)
                - distance: 距离 (米)
        """
        task_type = task_payload.get("task_type", "navigate")
        
        if task_type == "navigate":
            return self._plan_path(task_payload)
        elif task_type == "locate":
            return self._get_location(task_payload)
        elif task_type == "map":
            return self._get_map_data(task_payload)
        else:
            raise ValueError(f"未知任务类型：{task_type}")
    
    def _plan_path(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """规划路径
        
        Args:
            task_payload: 包含 destination, start, avoid_obstacles 等
            
        Returns:
            路径规划结果
        """
        destination = task_payload.get("destination")
        start = task_payload.get("start", self.current_location)
        avoid_obstacles = task_payload.get("avoid_obstacles", True)
        obstacles = task_payload.get("obstacles", [])
        
        if not destination:
            raise ValueError("导航任务必须指定 destination")
        
        if destination not in self.LOCATIONS:
            raise ValueError(f"未知目的地：{destination}。可用位置：{list(self.LOCATIONS.keys())}")
        
        if start not in self.LOCATIONS:
            raise ValueError(f"未知起点：{start}")
        
        # 获取坐标
        start_pos = self.LOCATIONS[start]
        end_pos = self.LOCATIONS[destination]
        
        # 生成路径点
        path = self._generate_path(start_pos, end_pos, avoid_obstacles, obstacles)
        
        # 计算距离和时间
        distance = self._calculate_distance(path)
        estimated_time = distance * 2.0  # 假设速度 0.5m/s
        
        # 更新当前位置
        self.current_location = destination
        
        logger.info(f"[EXPERT_NAV] 路径规划：{start} -> {destination}, 距离：{distance:.2f}m")
        
        return {
            "path": path,
            "start": start,
            "destination": destination,
            "distance": distance,
            "estimated_time": estimated_time,
            "obstacles_avoided": len(obstacles) if avoid_obstacles else 0
        }
    
    def _get_location(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """获取当前位置
        
        Args:
            task_payload: 包含 use_gps (可选)
            
        Returns:
            当前位置信息
        """
        use_gps = task_payload.get("use_gps", False)
        
        current_pos = self.LOCATIONS[self.current_location]
        
        # 模拟 GPS 误差
        if use_gps:
            noise_x = random.uniform(-0.1, 0.1)
            noise_y = random.uniform(-0.1, 0.1)
            current_pos = (current_pos[0] + noise_x, current_pos[1] + noise_y)
        
        logger.info(f"[EXPERT_NAV] 定位：{self.current_location}, 坐标：{current_pos}")
        
        return {
            "current_location": self.current_location,
            "coordinates": current_pos,
            "accuracy": 0.95 if not use_gps else 0.85,
            "timestamp": time.time()
        }
    
    def _get_map_data(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """获取地图数据
        
        Args:
            task_payload: 包含 include_obstacles (可选)
            
        Returns:
            地图数据
        """
        include_obstacles = task_payload.get("include_obstacles", False)
        
        map_data = {
            "locations": list(self.LOCATIONS.keys()),
            "coordinates": self.LOCATIONS,
            "map_size": (10, 10),  # 10m x 10m
            "resolution": 0.1  # 10cm 分辨率
        }
        
        if include_obstacles:
            # 模拟障碍物
            map_data["obstacles"] = [
                {"name": "桌子", "position": (2, 1)},
                {"name": "椅子", "position": (3, 2)}
            ]
        
        logger.info(f"[EXPERT_NAV] 地图数据：{len(map_data['locations'])} 个位置")
        
        return map_data
    
    def _generate_path(
        self, 
        start: Tuple[float, float], 
        end: Tuple[float, float],
        avoid_obstacles: bool,
        obstacles: List[Dict]
    ) -> List[Tuple[float, float]]:
        """生成路径点列表（简化版 A*算法模拟）
        
        Args:
            start: 起点坐标
            end: 终点坐标
            avoid_obstacles: 是否避障
            obstacles: 障碍物列表
            
        Returns:
            路径点列表 [(x1,y1), (x2,y2), ...]
        """
        path = [start]
        
        # 简单直线路径（实际应该用 A*或 Dijkstra）
        steps = 10
        dx = (end[0] - start[0]) / steps
        dy = (end[1] - start[1]) / steps
        
        for i in range(1, steps + 1):
            x = start[0] + dx * i
            y = start[1] + dy * i
            
            # 避障处理
            if avoid_obstacles and obstacles:
                for obstacle in obstacles:
                    obs_pos = obstacle.get("position", (0, 0))
                    dist_to_obstacle = ((x - obs_pos[0])**2 + (y - obs_pos[1])**2)**0.5
                    if dist_to_obstacle < 0.5:  # 距离障碍物太近
                        # 绕行
                        x += 0.3
                        y += 0.3
            
            path.append((round(x, 2), round(y, 2)))
        
        return path
    
    def _calculate_distance(self, path: List[Tuple[float, float]]) -> float:
        """计算路径总距离
        
        Args:
            path: 路径点列表
            
        Returns:
            总距离（米）
        """
        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            total += (dx**2 + dy**2)**0.5
        return round(total, 2)
