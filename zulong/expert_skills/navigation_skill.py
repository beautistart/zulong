# File: zulong/expert_skills/navigation_skill.py
# L3 导航专家技能 - 路径规划、避障、地图构建

"""
祖龙 (ZULONG) L3 导航专家技能

对应 TSD v1.7:
- 2.3.2 专家模型层：L3 专家技能池
- 导航专家：路径规划、避障、地图构建
- Phase 6: 完整 DWA 动态窗口避障算法

功能:
- A* 路径规划算法
- 动态避障（DWA 动态窗口法，Phase 6）
- 栅格地图管理
- 位置跟踪
- 导航历史
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import time

# Phase 6: DWA 动态窗口避障算法
from .dwa_planner import DWADynamicWindowApproach, DWAConfig
import math
import heapq
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NavigationGoal:
    """导航目标点"""
    x: float  # X 坐标（米）
    y: float  # Y 坐标（米）
    theta: float = 0.0  # 朝向角度（弧度）
    priority: int = 0  # 优先级
    created_at: float = field(default_factory=time.time)


@dataclass
class NavigationResult:
    """导航结果"""
    success: bool
    path: List[Tuple[float, float]]  # 路径点列表 [(x1,y1), (x2,y2), ...]
    distance: float  # 总距离（米）
    estimated_time: float  # 预估时间（秒）
    obstacles_avoided: int = 0  # 避障数量
    message: str = ""


class NavigationSkill:
    """L3 导航专家技能
    
    TSD v1.7 对应规则:
    - L3 专家技能池：导航专家
    - 路径规划、地图构建、避障
    - 支持 L2 调用
    
    功能:
    - 栅格地图管理
    - A* 路径规划
    - 动态避障
    - 导航历史
    """
    
    def __init__(self, 
                 skill_id: str = "navigation_expert",
                 map_size: Tuple[int, int] = (100, 100),
                 resolution: float = 0.1,
                 use_dwa: bool = True):
        """初始化导航专家技能
        
        Args:
            skill_id: 技能 ID
            map_size: 地图大小（栅格数）
            resolution: 地图分辨率（米/栅格）
            use_dwa: 是否使用 DWA 算法（Phase 6）
        """
        self.skill_id = skill_id
        self.map_size = map_size  # (width, height)
        self.resolution = resolution  # 米/栅格
        self.use_dwa = use_dwa
        
        # 栅格地图（0=空闲，1=障碍）
        self.grid_map = np.zeros(map_size, dtype=np.int8)
        
        # 机器人当前位置
        self.current_position = (0.0, 0.0, 0.0)  # (x, y, theta)
        
        # Phase 6: DWA 规划器
        self._dwa_planner = None
        if use_dwa:
            try:
                dwa_config = DWAConfig()
                self._dwa_planner = DWADynamicWindowApproach(dwa_config)
                logger.info(f"[NavigationSkill] DWA 规划器已初始化：{skill_id}")
            except Exception as e:
                logger.warning(f"[NavigationSkill] DWA 初始化失败，使用简化避障：{e}")
                self.use_dwa = False
        
        # 导航历史
        self.navigation_history: List[Dict[str, Any]] = []
        
        # 统计信息
        self.stats = {
            'total_navigations': 0,
            'successful_navigations': 0,
            'failed_navigations': 0,
            'total_distance': 0.0,
            'obstacles_avoided': 0,
            'dwa_planning_cycles': 0 if use_dwa else None
        }
        
        logger.info(f"[NavigationSkill] 初始化完成：id={skill_id}, "
                   f"map_size={map_size}, resolution={resolution}m, dwa={use_dwa}")
    
    # ========== 地图管理 ==========
    
    def update_map(self, 
                   obstacles: List[Tuple[float, float]],
                   free_space: Optional[List[Tuple[float, float]]] = None):
        """更新地图
        
        Args:
            obstacles: 障碍物位置列表 [(x1,y1), (x2,y2), ...]
            free_space: 自由空间位置列表（可选）
        """
        # 重置地图
        self.grid_map.fill(0)
        
        # 标记障碍物
        for x, y in obstacles:
            grid_x, grid_y = self._world_to_grid(x, y)
            if 0 <= grid_x < self.map_size[0] and 0 <= grid_y < self.map_size[1]:
                self.grid_map[grid_x, grid_y] = 1  # 障碍物
        
        # 标记自由空间（可选）
        if free_space:
            for x, y in free_space:
                grid_x, grid_y = self._world_to_grid(x, y)
                if 0 <= grid_x < self.map_size[0] and 0 <= grid_y < self.map_size[1]:
                    self.grid_map[grid_x, grid_y] = 0  # 自由空间
        
        logger.debug(f"[NavigationSkill] 地图更新：{len(obstacles)} 个障碍物")
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        grid_x = int((x + self.map_size[0] * self.resolution / 2) / self.resolution)
        grid_y = int((y + self.map_size[1] * self.resolution / 2) / self.resolution)
        return (grid_x, grid_y)
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        x = (grid_x * self.resolution) - (self.map_size[0] * self.resolution / 2)
        y = (grid_y * self.resolution) - (self.map_size[1] * self.resolution / 2)
        return (x, y)
    
    # ========== 路径规划（A* 算法） ==========
    
    def plan_path(self, 
                  start: Tuple[float, float],
                  goal: Tuple[float, float]) -> NavigationResult:
        """规划路径（A* 算法）
        
        Args:
            start: 起点 (x, y)
            goal: 目标点 (x, y)
            
        Returns:
            NavigationResult: 导航结果
        """
        logger.info(f"[NavigationSkill] 路径规划：起点={start}, 目标={goal}")
        
        # 转换为栅格坐标
        start_grid = self._world_to_grid(*start)
        goal_grid = self._world_to_grid(*goal)
        
        # 检查起点和终点是否可达
        if self._is_obstacle(start_grid):
            return NavigationResult(
                success=False,
                path=[],
                distance=0.0,
                estimated_time=0.0,
                message="起点有障碍物"
            )
        
        if self._is_obstacle(goal_grid):
            return NavigationResult(
                success=False,
                path=[],
                distance=0.0,
                estimated_time=0.0,
                message="目标点有障碍物"
            )
        
        # A* 算法
        path_grid = self._a_star(start_grid, goal_grid)
        
        if not path_grid:
            return NavigationResult(
                success=False,
                path=[],
                distance=0.0,
                estimated_time=0.0,
                message="无法找到可行路径"
            )
        
        # 转换为世界坐标
        path_world = [self._grid_to_world(gx, gy) for gx, gy in path_grid]
        
        # 计算距离和时间
        distance = self._calculate_path_length(path_world)
        estimated_time = distance / 0.5  # 假设速度 0.5m/s
        
        # 更新统计
        self.stats['total_navigations'] += 1
        self.stats['successful_navigations'] += 1
        self.stats['total_distance'] += distance
        
        logger.info(f"[NavigationSkill] 路径规划成功：距离={distance:.2f}m, "
                   f"时间={estimated_time:.1f}s, 路径点={len(path_world)}")
        
        return NavigationResult(
            success=True,
            path=path_world,
            distance=distance,
            estimated_time=estimated_time,
            obstacles_avoided=len(path_grid)  # 简化：假设每个路径点都避障
        )
    
    def _is_obstacle(self, grid_pos: Tuple[int, int]) -> bool:
        """检查是否为障碍物"""
        gx, gy = grid_pos
        if 0 <= gx < self.map_size[0] and 0 <= gy < self.map_size[1]:
            return self.grid_map[gx, gy] == 1
        return True  # 超出边界视为障碍
    
    def _a_star(self, 
                start: Tuple[int, int], 
                goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* 路径规划算法"""
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            """启发式函数（欧几里得距离）"""
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        # 优先队列
        frontier = []
        heapq.heappush(frontier, (0, start))
        
        # 路径追踪
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        # 8 方向移动
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
            
            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)
                
                # 检查边界和障碍
                if self._is_obstacle(next_pos):
                    continue
                
                # 计算代价
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # 重构路径
        if goal not in came_from:
            return []
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        
        path.reverse()
        return path
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """计算路径长度"""
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            total += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return total
    
    # ========== 避障算法 ==========
    
    def avoid_obstacles(self, 
                        current_pos: Tuple[float, float],
                        target_pos: Tuple[float, float],
                        sensor_data: Dict[str, Any]) -> Tuple[float, float]:
        """动态避障
        
        Args:
            current_pos: 当前位置 (x, y)
            target_pos: 目标位置 (x, y)
            sensor_data: 传感器数据（激光雷达等）
            
        Returns:
            Tuple[float, float]: 避障后的速度命令 (v, w)
        """
        logger.info(f"[NavigationSkill] 避障：current={current_pos}, "
                   f"target={target_pos}, use_dwa={self.use_dwa}")
        
        # Phase 6: 使用 DWA 算法
        if self.use_dwa and self._dwa_planner is not None:
            try:
                # 设置机器人状态
                theta = self.current_position[2] if len(self.current_position) > 2 else 0.0
                self._dwa_planner.set_robot_state(
                    position=current_pos,
                    orientation=theta
                )
                
                # 设置目标
                self._dwa_planner.set_goal(target_pos)
                
                # 更新障碍物
                obstacles = sensor_data.get('obstacles', [])
                self._dwa_planner.update_obstacles(obstacles)
                
                # 执行 DWA 规划
                v, w = self._dwa_planner.plan(sensor_data)
                
                # 更新统计
                if self.stats.get('dwa_planning_cycles') is not None:
                    dwa_stats = self._dwa_planner.get_stats()
                    self.stats['dwa_planning_cycles'] = dwa_stats.get('total_planning_cycles', 0)
                
                logger.info(f"[NavigationSkill] DWA 避障：v={v:.2f}m/s, w={w:.2f}rad/s")
                
                return (v, w)
                
            except Exception as e:
                logger.error(f"[NavigationSkill] DWA 避障失败，降级到简化模式：{e}")
                # 降级到简化模式
        
        # 简化模式（向后兼容）
        return self._avoid_obstacles_simple(current_pos, target_pos, sensor_data)
    
    def _avoid_obstacles_simple(self,
                                current_pos: Tuple[float, float],
                                target_pos: Tuple[float, float],
                                sensor_data: Dict[str, Any]) -> Tuple[float, float]:
        """简化避障算法（向后兼容）
        
        Returns:
            Tuple[float, float]: 方向向量 (dx, dy)
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # 检查前方是否有障碍
        obstacles = sensor_data.get('obstacles', [])
        for obs in obstacles:
            dist_to_obs = math.sqrt((obs[0] - current_pos[0])**2 + 
                                   (obs[1] - current_pos[1])**2)
            if dist_to_obs < 0.5:  # 安全距离 0.5m
                angle = math.atan2(dy, dx)
                angle += math.pi / 4  # 右转 45 度
                dx = math.cos(angle)
                dy = math.sin(angle)
                self.stats['obstacles_avoided'] += 1
                break
        
        # 归一化
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
        
        return (dx, dy)
    
    # ========== 位置跟踪 ==========
    
    def update_position(self, 
                        x: float, 
                        y: float, 
                        theta: Optional[float] = None):
        """更新机器人位置
        
        Args:
            x: X 坐标
            y: Y 坐标
            theta: 朝向角度（可选）
        """
        if theta is not None:
            self.current_position = (x, y, theta)
        else:
            self.current_position = (x, y, self.current_position[2])
        
        logger.debug(f"[NavigationSkill] 位置更新：({x:.2f}, {y:.2f}, "
                    f"{self.current_position[2]:.2f})")
    
    def get_position(self) -> Tuple[float, float, float]:
        """获取当前位置"""
        return self.current_position
    
    # ========== 导航历史 ==========
    
    def record_navigation(self, 
                          start: Tuple[float, float],
                          goal: Tuple[float, float],
                          result: NavigationResult):
        """记录导航历史
        
        Args:
            start: 起点
            goal: 目标点
            result: 导航结果
        """
        record = {
            'timestamp': time.time(),
            'start': start,
            'goal': goal,
            'success': result.success,
            'distance': result.distance,
            'path_length': len(result.path)
        }
        
        self.navigation_history.append(record)
        
        # 保留最近 100 条
        if len(self.navigation_history) > 100:
            self.navigation_history.pop(0)
    
    def get_navigation_history(self, 
                               limit: int = 10) -> List[Dict[str, Any]]:
        """获取导航历史
        
        Args:
            limit: 返回数量
            
        Returns:
            List[Dict]: 历史记录列表
        """
        return self.navigation_history[-limit:]
    
    # ========== 统计信息 ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'map_size': self.map_size,
            'resolution': self.resolution,
            'current_position': self.current_position,
            'history_count': len(self.navigation_history)
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_navigations': 0,
            'successful_navigations': 0,
            'failed_navigations': 0,
            'total_distance': 0.0,
            'obstacles_avoided': 0
        }
        self.navigation_history.clear()
        logger.info("[NavigationSkill] 统计信息已重置")
