# File: zulong/expert_skills/dwa_planner.py
# DWA 动态窗口避障算法

"""
祖龙 (ZULONG) DWA 动态窗口避障算法

对应 TSD v1.7:
- 2.3.2 专家模型层：L3 专家技能池
- 导航专家：Phase 6 DWA 算法增强

功能:
- 速度空间采样（线速度/角速度）
- 轨迹评估函数
- 动态障碍物预测
- 实时路径重规划
- 符合 RTX 3060 6GB 限制（CPU 运行）
"""

import logging
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DWAConfig:
    """DWA 算法配置"""
    # 机器人运动学参数
    max_linear_velocity: float = 1.0  # 最大线速度 (m/s)
    min_linear_velocity: float = 0.0  # 最小线速度 (m/s)
    max_angular_velocity: float = 1.0  # 最大角速度 (rad/s)
    min_angular_velocity: float = -1.0  # 最小角速度 (rad/s)
    linear_acceleration: float = 0.5  # 线加速度 (m/s²)
    angular_acceleration: float = 1.0  # 角加速度 (rad/s²)
    
    # 采样参数
    num_linear_samples: int = 10  # 线速度采样数
    num_angular_samples: int = 20  # 角速度采样数
    prediction_time: float = 2.0  # 预测时间 (s)
    time_step: float = 0.1  # 时间步长 (s)
    
    # 评估函数权重
    heading_weight: float = 0.3  # 目标朝向权重
    distance_weight: float = 0.5  # 障碍物距离权重
    velocity_weight: float = 0.2  # 速度权重
    
    # 安全参数
    safety_distance: float = 0.5  # 安全距离 (m)
    stop_distance: float = 0.3  # 停止距离 (m)
    
    # 目标参数
    goal_radius: float = 0.3  # 目标半径 (m)


@dataclass
class TrajectorySample:
    """轨迹样本"""
    linear_velocity: float  # 线速度
    angular_velocity: float  # 角速度
    trajectory: List[Tuple[float, float]] = field(default_factory=list)  # 轨迹点
    distance_to_obstacle: float = float('inf')  # 到最近障碍物的距离
    distance_to_goal: float = float('inf')  # 到目标的距离
    final_heading: float = 0.0  # 最终朝向
    score: float = 0.0  # 评估分数


class DWADynamicWindowApproach:
    """DWA 动态窗口避障算法
    
    TSD v1.7 对应规则:
    - Phase 6: 导航算法优化
    - 完整 DWA 实现
    - CPU 运行，实时避障
    
    功能:
    - 速度空间采样
    - 轨迹评估
    - 动态障碍物预测
    - 实时避障决策
    """
    
    def __init__(self, config: Optional[DWAConfig] = None):
        """初始化 DWA 规划器
        
        Args:
            config: DWA 配置参数
        """
        self.config = config or DWAConfig()
        
        # 当前状态
        self.current_position: Tuple[float, float] = (0.0, 0.0)
        self.current_orientation: float = 0.0  # 朝向 (rad)
        self.current_linear_velocity: float = 0.0
        self.current_angular_velocity: float = 0.0
        
        # 目标
        self.goal_position: Optional[Tuple[float, float]] = None
        
        # 障碍物地图
        self.obstacle_map: Dict[Tuple[float, float], bool] = {}
        
        # 统计信息
        self.stats = {
            'total_planning_cycles': 0,
            'total_obstacles_avoided': 0,
            'avg_planning_time_ms': 0.0,
            'last_planning_time': 0.0,
            'trajectories_evaluated': 0
        }
        
        logger.info(f"[DWADynamicWindowApproach] 初始化完成")
    
    def set_robot_state(self, 
                       position: Tuple[float, float],
                       orientation: float,
                       linear_velocity: float = 0.0,
                       angular_velocity: float = 0.0):
        """设置机器人状态
        
        Args:
            position: 当前位置 (x, y)
            orientation: 当前朝向 (rad)
            linear_velocity: 当前线速度 (m/s)
            angular_velocity: 当前角速度 (rad/s)
        """
        self.current_position = position
        self.current_orientation = orientation
        self.current_linear_velocity = linear_velocity
        self.current_angular_velocity = angular_velocity
        
        logger.debug(f"[DWA] 机器人状态更新：pos={position}, "
                    f"orient={orientation:.2f}rad, v={linear_velocity:.2f}m/s")
    
    def set_goal(self, goal: Tuple[float, float]):
        """设置目标位置
        
        Args:
            goal: 目标位置 (x, y)
        """
        self.goal_position = goal
        logger.debug(f"[DWA] 目标设置：{goal}")
    
    def update_obstacles(self, obstacles: List[Tuple[float, float]]):
        """更新障碍物位置
        
        Args:
            obstacles: 障碍物位置列表 [(x1, y1), (x2, y2), ...]
        """
        self.obstacle_map.clear()
        for obs in obstacles:
            # 将障碍物标记到栅格中
            grid_pos = (round(obs[0] / 0.1), round(obs[1] / 0.1))
            self.obstacle_map[grid_pos] = True
        
        logger.debug(f"[DWA] 障碍物更新：{len(obstacles)} 个")
    
    def plan(self, 
             sensor_data: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """执行 DWA 规划
        
        Args:
            sensor_data: 传感器数据（可选）
            
        Returns:
            Tuple[float, float]: 最佳速度命令 (v, w)
        """
        start_time = time.time()
        
        if self.goal_position is None:
            logger.warning("[DWA] 未设置目标，返回零速度")
            return (0.0, 0.0)
        
        # 检查是否到达目标
        if self._is_goal_reached():
            logger.info("[DWA] 已到达目标")
            return (0.0, 0.0)
        
        # 生成速度空间样本
        velocity_samples = self._generate_velocity_samples()
        
        # 评估每个样本
        best_sample = None
        best_score = -float('inf')
        
        for sample in velocity_samples:
            # 模拟轨迹
            self._simulate_trajectory(sample)
            
            # 检查碰撞
            if self._check_collision(sample):
                continue
            
            # 评估分数
            score = self._evaluate_trajectory(sample)
            
            if score > best_score:
                best_score = score
                best_sample = sample
        
        # 更新统计
        self._update_stats(start_time, len(velocity_samples))
        
        if best_sample is None:
            logger.warning("[DWA] 无可行轨迹，停止")
            return (0.0, 0.0)
        
        logger.debug(f"[DWA] 最佳速度：v={best_sample.linear_velocity:.2f}m/s, "
                    f"w={best_sample.angular_velocity:.2f}rad/s, "
                    f"score={best_score:.2f}")
        
        return (best_sample.linear_velocity, best_sample.angular_velocity)
    
    def _generate_velocity_samples(self) -> List[TrajectorySample]:
        """生成速度空间样本
        
        Returns:
            List[TrajectorySample]: 速度样本列表
        """
        samples = []
        
        # 计算可达速度范围（考虑加速度限制）
        dt = self.config.time_step
        
        v_min = max(
            self.config.min_linear_velocity,
            self.current_linear_velocity - self.config.linear_acceleration * dt
        )
        v_max = min(
            self.config.max_linear_velocity,
            self.current_linear_velocity + self.config.linear_acceleration * dt
        )
        
        w_min = max(
            self.config.min_angular_velocity,
            self.current_angular_velocity - self.config.angular_acceleration * dt
        )
        w_max = min(
            self.config.max_angular_velocity,
            self.current_angular_velocity + self.config.angular_acceleration * dt
        )
        
        # 均匀采样
        v_step = (v_max - v_min) / max(1, self.config.num_linear_samples - 1)
        w_step = (w_max - w_min) / max(1, self.config.num_angular_samples - 1)
        
        for i in range(self.config.num_linear_samples):
            v = v_min + i * v_step
            
            for j in range(self.config.num_angular_samples):
                w = w_min + j * w_step
                
                sample = TrajectorySample(
                    linear_velocity=v,
                    angular_velocity=w
                )
                samples.append(sample)
        
        logger.debug(f"[DWA] 生成速度样本：{len(samples)} 个")
        
        return samples
    
    def _simulate_trajectory(self, sample: TrajectorySample):
        """模拟轨迹
        
        Args:
            sample: 速度样本
        """
        trajectory = []
        
        x, y = self.current_position
        theta = self.current_orientation
        
        v = sample.linear_velocity
        w = sample.angular_velocity
        
        # 模拟运动
        num_steps = int(self.config.prediction_time / self.config.time_step)
        
        for _ in range(num_steps):
            # 运动学模型
            if abs(w) < 1e-6:
                # 直线运动
                x += v * self.config.time_step * math.cos(theta)
                y += v * self.config.time_step * math.sin(theta)
            else:
                # 曲线运动
                radius = v / w
                theta += w * self.config.time_step
                x += radius * (math.sin(theta) - math.sin(theta - w * self.config.time_step))
                y += radius * (math.cos(theta - w * self.config.time_step) - math.cos(theta))
            
            trajectory.append((x, y))
        
        sample.trajectory = trajectory
        
        # 计算到目标的距离
        if self.goal_position:
            dx = self.goal_position[0] - x
            dy = self.goal_position[1] - y
            sample.distance_to_goal = math.sqrt(dx**2 + dy**2)
            sample.final_heading = math.atan2(dy, dx)
    
    def _check_collision(self, sample: TrajectorySample) -> bool:
        """检查碰撞
        
        Args:
            sample: 速度样本
            
        Returns:
            bool: 是否碰撞
        """
        safety_radius = self.config.safety_distance
        
        for point in sample.trajectory:
            # 检查是否在障碍物附近
            grid_pos = (round(point[0] / 0.1), round(point[1] / 0.1))
            
            # 检查周围栅格
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    check_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
                    if check_pos in self.obstacle_map:
                        # 计算实际距离
                        actual_x = check_pos[0] * 0.1
                        actual_y = check_pos[1] * 0.1
                        dist = math.sqrt((point[0] - actual_x)**2 + 
                                       (point[1] - actual_y)**2)
                        
                        if dist < safety_radius:
                            sample.distance_to_obstacle = min(
                                sample.distance_to_obstacle, 
                                dist
                            )
                            return True
        
        # 更新最近障碍物距离
        if sample.distance_to_obstacle == float('inf'):
            sample.distance_to_obstacle = self.config.safety_distance * 2
        
        return False
    
    def _evaluate_trajectory(self, sample: TrajectorySample) -> float:
        """评估轨迹
        
        Args:
            sample: 速度样本
            
        Returns:
            float: 评估分数
        """
        # 1. 目标朝向分数
        heading_score = self._calculate_heading_score(sample)
        
        # 2. 障碍物距离分数
        distance_score = self._calculate_distance_score(sample)
        
        # 3. 速度分数
        velocity_score = self._calculate_velocity_score(sample)
        
        # 加权总分
        total_score = (
            self.config.heading_weight * heading_score +
            self.config.distance_weight * distance_score +
            self.config.velocity_weight * velocity_score
        )
        
        sample.score = total_score
        
        logger.debug(f"[DWA] 轨迹评估：heading={heading_score:.2f}, "
                    f"dist={distance_score:.2f}, vel={velocity_score:.2f}, "
                    f"total={total_score:.2f}")
        
        return total_score
    
    def _calculate_heading_score(self, sample: TrajectorySample) -> float:
        """计算目标朝向分数
        
        Args:
            sample: 速度样本
            
        Returns:
            float: 分数 (0-1)
        """
        if sample.final_heading == 0.0 and sample.distance_to_goal > 1e-6:
            return 0.0
        
        # 计算期望朝向
        if self.goal_position:
            dx = self.goal_position[0] - sample.trajectory[-1][0]
            dy = self.goal_position[1] - sample.trajectory[-1][1]
            expected_heading = math.atan2(dy, dx)
        else:
            return 0.0
        
        # 角度差
        heading_diff = abs(self._normalize_angle(
            sample.final_heading - expected_heading
        ))
        
        # 归一化为分数 (0-1)
        score = 1.0 - (heading_diff / math.pi)
        
        return score
    
    def _calculate_distance_score(self, sample: TrajectorySample) -> float:
        """计算障碍物距离分数
        
        Args:
            sample: 速度样本
            
        Returns:
            float: 分数 (0-1)
        """
        dist = sample.distance_to_obstacle
        
        if dist >= self.config.safety_distance * 2:
            return 1.0
        elif dist <= self.config.stop_distance:
            return 0.0
        else:
            # 线性插值
            return (dist - self.config.stop_distance) / \
                   (self.config.safety_distance * 2 - self.config.stop_distance)
    
    def _calculate_velocity_score(self, sample: TrajectorySample) -> float:
        """计算速度分数
        
        Args:
            sample: 速度样本
            
        Returns:
            float: 分数 (0-1)
        """
        # 速度越快分数越高
        return sample.linear_velocity / self.config.max_linear_velocity
    
    def _is_goal_reached(self) -> bool:
        """检查是否到达目标
        
        Returns:
            bool: 是否到达
        """
        if self.goal_position is None:
            return False
        
        dx = self.goal_position[0] - self.current_position[0]
        dy = self.goal_position[1] - self.current_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        return distance < self.config.goal_radius
    
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到 [-π, π]
        
        Args:
            angle: 角度 (rad)
            
        Returns:
            float: 归一化角度
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _update_stats(self, start_time: float, num_samples: int):
        """更新统计信息
        
        Args:
            start_time: 开始时间
            num_samples: 样本数量
        """
        planning_time = (time.time() - start_time) * 1000  # ms
        
        self.stats['total_planning_cycles'] += 1
        self.stats['trajectories_evaluated'] += num_samples
        
        # 更新平均规划时间
        total = self.stats['total_planning_cycles']
        old_avg = self.stats['avg_planning_time_ms']
        self.stats['avg_planning_time_ms'] = (
            (old_avg * (total - 1) + planning_time) / total
        )
        
        self.stats['last_planning_time'] = planning_time
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            **self.stats,
            'current_position': self.current_position,
            'goal_position': self.goal_position,
            'num_obstacles': len(self.obstacle_map)
        }
    
    def get_predicted_trajectory(self, 
                                 v: float, 
                                 w: float) -> List[Tuple[float, float]]:
        """获取预测轨迹
        
        Args:
            v: 线速度 (m/s)
            w: 角速度 (rad/s)
            
        Returns:
            List[Tuple[float, float]]: 预测轨迹点
        """
        sample = TrajectorySample(
            linear_velocity=v,
            angular_velocity=w
        )
        
        self._simulate_trajectory(sample)
        
        return sample.trajectory
