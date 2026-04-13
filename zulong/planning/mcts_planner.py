# File: zulong/planning/mcts_planner.py
# MCTS (蒙特卡洛树搜索) 规划器 (Phase 9.4)
# 用于长期任务规划和决策

import math
import random
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy

logger = logging.getLogger(__name__)


@dataclass
class MCTSAction:
    """MCTS 动作"""
    name: str
    description: str
    cost: float               # 动作代价
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCTSState:
    """MCTS 状态"""
    state_data: Dict[str, Any]  # 状态数据
    is_terminal: bool = False   # 是否终止状态
    reward: float = 0.0         # 状态奖励
    
    def clone(self):
        """克隆状态"""
        return MCTSState(
            state_data=copy.deepcopy(self.state_data),
            is_terminal=self.is_terminal,
            reward=self.reward
        )


@dataclass
class MCTSNode:
    """MCTS 节点"""
    state: MCTSState
    parent: Optional['MCTSNode'] = None
    action: Optional[MCTSAction] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[MCTSAction] = field(default_factory=list)
    
    def is_fully_expanded(self) -> bool:
        """是否完全展开"""
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """
        选择最佳子节点 (UCT 公式)
        
        UCT = Q/N + C * sqrt(ln(parent_N) / N)
        """
        if not self.children:
            return None
        
        def uct_score(child):
            if child.visits == 0:
                return float('inf')
            
            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )
            return exploitation + exploration
        
        return max(self.children, key=uct_score)


class MCTSPlanner:
    """
    MCTS (蒙特卡洛树搜索) 规划器
    
    功能:
    - 长期任务规划
    - 多步骤决策
    - 资源调度优化
    - 支持自定义状态评估和动作生成
    
    使用示例:
    ```python
    planner = MCTSPlanner()
    
    # 定义初始状态
    initial_state = MCTSState(
        state_data={"task": "clean_house", "rooms": ["living", "bedroom"]},
        is_terminal=False,
        reward=0.0
    )
    
    # 定义可用动作
    actions = [
        MCTSAction(name="clean_living", description="Clean living room", cost=10),
        MCTSAction(name="clean_bedroom", description="Clean bedroom", cost=8)
    ]
    
    # 规划
    plan = planner.plan(
        initial_state=initial_state,
        available_actions=actions,
        iterations=1000,
        max_depth=10
    )
    ```
    """
    
    def __init__(self):
        """初始化 MCTS 规划器"""
        self._iteration_count = 0
        logger.info("[MCTSPlanner] 初始化完成")
    
    def plan(
        self,
        initial_state: MCTSState,
        available_actions: List[MCTSAction],
        iterations: int = 1000,
        max_depth: int = 10,
        exploration_weight: float = 1.414,
        timeout: float = 5.0
    ) -> List[MCTSAction]:
        """
        执行 MCTS 规划
        
        Args:
            initial_state: 初始状态
            available_actions: 可用动作列表
            iterations: 迭代次数
            max_depth: 最大深度
            exploration_weight: 探索权重 (UCT 公式中的 C)
            timeout: 超时时间 (秒)
            
        Returns:
            List[MCTSAction]: 最优动作序列
        """
        start_time = time.time()
        
        # 创建根节点
        root = MCTSNode(
            state=initial_state,
            untried_actions=available_actions.copy()
        )
        
        self._iteration_count = 0
        
        # MCTS 主循环
        for i in range(iterations):
            # 检查超时
            if time.time() - start_time > timeout:
                logger.info(f"[MCTSPlanner] 超时，已完成 {i} 次迭代")
                break
            
            # 1. Selection: 选择节点
            node = self._select(root, exploration_weight)
            
            # 2. Expansion: 扩展节点
            if node.state.is_terminal or node.is_fully_expanded():
                # 如果是终止状态或已完全展开，直接模拟
                reward = self._simulate(node.state, max_depth - self._get_depth(node))
            else:
                node, reward = self._expand(node, max_depth)
            
            # 3. Backpropagation: 反向传播
            self._backpropagate(node, reward)
            
            self._iteration_count += 1
            
            if i % 100 == 0:
                logger.debug(f"[MCTSPlanner] 迭代 {i}/{iterations}")
        
        # 返回最优路径
        plan = self._extract_plan(root)
        
        elapsed = time.time() - start_time
        logger.info(f"[MCTSPlanner] 规划完成: {self._iteration_count} 次迭代, {elapsed:.3f}s")
        
        return plan
    
    def _select(self, node: MCTSNode, exploration_weight: float) -> MCTSNode:
        """
        Selection 阶段: 选择最有潜力的节点
        
        Args:
            node: 当前节点
            exploration_weight: 探索权重
            
        Returns:
            MCTSNode: 选中的节点
        """
        while node.children and node.is_fully_expanded():
            node = node.best_child(exploration_weight)
        
        return node
    
    def _expand(
        self,
        node: MCTSNode,
        max_depth: int
    ) -> Tuple[MCTSNode, float]:
        """
        Expansion 阶段: 扩展节点
        
        Args:
            node: 当前节点
            max_depth: 最大深度
            
        Returns:
            Tuple[MCTSNode, float]: (新节点, 奖励)
        """
        if not node.untried_actions:
            # 没有未尝试的动作，直接模拟
            depth = self._get_depth(node)
            reward = self._simulate(node.state, max_depth - depth)
            return node, reward
        
        # 随机选择一个未尝试的动作
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # 应用动作，得到新状态
        new_state = self._apply_action(node.state, action)
        
        # 创建子节点
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action,
            untried_actions=[]  # 简化：不预先生成子节点的动作
        )
        
        node.children.append(child)
        
        # 模拟并获取奖励
        depth = self._get_depth(child)
        reward = self._simulate(new_state, max_depth - depth)
        
        return child, reward
    
    def _simulate(self, state: MCTSState, remaining_depth: int) -> float:
        """
        Simulation 阶段: 随机模拟
        
        Args:
            state: 当前状态
            remaining_depth: 剩余深度
            
        Returns:
            float: 模拟奖励
        """
        if remaining_depth <= 0 or state.is_terminal:
            return state.reward
        
        # 简化：使用启发式方法估计奖励
        # 实际应用中应该使用领域特定的模拟策略
        current_state = state.clone()
        total_reward = state.reward
        
        for _ in range(remaining_depth):
            if current_state.is_terminal:
                break
            
            # 随机衰减 (模拟不确定性)
            decay = random.uniform(0.5, 0.9)
            total_reward += current_state.reward * decay
        
        return total_reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation 阶段: 反向传播奖励
        
        Args:
            node: 起始节点
            reward: 奖励值
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def _apply_action(self, state: MCTSState, action: MCTSAction) -> MCTSState:
        """
        应用动作到状态
        
        Args:
            state: 当前状态
            action: 动作
            
        Returns:
            MCTSState: 新状态
        """
        new_state = state.clone()
        
        # 简化实现：根据动作名称更新状态
        # 实际应用中应该使用领域特定的状态转换函数
        if 'task' in new_state.state_data:
            new_state.state_data['last_action'] = action.name
            new_state.state_data['action_cost'] = action.cost
        
        # 减少奖励 (动作代价)
        new_state.reward -= action.cost * 0.1
        
        return new_state
    
    def _get_depth(self, node: MCTSNode) -> int:
        """获取节点深度"""
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth
    
    def _extract_plan(self, root: MCTSNode) -> List[MCTSAction]:
        """
        提取最优动作序列
        
        Args:
            root: 根节点
            
        Returns:
            List[MCTSAction]: 最优动作序列
        """
        plan = []
        node = root
        
        while node.children:
            # 选择访问次数最多的子节点 (最稳定)
            best_child = max(node.children, key=lambda c: c.visits)
            if best_child.action:
                plan.append(best_child.action)
            node = best_child
        
        return plan
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取规划统计信息"""
        return {
            "iterations": self._iteration_count
        }
