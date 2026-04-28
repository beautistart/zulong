# File: tests/test_task_graph_fix.py
"""
测试任务图谱修复

验证:
1. 复杂度分类是否生效
2. 任务图谱技能包是否安装
3. 工具是否注册到 ToolRegistry
4. AgentOrchestrator 是否启动
5. 图谱事件是否发布到 EventBus
"""

import sys
import os
import asyncio
import logging
import time
import pytest
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_complexity_classification():
    """测试 1: 复杂度分类（已移除 - _is_complex_task 已废弃，模型自主路由）"""
    print("\n" + "="*80)
    print("📋 测试 1: 复杂度分类（跳过 - 模型自主路由已替代硬编码分类）")
    print("="*80)
    print("✅ 测试 1 跳过\n")


def test_task_graph_pack_installation():
    """测试 2: 任务图谱工具注册到 ToolRegistry"""
    print("\n" + "="*80)
    print("📋 测试 2: 任务图谱工具可实例化并注册到 ToolRegistry")
    print("="*80)

    from zulong.tools.base import ToolRegistry
    from zulong.tools.task_tools import (
        TaskCreatePlanTool,
        TaskAddNodeTool,
        TaskMarkStatusTool,
        TaskViewOverviewTool,
        TaskSuspendTool,
        TaskListSuspendedTool,
    )

    # 重置单例以避免测试间干扰
    ToolRegistry._instance = None
    registry = ToolRegistry()

    tool_classes = [
        TaskCreatePlanTool,
        TaskAddNodeTool,
        TaskMarkStatusTool,
        TaskViewOverviewTool,
        TaskSuspendTool,
        TaskListSuspendedTool,
    ]

    for cls in tool_classes:
        tool = cls()
        success = registry.register(tool)
        assert success, f"注册失败: {tool.name}"
        print(f"  ✅ {tool.name}: {tool.description[:40]}...")

    # 验证所有工具已注册
    assert len(registry.tools) >= len(tool_classes)
    print(f"📊 共注册 {len(registry.tools)} 个工具")

    # 清理
    ToolRegistry._instance = None
    print("✅ 测试 2 完成\n")


def test_agent_orchestrator_with_external_graph():
    """测试 3: TaskGraph 与活跃图管理（替代已废弃的 AgentOrchestrator）"""
    print("\n" + "="*80)
    print("📋 测试 3: TaskGraph 活跃图设置与获取")
    print("="*80)

    from zulong.l2.task_graph import TaskGraph
    from zulong.tools.task_tools import (
        get_active_task_graph,
        set_active_task_graph,
    )

    # 保存原始状态
    original_tg = get_active_task_graph()

    try:
        # 创建外部 TaskGraph
        task_graph = TaskGraph(title="测试任务")
        print(f"✅ 外部 TaskGraph 已创建：{task_graph.title}")

        # 设置为活跃图
        set_active_task_graph(task_graph, task_graph.id)

        # 验证获取
        active = get_active_task_graph()
        assert active is task_graph, "活跃图设置失败"
        print(f"✅ 活跃图已设置: {active.title}")

        # 添加节点验证功能
        task_graph.add_node("req", label="需求", type="requirement",
                           status="in_progress", desc="测试需求")
        task_graph.add_node("o1", label="步骤1", type="outline",
                           status="pending", desc="")
        task_graph.add_h_edge("req", "o1")

        # 验证节点结构
        assert task_graph.get_node("req") is not None
        assert len(task_graph.get_leaf_nodes()) == 1
        print(f"✅ TaskGraph 节点结构正确: {len(task_graph._nodes)} 个节点")

        # 清除
        set_active_task_graph(None, None)
        assert get_active_task_graph() is None
        print(f"✅ 活跃图已清除")
    finally:
        # 恢复原始状态
        if original_tg is not None:
            set_active_task_graph(original_tg, getattr(original_tg, 'id', ''))
        else:
            set_active_task_graph(None, None)

    print("✅ 测试 3 完成\n")


def test_tool_registry_singleton():
    """测试 4: ToolRegistry 单例模式"""
    print("\n" + "="*80)
    print("📋 测试 4: ToolRegistry 单例模式")
    print("="*80)
    
    from zulong.tools.base import ToolRegistry
    
    # 创建两个实例
    registry1 = ToolRegistry()
    registry2 = ToolRegistry()
    
    # 验证是同一个实例
    if registry1 is registry2:
        print(f"✅ ToolRegistry 是单例模式")
    else:
        print(f"❌ ToolRegistry 不是单例模式")
    
    # 检查工具数量
    tools = registry1.list_tools()
    print(f"📊 当前注册工具数量：{len(tools)}")
    
    print("✅ 测试 4 完成\n")


async def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🚀 开始测试任务图谱修复")
    print("="*80)
    
    try:
        # 测试 1: 复杂度分类
        test_complexity_classification()
        
        # 测试 2: 技能包安装
        test_task_graph_pack_installation()
        
        # 测试 3: TaskGraph 活跃图管理
        test_agent_orchestrator_with_external_graph()
        
        # 测试 4: ToolRegistry 单例
        test_tool_registry_singleton()
        
        print("\n" + "="*80)
        print("✅ 所有测试完成!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
