"""
Zulong 系统测试 - 共享 Fixtures

提供单例重置、临时目录、预构建对象等公共 fixture。
"""

import os
import sys
import shutil
import tempfile

import pytest

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── 单例重置 ──

@pytest.fixture(autouse=True)
def reset_memory_graph_singleton():
    """每个测试前/后重置 MemoryGraph 单例"""
    try:
        from zulong.memory.memory_graph import MemoryGraph
        MemoryGraph._instance = None
        if hasattr(MemoryGraph, '_initialized'):
            del MemoryGraph._initialized
    except ImportError:
        pass
    yield
    try:
        from zulong.memory.memory_graph import MemoryGraph
        MemoryGraph._instance = None
        if hasattr(MemoryGraph, '_initialized'):
            del MemoryGraph._initialized
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def reset_task_suspension_singleton():
    """每个测试前/后重置 TaskSuspensionManager 单例"""
    try:
        from zulong.l2.task_suspension import TaskSuspensionManager
        TaskSuspensionManager._instance = None
        if hasattr(TaskSuspensionManager, '_initialized'):
            # _initialized 是实例属性，通过清除 _instance 自然重置
            pass
    except ImportError:
        pass
    yield
    try:
        from zulong.l2.task_suspension import TaskSuspensionManager
        TaskSuspensionManager._instance = None
    except ImportError:
        pass


# ── 临时目录 ──

@pytest.fixture
def temp_dir():
    """提供一个临时目录，测试结束后自动清理"""
    d = tempfile.mkdtemp(prefix="zulong_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── MemoryGraph 隔离实例 ──

@pytest.fixture
def temp_memory_graph(temp_dir):
    """创建一个使用临时目录的隔离 MemoryGraph 实例"""
    from zulong.memory.memory_graph import MemoryGraph
    graph = MemoryGraph(persist_path=temp_dir)
    return graph


# ── 预构建 TaskGraph ──

@pytest.fixture
def sample_task_graph():
    """预构建 TaskGraph: req -> analysis -> 3 outline 节点 + 依赖边"""
    from zulong.l2.task_graph import TaskGraph

    tg = TaskGraph(title="测试任务", graph_id="test_tg_001")

    # 根节点
    tg.add_node(id="req", label="需求根节点", type="requirement",
                status="in_progress", desc="做一个测试项目")
    # 分析节点
    tg.add_node(id="analysis", label="需求分析", type="analysis",
                status="in_progress", desc="分析测试需求")
    # 大纲节点
    tg.add_node(id="o1", label="模块A", type="outline",
                status="pending", desc="实现模块A")
    tg.add_node(id="o2", label="模块B", type="outline",
                status="pending", desc="实现模块B")
    tg.add_node(id="o3", label="集成测试", type="outline",
                status="pending", desc="执行集成测试")

    # 层级边
    tg.add_h_edge("req", "analysis")
    tg.add_h_edge("analysis", "o1")
    tg.add_h_edge("analysis", "o2")
    tg.add_h_edge("analysis", "o3")

    # 依赖边
    tg.add_d_edge("o1", "o3", via="模块A输出", cross=False)
    tg.add_d_edge("o2", "o3", via="模块B输出", cross=False)

    return tg
