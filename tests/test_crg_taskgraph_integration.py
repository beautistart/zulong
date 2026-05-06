"""
CRG + TaskGraph 集成测试：模拟"深度理解编程项目"完整流程

验证目标：
1. TaskGraph 创建 + req 根节点
2. CRG 索引后正确注入多层结构节点（项目 → 模块 → 文件）
3. 节点层级结构完整（深度 >= 3）
4. 依赖关系完整可追溯
5. 节点 CRUD 操作正确
6. 线程安全（并发写入不崩溃）
7. on_change_callback 正确触发
8. 序列化/反序列化保持完整性
"""
import sys
import os
import time
import threading
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, List

# 确保 zulong 包可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from zulong.l2.task_graph import TaskGraph, TaskNode


# ─── Mock CodeGraph 数据 ─────────────────────────────────────
@dataclass
class MockSymbol:
    node_id: str
    name: str
    kind: str
    file_path: str
    start_line: int = 1
    end_line: int = 10
    qualified_name: str = ""
    parent_class: str = ""


@dataclass
class MockCodeGraph:
    root_dir: str = "/mock/project"
    project_name: str = "zulong_beta4"
    symbols: Dict[str, MockSymbol] = field(default_factory=dict)
    edges: list = field(default_factory=list)
    file_results: Dict[str, object] = field(default_factory=dict)
    directories: Dict[str, List[str]] = field(default_factory=dict)
    edge_count: int = 0

    @classmethod
    def create_realistic(cls):
        """创建仿真的 CodeGraph 数据（模拟 zulong 项目结构）"""
        graph = cls()
        # 模拟目录结构：顶层目录和子目录
        graph.directories = {
            "": ["__init__.py"],  # 根目录下的文件
            "zulong": ["zulong/__init__.py", "zulong/config.py"],
            "zulong/l2": [
                "zulong/l2/__init__.py",
                "zulong/l2/task_graph.py",
                "zulong/l2/attention_window.py",
                "zulong/l2/circuit_breaker.py",
            ],
            "zulong/ide": [
                "zulong/ide/__init__.py",
                "zulong/ide/ide_server.py",
                "zulong/ide/ide_fc_runner.py",
                "zulong/ide/ide_prompt_handler.py",
            ],
            "zulong/tools": [
                "zulong/tools/__init__.py",
                "zulong/tools/code_tools.py",
                "zulong/tools/task_tools.py",
                "zulong/tools/session_tool.py",
            ],
            "zulong/memory": [
                "zulong/memory/__init__.py",
                "zulong/memory/memory_graph.py",
                "zulong/memory/graph_adapters.py",
            ],
            "zulong/code": [
                "zulong/code/__init__.py",
                "zulong/code/graph_builder.py",
                "zulong/code/ast_parser.py",
            ],
            "tests": [
                "tests/test_task_graph.py",
                "tests/test_memory.py",
            ],
            "config": [
                "config/zulong_config.yaml",
            ],
        }
        # 创建符号
        symbols_data = [
            ("code:class:zulong/l2/task_graph.py:TaskGraph", "TaskGraph", "class", "zulong/l2/task_graph.py"),
            ("code:func:zulong/l2/task_graph.py:add_node", "add_node", "method", "zulong/l2/task_graph.py"),
            ("code:func:zulong/l2/task_graph.py:add_h_edge", "add_h_edge", "method", "zulong/l2/task_graph.py"),
            ("code:class:zulong/ide/ide_fc_runner.py:IDEFCRunner", "IDEFCRunner", "class", "zulong/ide/ide_fc_runner.py"),
            ("code:func:zulong/ide/ide_fc_runner.py:_finalize", "_finalize", "method", "zulong/ide/ide_fc_runner.py"),
            ("code:class:zulong/tools/code_tools.py:IndexProjectTool", "IndexProjectTool", "class", "zulong/tools/code_tools.py"),
            ("code:func:zulong/tools/task_tools.py:get_active_task_graph", "get_active_task_graph", "function", "zulong/tools/task_tools.py"),
            ("code:class:zulong/memory/memory_graph.py:MemoryGraph", "MemoryGraph", "class", "zulong/memory/memory_graph.py"),
            ("code:func:zulong/code/graph_builder.py:build", "build", "method", "zulong/code/graph_builder.py"),
        ]
        for node_id, name, kind, fp in symbols_data:
            graph.symbols[node_id] = MockSymbol(
                node_id=node_id, name=name, kind=kind, file_path=fp
            )
        # 模拟文件解析结果
        all_files = set()
        for files in graph.directories.values():
            all_files.update(files)
        graph.file_results = {f: object() for f in all_files}
        graph.edge_count = 15
        return graph


# ─── 测试函数 ────────────────────────────────────────────────


@pytest.fixture
def tg():
    """创建带 req 根节点的 TaskGraph fixture"""
    _tg = TaskGraph(title="深度理解编程项目", graph_id="tg_test_001")
    _tg.add_node(id="req", label="深度理解编程项目", type="requirement",
                 status="in_progress", desc="理解 zulong_beta4 项目架构")
    return _tg


def _inject_crg_data(tg: TaskGraph) -> TaskGraph:
    """辅助函数：向 TaskGraph 注入模拟 CRG 数据"""
    code_graph = MockCodeGraph.create_realistic()
    proj_name = code_graph.project_name or "project"
    struct_id = f"crg_{proj_name}"

    if not tg.get_node(struct_id):
        tg.add_node(
            id=struct_id, label=f"{proj_name} 项目结构",
            type="analysis", status="completed",
            desc=f"{len(code_graph.directories)} 模块, {len(code_graph.file_results)} 文件, {len(code_graph.symbols)} 符号",
        )
        tg.add_h_edge("req", struct_id)

    directories = code_graph.directories
    top_dirs = sorted(
        [(d, files) for d, files in directories.items() if d and "/" not in d],
        key=lambda x: -len(x[1])
    )
    for dir_name, dir_files in top_dirs[:20]:
        dir_node_id = f"crg_{proj_name}/{dir_name}"
        if not tg.get_node(dir_node_id):
            tg.add_node(id=dir_node_id, label=f"{dir_name}/",
                        type="subtask", status="completed", desc=f"模块 {dir_name}")
            tg.add_h_edge(struct_id, dir_node_id)

        sub_dirs = sorted(
            [(d2, f2) for d2, f2 in directories.items()
             if d2.startswith(dir_name + "/") and d2.count("/") == 1],
            key=lambda x: -len(x[1])
        )
        for sub_dir, sub_files in sub_dirs[:15]:
            sub_dir_id = f"crg_{proj_name}/{sub_dir}"
            if not tg.get_node(sub_dir_id):
                sub_name = sub_dir.split("/")[-1]
                tg.add_node(id=sub_dir_id, label=f"{sub_name}/",
                            type="subtask", status="completed", desc=f"子模块 {sub_dir}")
                tg.add_h_edge(dir_node_id, sub_dir_id)

            for file_rel_path in sub_files[:8]:
                base_name = file_rel_path.split("/")[-1]
                file_node_id = f"crg_{proj_name}/{file_rel_path}"
                if not tg.get_node(file_node_id):
                    tg.add_node(id=file_node_id, label=f"{base_name}",
                                type="subtask", status="completed",
                                desc=f"文件 {file_rel_path}")
                    tg.add_h_edge(sub_dir_id, file_node_id)

        for file_rel_path in dir_files[:5]:
            base_name = file_rel_path.split("/")[-1]
            file_node_id = f"crg_{proj_name}/{file_rel_path}"
            if not tg.get_node(file_node_id):
                tg.add_node(id=file_node_id, label=f"{base_name}",
                            type="subtask", status="completed",
                            desc=f"文件 {file_rel_path}")
                tg.add_h_edge(dir_node_id, file_node_id)
    return tg


@pytest.fixture
def tg_with_crg(tg):
    """已注入 CRG 数据的 TaskGraph fixture"""
    return _inject_crg_data(tg)


def test_01_taskgraph_creation():
    """测试 1: TaskGraph 创建和根节点"""
    print("\n=== 测试 1: TaskGraph 创建 ===")
    tg = TaskGraph(title="深度理解编程项目", graph_id="tg_test_001")
    tg.add_node(id="req", label="深度理解编程项目", type="requirement",
                status="in_progress", desc="理解 zulong_beta4 项目架构")
    
    assert tg.id == "tg_test_001"
    assert tg.get_node("req") is not None
    assert tg.get_node("req").status == "in_progress"
    assert tg.get_node("req").type == "requirement"
    print("  [PASS] TaskGraph 创建成功，req 根节点存在")
    return tg


def test_02_crg_injection(tg: TaskGraph):
    """测试 2: 模拟 CRG 注入流程（两级目录结构，匹配 code_tools._background_index 新逻辑）"""
    print("\n=== 测试 2: CRG 节点注入 ===")
    code_graph = MockCodeGraph.create_realistic()
    
    proj_name = code_graph.project_name or "project"
    struct_id = f"crg_{proj_name}"
    
    # 1) 创建项目结构根节点
    if not tg.get_node(struct_id):
        tg.add_node(
            id=struct_id,
            label=f"{proj_name} 项目结构",
            type="analysis",
            status="completed",
            desc=f"{len(code_graph.directories)} 模块, {len(code_graph.file_results)} 文件, {len(code_graph.symbols)} 符号",
        )
        tg.add_h_edge("req", struct_id)
    
    # 2) 构建两级目录树
    directories = code_graph.directories
    top_dirs = sorted(
        [(d, files) for d, files in directories.items() if d and "/" not in d],
        key=lambda x: -len(x[1])
    )
    
    created_dirs = 0
    for dir_name, dir_files in top_dirs[:20]:
        dir_node_id = f"crg_{proj_name}/{dir_name}"
        if not tg.get_node(dir_node_id):
            dir_sym_count = sum(
                1 for sym in code_graph.symbols.values()
                if sym.file_path.startswith(dir_name + "/")
            )
            tg.add_node(
                id=dir_node_id,
                label=f"{dir_name}/ ({len(dir_files)} files, {dir_sym_count} symbols)",
                type="subtask",
                status="completed",
                desc=f"模块 {dir_name}",
            )
            tg.add_h_edge(struct_id, dir_node_id)
            created_dirs += 1
        
        # 3a) 子目录（第二级: "zulong/l2", "zulong/ide" 等）
        sub_dirs = sorted(
            [(d2, f2) for d2, f2 in directories.items()
             if d2.startswith(dir_name + "/") and d2.count("/") == 1],
            key=lambda x: -len(x[1])
        )
        for sub_dir, sub_files in sub_dirs[:15]:
            sub_dir_id = f"crg_{proj_name}/{sub_dir}"
            if not tg.get_node(sub_dir_id):
                sub_name = sub_dir.split("/")[-1]
                sub_sym_count = sum(
                    1 for sym in code_graph.symbols.values()
                    if sym.file_path.startswith(sub_dir + "/")
                )
                tg.add_node(
                    id=sub_dir_id,
                    label=f"{sub_name}/ ({len(sub_files)} files, {sub_sym_count} symbols)",
                    type="subtask",
                    status="completed",
                    desc=f"子模块 {sub_dir}",
                )
                tg.add_h_edge(dir_node_id, sub_dir_id)
                created_dirs += 1

            # 3b) 子目录下的文件节点
            for file_rel_path in sub_files[:8]:
                base_name = file_rel_path.split("/")[-1]
                file_node_id = f"crg_{proj_name}/{file_rel_path}"
                if not tg.get_node(file_node_id):
                    file_sym_count = sum(
                        1 for sym in code_graph.symbols.values()
                        if sym.file_path == file_rel_path
                    )
                    tg.add_node(
                        id=file_node_id,
                        label=f"{base_name} ({file_sym_count} symbols)",
                        type="subtask",
                        status="completed",
                        desc=f"文件 {file_rel_path}",
                    )
                    tg.add_h_edge(sub_dir_id, file_node_id)

        # 3c) 顶层目录下直接的文件
        for file_rel_path in dir_files[:5]:
            base_name = file_rel_path.split("/")[-1]
            file_node_id = f"crg_{proj_name}/{file_rel_path}"
            if not tg.get_node(file_node_id):
                file_sym_count = sum(
                    1 for sym in code_graph.symbols.values()
                    if sym.file_path == file_rel_path
                )
                tg.add_node(
                    id=file_node_id,
                    label=f"{base_name} ({file_sym_count} symbols)",
                    type="subtask",
                    status="completed",
                    desc=f"文件 {file_rel_path}",
                )
                tg.add_h_edge(dir_node_id, file_node_id)
    
    print(f"  注入: {created_dirs} 模块/子模块")
    assert created_dirs > 0, "必须注入至少 1 个模块节点"
    assert tg.get_node(struct_id) is not None, "项目结构根节点必须存在"
    
    # 验证无路径双写
    for node in tg._nodes.values():
        if node.id.startswith("crg_") and "//" in node.id:
            assert False, f"路径双写: {node.id}"
        # 验证文件节点路径正确（不含目录名重复）
        if node.desc and node.desc.startswith("文件 "):
            file_path = node.desc.replace("文件 ", "")
            parts = file_path.split("/")
            for i in range(len(parts) - 1):
                assert parts[i] != parts[i+1], f"路径重复段: {file_path}"
    
    print(f"  [PASS] CRG 注入成功: {created_dirs} 模块/子模块, 无路径双写")
    return tg


def test_03_depth_structure(tg_with_crg: TaskGraph):
    """测试 3: 验证层级深度 >= 4（两级目录结构）"""
    print("\n=== 测试 3: 层级深度验证 ===")
    tg = tg_with_crg
    max_depth = 0
    depth_counts = {}
    
    for node in tg._nodes.values():
        d = tg.get_node_depth(node.id)
        max_depth = max(max_depth, d)
        depth_counts[d] = depth_counts.get(d, 0) + 1
    
    print(f"  深度分布: {depth_counts}")
    print(f"  最大深度: {max_depth}")
    
    assert max_depth >= 4, f"任务图最大深度必须 >= 4（两级目录），实际: {max_depth}"
    assert depth_counts.get(0, 0) == 1, "深度0 应只有 req 节点"
    assert depth_counts.get(1, 0) >= 1, "深度1 应有项目结构节点"
    assert depth_counts.get(2, 0) >= 1, "深度2 应有顶层模块节点"
    assert depth_counts.get(3, 0) >= 1, "深度3 应有子模块节点"
    assert depth_counts.get(4, 0) >= 1, "深度4 应有文件节点"
    print("  [PASS] 层级深度 >= 4，两级目录结构完整")


def test_04_h_edge_dedup(tg_with_crg: TaskGraph):
    """测试 4: 层级边去重"""
    print("\n=== 测试 4: 边去重验证 ===")
    tg = tg_with_crg
    initial_count = len(tg._h_edges)
    
    # 尝试添加重复边
    tg.add_h_edge("req", f"crg_{MockCodeGraph.create_realistic().project_name}")
    tg.add_h_edge("req", f"crg_{MockCodeGraph.create_realistic().project_name}")
    tg.add_h_edge("req", f"crg_{MockCodeGraph.create_realistic().project_name}")
    
    final_count = len(tg._h_edges)
    assert final_count == initial_count, f"重复边不应增加: {initial_count} -> {final_count}"
    
    # 自环检测
    tg.add_h_edge("req", "req")
    assert len(tg._h_edges) == initial_count, "自环不应被添加"
    
    print(f"  [PASS] 边去重正确，{initial_count} 条边保持不变")


def test_05_status_validation(tg: TaskGraph):
    """测试 5: 状态校验"""
    print("\n=== 测试 5: 状态校验 ===")
    
    # 合法状态
    assert tg.update_node_status("req", "in_progress") == True
    assert tg.get_node("req").status == "in_progress"
    
    # 非法状态应被拒绝
    result = tg.update_node_status("req", "invalid_state_xyz")
    assert result == False, "非法状态应返回 False"
    assert tg.get_node("req").status == "in_progress", "状态不应变更"
    
    # 不存在的节点
    result = tg.update_node_status("non_existent_node", "completed")
    assert result == False
    
    print("  [PASS] 状态校验正确")


def test_06_req_no_cascade(tg: TaskGraph):
    """测试 6: req 节点不被自动级联"""
    print("\n=== 测试 6: req 防级联 ===")
    
    # 确保 req 是 in_progress
    tg.update_node_status("req", "in_progress")
    
    # 将 CRG 根节点标记为 completed（模拟 CRG 索引完成）
    struct_id = f"crg_{MockCodeGraph.create_realistic().project_name}"
    tg.update_node_status(struct_id, "completed")
    
    # req 不应被级联为 completed
    assert tg.get_node("req").status == "in_progress", \
        f"req 不应被自动级联! 实际状态: {tg.get_node('req').status}"
    
    print("  [PASS] req 节点不被自动级联为 completed")


def test_07_on_change_callback(tg: TaskGraph):
    """测试 7: on_change_callback 正确触发"""
    print("\n=== 测试 7: 变化回调 ===")
    events = []
    
    def mock_callback(event_type, data):
        events.append((event_type, data))
    
    tg.on_change_callback = mock_callback
    
    # 添加新节点后修改状态应触发回调
    tg.add_node(id="test_cb", label="回调测试节点", type="task", status="pending")
    tg.add_h_edge("req", "test_cb")
    tg.update_node_status("test_cb", "in_progress")
    tg.update_node_status("test_cb", "completed", result="测试完成")
    
    # 至少应有 status 更新的回调
    status_events = [(t, d) for t, d in events if t == "node_update"]
    assert len(status_events) >= 2, f"应有至少 2 个 node_update 事件，实际: {len(status_events)}"
    
    # 验证回调数据完整
    last_event = status_events[-1]
    assert last_event[1]["node_id"] == "test_cb"
    assert last_event[1]["status"] == "completed"
    assert last_event[1]["result"] == "测试完成"
    
    print(f"  [PASS] 回调触发 {len(events)} 次，数据完整")
    tg.on_change_callback = None  # 清理


def test_08_thread_safety(tg: TaskGraph):
    """测试 8: 线程安全（并发写入不崩溃）"""
    print("\n=== 测试 8: 线程安全 ===")
    errors = []
    
    def writer_thread(thread_id, count):
        try:
            for i in range(count):
                node_id = f"thread_{thread_id}_node_{i}"
                tg.add_node(id=node_id, label=f"并发节点 {thread_id}-{i}",
                            type="subtask", status="pending")
                tg.add_h_edge("req", node_id)
                tg.update_node_status(node_id, "in_progress")
                tg.update_node_status(node_id, "completed", result=f"done by thread {thread_id}")
        except Exception as e:
            errors.append((thread_id, str(e)))
    
    def reader_thread(count):
        try:
            for _ in range(count):
                _ = tg.to_frontend_dict()
                _ = tg.get_leaf_nodes()
                _ = tg.get_children("req")
        except Exception as e:
            errors.append(("reader", str(e)))
    
    threads = []
    # 5 个写入线程 + 3 个读取线程并发
    for tid in range(5):
        t = threading.Thread(target=writer_thread, args=(tid, 10))
        threads.append(t)
    for _ in range(3):
        t = threading.Thread(target=reader_thread, args=(20,))
        threads.append(t)
    
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    elapsed = time.time() - start
    
    if errors:
        for tid, err in errors:
            print(f"  [ERROR] Thread {tid}: {err}")
        assert False, f"线程安全测试失败: {len(errors)} 个错误"
    
    print(f"  [PASS] 8 线程并发写入/读取无错误 ({elapsed:.2f}s)")


def test_09_serialize_deserialize(tg_with_crg: TaskGraph):
    """测试 9: 序列化/反序列化保持完整性"""
    print("\n=== 测试 9: 序列化完整性 ===")
    tg = tg_with_crg
    
    # 序列化
    data = tg.serialize()
    assert isinstance(data, dict)
    assert "nodes" in data
    assert "h_edges" in data
    assert "id" in data
    
    # 反序列化
    restored = TaskGraph.deserialize(data)
    
    # 验证节点数一致
    assert len(restored._nodes) == len(tg._nodes), \
        f"节点数不一致: {len(restored._nodes)} vs {len(tg._nodes)}"
    
    # 验证边数一致
    assert len(restored._h_edges) == len(tg._h_edges), \
        f"边数不一致: {len(restored._h_edges)} vs {len(tg._h_edges)}"
    
    # 验证去重集合被正确重建
    assert len(restored._h_edge_set) == len(restored._h_edges), \
        f"_h_edge_set 未正确重建: {len(restored._h_edge_set)} vs {len(restored._h_edges)}"
    
    # 验证关键节点存在且状态正确
    assert restored.get_node("req") is not None
    struct_id = f"crg_{MockCodeGraph.create_realistic().project_name}"
    assert restored.get_node(struct_id) is not None
    
    # 验证深度计算仍正确
    for node in restored._nodes.values():
        d_orig = tg.get_node_depth(node.id)
        d_rest = restored.get_node_depth(node.id)
        assert d_orig == d_rest, f"节点 {node.id} 深度不一致: {d_orig} vs {d_rest}"
    
    print(f"  [PASS] 序列化/反序列化: {len(data['nodes'])} 节点, {len(data['h_edges'])} 边")


def test_10_crud_operations(tg: TaskGraph):
    """测试 10: 节点增删改查"""
    print("\n=== 测试 10: CRUD 操作 ===")
    
    # CREATE
    tg.add_node(id="crud_test", label="CRUD测试", type="task",
                status="pending", desc="测试增删改查")
    tg.add_h_edge("req", "crud_test")
    assert tg.get_node("crud_test") is not None
    
    # READ
    node = tg.get_node("crud_test")
    assert node.label == "CRUD测试"
    assert node.status == "pending"
    depth = tg.get_node_depth("crud_test")
    assert depth == 1  # 直接挂在 req 下
    
    # UPDATE
    tg.update_node_status("crud_test", "in_progress")
    assert tg.get_node("crud_test").status == "in_progress"
    tg.update_node_status("crud_test", "completed", result="已完成CRUD验证")
    assert tg.get_node("crud_test").result == "已完成CRUD验证"
    
    # DELETE
    removed = tg.remove_node("crud_test")
    assert "crud_test" in removed
    assert tg.get_node("crud_test") is None
    
    # 验证 req 不能被删除
    removed = tg.remove_node("req")
    assert removed == []
    assert tg.get_node("req") is not None
    
    print("  [PASS] CRUD 操作全部正确")


def test_11_frontend_dict_format(tg: TaskGraph):
    """测试 11: to_frontend_dict 格式正确"""
    print("\n=== 测试 11: 前端输出格式 ===")
    
    fd = tg.to_frontend_dict()
    
    # 必要字段
    assert "id" in fd
    assert "title" in fd
    assert "nodes" in fd
    assert "hEdges" in fd
    assert "dEdges" in fd
    assert "graphAddress" in fd
    assert "createdAt" in fd
    
    # 节点格式
    for node_dict in fd["nodes"]:
        assert "id" in node_dict
        assert "label" in node_dict
        assert "status" in node_dict
        assert "type" in node_dict
        assert "address" in node_dict
        # 地址格式验证
        assert node_dict["address"].startswith("tg:"), f"地址格式错误: {node_dict['address']}"
    
    # 边格式
    for edge in fd["hEdges"]:
        assert len(edge) == 2
        assert isinstance(edge[0], str)
        assert isinstance(edge[1], str)
    
    # 验证可序列化为 JSON
    json_str = json.dumps(fd, ensure_ascii=False)
    assert len(json_str) > 100
    
    print(f"  [PASS] 前端格式正确: {len(fd['nodes'])} 节点, {len(fd['hEdges'])} 边")
    print(f"  JSON 大小: {len(json_str)} bytes")


def test_12_code_anchor_traceability(tg: TaskGraph):
    """测试 12: 代码锚定可追溯性"""
    print("\n=== 测试 12: 代码锚定 ===")
    
    # 模拟代码锚定：为文件节点关联源代码位置
    code_graph = MockCodeGraph.create_realistic()
    proj_name = code_graph.project_name
    
    # 找到一个文件节点并添加文件引用
    file_node_id = f"crg_{proj_name}/zulong/l2/task_graph.py"
    node = tg.get_node(file_node_id)
    if node:
        tg.add_file_to_node(
            file_node_id,
            file_name="task_graph.py",
            file_path="zulong/l2/task_graph.py"
        )
        # 验证文件引用
        refreshed = tg.get_node(file_node_id)
        assert len(refreshed.files) > 0, "文件引用应被添加"
        assert refreshed.files[0].name == "task_graph.py"
        assert refreshed.files[0].path == "zulong/l2/task_graph.py"
        print(f"  [PASS] 代码锚定: {file_node_id} → {refreshed.files[0].path}")
    else:
        # 如果之前测试中该节点被删除等情况
        print(f"  [SKIP] 文件节点 {file_node_id} 不存在（可能被其他测试影响）")


# ─── 主测试流程 ───────────────────────────────────────────────

def run_all_tests():
    """按顺序执行所有集成测试"""
    print("=" * 60)
    print("  CRG + TaskGraph 集成测试: '深度理解编程项目' 流程")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    try:
        tg = test_01_taskgraph_creation()
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] 测试 1: {e}")
        failed += 1
        return
    
    tests = [
        ("02_crg_injection", lambda: test_02_crg_injection(tg)),
        ("03_depth_structure", lambda: test_03_depth_structure(tg)),
        ("04_h_edge_dedup", lambda: test_04_h_edge_dedup(tg)),
        ("05_status_validation", lambda: test_05_status_validation(tg)),
        ("06_req_no_cascade", lambda: test_06_req_no_cascade(tg)),
        ("07_on_change_callback", lambda: test_07_on_change_callback(tg)),
        ("08_thread_safety", lambda: test_08_thread_safety(tg)),
        ("09_serialize_deserialize", lambda: test_09_serialize_deserialize(tg)),
        ("10_crud_operations", lambda: test_10_crud_operations(tg)),
        ("11_frontend_dict_format", lambda: test_11_frontend_dict_format(tg)),
        ("12_code_anchor", lambda: test_12_code_anchor_traceability(tg)),
    ]
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result is not None:
                tg = result  # 部分测试返回修改后的 tg
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] 测试 {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] 测试 {name}: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"  结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
