# File: tests/test_task_graph_recursive.py
"""
测试任务图谱的无限深度递归树架构

验证内容：
1. TaskGraph 基础操作：多级节点创建、层级边、叶子节点判定
2. depth_to_type：深度 -> 类型自动映射
3. get_children / get_leaf_nodes / get_node_depth
4. 状态聚合：非叶子节点从子节点动态聚合 status
5. to_frontend_dict：前端兼容导出
6. to_planning_table：递归规划表
7. 序列化/反序列化完整性
"""

import sys
import os
import json

# 确保可以导入 zulong 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zulong.l2.task_graph import TaskGraph, TaskNode


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def create_template_graph(title="测试图谱") -> TaskGraph:
    """创建包含模板节点（req + analysis）的基础图谱"""
    g = TaskGraph(title=title)
    g.add_node("req", label="原始需求", type="requirement", status="completed", desc="用户需求")
    g.add_node("analysis", label="需求分析", type="analysis", status="completed", desc="分析结果")
    g.add_h_edge("req", "analysis")
    return g


def build_4_level_tree(g: TaskGraph):
    """构建 4 级深度的任务树
    
    结构:
    req -> analysis -> o1 (outline)
                      ├── o1_1 (task)
                      │    ├── o1_1_1 (subtask)  <- 叶子
                      │    └── o1_1_2 (subtask)  <- 叶子
                      └── o1_2 (task)  <- 叶子
                  -> o2 (outline)
                      └── o2_1 (task)
                           └── o2_1_1 (subtask)
                                └── o2_1_1_1 (subtask)  <- 叶子 (第4级)
    """
    # 大纲层 (depth 2)
    g.add_node("o1", label="前端开发", type="outline", status="pending", desc="前端相关任务")
    g.add_node("o2", label="后端开发", type="outline", status="pending", desc="后端相关任务")
    g.add_h_edge("analysis", "o1")
    g.add_h_edge("analysis", "o2")

    # 任务层 (depth 3)
    g.add_node("o1_1", label="组件开发", type="task", status="pending", desc="React组件")
    g.add_node("o1_2", label="样式调整", type="task", status="pending", desc="CSS样式")
    g.add_h_edge("o1", "o1_1")
    g.add_h_edge("o1", "o1_2")

    g.add_node("o2_1", label="API开发", type="task", status="pending", desc="REST API")
    g.add_h_edge("o2", "o2_1")

    # 子任务层 (depth 4)
    g.add_node("o1_1_1", label="按钮组件", type="subtask", status="pending", desc="Button组件")
    g.add_node("o1_1_2", label="表单组件", type="subtask", status="pending", desc="Form组件")
    g.add_h_edge("o1_1", "o1_1_1")
    g.add_h_edge("o1_1", "o1_1_2")

    g.add_node("o2_1_1", label="数据库设计", type="subtask", status="pending", desc="Schema设计")
    g.add_h_edge("o2_1", "o2_1_1")

    # 子子任务层 (depth 5) — 第4级
    g.add_node("o2_1_1_1", label="用户表设计", type="subtask", status="pending", desc="users表")
    g.add_h_edge("o2_1_1", "o2_1_1_1")


def _print_tree(g: TaskGraph, node_id: str, indent: int = 0):
    """递归打印树形结构"""
    node = g.get_node(node_id)
    if not node:
        return
    prefix = "  " * indent + ("├── " if indent > 0 else "")
    children = g.get_children(node_id)
    is_leaf = len(children) == 0 and node_id not in ("req", "analysis")
    leaf_mark = " [LEAF]" if is_leaf else ""
    print(f"  {prefix}{node.id} ({node.type}) - {node.label}{leaf_mark}")
    for child in children:
        _print_tree(g, child.id, indent + 1)


# ═══════════════════════════════════════════════════════════
# 测试 1: depth_to_type 映射
# ═══════════════════════════════════════════════════════════

def test_depth_to_type():
    print("=" * 60)
    print("测试 1: depth_to_type 深度->类型映射")
    print("=" * 60)

    expected = {
        0: "requirement",
        1: "analysis",
        2: "outline",
        3: "task",
        4: "subtask",
        5: "subtask",
        10: "subtask",
        100: "subtask",
    }

    all_pass = True
    for depth, expected_type in expected.items():
        actual = TaskGraph.depth_to_type(depth)
        status = "PASS" if actual == expected_type else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  depth={depth:3d} -> {actual:12s}  (期望: {expected_type:12s})  [{status}]")

    print(f"\n  结果: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")
    return all_pass


# ═══════════════════════════════════════════════════════════
# 测试 2: 多级节点创建 + get_children
# ═══════════════════════════════════════════════════════════

def test_get_children():
    print("=" * 60)
    print("测试 2: 多级节点创建 + get_children")
    print("=" * 60)

    g = create_template_graph()
    build_4_level_tree(g)

    tests = [
        ("req", ["analysis"]),
        ("analysis", ["o1", "o2"]),
        ("o1", ["o1_1", "o1_2"]),
        ("o1_1", ["o1_1_1", "o1_1_2"]),
        ("o1_2", []),  # 叶子无子节点
        ("o2", ["o2_1"]),
        ("o2_1", ["o2_1_1"]),
        ("o2_1_1", ["o2_1_1_1"]),
        ("o2_1_1_1", []),  # 最深叶子
    ]

    all_pass = True
    for parent_id, expected_child_ids in tests:
        children = g.get_children(parent_id)
        actual_ids = [c.id for c in children]
        match = actual_ids == expected_child_ids
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  {parent_id:12s} -> children: {actual_ids}  (期望: {expected_child_ids})  [{status}]")

    print(f"\n  结果: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")
    return all_pass


# ═══════════════════════════════════════════════════════════
# 测试 3: get_leaf_nodes
# ═══════════════════════════════════════════════════════════

def test_get_leaf_nodes():
    print("=" * 60)
    print("测试 3: get_leaf_nodes 叶子节点判定")
    print("=" * 60)

    g = create_template_graph()
    build_4_level_tree(g)

    leaves = g.get_leaf_nodes()
    leaf_ids = sorted([n.id for n in leaves])
    expected_leaf_ids = sorted(["o1_1_1", "o1_1_2", "o1_2", "o2_1_1_1"])

    print(f"  叶子节点: {leaf_ids}")
    print(f"  期望:     {expected_leaf_ids}")

    match = leaf_ids == expected_leaf_ids
    print(f"  结果: {'PASS' if match else 'FAIL'}\n")

    # 验证模板节点不算叶子
    leaf_id_set = set(leaf_ids)
    assert "req" not in leaf_id_set, "req 不应该是叶子节点"
    assert "analysis" not in leaf_id_set, "analysis 不应该是叶子节点"

    # 验证容器节点不算叶子
    assert "o1" not in leaf_id_set, "o1 是容器节点，不应该是叶子"
    assert "o1_1" not in leaf_id_set, "o1_1 有子节点，不应该是叶子"
    assert "o2_1" not in leaf_id_set, "o2_1 有子节点，不应该是叶子"
    assert "o2_1_1" not in leaf_id_set, "o2_1_1 有子节点，不应该是叶子"

    print("  模板/容器节点排除验证: PASS\n")
    return match


# ═══════════════════════════════════════════════════════════
# 测试 4: get_node_depth
# ═══════════════════════════════════════════════════════════

def test_get_node_depth():
    print("=" * 60)
    print("测试 4: get_node_depth 节点深度")
    print("=" * 60)

    g = create_template_graph()
    build_4_level_tree(g)

    expected_depths = {
        "req": 0,
        "analysis": 1,
        "o1": 2,
        "o2": 2,
        "o1_1": 3,
        "o1_2": 3,
        "o2_1": 3,
        "o1_1_1": 4,
        "o1_1_2": 4,
        "o2_1_1": 4,
        "o2_1_1_1": 5,  # 第4级子任务
    }

    all_pass = True
    for node_id, expected_depth in expected_depths.items():
        actual = g.get_node_depth(node_id)
        match = actual == expected_depth
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        node = g.get_node(node_id)
        print(f"  {node_id:12s} depth={actual}  (期望: {expected_depth})  type={node.type if node else '?':12s}  [{status}]")

    print(f"\n  结果: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")
    return all_pass


# ═══════════════════════════════════════════════════════════
# 测试 5: 状态聚合 (_aggregate_status)
# ═══════════════════════════════════════════════════════════

def test_status_aggregation():
    print("=" * 60)
    print("测试 5: 非叶子节点状态聚合")
    print("=" * 60)

    g = create_template_graph()
    build_4_level_tree(g)

    # 场景 A: 所有叶子 pending -> 所有父节点 pending
    print("  场景A: 全部 pending")
    front = g.to_frontend_dict()
    node_map = {n["id"]: n for n in front["nodes"]}
    assert node_map["o1"]["status"] == "pending", f"o1 应该是 pending, 实际: {node_map['o1']['status']}"
    assert node_map["o2"]["status"] == "pending", f"o2 应该是 pending, 实际: {node_map['o2']['status']}"
    print("    o1=pending, o2=pending  [PASS]")

    # 场景 B: 部分叶子完成
    g.update_node_status("o1_1_1", "completed", result="按钮完成")
    g.update_node_status("o1_1_2", "in_progress")
    front = g.to_frontend_dict()
    node_map = {n["id"]: n for n in front["nodes"]}
    # o1_1 的子节点: o1_1_1=completed, o1_1_2=in_progress -> o1_1 应该是 in_progress
    assert node_map["o1_1"]["status"] == "in_progress", f"o1_1 应该是 in_progress, 实际: {node_map['o1_1']['status']}"
    # o1 的子节点: o1_1=in_progress (聚合), o1_2=pending -> o1 应该是 in_progress
    assert node_map["o1"]["status"] == "in_progress", f"o1 应该是 in_progress, 实际: {node_map['o1']['status']}"
    print("  场景B: 部分叶子执行中")
    print(f"    o1_1={node_map['o1_1']['status']}, o1={node_map['o1']['status']}  [PASS]")

    # 场景 C: 一个大纲下全部完成
    g.update_node_status("o1_1_2", "completed", result="表单完成")
    g.update_node_status("o1_2", "completed", result="样式完成")
    front = g.to_frontend_dict()
    node_map = {n["id"]: n for n in front["nodes"]}
    assert node_map["o1"]["status"] == "completed", f"o1 应该是 completed, 实际: {node_map['o1']['status']}"
    assert node_map["o2"]["status"] == "pending", f"o2 应该仍是 pending, 实际: {node_map['o2']['status']}"
    print("  场景C: o1 全部完成, o2 仍 pending")
    print(f"    o1={node_map['o1']['status']}, o2={node_map['o2']['status']}  [PASS]")

    # 场景 D: 深层嵌套完成传递
    g.update_node_status("o2_1_1_1", "completed", result="用户表完成")
    front = g.to_frontend_dict()
    node_map = {n["id"]: n for n in front["nodes"]}
    assert node_map["o2_1_1"]["status"] == "completed", f"o2_1_1 实际: {node_map['o2_1_1']['status']}"
    assert node_map["o2_1"]["status"] == "completed", f"o2_1 实际: {node_map['o2_1']['status']}"
    assert node_map["o2"]["status"] == "completed", f"o2 实际: {node_map['o2']['status']}"
    print("  场景D: 深层嵌套完成向上传递 (depth 5 -> 2)")
    print(f"    o2_1_1_1=completed -> o2_1_1={node_map['o2_1_1']['status']} -> o2_1={node_map['o2_1']['status']} -> o2={node_map['o2']['status']}  [PASS]")

    print(f"\n  结果: ALL PASS\n")
    return True


# ═══════════════════════════════════════════════════════════
# 测试 6: to_frontend_dict 格式
# ═══════════════════════════════════════════════════════════

def test_frontend_dict():
    print("=" * 60)
    print("测试 6: to_frontend_dict 前端兼容格式")
    print("=" * 60)

    g = create_template_graph()
    build_4_level_tree(g)

    front = g.to_frontend_dict()

    # 检查基本结构
    assert "id" in front, "缺少 id"
    assert "title" in front, "缺少 title"
    assert "nodes" in front, "缺少 nodes"
    assert "hEdges" in front, "缺少 hEdges"
    assert "dEdges" in front, "缺少 dEdges"
    print("  基本结构字段完整: PASS")

    # 检查节点数量 (2 模板 + 9 业务节点 = 11)
    node_count = len(front["nodes"])
    assert node_count == 11, f"期望 11 个节点, 实际 {node_count}"
    print(f"  节点数量: {node_count} (期望 11): PASS")

    # 检查 hEdges 数量
    hedge_count = len(front["hEdges"])
    assert hedge_count == 10, f"期望 10 条层级边, 实际 {hedge_count}"
    print(f"  层级边数量: {hedge_count} (期望 10): PASS")

    # 检查每个节点的字段
    for node in front["nodes"]:
        assert "id" in node, f"节点缺少 id: {node}"
        assert "label" in node, f"节点缺少 label: {node}"
        assert "type" in node, f"节点缺少 type: {node}"
        assert "status" in node, f"节点缺少 status: {node}"
    print("  所有节点字段完整: PASS")

    # JSON 序列化测试
    json_str = json.dumps(front, ensure_ascii=False, indent=2)
    assert len(json_str) > 0, "JSON序列化失败"
    print(f"  JSON 序列化成功: {len(json_str)} 字符")

    print(f"\n  结果: ALL PASS\n")
    return True


# ═══════════════════════════════════════════════════════════
# 测试 7: to_planning_table 递归规划表
# ═══════════════════════════════════════════════════════════

def test_planning_table():
    print("=" * 60)
    print("测试 7: to_planning_table 递归规划表")
    print("=" * 60)

    g = create_template_graph()
    build_4_level_tree(g)

    # 设置一些状态
    g.update_node_status("o1_1_1", "completed", result="按钮组件已完成，包含hover效果和disabled状态")
    g.update_node_status("o1_1_2", "in_progress")

    table = g.to_planning_table()
    print("  规划表输出:")
    for line in table.split("\n"):
        print(f"    {line}")

    # 验证结构
    assert "## 当前任务规划" in table, "缺少标题"
    assert "### 前端开发" in table, "缺少大纲: 前端开发"
    assert "### 后端开发" in table, "缺少大纲: 后端开发"
    assert "o1_1_1" in table, "缺少叶子节点 o1_1_1"
    assert "o1_1_2" in table, "缺少叶子节点 o1_1_2"
    assert "o1_2" in table, "缺少叶子节点 o1_2"
    assert "o2_1_1_1" in table, "缺少深层叶子节点 o2_1_1_1"
    assert "完成" in table, "缺少状态文本: 完成"
    assert "进行中" in table, "缺少状态文本: 进行中"
    # 非叶子节点应该作为分组标题
    assert "**组件开发**" in table, "非叶子节点 o1_1 应作为分组标题"

    print(f"\n  结果: ALL PASS\n")
    return True


# ═══════════════════════════════════════════════════════════
# 测试 8: 序列化/反序列化
# ═══════════════════════════════════════════════════════════

def test_serialization():
    print("=" * 60)
    print("测试 8: 序列化/反序列化保持完整性")
    print("=" * 60)

    g = create_template_graph("序列化测试")
    build_4_level_tree(g)
    g.update_node_status("o1_1_1", "completed", result="done")
    g.add_d_edge("o1_1_1", "o1_1_2", via="测试数据")

    # 序列化
    data = g.serialize()
    json_str = json.dumps(data, ensure_ascii=False)
    print(f"  序列化大小: {len(json_str)} 字符")

    # 反序列化
    g2 = TaskGraph.deserialize(data)

    # 验证节点数量
    assert len(g2.nodes) == len(g.nodes), "节点数量不匹配"
    print(f"  节点数量: {len(g2.nodes)} = {len(g.nodes)}: PASS")

    # 验证边数量
    assert len(g2.h_edges) == len(g.h_edges), "层级边数量不匹配"
    assert len(g2.d_edges) == len(g.d_edges), "依赖边数量不匹配"
    print(f"  层级边: {len(g2.h_edges)}, 依赖边: {len(g2.d_edges)}: PASS")

    # 验证深层节点
    deep_node = g2.get_node("o2_1_1_1")
    assert deep_node is not None, "反序列化后深层节点丢失"
    assert deep_node.label == "用户表设计"
    print(f"  深层节点 o2_1_1_1 保留: PASS")

    # 验证叶子节点
    leaves_orig = sorted([n.id for n in g.get_leaf_nodes()])
    leaves_new = sorted([n.id for n in g2.get_leaf_nodes()])
    assert leaves_orig == leaves_new, "叶子节点不匹配"
    print(f"  叶子节点保持一致: PASS")

    # 验证状态
    node = g2.get_node("o1_1_1")
    assert node.status == "completed", "状态未保留"
    assert node.result == "done", "结果未保留"
    print(f"  节点状态和结果保留: PASS")

    print(f"\n  结果: ALL PASS\n")
    return True


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  祖龙任务图谱 — 无限深度递归树 测试套件")
    print("=" * 60 + "\n")

    tests = [
        ("depth_to_type 映射", test_depth_to_type),
        ("get_children", test_get_children),
        ("get_leaf_nodes", test_get_leaf_nodes),
        ("get_node_depth", test_get_node_depth),
        ("状态聚合", test_status_aggregation),
        ("to_frontend_dict", test_frontend_dict),
        ("to_planning_table", test_planning_table),
        ("序列化/反序列化", test_serialization),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  !!! 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 汇总
    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    for name, p in results:
        icon = "PASS" if p else "FAIL"
        print(f"  [{icon}] {name}")

    print(f"\n  总计: {total} 个测试, {passed} 通过, {failed} 失败")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
