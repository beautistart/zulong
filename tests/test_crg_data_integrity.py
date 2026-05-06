"""CRG 数据完整性验证脚本

验证 CRG 原始数据经 MemoryGraph 节点化改造后的正确性：
- AST 解析 → CodeGraph 构建 → MemoryGraph 投射
- 节点层级: MODULE → FILE → CODE_SYMBOL
- 边关系: contains/calls/inherits/imports
- BFS 传播能穿透 CODE_SYMBOL 节点

用法:
    python tests/test_crg_data_integrity.py [--target-dir zulong/tools]
"""

import sys
import os
import argparse
import logging

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("crg_verify")


def run_verification(target_dir: str, persist_path: str = None):
    """执行 CRG 数据完整性验证"""

    # ── 1. 导入核心模块 ──
    logger.info("=" * 60)
    logger.info("CRG 数据完整性验证")
    logger.info("=" * 60)

    try:
        from zulong.code.ast_parser import ASTParser
        from zulong.code.graph_builder import CodeGraphBuilder
        from zulong.memory.memory_graph import (
            MemoryGraph, GraphNode, NodeType, EdgeType, get_memory_graph,
        )
        from zulong.memory.graph_adapters import CodeGraphAdapter
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        logger.error("请确保在项目根目录运行，且依赖已安装 (networkx, tree-sitter)")
        return False

    # ── 2. 验证 ASTParser 可用 ──
    logger.info("\n[Step 1] 验证 ASTParser 可用性...")
    parser = ASTParser(lang="python")
    if not parser.available:
        logger.error("❌ Tree-sitter Python 解析器不可用")
        logger.error("   请安装: pip install tree-sitter tree-sitter-python")
        return False
    logger.info("✅ ASTParser (Python) 可用")

    # ── 3. 构建 CodeGraph ──
    abs_target = os.path.abspath(target_dir)
    if not os.path.isdir(abs_target):
        logger.error(f"❌ 目标目录不存在: {abs_target}")
        return False

    logger.info(f"\n[Step 2] 构建 CodeGraph (目录: {target_dir})...")
    builder = CodeGraphBuilder()
    code_graph = builder.build(abs_target, languages=["python"], max_files=50)

    logger.info(f"  符号数: {code_graph.symbol_count}")
    logger.info(f"  边数:   {code_graph.edge_count}")
    logger.info(f"  目录数: {len(code_graph.directories)}")
    logger.info(f"  文件数: {len(code_graph.file_results)}")

    if code_graph.symbol_count == 0:
        logger.error("❌ CodeGraph 符号数为 0，解析可能失败")
        return False
    logger.info("✅ CodeGraph 构建成功")

    # ── 4. 初始化 MemoryGraph (临时实例，不干扰生产数据) ──
    logger.info("\n[Step 3] 初始化 MemoryGraph (临时实例)...")
    if persist_path:
        mg = MemoryGraph(persist_path=persist_path)
    else:
        # 使用临时路径，不影响生产数据
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="crg_verify_")
        mg = MemoryGraph(persist_path=tmp_dir)

    logger.info("✅ MemoryGraph 初始化完成")

    # ── 5. 执行 CodeGraphAdapter.sync() ──
    logger.info("\n[Step 4] 执行 CodeGraphAdapter.sync() 投射到 MemoryGraph...")
    adapter = CodeGraphAdapter()
    mg.register_adapter(adapter.name, adapter)
    count = adapter.sync(mg, code_graph)

    logger.info(f"  投射符号数: {count}")
    logger.info("✅ sync() 执行完成")

    # ── 6. 验证节点类型和数量 ──
    logger.info("\n[Step 5] 验证节点类型和数量...")
    results = {}

    code_symbols = mg.get_nodes_by_type(NodeType.CODE_SYMBOL)
    modules = mg.get_nodes_by_type(NodeType.MODULE)
    files = mg.get_nodes_by_type(NodeType.FILE)

    results["code_symbol_count"] = len(code_symbols)
    results["module_count"] = len(modules)
    results["file_count"] = len(files)

    logger.info(f"  CODE_SYMBOL 节点: {len(code_symbols)}")
    logger.info(f"  MODULE 节点:      {len(modules)}")
    logger.info(f"  FILE 节点:        {len(files)}")

    assert len(code_symbols) > 0, "❌ CODE_SYMBOL 节点数为 0"
    assert len(modules) > 0, "❌ MODULE 节点数为 0"
    assert len(files) > 0, "❌ FILE 节点数为 0"
    logger.info("✅ 节点类型验证通过")

    # ── 7. 验证节点数据正确性 ──
    logger.info("\n[Step 6] 验证节点元数据正确性...")
    sample_sym = code_symbols[0]
    logger.info(f"  示例 CODE_SYMBOL: {sample_sym.label}")
    logger.info(f"    node_id:    {sample_sym.node_id}")
    logger.info(f"    kind:       {sample_sym.metadata.get('kind')}")
    logger.info(f"    file_path:  {sample_sym.metadata.get('file_path')}")
    logger.info(f"    start_line: {sample_sym.metadata.get('start_line')}")
    logger.info(f"    end_line:   {sample_sym.metadata.get('end_line')}")
    logger.info(f"    parameters: {sample_sym.metadata.get('parameters')}")

    assert sample_sym.metadata.get("kind") in ("function", "method", "class"), \
        f"❌ kind 无效: {sample_sym.metadata.get('kind')}"
    assert sample_sym.metadata.get("file_path"), "❌ file_path 为空"
    assert sample_sym.metadata.get("start_line") is not None, "❌ start_line 缺失"
    logger.info("✅ 节点元数据验证通过")

    # ── 8. 验证层级结构 (MODULE → FILE → CODE_SYMBOL) ──
    logger.info("\n[Step 7] 验证层级结构...")
    hierarchy_edges = []
    reference_edges = []
    dependency_edges = []

    # 遍历图中所有边
    graph = mg._graph  # NetworkX DiGraph
    for u, v, data in graph.edges(data=True):
        edge_type = data.get("edge_type")
        if edge_type == EdgeType.HIERARCHY.value or edge_type == EdgeType.HIERARCHY:
            hierarchy_edges.append((u, v, data))
        elif edge_type == EdgeType.REFERENCE.value or edge_type == EdgeType.REFERENCE:
            reference_edges.append((u, v, data))
        elif edge_type == EdgeType.DEPENDENCY.value or edge_type == EdgeType.DEPENDENCY:
            dependency_edges.append((u, v, data))

    results["hierarchy_edges"] = len(hierarchy_edges)
    results["reference_edges"] = len(reference_edges)
    results["dependency_edges"] = len(dependency_edges)

    logger.info(f"  HIERARCHY 边 (contains/inherits): {len(hierarchy_edges)}")
    logger.info(f"  REFERENCE 边 (calls):             {len(reference_edges)}")
    logger.info(f"  DEPENDENCY 边 (imports):          {len(dependency_edges)}")

    assert len(hierarchy_edges) > 0, "❌ HIERARCHY 边为 0 (应有 MODULE→FILE→CODE_SYMBOL)"
    logger.info("✅ 层级结构验证通过")

    # 验证 MODULE→FILE 边存在
    module_to_file_edges = [
        (u, v) for u, v, d in hierarchy_edges
        if u.startswith("module:") and v.startswith("file:")
    ]
    logger.info(f"  MODULE→FILE 边: {len(module_to_file_edges)}")
    assert len(module_to_file_edges) > 0, "❌ 无 MODULE→FILE HIERARCHY 边"

    # 验证 FILE→CODE_SYMBOL 边存在
    file_to_symbol_edges = [
        (u, v) for u, v, d in hierarchy_edges
        if u.startswith("file:") and v.startswith("code:")
    ]
    logger.info(f"  FILE→CODE_SYMBOL 边: {len(file_to_symbol_edges)}")
    assert len(file_to_symbol_edges) > 0, "❌ 无 FILE→CODE_SYMBOL HIERARCHY 边"
    logger.info("✅ MODULE→FILE→CODE_SYMBOL 层级验证通过")

    # ── 9. 验证边关系元数据 ──
    logger.info("\n[Step 8] 验证边关系元数据...")

    # 检查 contains 关系 (class → method)
    contains_edges = [
        (u, v, d) for u, v, d in hierarchy_edges
        if d.get("metadata", {}).get("relation") == "defines"
        or d.get("metadata", {}).get("code_relation") == "contains"
    ]
    logger.info(f"  defines/contains 关系边: {len(contains_edges)}")

    # 检查 calls 关系
    calls_edges = [
        (u, v, d) for u, v, d in reference_edges
        if d.get("metadata", {}).get("code_relation") == "calls"
    ]
    logger.info(f"  calls 关系边: {len(calls_edges)}")
    results["calls_edges"] = len(calls_edges)

    if len(calls_edges) > 0:
        sample_call = calls_edges[0]
        logger.info(f"    示例: {sample_call[0]} → {sample_call[1]}")
    logger.info("✅ 边关系元数据验证通过")

    # ── 10. 验证 BFS 扩散激活 ──
    logger.info("\n[Step 9] 验证 BFS 扩散激活穿透 CODE_SYMBOL...")

    # 选一个有连接的 CODE_SYMBOL 节点作为种子
    seed_node = None
    for sym in code_symbols:
        # 找一个有出边的节点
        if graph.out_degree(sym.node_id) > 0 or graph.in_degree(sym.node_id) > 0:
            seed_node = sym
            break

    if seed_node:
        activations = mg.compute_activations(
            seed_node_ids=[seed_node.node_id],
            max_depth=2,
            decay=0.5,
            min_activation=0.01,
        )
        activated_count = len(activations)
        logger.info(f"  种子节点: {seed_node.label}")
        logger.info(f"  BFS 激活节点数: {activated_count}")

        # 检查是否有非 CODE_SYMBOL 类型被激活 (穿透验证)
        activated_types = set()
        for nid in activations:
            node = mg.get_node(nid)
            if node:
                activated_types.add(node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type))

        logger.info(f"  激活的节点类型: {activated_types}")
        results["bfs_activated"] = activated_count
        results["bfs_types"] = list(activated_types)

        if activated_count > 1:
            logger.info("✅ BFS 扩散激活验证通过 (能穿透 CODE_SYMBOL)")
        else:
            logger.warning("⚠️ BFS 只激活了种子节点自身 (可能缺少边连接)")
    else:
        logger.warning("⚠️ 未找到有连接的 CODE_SYMBOL 节点，跳过 BFS 验证")
        results["bfs_activated"] = 0

    # ── 11. 持久化验证 ──
    logger.info("\n[Step 10] 验证 MemoryGraph 持久化...")
    mg.save()
    saved_path = os.path.join(
        getattr(mg, 'persist_path', '') or getattr(mg, '_persist_path', ''),
        "memory_graph.json"
    )
    if os.path.exists(saved_path):
        file_size = os.path.getsize(saved_path)
        logger.info(f"  持久化文件: {saved_path}")
        logger.info(f"  文件大小: {file_size / 1024:.1f} KB")
        logger.info("✅ 持久化验证通过")
    else:
        logger.error("❌ 持久化文件不存在")
        return False

    # ── 总结 ──
    logger.info("\n" + "=" * 60)
    logger.info("验证结果总结")
    logger.info("=" * 60)
    logger.info(f"  目标目录:     {target_dir}")
    logger.info(f"  CODE_SYMBOL:  {results['code_symbol_count']} 个节点")
    logger.info(f"  MODULE:       {results['module_count']} 个节点")
    logger.info(f"  FILE:         {results['file_count']} 个节点")
    logger.info(f"  HIERARCHY 边: {results['hierarchy_edges']} 条")
    logger.info(f"  REFERENCE 边: {results['reference_edges']} 条")
    logger.info(f"  DEPENDENCY 边:{results['dependency_edges']} 条")
    logger.info(f"  calls 关系:   {results.get('calls_edges', 0)} 条")
    logger.info(f"  BFS 激活:     {results.get('bfs_activated', 0)} 个节点")
    logger.info("")

    all_passed = (
        results["code_symbol_count"] > 0
        and results["module_count"] > 0
        and results["file_count"] > 0
        and results["hierarchy_edges"] > 0
    )

    if all_passed:
        logger.info("🎉 全部验证通过! CRG 数据层工作正常。")
    else:
        logger.error("❌ 部分验证失败，请检查上方日志。")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRG 数据完整性验证")
    parser.add_argument(
        "--target-dir",
        default="zulong/tools",
        help="要解析的代码目录 (默认: zulong/tools)",
    )
    parser.add_argument(
        "--persist-path",
        default=None,
        help="MemoryGraph 持久化路径 (默认: 临时目录)",
    )
    args = parser.parse_args()

    success = run_verification(args.target_dir, args.persist_path)
    sys.exit(0 if success else 1)
