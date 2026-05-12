"""代码智能 FC 工具

提供基于 MemoryGraph CODE_SYMBOL 节点的代码导航能力：
- search_code_symbols: 搜索代码符号（函数/类/方法）
- get_symbol_context: 获取符号上下文（调用者/被调用者/所属类）
- get_impact_analysis: 分析修改某个符号的影响范围
"""

import logging
import time
import threading
from typing import Any, Dict, List

from .base import BaseTool, ToolRequest, ToolResult, ToolCategory

logger = logging.getLogger(__name__)


class SearchCodeSymbolsTool(BaseTool):
    """搜索代码符号

    在 MemoryGraph 的 CODE_SYMBOL 节点中搜索匹配的函数/类/方法。
    """

    def __init__(self):
        super().__init__(
            name="search_code_symbols",
            category=ToolCategory.CUSTOM,
        )
        self.description = (
            "搜索代码库中的函数、类或方法定义。"
            "支持按名称关键词、符号类型和文件路径过滤。"
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "符号名称或关键词",
                },
                "kind": {
                    "type": "string",
                    "description": "过滤类型: function, class, method",
                    "enum": ["function", "class", "method"],
                },
                "file_path": {
                    "type": "string",
                    "description": "限制搜索范围到指定文件路径（子串匹配）",
                },
            },
            "required": ["query"],
        }

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.time()
        params = request.parameters
        query = params.get("query", "").strip()
        kind_filter = params.get("kind", "")
        file_filter = params.get("file_path", "")

        if not query:
            return self._create_result(
                success=False,
                error="缺少搜索关键词 (query)",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        try:
            from zulong.memory.memory_graph import (
                get_memory_graph, NodeType,
            )
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start,
                    request_id=request.request_id,
                )

            # 搜索匹配节点
            results = []
            query_lower = query.lower()
            for node_id, node in mg._nodes.items():
                if node.node_type != NodeType.CODE_SYMBOL:
                    continue
                meta = node.metadata or {}

                # kind 过滤
                if kind_filter and meta.get("kind") != kind_filter:
                    continue

                # file_path 过滤
                if file_filter and file_filter not in meta.get("file_path", ""):
                    continue

                # 名称匹配（标签 = qualified_name）
                label_lower = node.label.lower()
                if query_lower in label_lower or query_lower in node_id.lower():
                    results.append({
                        "name": node.label,
                        "kind": meta.get("kind", ""),
                        "file": meta.get("file_path", ""),
                        "lines": f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}",
                        "docstring": (meta.get("docstring", "") or "")[:100],
                        "parameters": meta.get("parameters", []),
                    })

            # 限制结果数
            results = results[:20]

            return self._create_result(
                success=True,
                data={
                    "query": query,
                    "count": len(results),
                    "symbols": results,
                },
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        except Exception as e:
            return self._create_result(
                success=False,
                error=f"搜索失败: {e}",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )


class GetSymbolContextTool(BaseTool):
    """获取符号上下文

    返回指定符号的详细信息和关系网络：
    调用者/被调用者、所属类/方法、继承关系。
    """

    def __init__(self):
        super().__init__(
            name="get_symbol_context",
            category=ToolCategory.CUSTOM,
        )
        self.description = (
            "获取代码符号的上下文关系，包括调用者、被调用者、所属类和继承关系。"
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "符号的 qualified_name，如 'MyClass.my_method'",
                },
                "file_path": {
                    "type": "string",
                    "description": "用于精确定位同名符号的文件路径（子串匹配）",
                },
            },
            "required": ["symbol_name"],
        }

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.time()
        params = request.parameters
        symbol_name = params.get("symbol_name", "").strip()
        file_filter = params.get("file_path", "")

        if not symbol_name:
            return self._create_result(
                success=False,
                error="缺少符号名称 (symbol_name)",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        try:
            from zulong.memory.memory_graph import (
                get_memory_graph, NodeType, EdgeType,
            )
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start,
                    request_id=request.request_id,
                )

            # 查找目标节点
            target_node = None
            target_id = None
            name_lower = symbol_name.lower()

            for nid, node in mg._nodes.items():
                if node.node_type != NodeType.CODE_SYMBOL:
                    continue
                if node.label.lower() == name_lower:
                    meta = node.metadata or {}
                    if file_filter and file_filter not in meta.get("file_path", ""):
                        continue
                    target_node = node
                    target_id = nid
                    break

            if target_node is None:
                return self._create_result(
                    success=False,
                    error=f"未找到符号: {symbol_name}",
                    execution_time=time.time() - start,
                    request_id=request.request_id,
                )

            meta = target_node.metadata or {}

            # 收集关系
            callers = []   # 谁调用了这个符号
            callees = []   # 这个符号调用了谁
            contains = []  # 这个符号包含的子元素
            parent = None  # 这个符号的父元素

            # 入边（指向 target 的边）
            for src_id, _, data in mg._graph.in_edges(target_id, data=True):
                edge_type = data.get("edge_type", "")
                code_rel = data.get("metadata", {}).get("code_relation", "")
                src_node = mg.get_node(src_id)
                if src_node is None:
                    continue

                if code_rel == "calls":
                    callers.append(src_node.label)
                elif code_rel == "contains" or (
                    edge_type == EdgeType.HIERARCHY.value
                    and src_node.node_type == NodeType.CODE_SYMBOL
                ):
                    parent = src_node.label

            # 出边（从 target 出发的边）
            for _, dst_id, data in mg._graph.out_edges(target_id, data=True):
                edge_type = data.get("edge_type", "")
                code_rel = data.get("metadata", {}).get("code_relation", "")
                dst_node = mg.get_node(dst_id)
                if dst_node is None:
                    continue

                if code_rel == "calls":
                    callees.append(dst_node.label)
                elif code_rel == "contains":
                    contains.append(dst_node.label)

            context = {
                "name": target_node.label,
                "kind": meta.get("kind", ""),
                "file": meta.get("file_path", ""),
                "lines": f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}",
                "docstring": meta.get("docstring", ""),
                "parameters": meta.get("parameters", []),
                "bases": meta.get("bases", []),
            }
            if callers:
                context["called_by"] = callers[:10]
            if callees:
                context["calls"] = callees[:10]
            if contains:
                context["contains"] = contains[:20]
            if parent:
                context["parent"] = parent

            return self._create_result(
                success=True,
                data=context,
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        except Exception as e:
            return self._create_result(
                success=False,
                error=f"获取上下文失败: {e}",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )


class GetImpactAnalysisTool(BaseTool):
    """影响范围分析

    给定一个代码符号，分析修改它可能影响到的所有符号（调用者链 + 继承链）。
    帮助 LLM 在重构或 bug 修复前了解变更波及面。
    """

    def __init__(self):
        super().__init__(
            name="get_impact_analysis",
            category=ToolCategory.CUSTOM,
        )
        self.description = (
            "分析修改某个代码符号的影响范围。"
            "返回所有直接和间接依赖该符号的调用者、子类等。"
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "要分析影响范围的符号 qualified_name",
                },
                "file_path": {
                    "type": "string",
                    "description": "用于精确定位同名符号的文件路径",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "最大追溯深度 (默认 3)",
                    "default": 3,
                },
            },
            "required": ["symbol_name"],
        }

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.time()
        params = request.parameters
        symbol_name = params.get("symbol_name", "").strip()
        file_filter = params.get("file_path", "")
        max_depth = min(int(params.get("max_depth", 3)), 5)

        if not symbol_name:
            return self._create_result(
                success=False,
                error="缺少符号名称 (symbol_name)",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        try:
            from zulong.memory.memory_graph import (
                get_memory_graph, NodeType,
            )
            mg = get_memory_graph()
            if mg is None:
                return self._create_result(
                    success=False,
                    error="MemoryGraph 未初始化",
                    execution_time=time.time() - start,
                    request_id=request.request_id,
                )

            # 查找目标节点
            target_id = None
            name_lower = symbol_name.lower()
            for nid, node in mg._nodes.items():
                if node.node_type != NodeType.CODE_SYMBOL:
                    continue
                if node.label.lower() == name_lower:
                    meta = node.metadata or {}
                    if file_filter and file_filter not in meta.get("file_path", ""):
                        continue
                    target_id = nid
                    break

            if target_id is None:
                return self._create_result(
                    success=False,
                    error=f"未找到符号: {symbol_name}",
                    execution_time=time.time() - start,
                    request_id=request.request_id,
                )

            # BFS 反向追溯: 找到所有直接/间接依赖此符号的节点
            visited = set()
            queue = [(target_id, 0)]
            impact_nodes: List[Dict] = []

            while queue:
                current_id, depth = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)

                current_node = mg.get_node(current_id)
                if current_node is None:
                    continue

                if current_id != target_id:
                    meta = current_node.metadata or {}
                    impact_nodes.append({
                        "name": current_node.label,
                        "kind": meta.get("kind", ""),
                        "file": meta.get("file_path", ""),
                        "lines": f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}",
                        "depth": depth,
                        "relation": "caller/dependent",
                    })

                if depth >= max_depth:
                    continue

                # 反向边: 谁调用/依赖了 current
                for src_id, _, data in mg._graph.in_edges(current_id, data=True):
                    code_rel = data.get("metadata", {}).get("code_relation", "")
                    if code_rel in ("calls", "inherits", "imports"):
                        if src_id not in visited:
                            queue.append((src_id, depth + 1))

            # 统计影响的文件
            affected_files = sorted(set(
                n["file"] for n in impact_nodes if n.get("file")
            ))

            return self._create_result(
                success=True,
                data={
                    "symbol": symbol_name,
                    "impact_count": len(impact_nodes),
                    "affected_files": affected_files,
                    "affected_symbols": impact_nodes[:30],
                },
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        except Exception as e:
            return self._create_result(
                success=False,
                error=f"影响分析失败: {e}",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )


class IndexCodeFileTool(BaseTool):
    """代码图谱索引工具

    对指定文件执行 Tree-sitter 解析，提取符号和关系，
    并增量同步到 MemoryGraph 的 CODE_SYMBOL 节点。
    支持单文件或目录批量索引。
    """

    def __init__(self):
        super().__init__(
            name="index_code_file",
            category=ToolCategory.CUSTOM,
        )
        self.description = (
            "对代码文件执行 AST 解析并构建代码图谱（符号、调用关系、继承等）。"
            "索引后可使用 search_code_symbols 和 get_symbol_context 查询。"
        )
        # 内容哈希缓存：避免未变更文件重复索引
        self._indexed_hashes: Dict[str, str] = {}

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要索引的代码文件路径（相对路径）",
                },
                "source_content": {
                    "type": "string",
                    "description": "文件源码内容（如已通过 read_file 获取，可直接传入避免重复读取）",
                },
                "force": {
                    "type": "boolean",
                    "description": "强制重新索引（忽略缓存），默认 false",
                    "default": False,
                },
            },
            "required": ["file_path"],
        }

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        import hashlib
        import os
        import json as _json
        start = time.time()
        params = request.parameters
        file_path = params.get("file_path", "").strip()
        source_content = params.get("source_content", "")
        force = params.get("force", False)

        if not file_path:
            return self._create_result(
                success=False,
                error="缺少 file_path 参数",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        # 归一化路径
        file_path = file_path.replace("\\", "/")

        # 检查扩展名
        ext = os.path.splitext(file_path)[1]
        try:
            from zulong.code.graph_builder import ext_to_lang
        except ImportError:
            return self._create_result(
                success=False,
                error="code graph_builder 模块不可用",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        lang = ext_to_lang(ext)
        if not lang:
            return self._create_result(
                success=False,
                error=f"不支持的文件类型: {ext}（支持: .py, .js, .ts, .tsx, .jsx）",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        # 如果没传 source_content，尝试从工作区读取
        if not source_content:
            try:
                from pathlib import Path
                from zulong.tools.task_tools import get_active_workspace_dir
                workspace = get_active_workspace_dir()
                candidates = [Path(file_path)]
                if workspace:
                    candidates.insert(0, Path(workspace) / file_path)
                candidates.append(Path(".") / file_path)
                for p in candidates:
                    if p.exists():
                        source_content = p.read_text(encoding="utf-8", errors="replace")
                        break
            except Exception:
                pass

        if not source_content:
            return self._create_result(
                success=False,
                error=f"无法获取文件内容: {file_path}（请通过 source_content 参数传入）",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        # 哈希去重
        content_hash = hashlib.md5(
            source_content.encode("utf-8", errors="replace")).hexdigest()
        if not force and self._indexed_hashes.get(file_path) == content_hash:
            return self._create_result(
                success=True,
                data={
                    "file_path": file_path,
                    "status": "skipped",
                    "reason": "内容未变更，无需重新索引",
                },
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        # 解析
        try:
            from zulong.code.ast_parser import ASTParser
            from zulong.code.graph_builder import CodeGraphBuilder, CodeEdge
            from zulong.memory.memory_graph import get_memory_graph, NodeType
        except ImportError as ie:
            return self._create_result(
                success=False,
                error=f"依赖模块不可用: {ie}",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        mg = get_memory_graph()
        if mg is None:
            return self._create_result(
                success=False,
                error="MemoryGraph 未初始化",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        parser = ASTParser(lang)
        if not parser.available:
            return self._create_result(
                success=False,
                error=f"Tree-sitter {lang} 解析器不可用（需安装 tree-sitter 和对应语言包）",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        source_bytes = source_content.encode("utf-8", errors="replace")
        result = parser.parse_source(source_bytes, file_path)
        if result.parse_error:
            return self._create_result(
                success=False,
                error=f"解析失败: {result.parse_error}",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        # 更新符号的 file_path
        for sym in result.symbols:
            sym.file_path = file_path

        # 构建文件内部边
        edges = CodeGraphBuilder._build_edges_for_file(result)

        # 跨文件边解析
        local_sym_names = {s.name for s in result.symbols}
        local_sym_names.update(s.qualified_name for s in result.symbols)
        local_node_ids = {s.node_id for s in result.symbols}

        global_sym_index = {}
        for nid, node in mg._nodes.items():
            if node.node_type == NodeType.CODE_SYMBOL:
                global_sym_index[node.label] = nid
                short = node.label.rsplit(".", 1)[-1]
                if short not in global_sym_index:
                    global_sym_index[short] = nid

        # import 边
        file_node_id = f"file:{file_path}"
        for imp in result.imports:
            if imp.is_from and imp.names:
                for name in imp.names:
                    target_id = global_sym_index.get(name)
                    if target_id and target_id not in local_node_ids:
                        edges.append(CodeEdge(
                            source_id=file_node_id,
                            target_id=target_id,
                            edge_type="imports",
                            metadata={"line": imp.line, "module": imp.module},
                        ))

        # 跨文件 call 边
        for call in result.calls:
            if call.callee in local_sym_names:
                continue
            target_id = global_sym_index.get(call.callee)
            if target_id:
                caller_id = None
                for s in result.symbols:
                    if s.qualified_name == call.caller:
                        caller_id = s.node_id
                        break
                if caller_id:
                    edges.append(CodeEdge(
                        source_id=caller_id,
                        target_id=target_id,
                        edge_type="calls",
                        metadata={"line": call.line, "cross_file": True},
                    ))

        # 增量同步
        adapter = mg._adapters.get("code_graph")
        if adapter is None:
            # 防御性回退：自动注册 CodeGraphAdapter
            try:
                from zulong.memory.graph_adapters import CodeGraphAdapter, register_all_adapters
                register_all_adapters(mg)
                adapter = mg._adapters.get("code_graph")
                logger.info("[IndexCodeFileTool] CodeGraphAdapter 自动补注册成功")
            except Exception as reg_err:
                return self._create_result(
                    success=False,
                    error=f"CodeGraphAdapter 注册失败: {reg_err}",
                    execution_time=time.time() - start,
                    request_id=request.request_id,
                )

        adapter.incremental_sync(mg, "file_updated", {
            "file_path": file_path,
            "symbols": result.symbols,
            "edges": edges,
        })

        # 记录哈希
        self._indexed_hashes[file_path] = content_hash

        cross_count = sum(
            1 for e in edges
            if e.metadata.get("cross_file") or e.edge_type == "imports"
        )

        return self._create_result(
            success=True,
            data={
                "file_path": file_path,
                "status": "indexed",
                "symbols_count": len(result.symbols),
                "edges_count": len(edges),
                "cross_file_edges": cross_count,
                "symbols": [
                    {"name": s.qualified_name, "kind": s.kind,
                     "lines": f"{s.start_line}-{s.end_line}"}
                    for s in result.symbols[:15]
                ],
            },
            execution_time=time.time() - start,
            request_id=request.request_id,
        )


class IndexProjectTool(BaseTool):
    """项目级代码图谱全量构建

    对指定目录执行递归扫描 + Tree-sitter 解析，
    构建完整的 PROJECT → MODULE → FILE → CLASS → METHOD 层次链，
    并将所有符号节点和关系边同步到 MemoryGraph。

    适用场景：LLM 首次接触新项目时，一键构建完整代码结构记忆。
    """

    def __init__(self):
        super().__init__(
            name="index_project",
            category=ToolCategory.CUSTOM,
        )
        self.description = (
            "对项目目录执行全量代码扫描和索引。"
            "构建完整的项目结构记忆（目录→文件→类→方法），"
            "扫描后可通过 search_code_symbols 搜索任何符号。"
            "首次理解新项目时调用此工具。"
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "root_dir": {
                    "type": "string",
                    "description": "项目根目录路径（相对或绝对路径）",
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要索引的语言列表，默认 ['python']。支持: python, javascript, typescript",
                    "default": ["python"],
                },
                "max_files": {
                    "type": "integer",
                    "description": "最大扫描文件数（防止超大项目卡死），默认 300",
                    "default": 300,
                },
                "summary_only": {
                    "type": "boolean",
                    "description": "仅扫描目录结构和文件列表，不做 AST 解析（适用于超大项目先览全貌）",
                    "default": False,
                },
            },
            "required": ["root_dir"],
        }

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        import os
        start = time.time()
        params = request.parameters
        root_dir = params.get("root_dir", "").strip()
        languages = params.get("languages", ["python"])
        max_files = params.get("max_files", 300)
        summary_only = params.get("summary_only", False)

        if not root_dir:
            return self._create_result(
                success=False,
                error="缺少 root_dir 参数",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        # 路径解析
        from pathlib import Path
        root_path = Path(root_dir)
        if not root_path.is_absolute():
            # 优先使用活跃工作目录（IDE 会话的 cwd）
            try:
                from zulong.tools.task_tools import get_active_workspace_dir
                workspace = get_active_workspace_dir()
                if workspace:
                    root_path = Path(workspace) / root_dir
                else:
                    root_path = Path(".") / root_dir
            except Exception:
                root_path = Path(".") / root_dir
        if not root_path.exists():
            return self._create_result(
                success=False,
                error=f"目录不存在: {root_dir}",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        try:
            from zulong.code.graph_builder import CodeGraphBuilder
            from zulong.memory.memory_graph import get_memory_graph
        except ImportError as ie:
            return self._create_result(
                success=False,
                error=f"依赖模块不可用: {ie}",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        # summary_only 模式：只扫描目录结构，不做 AST 解析
        if summary_only:
            return self._summary_only(root_path, languages, start, request)

        mg = get_memory_graph()
        if mg is None:
            return self._create_result(
                success=False,
                error="MemoryGraph 未初始化",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        # 全量构建 CodeGraph + 同步到 MemoryGraph 放入后台线程
        # 避免阻塞 FC 循环（大型项目可能耗时 60~90 秒）
        resolved_root = str(root_path.resolve())

        def _inject_symbols_to_tg(tg, code_graph, proj_name, file_rel_path, file_node_id):
            """将文件中的类/函数/方法符号注入到TaskGraph中"""
            _MAX_CLASSES_PER_FILE = 20
            _MAX_METHODS_PER_CLASS = 15
            _MAX_TOP_FUNCTIONS = 20

            file_syms = [
                sym for sym in code_graph.symbols.values()
                if sym.file_path == file_rel_path
            ]

            # 按类型分组
            classes = [s for s in file_syms if s.kind == "class"]
            top_funcs = [s for s in file_syms if s.kind == "function"]

            # 注入顶层函数
            for sym in top_funcs[:_MAX_TOP_FUNCTIONS]:
                sym_node_id = f"crg_{proj_name}/sym:{sym.node_id}"
                if tg.get_node(sym_node_id):
                    continue
                param_str = f"({', '.join(sym.parameters)})" if sym.parameters else "()"
                tg.add_node(
                    id=sym_node_id,
                    label=f"F {sym.name}{param_str}",
                    type="subtask",
                    status="completed",
                    desc=f"函数 {sym.qualified_name}{param_str}\n行 {sym.start_line}-{sym.end_line}",
                )
                tg.add_h_edge(file_node_id, sym_node_id)

            # 注入类及其方法
            for cls_sym in classes[:_MAX_CLASSES_PER_FILE]:
                cls_node_id = f"crg_{proj_name}/sym:{cls_sym.node_id}"
                if tg.get_node(cls_node_id):
                    continue
                bases_str = f" extends {', '.join(cls_sym.bases)}" if cls_sym.bases else ""
                tg.add_node(
                    id=cls_node_id,
                    label=f"C {cls_sym.name}{bases_str}",
                    type="subtask",
                    status="completed",
                    desc=f"类 {cls_sym.name}{bases_str}\n行 {cls_sym.start_line}-{cls_sym.end_line}",
                )
                tg.add_h_edge(file_node_id, cls_node_id)

                # 注入类方法
                methods = [s for s in file_syms
                           if s.kind == "method" and s.parent_class == cls_sym.name]
                for meth_sym in methods[:_MAX_METHODS_PER_CLASS]:
                    meth_node_id = f"crg_{proj_name}/sym:{meth_sym.node_id}"
                    if tg.get_node(meth_node_id):
                        continue
                    param_str = f"({', '.join(meth_sym.parameters)})" if meth_sym.parameters else "()"
                    tg.add_node(
                        id=meth_node_id,
                        label=f"M {meth_sym.name}{param_str}",
                        type="subtask",
                        status="completed",
                        desc=f"方法 {meth_sym.qualified_name}{param_str}\n行 {meth_sym.start_line}-{meth_sym.end_line}",
                    )
                    tg.add_h_edge(cls_node_id, meth_node_id)

        def _background_index():
            """后台执行 AST 解析 + MemoryGraph 同步 + 注入任务图"""
            try:
                from zulong.code.graph_builder import CodeGraphBuilder
                from zulong.code.ast_parser import ASTParser
                
                # 🔥 修复：检查语言包可用性，过滤掉不支持的语言
                available_languages = []
                for lang in languages:
                    parser = ASTParser(lang)
                    if parser.available:
                        available_languages.append(lang)
                    else:
                        logger.warning(
                            f"[IndexProjectTool] 跳过语言 {lang}：tree-sitter 语言包未安装"
                        )
                
                if not available_languages:
                    logger.error("[IndexProjectTool] 没有可用的语言包，索引终止")
                    return
                
                if len(available_languages) < len(languages):
                    logger.info(
                        f"[IndexProjectTool] 语言包检查："
                        f"请求{len(languages)}种语言，可用{len(available_languages)}种 "
                        f"({', '.join(available_languages)})"
                    )
                
                bg_builder = CodeGraphBuilder()
                code_graph = bg_builder.build(
                    resolved_root,
                    languages=available_languages,  # 🔥 使用过滤后的语言列表
                    max_files=max_files,
                )

                # 同步到 MemoryGraph（后端存储）
                adapter = mg._adapters.get("code_graph")
                if adapter is None:
                    try:
                        from zulong.memory.graph_adapters import register_all_adapters
                        register_all_adapters(mg)
                        adapter = mg._adapters.get("code_graph")
                        logger.info("[IndexProjectTool] CodeGraphAdapter 自动补注册成功(bg)")
                    except Exception as reg_err:
                        logger.error(f"[IndexProjectTool] 后台索引注册失败: {reg_err}")
                        return

                if adapter is None:
                    logger.error("[IndexProjectTool] 后台索引: adapter 为空")
                    return

                sym_count = adapter.sync(mg, code_graph)

                # ── 直接从 CodeGraph 数据注入任务图（不依赖 on_batch）──
                try:
                    from zulong.tools.task_tools import get_active_task_graph
                    tg = get_active_task_graph()
                    if tg and tg.get_node("req"):
                        proj_name = code_graph.project_name or "project"
                        struct_id = f"crg_{proj_name}"

                        # 1) 创建项目结构根节点
                        if not tg.get_node(struct_id):
                            tg.add_node(
                                id=struct_id,
                                label=f"{proj_name} 项目结构",
                                type="analysis",
                                status="completed",
                                desc=f"{len(code_graph.directories)} 模块, {len(code_graph.file_results)} 文件, {sym_count} 符号",
                            )
                            tg.add_h_edge("req", struct_id)

                        # 2) 构建两级目录树：顶层目录 + 子模块
                        directories = code_graph.directories
                        # 顶层目录: 路径中无 /
                        top_dirs = sorted(
                            [(d, files) for d, files in directories.items() if d and "/" not in d],
                            key=lambda x: -len(x[1])
                        )
                        created_dirs = 0
                        for dir_name, dir_files in top_dirs[:20]:
                            dir_node_id = f"crg_{proj_name}/{dir_name}"
                            if not tg.get_node(dir_node_id):
                                # 统计该顶层目录下所有符号（含子目录）
                                dir_sym_count = sum(
                                    1 for sym in code_graph.symbols.values()
                                    if sym.file_path.startswith(dir_name + "/")
                                )
                                # 生成模块摘要描述：列出主要类和函数
                                dir_classes = [
                                    sym.name for sym in code_graph.symbols.values()
                                    if sym.file_path.startswith(dir_name + "/") and sym.kind == "class"
                                ][:8]
                                dir_funcs = [
                                    sym.name for sym in code_graph.symbols.values()
                                    if sym.file_path.startswith(dir_name + "/") and sym.kind == "function"
                                ][:8]
                                desc_parts = [f"模块 {dir_name} ({len(dir_files)} 文件, {dir_sym_count} 符号)"]
                                if dir_classes:
                                    desc_parts.append(f"主要类: {', '.join(dir_classes)}")
                                if dir_funcs:
                                    desc_parts.append(f"主要函数: {', '.join(dir_funcs)}")
                                tg.add_node(
                                    id=dir_node_id,
                                    label=f"{dir_name}/ ({len(dir_files)} files, {dir_sym_count} symbols)",
                                    type="subtask",
                                    status="completed",
                                    desc="\n".join(desc_parts),
                                )
                                tg.add_h_edge(struct_id, dir_node_id)
                                created_dirs += 1

                            # 3a) 查找该顶层目录的子目录（深度=2: "zulong/l2", "zulong/ide" 等）
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
                                    # 生成子模块摘要描述
                                    sub_classes = [
                                        sym.name for sym in code_graph.symbols.values()
                                        if sym.file_path.startswith(sub_dir + "/") and sym.kind == "class"
                                    ][:6]
                                    sub_funcs = [
                                        sym.name for sym in code_graph.symbols.values()
                                        if sym.file_path.startswith(sub_dir + "/") and sym.kind == "function"
                                    ][:6]
                                    sub_desc_parts = [f"子模块 {sub_dir} ({len(sub_files)} 文件, {sub_sym_count} 符号)"]
                                    if sub_classes:
                                        sub_desc_parts.append(f"类: {', '.join(sub_classes)}")
                                    if sub_funcs:
                                        sub_desc_parts.append(f"函数: {', '.join(sub_funcs)}")
                                    tg.add_node(
                                        id=sub_dir_id,
                                        label=f"{sub_name}/ ({len(sub_files)} files, {sub_sym_count} symbols)",
                                        type="subtask",
                                        status="completed",
                                        desc="\n".join(sub_desc_parts),
                                    )
                                    tg.add_h_edge(dir_node_id, sub_dir_id)
                                    created_dirs += 1

                                # 3b) 子目录下的文件节点
                                for file_rel_path in sub_files[:8]:
                                    base_name = file_rel_path.split("/")[-1]
                                    file_node_id = f"crg_{proj_name}/{file_rel_path}"
                                    if not tg.get_node(file_node_id):
                                        file_syms = [
                                            sym for sym in code_graph.symbols.values()
                                            if sym.file_path == file_rel_path
                                        ]
                                        file_sym_count = len(file_syms)
                                        sym_lines = []
                                        for sym in file_syms[:10]:
                                            kind_icon = {"class": "C", "function": "F", "method": "M"}.get(sym.kind, "?")
                                            sym_lines.append(f"[{kind_icon}] {sym.name} (L{sym.start_line}-{sym.end_line})")
                                        desc = f"文件 {file_rel_path}\n" + "\n".join(sym_lines) if sym_lines else f"文件 {file_rel_path}"
                                        tg.add_node(
                                            id=file_node_id,
                                            label=f"{base_name} ({file_sym_count} symbols)",
                                            type="subtask",
                                            status="completed",
                                            desc=desc,
                                        )
                                        tg.add_h_edge(sub_dir_id, file_node_id)
                                        tg.add_file_to_node(file_node_id, base_name, file_rel_path)
                                    # 3b-sym) 文件下的类/函数符号节点
                                    _inject_symbols_to_tg(tg, code_graph, proj_name, file_rel_path, file_node_id)

                            # 3c) 顶层目录下直接的文件（不在子目录中的）
                            for file_rel_path in dir_files[:5]:
                                base_name = file_rel_path.split("/")[-1]
                                file_node_id = f"crg_{proj_name}/{file_rel_path}"
                                if not tg.get_node(file_node_id):
                                    file_syms = [
                                        sym for sym in code_graph.symbols.values()
                                        if sym.file_path == file_rel_path
                                    ]
                                    file_sym_count = len(file_syms)
                                    sym_lines = []
                                    for sym in file_syms[:10]:
                                        kind_icon = {"class": "C", "function": "F", "method": "M"}.get(sym.kind, "?")
                                        sym_lines.append(f"[{kind_icon}] {sym.name} (L{sym.start_line}-{sym.end_line})")
                                    desc = f"文件 {file_rel_path}\n" + "\n".join(sym_lines) if sym_lines else f"文件 {file_rel_path}"
                                    tg.add_node(
                                        id=file_node_id,
                                        label=f"{base_name} ({file_sym_count} symbols)",
                                        type="subtask",
                                        status="completed",
                                        desc=desc,
                                    )
                                    tg.add_h_edge(dir_node_id, file_node_id)
                                    tg.add_file_to_node(file_node_id, base_name, file_rel_path)
                                # 3c-sym) 文件下的类/函数符号节点
                                _inject_symbols_to_tg(tg, code_graph, proj_name, file_rel_path, file_node_id)

                        # 4) 注入跨文件依赖边（calls/inherits/imports → d_edge）
                        dep_count = 0
                        seen_dep_pairs = set()
                        for edge in code_graph.edges:
                            if edge.edge_type == "contains":
                                continue  # contains 边是同文件层级关系，跳过

                            if edge.edge_type == "imports":
                                # imports 边的 source_id/target_id 是 "file:xxx" 格式
                                # 需要去掉 "file:" 前缀匹配 TaskGraph 中的 crg_ 节点
                                src_path = edge.source_id.replace("file:", "", 1) if edge.source_id.startswith("file:") else None
                                tgt_path = edge.target_id.replace("file:", "", 1) if edge.target_id.startswith("file:") else None
                                # target 也可能是符号 node_id（from X import Y 的精确边）
                                if not src_path:
                                    continue
                                if not tgt_path:
                                    tgt_sym = code_graph.symbols.get(edge.target_id)
                                    if tgt_sym:
                                        tgt_path = tgt_sym.file_path
                                    else:
                                        continue
                                if src_path == tgt_path:
                                    continue
                                src_file_node = f"crg_{proj_name}/{src_path}"
                                tgt_file_node = f"crg_{proj_name}/{tgt_path}"
                                pair_key = (src_file_node, tgt_file_node)
                                if pair_key in seen_dep_pairs:
                                    continue
                                if tg.get_node(src_file_node) and tg.get_node(tgt_file_node):
                                    seen_dep_pairs.add(pair_key)
                                    via_label = edge.metadata.get("symbol", "") or edge.metadata.get("module", "import")
                                    tg.add_d_edge(src_file_node, tgt_file_node, via=f"imports {via_label}", cross=True)
                                    dep_count += 1
                            else:
                                # calls/inherits 边：优先在符号节点间建立依赖边
                                src_sym = code_graph.symbols.get(edge.source_id)
                                tgt_sym = code_graph.symbols.get(edge.target_id)
                                if not src_sym or not tgt_sym:
                                    continue
                                if src_sym.file_path == tgt_sym.file_path:
                                    continue
                                # 尝试符号级依赖边
                                src_sym_node = f"crg_{proj_name}/sym:{edge.source_id}"
                                tgt_sym_node = f"crg_{proj_name}/sym:{edge.target_id}"
                                if tg.get_node(src_sym_node) and tg.get_node(tgt_sym_node):
                                    sym_pair = (src_sym_node, tgt_sym_node)
                                    if sym_pair not in seen_dep_pairs:
                                        seen_dep_pairs.add(sym_pair)
                                        via_label = f"{src_sym.name}→{tgt_sym.name} ({edge.edge_type})"
                                        tg.add_d_edge(src_sym_node, tgt_sym_node, via=via_label, cross=True)
                                        dep_count += 1
                                # 同时在文件级也建立依赖边（兜底）
                                src_file_node = f"crg_{proj_name}/{src_sym.file_path}"
                                tgt_file_node = f"crg_{proj_name}/{tgt_sym.file_path}"
                                pair_key = (src_file_node, tgt_file_node)
                                if pair_key not in seen_dep_pairs:
                                    if tg.get_node(src_file_node) and tg.get_node(tgt_file_node):
                                        seen_dep_pairs.add(pair_key)
                                        via_label = f"{src_sym.name}→{tgt_sym.name} ({edge.edge_type})"
                                        tg.add_d_edge(src_file_node, tgt_file_node, via=via_label, cross=True)
                                        dep_count += 1

                        logger.info(f"[IndexProjectTool] 已注入任务图: {created_dirs} 模块, {len(code_graph.file_results)} 文件, {len(code_graph.symbols)} 符号, {dep_count} 依赖边")

                        # 推送完整任务图更新到前端
                        if tg.on_change_callback:
                            tg.on_change_callback("crg_structure_complete", {
                                "project_name": proj_name,
                                "module_count": created_dirs,
                                "file_count": len(code_graph.file_results),
                                "symbol_count": sym_count,
                            })
                except Exception as tg_err:
                    logger.warning(f"[IndexProjectTool] 注入任务图失败(非致命): {tg_err}")

                # 广播 CRG 全量索引完成事件
                crg_payload = {
                    "project_name": code_graph.project_name,
                    "root_dir": code_graph.root_dir,
                    "symbol_count": sym_count,
                    "edge_count": code_graph.edge_count,
                    "file_count": len(code_graph.file_results),
                    "dir_count": len(code_graph.directories),
                }
                try:
                    from zulong.ide.ide_server import _broadcast_sync
                    _broadcast_sync("CRG_INDEX_COMPLETE", crg_payload)
                except Exception:
                    pass
                try:
                    from zulong.launcher.web_chat_router import _schedule_broadcast
                    _schedule_broadcast({
                        "type": "CRG_INDEX_COMPLETE",
                        "payload": crg_payload,
                    })
                except Exception:
                    pass
                logger.info(
                    f"[IndexProjectTool] 后台索引完成: {sym_count} symbols, "
                    f"{code_graph.edge_count} edges, {len(code_graph.file_results)} files"
                )
            except Exception as bg_err:
                logger.error(f"[IndexProjectTool] 后台索引异常: {bg_err}")
                # 🔥 修复：广播失败事件到前端
                try:
                    from zulong.ide.ide_server import _broadcast_sync
                    _broadcast_sync("CRG_INDEX_FAILED", {
                        "error": str(bg_err),
                        "project_name": root_path.name,
                    })
                except Exception:
                    pass
                try:
                    from zulong.launcher.web_chat_router import _schedule_broadcast
                    _schedule_broadcast({
                        "type": "CRG_INDEX_FAILED",
                        "payload": {
                            "error": str(bg_err),
                            "project_name": root_path.name,
                        },
                    })
                except Exception:
                    pass

        # 启动后台线程
        t = threading.Thread(target=_background_index, daemon=True, name="CRG-IndexProject")
        t.start()

        return self._create_result(
            success=True,
            data={
                "project_name": root_path.name,
                "root_dir": resolved_root,
                "status": "indexing_started",
                "message": (
                    f"后台索引已启动（languages={languages}, max_files={max_files}）。"
                    "索引完成后将自动同步到 MemoryGraph 并广播通知。"
                    "你可以继续进行任务规划，无需等待索引完成。"
                ),
            },
            execution_time=time.time() - start,
            request_id=request.request_id,
        )

    def _summary_only(self, root_path, languages, start, request):
        """仅扫描目录结构，不做 AST 解析。返回项目结构概览。"""
        import os
        from zulong.code.graph_builder import _LANG_EXTENSIONS, _IGNORE_DIRS

        extensions = set()
        for lang in languages:
            extensions.update(_LANG_EXTENSIONS.get(lang, set()))

        dir_tree = {}  # dir_path → file list
        total_files = 0
        for dirpath, dirnames, filenames in os.walk(str(root_path.resolve())):
            dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
            rel_dir = os.path.relpath(dirpath, str(root_path.resolve())).replace("\\", "/")
            if rel_dir == ".":
                rel_dir = ""
            code_files = [f for f in filenames if os.path.splitext(f)[1] in extensions]
            if code_files:
                dir_tree[rel_dir] = code_files
                total_files += len(code_files)

        # 构建树状概览
        top_dirs = sorted(dir_tree.items(), key=lambda x: -len(x[1]))[:20]

        return self._create_result(
            success=True,
            data={
                "project_name": root_path.name,
                "root_dir": str(root_path.resolve()),
                "status": "summary",
                "mode": "summary_only",
                "stats": {
                    "directories": len(dir_tree),
                    "files": total_files,
                    "languages": languages,
                },
                "directory_tree": [
                    {"path": d or "(root)", "files": files[:10], "file_count": len(files)}
                    for d, files in top_dirs
                ],
                "hint": "使用 index_project(summary_only=false) 全量索引，或用 analyze_module 逐模块深入",
            },
            execution_time=time.time() - start,
            request_id=request.request_id,
        )


class AnalyzeModuleTool(BaseTool):
    """模块级结构分析

    对指定目录（模块）执行 AST 解析并返回结构概览，
    默认不同步到 MemoryGraph（轻量级分析，无副作用）。

    适用于渐进式理解大项目：先 summary_only 看全貌，
    再对感兴趣的模块逐个 analyze_module 深入。
    """

    def __init__(self):
        super().__init__(
            name="analyze_module",
            category=ToolCategory.CUSTOM,
        )
        self.description = (
            "对指定目录/模块执行结构分析，返回文件列表、"
            "顶层类/函数、依赖关系（默认不写入 MemoryGraph）。"
            "用于渐进式理解大项目的某个子模块。"
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "module_dir": {
                    "type": "string",
                    "description": "目标模块/目录路径（相对或绝对）",
                },
                "language": {
                    "type": "string",
                    "description": "语言，默认 python",
                    "default": "python",
                },
                "max_files": {
                    "type": "integer",
                    "description": "最大分析文件数，默认 30",
                    "default": 30,
                },
                "index_to_memory": {
                    "type": "boolean",
                    "description": "是否同步到 MemoryGraph（默认 false，仅返回分析结果）",
                    "default": False,
                },
            },
            "required": ["module_dir"],
        }

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        import os
        start = time.time()
        params = request.parameters
        module_dir = params.get("module_dir", "").strip()
        language = params.get("language", "python")
        max_files = params.get("max_files", 30)
        index_to_memory = params.get("index_to_memory", False)

        if not module_dir:
            return self._create_result(
                success=False,
                error="缺少 module_dir 参数",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        from pathlib import Path
        mod_path = Path(module_dir)
        if not mod_path.is_absolute():
            try:
                from zulong.tools.task_tools import get_active_workspace_dir
                workspace = get_active_workspace_dir()
                if workspace:
                    mod_path = Path(workspace) / module_dir
                else:
                    mod_path = Path(".") / module_dir
            except Exception:
                mod_path = Path(".") / module_dir

        if not mod_path.exists():
            return self._create_result(
                success=False,
                error=f"目录不存在: {module_dir}",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        try:
            from zulong.code.graph_builder import CodeGraphBuilder
        except ImportError as ie:
            return self._create_result(
                success=False,
                error=f"依赖模块不可用: {ie}",
                execution_time=time.time() - start,
                request_id=request.request_id,
            )

        # 构建 CodeGraph
        builder = CodeGraphBuilder()
        code_graph = builder.build(
            str(mod_path.resolve()),
            languages=[language],
            max_files=max_files,
        )

        # 提取结构概览
        classes = []
        functions = []
        for sym in code_graph.symbols.values():
            entry = {
                "name": sym.qualified_name,
                "file": sym.file_path,
                "lines": f"{sym.start_line}-{sym.end_line}",
            }
            if sym.kind == "class":
                entry["bases"] = sym.bases
                entry["docstring"] = (sym.docstring or "")[:100]
                classes.append(entry)
            elif sym.kind == "function":
                entry["params"] = sym.parameters
                entry["docstring"] = (sym.docstring or "")[:80]
                functions.append(entry)

        # 提取 imports 概览
        all_imports = []
        for fp, result in code_graph.file_results.items():
            for imp in result.imports:
                all_imports.append({
                    "file": fp,
                    "module": imp.module,
                    "names": imp.names[:5],
                })

        # 边统计
        from collections import Counter
        edge_types = Counter(e.edge_type for e in code_graph.edges)

        # 可选：同步到 MemoryGraph
        synced = 0
        if index_to_memory:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg:
                adapter = mg._adapters.get("code_graph")
                if adapter is None:
                    try:
                        from zulong.memory.graph_adapters import register_all_adapters
                        register_all_adapters(mg)
                        adapter = mg._adapters.get("code_graph")
                    except Exception:
                        pass
                if adapter:
                    synced = adapter.sync(mg, code_graph)

        return self._create_result(
            success=True,
            data={
                "module_name": mod_path.name,
                "module_path": str(mod_path.resolve()),
                "stats": {
                    "files": len(code_graph.file_results),
                    "symbols": code_graph.symbol_count,
                    "edges": code_graph.edge_count,
                    "edge_types": dict(edge_types),
                },
                "classes": classes[:20],
                "top_functions": functions[:30],
                "imports": all_imports[:20],
                "indexed_to_memory": synced if index_to_memory else None,
            },
            execution_time=time.time() - start,
            request_id=request.request_id,
        )
