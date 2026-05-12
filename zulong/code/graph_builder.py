"""代码图谱构建器

扫描代码目录 → Tree-sitter 解析 → 提取符号和关系 → 生成可投射到 MemoryGraph 的结构。

用法:
    builder = CodeGraphBuilder()
    code_graph = builder.build("zulong/", languages=["python"])
    # code_graph.symbols, code_graph.edges 可直接由 CodeGraphAdapter 消费
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .ast_parser import (
    ASTParser, CodeSymbol, ImportInfo, CallInfo, FileParseResult,
)

logger = logging.getLogger(__name__)

# 默认忽略的目录
_IGNORE_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".egg-info",
}

# 语言 → 文件扩展名
_LANG_EXTENSIONS = {
    "python": {".py"},
    "javascript": {".js", ".jsx"},
    "typescript": {".ts"},
    "tsx": {".tsx"},
}


@dataclass
class CodeEdge:
    """代码关系边"""
    source_id: str           # 源符号 node_id
    target_id: str           # 目标符号 node_id
    edge_type: str           # "contains", "calls", "inherits", "imports"
    metadata: Dict = field(default_factory=dict)


@dataclass
class CodeGraph:
    """代码图谱（AST 解析中间结果）

    由 CodeGraphBuilder.build() 产生，
    供 CodeGraphAdapter.sync() 消费。
    """
    root_dir: str
    symbols: Dict[str, CodeSymbol] = field(default_factory=dict)  # node_id → symbol
    edges: List[CodeEdge] = field(default_factory=list)
    file_results: Dict[str, FileParseResult] = field(default_factory=dict)
    # 目录层次：dir_path → 其下的文件相对路径列表
    directories: Dict[str, List[str]] = field(default_factory=dict)
    # 项目名
    project_name: str = ""

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


class CodeGraphBuilder:
    """代码图谱构建器"""

    def __init__(self):
        self._parsers: Dict[str, ASTParser] = {}

    def _get_parser(self, lang: str) -> ASTParser:
        if lang not in self._parsers:
            self._parsers[lang] = ASTParser(lang)
        return self._parsers[lang]

    def build(
        self,
        root_dir: str,
        languages: Optional[List[str]] = None,
        max_files: int = 500,
    ) -> CodeGraph:
        """扫描目录并构建代码图谱。

        Args:
            root_dir: 根目录路径
            languages: 要解析的语言列表，默认 ["python"]
            max_files: 最大文件数限制
        """
        if languages is None:
            languages = ["python"]

        root_path = Path(root_dir).resolve()
        graph = CodeGraph(root_dir=str(root_path))
        graph.project_name = root_path.name

        # 收集文件
        files = self._collect_files(root_path, languages, max_files)
        logger.info(
            f"[CodeGraph] 扫描到 {len(files)} 个文件 ({', '.join(languages)})"
        )

        # 解析每个文件
        for lang, file_path in files:
            parser = self._get_parser(lang)
            if not parser.available:
                continue
            result = parser.parse_file(str(file_path))
            if result.parse_error:
                logger.debug(
                    f"[CodeGraph] 解析跳过 {file_path}: {result.parse_error}"
                )
                continue
            # 使用相对路径作为文件标识
            try:
                rel_path = str(file_path.relative_to(root_path))
            except ValueError:
                rel_path = str(file_path)
            # 统一路径分隔符
            rel_path = rel_path.replace("\\", "/")

            # 更新符号的 file_path 为相对路径
            for sym in result.symbols:
                sym.file_path = rel_path
                graph.symbols[sym.node_id] = sym

            graph.file_results[rel_path] = result

            # 构建目录层次索引
            dir_path = "/".join(rel_path.split("/")[:-1]) if "/" in rel_path else ""
            if dir_path not in graph.directories:
                graph.directories[dir_path] = []
            graph.directories[dir_path].append(rel_path)

        # 确保所有父级目录都被记录（包含中间层）
        all_dirs = set(graph.directories.keys())
        for d in list(all_dirs):
            parts = d.split("/") if d else []
            for i in range(len(parts)):
                parent = "/".join(parts[:i])
                if parent not in graph.directories:
                    graph.directories[parent] = []

        # 构建边
        self._build_containment_edges(graph)
        self._build_call_edges(graph)
        self._build_inheritance_edges(graph)
        self._build_import_edges(graph)

        logger.info(
            f"[CodeGraph] 构建完成: "
            f"{graph.symbol_count} 符号, {graph.edge_count} 边, "
            f"{len(graph.directories)} 目录"
        )
        return graph

    def _collect_files(
        self,
        root: Path,
        languages: List[str],
        max_files: int,
    ) -> List[Tuple[str, Path]]:
        """收集指定语言的文件"""
        extensions: Dict[str, str] = {}
        for lang in languages:
            for ext in _LANG_EXTENSIONS.get(lang, set()):
                extensions[ext] = lang

        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            # 过滤忽略目录
            dirnames[:] = [
                d for d in dirnames if d not in _IGNORE_DIRS
            ]
            for fname in filenames:
                if len(files) >= max_files:
                    return files
                ext = os.path.splitext(fname)[1]
                if ext in extensions:
                    files.append(
                        (extensions[ext], Path(dirpath) / fname)
                    )
        return files

    def _build_containment_edges(self, graph: CodeGraph):
        """构建 class → method 的包含关系边"""
        for sym in graph.symbols.values():
            if sym.kind == "method" and sym.parent_class:
                # 查找所属类
                class_id = f"code:class:{sym.file_path}:{sym.parent_class}"
                if class_id in graph.symbols:
                    graph.edges.append(CodeEdge(
                        source_id=class_id,
                        target_id=sym.node_id,
                        edge_type="contains",
                    ))

    def _build_call_edges(self, graph: CodeGraph):
        """构建函数调用关系边"""
        # 建立 name → node_id 的索引（最后定义的优先）
        name_index: Dict[str, str] = {}
        for node_id, sym in graph.symbols.items():
            name_index[sym.name] = node_id
            name_index[sym.qualified_name] = node_id

        for rel_path, result in graph.file_results.items():
            for call in result.calls:
                # 查找 caller（使用 rel_path 匹配符号的 file_path）
                caller_candidates = [
                    nid for nid, s in graph.symbols.items()
                    if s.qualified_name == call.caller
                    and s.file_path == rel_path
                ]
                if not caller_candidates:
                    continue
                caller_id = caller_candidates[0]

                # 查找 callee
                callee_id = name_index.get(call.callee)
                if callee_id and callee_id != caller_id:
                    graph.edges.append(CodeEdge(
                        source_id=caller_id,
                        target_id=callee_id,
                        edge_type="calls",
                        metadata={"line": call.line},
                    ))

    def _build_inheritance_edges(self, graph: CodeGraph):
        """构建类继承关系边"""
        # 建立类名索引
        class_index: Dict[str, str] = {}
        for node_id, sym in graph.symbols.items():
            if sym.kind == "class":
                class_index[sym.name] = node_id

        for node_id, sym in graph.symbols.items():
            if sym.kind == "class" and sym.bases:
                for base in sym.bases:
                    # 去除模块前缀
                    base_name = base.split(".")[-1]
                    base_id = class_index.get(base_name)
                    if base_id:
                        graph.edges.append(CodeEdge(
                            source_id=node_id,
                            target_id=base_id,
                            edge_type="inherits",
                        ))

    def _build_import_edges(self, graph: CodeGraph):
        """构建跨文件 import 依赖边

        将 FileParseResult 中的 ImportInfo 转化为 file→file 层级的 DEPENDENCY 边。
        逻辑：
          1. 将 import 的模块路径（如 "zulong.l2.task_graph"）转化为文件路径
          2. 如果目标文件在图中存在，创建从源文件 FILE 节点到目标文件 FILE 节点的边
          3. 如果能精确定位到符号级别（from X import Y），创建符号→符号的边
        """
        # 建立模块路径 → 文件相对路径的映射
        # 同时记录带项目名前缀的路径（处理 "zulong.ide.xxx" vs "ide.xxx" 差异）
        module_to_file: Dict[str, str] = {}
        project_name = graph.project_name  # e.g., "zulong"
        for rel_path in graph.file_results:
            # "ide/ide_session.py" → "ide.ide_session"
            mod_path = rel_path.replace("/", ".").replace("\\", ".")
            if mod_path.endswith(".py"):
                mod_path = mod_path[:-3]
            module_to_file[mod_path] = rel_path
            # 也记录包路径（去掉 __init__）
            if mod_path.endswith(".__init__"):
                pkg_path = mod_path[:-9]
                module_to_file[pkg_path] = rel_path
            # 带项目名前缀的版本: "zulong.ide.ide_session" → "ide/ide_session.py"
            if project_name:
                full_mod = f"{project_name}.{mod_path}"
                module_to_file[full_mod] = rel_path
                if mod_path.endswith(".__init__"):
                    module_to_file[f"{project_name}.{mod_path[:-9]}"] = rel_path
            module_to_file[mod_path] = rel_path
            # 也记录包路径（去掉 __init__）
            if mod_path.endswith(".__init__"):
                pkg_path = mod_path[:-9]  # 去掉 ".__init__"
                module_to_file[pkg_path] = rel_path

        # 建立符号名 → node_id 的索引
        name_to_id: Dict[str, str] = {}
        for nid, sym in graph.symbols.items():
            name_to_id[sym.name] = nid
            name_to_id[sym.qualified_name] = nid

        seen_edges: Set[Tuple[str, str]] = set()

        for src_file, result in graph.file_results.items():
            for imp in result.imports:
                # 解析目标模块路径
                target_module = imp.module
                target_file = module_to_file.get(target_module)

                # 如果精确模块匹配失败，尝试前缀匹配
                if not target_file:
                    for mod_key, fpath in module_to_file.items():
                        if mod_key.startswith(target_module + ".") or target_module.startswith(mod_key + "."):
                            target_file = fpath
                            break

                if not target_file or target_file == src_file:
                    continue  # 跳过自引用或无法解析的外部包

                # 创建 FILE→FILE 级别的 import 边
                src_file_id = f"file:{src_file}"
                tgt_file_id = f"file:{target_file}"
                edge_key = (src_file_id, tgt_file_id)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    graph.edges.append(CodeEdge(
                        source_id=src_file_id,
                        target_id=tgt_file_id,
                        edge_type="imports",
                        metadata={"module": imp.module, "line": imp.line},
                    ))

                # 如果是 from X import Y，尝试创建符号→符号级别的精确依赖
                if imp.is_from and imp.names:
                    for sym_name in imp.names:
                        target_sym_id = name_to_id.get(sym_name)
                        if target_sym_id:
                            # 找源文件中使用该 import 的符号（简化：连接到该文件的第一个符号）
                            src_symbols = [
                                s for s in graph.symbols.values()
                                if s.file_path == src_file
                            ]
                            if src_symbols:
                                # 用文件节点作为源（更准确的语义：文件依赖符号）
                                sym_edge_key = (src_file_id, target_sym_id)
                                if sym_edge_key not in seen_edges:
                                    seen_edges.add(sym_edge_key)
                                    graph.edges.append(CodeEdge(
                                        source_id=src_file_id,
                                        target_id=target_sym_id,
                                        edge_type="imports",
                                        metadata={
                                            "module": imp.module,
                                            "symbol": sym_name,
                                            "line": imp.line,
                                        },
                                    ))

    @staticmethod
    def _build_edges_for_file(result: FileParseResult) -> List[CodeEdge]:
        """从单个 FileParseResult 构建文件内的关系边

        用于事件驱动懒索引：文件被读取时解析单文件并构建边。
        仅处理文件内部可确定的边 (containment, intra-file calls, inheritance)。
        跨文件的 call 边需要目标节点已在 MemoryGraph 中存在。
        """
        edges: List[CodeEdge] = []
        symbols = result.symbols

        # 建立本文件的符号索引
        sym_by_name: Dict[str, str] = {}  # name/qualified_name → node_id
        sym_by_qn: Dict[str, CodeSymbol] = {}
        for sym in symbols:
            sym_by_name[sym.name] = sym.node_id
            sym_by_name[sym.qualified_name] = sym.node_id
            sym_by_qn[sym.qualified_name] = sym

        # containment: class → method
        for sym in symbols:
            if sym.kind == "method" and sym.parent_class:
                class_id = f"code:class:{sym.file_path}:{sym.parent_class}"
                if class_id in {s.node_id for s in symbols}:
                    edges.append(CodeEdge(
                        source_id=class_id,
                        target_id=sym.node_id,
                        edge_type="contains",
                    ))

        # calls: caller → callee (仅文件内已知的)
        for call in result.calls:
            caller_id = sym_by_name.get(call.caller)
            callee_id = sym_by_name.get(call.callee)
            if caller_id and callee_id and caller_id != callee_id:
                edges.append(CodeEdge(
                    source_id=caller_id,
                    target_id=callee_id,
                    edge_type="calls",
                    metadata={"line": call.line},
                ))

        # inheritance: subclass → base (仅文件内已知的)
        class_index = {s.name: s.node_id for s in symbols if s.kind == "class"}
        for sym in symbols:
            if sym.kind == "class" and sym.bases:
                for base in sym.bases:
                    base_name = base.split(".")[-1]
                    base_id = class_index.get(base_name)
                    if base_id:
                        edges.append(CodeEdge(
                            source_id=sym.node_id,
                            target_id=base_id,
                            edge_type="inherits",
                        ))

        return edges


# ── 辅助函数 ──

_EXT_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}

def ext_to_lang(ext: str) -> str:
    """文件扩展名 → tree-sitter 语言标识"""
    return _EXT_LANG_MAP.get(ext.lower(), "")
