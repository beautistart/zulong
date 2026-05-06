"""Tree-sitter AST 解析器

从源代码文件中提取结构化符号信息：
- 函数/方法定义（名称、参数、行范围、装饰器）
- 类定义（名称、基类、行范围）
- 导入语句（模块、别名）
- 函数调用关系

当前支持 Python，可扩展其他语言。
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# 延迟导入 tree-sitter
_parsers: Dict[str, "Parser"] = {}


def _get_parser(lang: str = "python"):
    """获取或创建指定语言的 tree-sitter Parser（单例缓存）"""
    if lang in _parsers:
        return _parsers[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "python":
            import tree_sitter_python as tsp
            language = Language(tsp.language())
        else:
            raise ValueError(f"不支持的语言: {lang}")
        parser = Parser(language)
        _parsers[lang] = parser
        return parser
    except ImportError as e:
        logger.warning(f"[ASTParser] tree-sitter 未安装: {e}")
        return None


@dataclass
class CodeSymbol:
    """代码符号（函数、类、方法）"""
    name: str
    kind: str               # "function", "class", "method"
    file_path: str
    start_line: int
    end_line: int
    qualified_name: str      # 含父类前缀: "MyClass.my_method"
    parameters: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)        # 仅 class
    docstring: Optional[str] = None
    parent_class: Optional[str] = None  # 方法所属类

    @property
    def node_id(self) -> str:
        """生成 MemoryGraph 节点 ID"""
        return f"code:{self.kind}:{self.file_path}:{self.qualified_name}"


@dataclass
class ImportInfo:
    """导入信息"""
    module: str              # "os.path" 或 "zulong.l2.fc_graph"
    names: List[str]         # ["run_fc_loop", "build_fc_graph"] 或 [] (整模块导入)
    alias: Optional[str] = None
    is_from: bool = False    # True = "from X import Y"
    line: int = 0


@dataclass
class CallInfo:
    """函数调用信息"""
    caller: str              # 调用者 qualified_name
    callee: str              # 被调用函数名
    line: int = 0


@dataclass
class FileParseResult:
    """单个文件的解析结果"""
    file_path: str
    symbols: List[CodeSymbol] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    calls: List[CallInfo] = field(default_factory=list)
    parse_error: Optional[str] = None
    content_hash: str = ""  # 文件内容的 MD5 哈希，用于增量更新判断


class ASTParser:
    """Tree-sitter 源码解析器"""

    def __init__(self, lang: str = "python"):
        self.lang = lang
        self._parser = _get_parser(lang)

    @property
    def available(self) -> bool:
        return self._parser is not None

    def parse_file(self, file_path: str) -> FileParseResult:
        """解析单个源码文件，提取所有符号和关系。"""
        result = FileParseResult(file_path=file_path)

        if not self._parser:
            result.parse_error = "tree-sitter 未安装"
            return result

        try:
            source = Path(file_path).read_bytes()
        except Exception as e:
            result.parse_error = f"读取文件失败: {e}"
            return result

        result.content_hash = hashlib.md5(source).hexdigest()

        try:
            tree = self._parser.parse(source)
        except Exception as e:
            result.parse_error = f"解析失败: {e}"
            return result

        self._extract_symbols(tree.root_node, source, file_path, result)
        self._extract_imports(tree.root_node, source, result)
        self._extract_calls(tree.root_node, source, result)

        return result

    def parse_source(self, source: bytes, file_path: str = "<string>") -> FileParseResult:
        """解析源码字节串（用于测试）。"""
        result = FileParseResult(file_path=file_path)
        if not self._parser:
            result.parse_error = "tree-sitter 未安装"
            return result
        try:
            tree = self._parser.parse(source)
        except Exception as e:
            result.parse_error = f"解析失败: {e}"
            return result
        self._extract_symbols(tree.root_node, source, file_path, result)
        self._extract_imports(tree.root_node, source, result)
        self._extract_calls(tree.root_node, source, result)
        return result

    # ── 符号提取 ──────────────────────────────────────────

    def _extract_symbols(self, root, source: bytes, file_path: str,
                         result: FileParseResult,
                         parent_class: Optional[str] = None):
        """递归提取函数和类定义。"""
        for child in root.children:
            if child.type == "function_definition":
                sym = self._parse_function(child, source, file_path, parent_class)
                if sym:
                    result.symbols.append(sym)

            elif child.type == "class_definition":
                cls_sym = self._parse_class(child, source, file_path)
                if cls_sym:
                    result.symbols.append(cls_sym)
                    # 递归提取方法
                    body = child.child_by_field_name("body")
                    if body:
                        self._extract_symbols(
                            body, source, file_path, result,
                            parent_class=cls_sym.name,
                        )

            elif child.type == "decorated_definition":
                # 装饰器下的函数/类
                for sub in child.children:
                    if sub.type == "function_definition":
                        sym = self._parse_function(sub, source, file_path, parent_class)
                        if sym:
                            sym.decorators = self._get_decorators(child, source)
                            result.symbols.append(sym)
                    elif sub.type == "class_definition":
                        cls_sym = self._parse_class(sub, source, file_path)
                        if cls_sym:
                            cls_sym.decorators = self._get_decorators(child, source)
                            result.symbols.append(cls_sym)
                            body = sub.child_by_field_name("body")
                            if body:
                                self._extract_symbols(
                                    body, source, file_path, result,
                                    parent_class=cls_sym.name,
                                )

    def _parse_function(self, node, source: bytes, file_path: str,
                        parent_class: Optional[str]) -> Optional[CodeSymbol]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = name_node.text.decode("utf-8", errors="replace")

        # 参数
        params = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for p in params_node.children:
                if p.type in ("identifier", "typed_parameter",
                              "default_parameter", "typed_default_parameter",
                              "list_splat_pattern", "dictionary_splat_pattern"):
                    pname = p.child_by_field_name("name") or p
                    params.append(pname.text.decode("utf-8", errors="replace"))

        kind = "method" if parent_class else "function"
        qname = f"{parent_class}.{name}" if parent_class else name

        # docstring
        docstring = self._get_docstring(node, source)

        return CodeSymbol(
            name=name,
            kind=kind,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            qualified_name=qname,
            parameters=params,
            parent_class=parent_class,
            docstring=docstring,
        )

    def _parse_class(self, node, source: bytes,
                     file_path: str) -> Optional[CodeSymbol]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = name_node.text.decode("utf-8", errors="replace")

        # 基类
        bases = []
        superclasses = node.child_by_field_name("superclasses")
        if superclasses:
            for arg in superclasses.children:
                if arg.type in ("identifier", "attribute"):
                    bases.append(arg.text.decode("utf-8", errors="replace"))

        docstring = self._get_docstring(node, source)

        return CodeSymbol(
            name=name,
            kind="class",
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            qualified_name=name,
            bases=bases,
            docstring=docstring,
        )

    def _get_docstring(self, node, source: bytes) -> Optional[str]:
        """提取函数/类的 docstring"""
        body = node.child_by_field_name("body")
        if not body or not body.children:
            return None
        first = body.children[0]
        if first.type == "expression_statement":
            expr = first.children[0] if first.children else None
            if expr and expr.type == "string":
                raw = expr.text.decode("utf-8", errors="replace")
                # 去除引号
                for q in ('"""', "'''", '"', "'"):
                    if raw.startswith(q) and raw.endswith(q):
                        raw = raw[len(q):-len(q)]
                        break
                return raw.strip()[:200]  # 截断过长 docstring
        return None

    def _get_decorators(self, decorated_node, source: bytes) -> List[str]:
        """提取装饰器列表"""
        decorators = []
        for child in decorated_node.children:
            if child.type == "decorator":
                text = child.text.decode("utf-8", errors="replace")
                decorators.append(text.lstrip("@").strip())
        return decorators

    # ── 导入提取 ──────────────────────────────────────────

    def _extract_imports(self, root, source: bytes, result: FileParseResult):
        """提取顶层 import 语句"""
        for child in root.children:
            if child.type == "import_statement":
                # import X, Y
                for name_node in child.children:
                    if name_node.type in ("dotted_name", "aliased_import"):
                        text = name_node.text.decode("utf-8", errors="replace")
                        parts = text.split(" as ")
                        result.imports.append(ImportInfo(
                            module=parts[0].strip(),
                            names=[],
                            alias=parts[1].strip() if len(parts) > 1 else None,
                            is_from=False,
                            line=child.start_point[0] + 1,
                        ))

            elif child.type == "import_from_statement":
                # from X import Y, Z
                module_node = child.child_by_field_name("module_name")
                module = (module_node.text.decode("utf-8", errors="replace")
                          if module_node else "")
                names = []
                for sub in child.children:
                    if sub.type in ("dotted_name", "aliased_import"):
                        if sub != module_node:
                            text = sub.text.decode("utf-8", errors="replace")
                            names.append(text.split(" as ")[0].strip())
                    elif sub.type == "identifier" and sub != module_node:
                        names.append(sub.text.decode("utf-8", errors="replace"))
                result.imports.append(ImportInfo(
                    module=module,
                    names=names,
                    is_from=True,
                    line=child.start_point[0] + 1,
                ))

    # ── 调用关系提取 ──────────────────────────────────────

    def _extract_calls(self, root, source: bytes, result: FileParseResult):
        """提取函数调用关系（在每个函数/方法体内查找 call 表达式）"""
        for sym in result.symbols:
            if sym.kind in ("function", "method"):
                # 重新定位函数体的 AST 节点
                calls = self._find_calls_in_range(
                    root, source,
                    sym.start_line - 1, sym.end_line - 1,
                    sym.qualified_name,
                )
                result.calls.extend(calls)

    def _find_calls_in_range(self, node, source: bytes,
                             start_row: int, end_row: int,
                             caller_qname: str) -> List[CallInfo]:
        """在行范围内查找所有函数调用"""
        calls = []
        self._walk_calls(node, source, start_row, end_row,
                         caller_qname, calls, set())
        return calls

    def _walk_calls(self, node, source: bytes,
                    start_row: int, end_row: int,
                    caller: str, calls: List[CallInfo],
                    seen: Set[Tuple[str, str]]):
        """递归遍历查找 call 表达式"""
        if node.start_point[0] > end_row or node.end_point[0] < start_row:
            return

        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node:
                callee = func_node.text.decode("utf-8", errors="replace")
                # 去除 self. 前缀
                if callee.startswith("self."):
                    callee = callee[5:]
                key = (caller, callee)
                if key not in seen:
                    seen.add(key)
                    calls.append(CallInfo(
                        caller=caller,
                        callee=callee,
                        line=node.start_point[0] + 1,
                    ))

        for child in node.children:
            self._walk_calls(child, source, start_row, end_row,
                             caller, calls, seen)
