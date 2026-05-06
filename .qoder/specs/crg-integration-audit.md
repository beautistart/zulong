# CRG 深度融合方案 -- 事件驱动懒索引

## Context

祖龙系统已有完整的 CRG 基础设施（AST 解析器、图构建器、CodeGraphAdapter、CODE_SYMBOL 节点类型），但管线从未被触发，MemoryGraph 中有 0 个 CODE_SYMBOL 节点。

**用户核心要求**：
1. 不要每次启动都全量索引（浪费）
2. CRG 构建的图谱应直接成为系统标准图谱（深度融合，非外挂管线）
3. 不需要 LLM 手动调用索引工具（自动化）

**设计哲学**：文件被触摸时自动索引（懒加载），利用已有的 `incremental_sync()` 适配器模式，让 CODE_SYMBOL 节点像 TASK 节点一样成为 MemoryGraph 的原生公民。

---

## 架构：事件驱动懒索引

```
LLM 调用 read_file("foo.py")
    │
    ▼
IDE 执行 → 文件内容返回
    │
    ▼
_inject_tool_results()                          ← 已有方法
    ├── 追加到 messages                          (已有)
    ├── 注册到 attention_window                  (已有)
    ├── _auto_index_touched_file(path, content)  ← 新增钩子
    │       │
    │       ├── ASTParser.parse_source(content)  (已有)
    │       │       → symbols, imports, calls
    │       │
    │       └── CodeGraphAdapter.incremental_sync(mg, "file_updated", {
    │               file_path, symbols, edges    ← 新增实现
    │           })
    │           │
    │           ├── 移除旧 CODE_SYMBOL 节点 (该文件的)
    │           ├── 创建新 CODE_SYMBOL 节点
    │           │   └── mg.add_node() 自动触发:
    │           │       ├── _auto_embed_node()     → 向量化
    │           │       └── discover_semantic_neighbors() → 语义边
    │           ├── 创建 FILE 节点 + HIERARCHY 边
    │           └── 创建 calls/contains/inherits 边
    │
    └── BFS 激活时 CODE_SYMBOL 自然参与扩散      (已有机制)
```

**关键区别**：不是"LLM 调 index_project → 建图 → 再查"，而是"文件被读/写时自动解析入图，LLM 查询时图已就绪"。

---

## 修改清单 (6 个文件)

### 1. `zulong/memory/graph_adapters.py` -- CodeGraphAdapter 增量同步

**核心修改**：为 `CodeGraphAdapter` 实现 `incremental_sync()`（参照 `TaskGraphAdapter` 第 210-278 行的模式）

```python
def incremental_sync(self, graph: MemoryGraph, event_type: str, data: dict):
    """代码图谱增量同步 -- 文件粒度"""
    
    if event_type == "file_updated":
        file_path = data["file_path"]        # 相对路径
        symbols = data.get("symbols", [])     # List[CodeSymbol]
        edges = data.get("edges", [])         # List[CodeEdge]
        
        # 1. 清除该文件的旧 CODE_SYMBOL 节点
        stale_ids = [nid for nid, n in graph._nodes.items()
                     if n.node_type == NodeType.CODE_SYMBOL
                     and n.metadata.get("file_path") == file_path]
        for nid in stale_ids:
            graph.remove_node(nid)
        
        # 2. 创建/更新 FILE 节点
        file_node_id = f"file:{file_path}"
        file_node = GraphNode(node_id=file_node_id, node_type=NodeType.FILE, ...)
        graph.add_node(file_node, touch=False)
        
        # 3. 创建 CODE_SYMBOL 节点 + FILE→SYMBOL HIERARCHY 边
        for sym in symbols:
            gnode = GraphNode(
                node_id=sym.node_id,  # "code:{kind}:{file_path}:{qualified_name}"
                node_type=NodeType.CODE_SYMBOL,
                label=sym.qualified_name,
                metadata={kind, file_path, start_line, end_line, parameters, bases, docstring, parent_class},
            )
            graph.add_node(gnode)  # 自动触发 embedding + 语义邻居发现
            graph.add_edge(file_node_id, sym.node_id, EdgeType.HIERARCHY, protected=True)
        
        # 4. 创建关系边 (calls, contains, inherits)
        for edge in edges:
            if graph.has_node(edge.source_id) and graph.has_node(edge.target_id):
                mg_edge_type, protected = self._EDGE_MAP.get(edge.edge_type, ...)
                graph.add_edge(edge.source_id, edge.target_id, mg_edge_type, ...)
    
    elif event_type == "file_removed":
        # 清除该文件所有 CODE_SYMBOL 节点
```

**新增 `remove_file_nodes()` 方法**: 辅助清除单个文件的旧节点

### 2. `zulong/ide/ide_fc_runner.py` -- 文件触摸自动索引钩子

**修改 `_inject_tool_results()` (行 851-852 之后)**: 插入自动索引钩子

```python
# 行 852 之后，注册 attention_window 之前:
# 自动索引被触摸的代码文件
if tool_name in ("read_file", "write_to_file", "replace_in_file"):
    self._auto_index_touched_file(tool_name, func_info, content)
```

**新增 `_auto_index_touched_file()` 私有方法**:

```python
def _auto_index_touched_file(self, tool_name: str, func_info: dict, content: str):
    """文件被读/写时自动解析并同步到 MemoryGraph（事件驱动懒索引）"""
    try:
        args = json.loads(func_info.get("arguments", "{}"))
        file_path = args.get("path", "")
        if not file_path:
            return
        
        # 仅处理支持的代码文件
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".py", ".js", ".ts", ".tsx", ".jsx"):
            return
        
        # 解析文件内容 (对 read_file 使用返回内容；write/replace 需从文件重新读取)
        from zulong.code.ast_parser import ASTParser
        parser = ASTParser(lang=_ext_to_lang(ext))
        if not parser.available:
            return  # tree-sitter 未安装，静默跳过
        
        if tool_name == "read_file":
            result = parser.parse_source(content.encode("utf-8"), file_path)
        else:
            # write/replace 后内容已变，用工具返回的确认信息无法解析
            # 标记为需要下次 read 时重新索引
            self._pending_reindex_files.add(file_path)
            return
        
        if not result or not result.symbols:
            return
        
        # 构建单文件的边关系
        from zulong.code.graph_builder import CodeGraphBuilder
        edges = CodeGraphBuilder._build_edges_for_file(result)
        
        # 通过 incremental_sync 推送到 MemoryGraph
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if mg:
            adapter = mg._adapters.get("code_graph")
            if adapter:
                adapter.incremental_sync(mg, "file_updated", {
                    "file_path": file_path.replace("\\", "/"),
                    "symbols": result.symbols,
                    "edges": edges,
                })
                logger.info(
                    f"[IDEFCRunner] 自动索引: {file_path} → "
                    f"{len(result.symbols)} 符号")
    except Exception as e:
        logger.debug(f"[IDEFCRunner] 自动索引跳过: {e}")
```

**新增 `__init__` 中**: `self._pending_reindex_files: Set[str] = set()`

**增强 `_inject_tool_results()` 开头**: 检查 pending reindex -- 如果本次是 `read_file` 且文件在 pending 集合中，执行索引后从集合移除

### 3. `zulong/code/graph_builder.py` -- 新增单文件边构建

**新增静态方法**: `_build_edges_for_file(result: FileParseResult) -> List[CodeEdge]`
- 从单个 FileParseResult 构建 containment + call + inheritance 边
- 提取自现有 `build()` 中的边构建逻辑，不依赖全局符号索引
- 跨文件的 call 边仅在目标已存在于 MemoryGraph 时创建 (通过 node_id 查找)

**新增辅助函数**: `_ext_to_lang(ext: str) -> str`
- `.py` → `"python"`, `.js/.jsx` → `"javascript"`, `.ts/.tsx` → `"typescript"`

### 4. `zulong/tools/code_tools.py` -- 修复参数 schema + 保留手动工具

**修复**: 为 `SearchCodeSymbolsTool` 和 `GetSymbolContextTool` 添加 `_get_parameters_schema()`:
- 当前基类返回空 schema，LLM 不知道如何传参
- `SearchCodeSymbolsTool`: `{query: str (required), kind?: str, file_path?: str}`
- `GetSymbolContextTool`: `{symbol_name: str (required), file_path?: str}`

**新增 `GetImpactAnalysisTool`** (P4-3 规划要求):
- 名称: `"get_impact_analysis"`
- 从符号出发 BFS 扩散，返回影响范围内的符号和文件
- 利用 `mg.compute_activations()` 已有能力
- 返回: `{source, impact_files: {path: [symbols...]}, total_affected: N}`

**不再需要**:
- ~~`index_project` 工具~~ -- 自动索引替代了手动全量索引
- ~~`index_file` 工具~~ -- 文件触摸时自动索引

### 5. `zulong/tools/tool_engine.py:633-643` -- 注册新工具

在现有代码工具注册块追加 `GetImpactAnalysisTool`:
```python
from zulong.tools.code_tools import (
    SearchCodeSymbolsTool, GetSymbolContextTool,
    GetImpactAnalysisTool,  # 新增
)
```

### 6. `zulong/ide/ide_fc_runner.py` -- BFS 激活增强 (可选)

**增强 `_run_bfs_activation()` (行 1866-1872)**:
在构建 BFS seed 列表时，如果 in_progress 任务的 FileRef 关联了已索引的文件，将该文件的 CODE_SYMBOL 节点也加入 seed:

```python
# 在现有 seeds 构建之后:
# 增强: 将活跃任务关联的 CODE_SYMBOL 节点加入 seed
for n in ip:
    for f in (getattr(tg.get_node(n.id), 'files', None) or []):
        file_node_id = f"file:{f.path}"
        if mg.has_node(file_node_id):
            # 获取该文件下的 CODE_SYMBOL 子节点
            for child in mg.get_children(file_node_id, EdgeType.HIERARCHY):
                if child.node_type == NodeType.CODE_SYMBOL and child.node_id not in seen:
                    seeds.append(child.node_id)
                    seen.add(child.node_id)
```

---

## 数据流对比

### 之前 (管线断裂)
```
CodeGraphBuilder.build() ← 从未调用
    ↓ (断裂)
CodeGraphAdapter.sync()  ← 从未接收数据
    ↓ (断裂)
MemoryGraph: 0 个 CODE_SYMBOL 节点
    ↓
search_code_symbols → 空结果
```

### 之后 (深度融合)
```
LLM 调用 read_file("scene.py") → IDE 返回内容
    ↓ (_inject_tool_results 钩子)
ASTParser.parse_source(content) → 20 个符号
    ↓ (incremental_sync)
MemoryGraph: 20 个 CODE_SYMBOL 节点 + 自动 embedding + 语义边
    ↓ (BFS 激活自然扩散)
search_code_symbols("Scene") → 找到 Scene 类及其方法
get_symbol_context("Scene.render") → 调用者/被调用者
get_impact_analysis("Scene.render") → 影响范围分析
```

---

## 健壮性

| 场景 | 处理 |
|------|------|
| tree-sitter 未安装 | `parser.available` 检查，静默跳过 |
| 非代码文件 (txt/md/json) | 扩展名过滤，不触发索引 |
| 文件内容被截断 (MAX_TOOL_RESULT_CHARS) | 用截断内容解析，可能少几个符号但不崩溃 |
| write_to_file 后内容未知 | 加入 pending 集合，下次 read_file 时补索引 |
| 同一文件多次读取 | incremental_sync 先清旧节点再建新节点，幂等 |
| 大文件 (>10000 行) | ASTParser 已有解析超时保护 |
| Windows 路径 | 统一 `replace("\\", "/")` |

---

## 验证步骤

1. **Python 语法**: `py_compile` 所有修改文件
2. **TypeScript**: `npx tsc --noEmit`
3. **功能验证**:
   - 启动 IDE 会话，LLM 调用 `read_file("zulong/l2/task_graph.py")`
   - 检查日志: `[IDEFCRunner] 自动索引: zulong/l2/task_graph.py → N 符号`
   - LLM 调用 `search_code_symbols(query="TaskGraph")` → 返回结果
   - LLM 调用 `get_symbol_context(symbol_name="TaskGraph.add_node")` → 返回调用关系
4. **构建 VSIX**: esbuild + vsce package + 安装
