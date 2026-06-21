"""
IDE 工具分类注册表

将工具分为两类：
- 内部工具 (internal): 祖龙服务端直接执行（task/memory/attention 等）
- 远程工具 (remote): 暂停 FC 循环，转译为 XML 返回给 IDE 插件 客户端执行
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Any, Optional

logger = logging.getLogger(__name__)

# IDE 的远程工具名（客户端执行）
IDE_REMOTE_TOOLS = {
    "read_file",
    "write_to_file",
    "replace_in_file",
    "delete_file",
    "execute_command",
    "search_files",
    "list_files",
    "list_code_definition_names",
    "browser_action",
    "ask_followup_question",
    "attempt_completion",
    # ===== VS Code 完整控制工具 (TSD v2.7 扩充) =====
    "vscode_run_command",
    "get_diagnostics",
    "ask_user_input",
    "ask_user_select_file",
    "vscode_manage_extension",
    "open_settings",
    "open_problems",
    # ===== 修复已有缺失 =====
    "create_directory",
}

# 祖龙内部工具中与 IDE 远程工具功能重叠的（IDE 模式下禁用）
_ZULONG_TOOLS_DISABLED_IN_IDE_MODE = {
    "exec_write_file",
    "exec_run_command",
    "exec_read_file",
    # ToolRAG 入口不参与首轮 IDE 工具注入；工具不足时走 request_tool_supplement。
    "search_tools",
}

# 继续已有任务图时物理排除的内部工具（防止 LLM 重新创建/扩展已恢复的任务图）
_CONTINUE_GRAPH_EXCLUDED_INTERNAL_TOOLS = {
    "task_create_plan",
    "task_add_node",
}

ANNOUNCE_STEP_TOOL_NAME = "announce_step"

ANNOUNCE_STEP_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ANNOUNCE_STEP_TOOL_NAME,
        "description": (
            "零副作用步骤说明工具。准备调用真实工具前，如果 assistant.content "
            "无法保留普通可见说明，先调用本工具用一句中文告诉用户本步将做什么。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "8-80 个中文字符优先，最多 160 字；只说明本步将做什么，不写推理过程。",
                    "maxLength": 160,
                },
                "expected_actions": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 48},
                    "description": "可选，最多 3 项，用于生成用户可见清单。",
                    "maxItems": 3,
                },
            },
            "required": ["message"],
        },
    },
}


@dataclass
class CachedSchema:
    """缓存的工具Schema"""
    schema: Dict[str, Any]
    definition_hash: str
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


class SchemaCache:
    """工具Schema缓存管理器"""

    def __init__(self):
        self._cache: Dict[str, CachedSchema] = {}
        self._total_requests = 0
        self._cache_hits = 0

    @staticmethod
    def _compute_hash(definition: Any) -> str:
        """计算工具定义的SHA-256哈希值（取前16位）"""
        try:
            content = str(definition)
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception:
            return ""

    def get(self, tool_name: str, definition: Any) -> Optional[Dict[str, Any]]:
        """从缓存获取Schema，检查哈希值是否匹配"""
        self._total_requests += 1

        cached = self._cache.get(tool_name)
        if not cached:
            return None

        current_hash = self._compute_hash(definition)
        if cached.definition_hash != current_hash:
            del self._cache[tool_name]
            logger.debug(f"[SchemaCache] 工具 {tool_name} 定义变更，缓存失效")
            return None

        self._cache_hits += 1
        cached.hit_count += 1
        logger.debug(f"[SchemaCache] 命中: {tool_name} (hit_count={cached.hit_count})")
        return cached.schema

    def set(self, tool_name: str, schema: Dict[str, Any], definition: Any) -> None:
        """写入Schema缓存"""
        definition_hash = self._compute_hash(definition)
        self._cache[tool_name] = CachedSchema(
            schema=schema,
            definition_hash=definition_hash,
        )
        logger.debug(f"[SchemaCache] 写入: {tool_name}")

    def get_hit_rate(self) -> float:
        """计算缓存命中率"""
        if self._total_requests == 0:
            return 0.0
        return self._cache_hits / self._total_requests

    def get_stats(self) -> Dict[str, Any]:
        """返回缓存统计信息"""
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "hit_rate": f"{self.get_hit_rate():.2%}",
            "cached_tools": len(self._cache),
        }

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._total_requests = 0
        self._cache_hits = 0
        logger.info("[SchemaCache] 缓存已清空")

# IDE 远程工具的 OpenAI Function Calling Schema
# 参考 IDE 源码中的 XML 工具定义
_IDE_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容。可选指定行范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件绝对路径"},
                    "start_line": {"type": "integer", "description": "起始行号（可选）"},
                    "end_line": {"type": "integer", "description": "结束行号（可选）"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_to_file",
            "description": (
                "将内容写入文件。默认覆盖；长文件必须按 800-1200 字符分片写入："
                "第一片 mode=overwrite，后续 mode=append。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件绝对路径"},
                    "content": {"type": "string", "description": "本次写入的内容。长文件不要一次性传完整内容，单片建议 800-1200 字符。"},
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "写入模式。默认 overwrite 覆盖；append 会追加到现有内容后再写入。",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "在文件中搜索并替换内容。支持多个替换块。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件绝对路径"},
                    "diff": {"type": "string", "description": "SEARCH/REPLACE 格式的替换块"},
                },
                "required": ["path", "diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "删除指定文件。需谨慎使用，删除后不可恢复。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要删除的文件绝对路径"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "在终端执行命令。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"},
                    "requires_approval": {
                        "type": "boolean",
                        "description": "是否需要用户批准（默认 true）",
                    },
                    "shell": {
                        "type": "string",
                        "enum": ["auto", "cmd", "powershell", "git_bash"],
                        "description": "可选。指定终端 shell：auto/cmd/powershell/git_bash。",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "在文件中搜索正则表达式模式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "搜索目录绝对路径"},
                    "regex": {"type": "string", "description": "正则表达式模式"},
                    "file_pattern": {"type": "string", "description": "文件名 glob 模式（可选）"},
                },
                "required": ["path", "regex"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录中的文件和子目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录绝对路径"},
                    "recursive": {
                        "type": "boolean",
                        "description": "是否递归列出（默认 false）",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_code_definition_names",
            "description": "列出目录中所有文件的代码定义名称（类、函数、方法等）。path 必须是目录路径，不能是文件路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径（不能是文件路径）"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_action",
            "description": "在浏览器中执行操作（启动、点击、输入、截图等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["launch", "click", "type", "scroll_down",
                                 "scroll_up", "screenshot", "close"],
                        "description": "浏览器操作类型",
                    },
                    "url": {"type": "string", "description": "启动时的 URL（action=launch 时）"},
                    "coordinate": {"type": "string", "description": "点击坐标 'x,y'（action=click 时）"},
                    "text": {"type": "string", "description": "输入文本（action=type 时）"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_followup_question",
            "description": "向用户提问以获取更多信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要问用户的问题"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attempt_completion",
            "description": "当任务完成时，向用户提交最终结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "任务完成结果的描述"},
                    "command": {"type": "string", "description": "可选的演示命令"},
                },
                "required": ["result"],
            },
        },
    },
    # ===== VS Code 完整控制工具 (TSD v2.7 扩充) =====
    {
        "type": "function",
        "function": {
            "name": "vscode_run_command",
            "description": "执行 VS Code 内置或扩展命令。可用于格式化代码、重构、运行测试、Git 操作、打开面板等。部分高风险命令需要用户审批。常用命令: editor.action.formatDocument, editor.action.organizeImports, workbench.action.tasks.build, git.stage 等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "VS Code 命令 ID"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "命令参数（可选）"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_diagnostics",
            "description": "获取工作区文件的 linter/编译器诊断信息（Error/Warning/Info/Hint）。修改代码后用于检查是否引入错误。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件绝对路径（可选，不传则返回所有文件）"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_input",
            "description": "弹出 VS Code 输入框向用户询问信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "提示文字"},
                    "placeholder": {"type": "string", "description": "占位文字（可选）"},
                    "default_value": {"type": "string", "description": "默认值（可选）"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_select_file",
            "description": "弹出系统文件/文件夹选择对话框，让用户选择路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "对话框标题"},
                    "type": {"type": "string", "enum": ["file", "folder"], "description": "选择文件还是文件夹"},
                },
                "required": ["title", "type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vscode_manage_extension",
            "description": "管理 VS Code 扩展（安装/卸载/启用/禁用/查询列表）。高风险操作需要用户审批。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["install", "uninstall", "enable", "disable", "list"], "description": "操作类型"},
                    "extension_id": {"type": "string", "description": "扩展 ID（install/uninstall/enable/disable 时必填）"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_settings",
            "description": "打开 VS Code 设置面板。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "enum": ["user", "workspace"], "description": "用户设置或工作区设置"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_problems",
            "description": "打开 VS Code 问题面板，展示诊断结果。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # ===== 修复已有缺失 =====
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "创建工作目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录绝对路径"},
                },
                "required": ["path"],
            },
        },
    },
]


class IDEToolRegistry:
    """IDE 工具分类注册表

    合并祖龙内部工具和 IDE 远程工具的定义，
    为 LLM 提供统一的 OpenAI FC tool schema。
    """

    def __init__(self, tool_engine=None):
        """
        Args:
            tool_engine: 祖龙 ToolEngine 实例（用于获取内部工具 schema）
        """
        self.tool_engine = tool_engine
        self._schema_cache = SchemaCache()

    def classify(self, tool_name: str) -> Literal["internal", "remote"]:
        """判断工具是内部执行还是远程执行"""
        if tool_name in IDE_REMOTE_TOOLS:
            return "remote"
        return "internal"

    def get_combined_tool_definitions(self) -> List[Dict[str, Any]]:
        """合并祖龙内部工具 + IDE 远程工具的 OpenAI FC schema"""
        internal = self._get_filtered_internal_tools()
        remote = list(_IDE_TOOL_SCHEMAS)
        combined = [ANNOUNCE_STEP_TOOL_SCHEMA] + internal + remote
        logger.info(
            f"[IDEToolRegistry] 合并工具定义: "
            f"内部={len(internal)}, 远程={len(remote)}, 总计={len(combined)}"
        )
        return combined

    def update_remote_tool_schema(self, tool_name: str, new_schema: Dict[str, Any]) -> bool:
        """P2-21: 动态更新远程工具Schema（前端工具定义变更时调用）

        Args:
            tool_name: 工具名称
            new_schema: 完整的OpenAI FC schema（含function.name, function.parameters等）
        """
        global _IDE_TOOL_SCHEMAS
        for i, schema in enumerate(_IDE_TOOL_SCHEMAS):
            func = schema.get("function", {})
            if func.get("name") == tool_name:
                _IDE_TOOL_SCHEMAS[i] = new_schema
                logger.info(f"[IDEToolRegistry] 已更新远程工具Schema: {tool_name}")
                return True
        logger.warning(f"[IDEToolRegistry] 未找到远程工具: {tool_name}")
        return False

    def get_combined_tool_definitions_for_policy(
        self, task_graph_policy: str = "none"
    ) -> List[Dict[str, Any]]:
        """按 L1-B 任务图策略返回工具定义。"""
        task_graph_policy = (task_graph_policy or "none").lower()
        extra_exclude = (
            _CONTINUE_GRAPH_EXCLUDED_INTERNAL_TOOLS
            if task_graph_policy in {"reuse", "inspect", "continue"}
            else None
        )
        internal = self._get_filtered_internal_tools(
            extra_exclude=extra_exclude,
        )
        remote = list(_IDE_TOOL_SCHEMAS)
        combined = [ANNOUNCE_STEP_TOOL_SCHEMA] + internal + remote
        logger.info(
            "[IDEToolRegistry] 策略工具定义: policy=%s, 内部=%s, 远程=%s, 总计=%s",
            task_graph_policy,
            len(internal),
            len(remote),
            len(combined),
        )
        return combined

    def _get_filtered_internal_tools(
        self, extra_exclude: Optional[set] = None,
        extra_include: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """获取过滤后的祖龙内部工具 schema

        IDE 模式下禁用与远程工具功能重叠的执行工具。
        extra_exclude 用于继续已有任务图时排除 task_create_plan/task_add_node。
        extra_include 用于显式工具白名单。
        """
        if not self.tool_engine:
            return []

        tool_definitions = []
        for name, tool in self.tool_engine.registry.tools.items():
            if not tool.enabled:
                continue
            # 显式白名单：仅加载 extra_include 中的工具
            if extra_include is not None:
                if name not in extra_include:
                    continue
            else:
                # 正常过滤
                if name in _ZULONG_TOOLS_DISABLED_IN_IDE_MODE:
                    logger.debug(f"[IDEToolRegistry] 跳过 IDE 模式禁用工具: {name}")
                    continue
                if name in IDE_REMOTE_TOOLS:
                    continue
                if extra_exclude and name in extra_exclude:
                    logger.info(f"[IDEToolRegistry] 继续任务图策略排除工具: {name}")
                    continue

            # 使用缓存获取schema
            cached_schema = self._schema_cache.get(name, tool)
            if cached_schema:
                tool_definitions.append(cached_schema)
                continue

            try:
                schema = tool.get_function_schema()
                self._schema_cache.set(name, schema, tool)
                tool_definitions.append(schema)
            except Exception as e:
                logger.warning(f"[IDEToolRegistry] 工具 {name} schema 获取失败: {e}")

        # 显式白名单模式不加载 CRUD 工具
        if extra_include is None:
            # 追加 TaskGraph CRUD 工具 schema
            try:
                from zulong.ide.graph_crud_tools import get_crud_tool_schemas
                crud_schemas = get_crud_tool_schemas()
                tool_definitions.extend(crud_schemas)
            except Exception as e:
                logger.warning(f"[IDEToolRegistry] CRUD工具schema加载失败: {e}")

        return tool_definitions

    def get_remote_tool_schemas(self) -> List[Dict[str, Any]]:
        """仅获取 IDE 远程工具的 schema"""
        return list(_IDE_TOOL_SCHEMAS)

    def get_internal_tool_names(self) -> List[str]:
        """获取所有内部工具名称"""
        if not self.tool_engine:
            return []
        return [
            name for name, tool in self.tool_engine.registry.tools.items()
            if tool.enabled
            and name not in _ZULONG_TOOLS_DISABLED_IN_IDE_MODE
            and name not in IDE_REMOTE_TOOLS
        ]

    def get_cache_stats(self) -> Dict[str, Any]:
        """返回Schema缓存统计信息"""
        return self._schema_cache.get_stats()
