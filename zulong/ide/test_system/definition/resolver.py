from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any, Optional

from .models import StepDefinition


class CyclicDependencyError(Exception):
    pass


class StepDependencyResolver:
    def __init__(self, steps: list[StepDefinition]):
        self.steps = steps
        self.step_map: dict[str, StepDefinition] = {s.step_id: s for s in steps}
        self._adj: dict[str, list[str]] = defaultdict(list)
        self._in_degree: dict[str, int] = defaultdict(int)
        self._build_graph()

    def _build_graph(self):
        for step in self.steps:
            self._in_degree.setdefault(step.step_id, 0)
            for dep in step.depends_on:
                if dep not in self.step_map:
                    raise ValueError(f"步骤 {step.step_id} 依赖不存在的步骤: {dep}")
                self._adj[dep].append(step.step_id)
                self._in_degree[step.step_id] += 1

    def topological_sort(self) -> list[str]:
        in_degree = dict(self._in_degree)
        queue = deque(
            step_id for step_id, deg in in_degree.items() if deg == 0
        )
        result = []

        while queue:
            step_id = queue.popleft()
            result.append(step_id)
            for neighbor in self._adj[step_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.steps):
            raise CyclicDependencyError("检测到循环依赖，无法确定执行顺序")

        return result

    def get_execution_order(self) -> list[StepDefinition]:
        order = self.topological_sort()
        return [self.step_map[sid] for sid in order]

    def evaluate_condition(
        self, condition: str, step_outputs: dict[str, Any]
    ) -> bool:
        if not condition:
            return True

        resolved = self._resolve_references(condition, step_outputs)

        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-truefalse!=<> ")
        if not all(c in allowed_chars for c in resolved):
            raise ValueError(f"条件表达式包含不安全字符: {resolved}")

        try:
            return bool(eval(resolved, {"__builtins__": {}}, {}))
        except Exception as e:
            raise ValueError(f"条件表达式求值失败: {resolved}, 错误: {e}")

    def resolve_step_input(
        self, tool_input: dict, step_outputs: dict[str, Any]
    ) -> dict:
        resolved = {}
        for key, value in tool_input.items():
            if isinstance(value, str):
                resolved[key] = self._resolve_references(value, step_outputs)
            elif isinstance(value, dict):
                resolved[key] = self.resolve_step_input(value, step_outputs)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_references(v, step_outputs) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved

    @staticmethod
    def _resolve_references(text: str, step_outputs: dict[str, Any]) -> str:
        def replacer(match: re.Match) -> str:
            step_id = match.group(1)
            field_path = match.group(2)
            output = step_outputs.get(step_id)
            if output is None:
                return match.group(0)
            if not field_path:
                return str(output)
            value = output
            for part in field_path.split("."):
                if isinstance(value, dict):
                    value = value.get(part, "")
                else:
                    return match.group(0)
            return str(value)

        pattern = r"\$([a-zA-Z0-9_-]+)\.output(?:\.([a-zA-Z0-9_.-]+))?"
        return re.sub(pattern, replacer, text)
