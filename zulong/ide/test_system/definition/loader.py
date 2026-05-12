from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from .enums import TestCaseType
from .models import TestCaseDefinition, TestConfig, StepDefinition, AssertionDef


_MALICIOUS_PATTERNS = [
    re.compile(r"import\s+os\b"),
    re.compile(r"import\s+subprocess\b"),
    re.compile(r"__import__\b"),
    re.compile(r"eval\s*\("),
    re.compile(r"exec\s*\("),
    re.compile(r"open\s*\("),
    re.compile(r"shutil\."),
    re.compile(r"sys\.exit"),
    re.compile(r"rm\s+-rf"),
]

_STEP_REF_PATTERN = re.compile(r"\$step_(\d+)\.output")


class TestCaseLoader:
    def __init__(self, cases_dir: str | Path):
        self.cases_dir = Path(cases_dir)

    def load(self, file_path: str) -> TestCaseDefinition:
        full_path = self.cases_dir / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"测试用例文件不存在: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return self._parse_and_validate(raw)

    def load_all(self) -> list[TestCaseDefinition]:
        results = []
        for path in sorted(self.cases_dir.glob("*.yaml")):
            try:
                results.append(self.load(path.name))
            except Exception as e:
                print(f"[TestCaseLoader] 跳过无效用例 {path.name}: {e}")
        for path in sorted(self.cases_dir.glob("*.yml")):
            try:
                results.append(self.load(path.name))
            except Exception as e:
                print(f"[TestCaseLoader] 跳过无效用例 {path.name}: {e}")
        return results

    def _parse_and_validate(self, raw: dict) -> TestCaseDefinition:
        test_case_id = raw.get("test_case_id", "")
        self._validate_id(test_case_id)

        name = raw.get("name", "")
        self._validate_name(name)

        type_str = raw.get("type", "")
        self._validate_type(type_str)

        steps_raw = raw.get("steps", [])
        self._validate_steps(steps_raw)

        for step in steps_raw:
            self._validate_step_safety(step)

        steps = [self._parse_step(s) for s in steps_raw]
        config = TestConfig.from_dict(raw.get("config", {}))

        return TestCaseDefinition(
            test_case_id=test_case_id,
            name=name,
            type=TestCaseType(type_str),
            description=raw.get("description", ""),
            steps=steps,
            config=config,
        )

    def _parse_step(self, raw: dict) -> StepDefinition:
        assertions = []
        for a in raw.get("assertions", []):
            assertions.append(AssertionDef.from_dict(a))
        return StepDefinition(
            step_id=raw["step_id"],
            name=raw["name"],
            tool=raw["tool"],
            tool_input=raw.get("tool_input", {}),
            depends_on=raw.get("depends_on", []),
            condition=raw.get("condition"),
            timeout_seconds=raw.get("timeout_seconds", 300),
            assertions=assertions,
        )

    @staticmethod
    def _validate_id(test_case_id: str):
        if not re.match(r"^[a-zA-Z0-9_-]+$", test_case_id):
            raise ValueError(f"test_case_id 格式无效: {test_case_id}")

    @staticmethod
    def _validate_name(name: str):
        if not name or len(name) > 100:
            raise ValueError(f"name 非空且长度≤100: '{name}'")

    @staticmethod
    def _validate_type(type_str: str):
        valid = {t.value for t in TestCaseType}
        if type_str not in valid:
            raise ValueError(f"type 必须为 {valid}: {type_str}")

    @staticmethod
    def _validate_steps(steps: list):
        if not steps:
            raise ValueError("steps 列表不能为空")

    @staticmethod
    def _validate_step_safety(step: dict):
        tool = step.get("tool", "")
        for pat in _MALICIOUS_PATTERNS:
            if pat.search(tool):
                raise ValueError(f"步骤包含恶意命令: tool={tool}")
        tool_input = step.get("tool_input", {})
        input_str = str(tool_input)
        for pat in _MALICIOUS_PATTERNS:
            if pat.search(input_str):
                raise ValueError(f"步骤输入包含恶意内容: {input_str[:100]}")
        refs = _STEP_REF_PATTERN.findall(input_str)
        if refs:
            for ref_num in refs:
                step_ids = {s.get("step_id", "") for s in [step]}
                if f"step_{ref_num}" not in step_ids and ref_num not in [
                    str(i) for i in range(len([step]))
                ]:
                    pass
