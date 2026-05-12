"""
代码审查钩子：检查 except pass 违规

扫描所有Python文件，检测 "except ...: pass" 模式，
确保异常处理遵循结构化日志记录规范。
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_python_files(root_dir: str) -> List[str]:
    """递归查找所有Python文件"""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def check_except_pass(file_path: str) -> List[Tuple[int, str]]:
    """
    检查文件中的 except pass 语句

    Returns:
        List of (line_number, line_content) tuples
    """
    violations = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            if re.match(r"except\s*.*:\s*pass\s*(#.*)?$", stripped):
                violations.append((i, stripped))

    except Exception as e:
        print(f"[ERROR] 无法读取文件 {file_path}: {e}", file=sys.stderr)

    return violations


def scan_directory(root_dir: str) -> Tuple[int, List[Tuple[str, List[Tuple[int, str]]]]]:
    """
    扫描目录下的所有Python文件

    Returns:
        (total_violations, violation_details)
    """
    python_files = find_python_files(root_dir)
    total_violations = 0
    violation_details = []

    for file_path in python_files:
        violations = check_except_pass(file_path)
        if violations:
            total_violations += len(violations)
            violation_details.append((file_path, violations))

    return total_violations, violation_details


def main():
    """命令行入口"""
    if len(sys.argv) < 2:
        root_dir = os.path.join(os.path.dirname(__file__), "..", "ide")
    else:
        root_dir = sys.argv[1]

    if not os.path.isdir(root_dir):
        print(f"[ERROR] 目录不存在: {root_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 扫描目录: {root_dir}")
    total_violations, violation_details = scan_directory(root_dir)

    if total_violations == 0:
        print("[SUCCESS] 未发现 except pass 违规")
        sys.exit(0)

    print(f"\n[WARNING] 发现 {total_violations} 处 except pass 违规:\n")

    for file_path, violations in violation_details:
        rel_path = os.path.relpath(file_path, root_dir)
        print(f"📄 {rel_path}:")
        for line_num, line_content in violations:
            print(f"   Line {line_num}: {line_content}")
        print()

    print(f"❌ 总计: {total_violations} 处违规")
    print("\n建议: 使用 ErrorHandler.handle_exception() 替换 except pass")
    print("示例:")
    print('  except Exception as e:')
    print('      ErrorHandler.handle_exception(e, ErrorCode.XXXX, context={...})')

    sys.exit(1)


if __name__ == "__main__":
    main()
