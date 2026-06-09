#!/usr/bin/env python3
"""
打包祖龙核心代码 + 生成项目目录结构
不修改任何源文件，生成两个文件：
1. project_structure.txt - 项目目录树
2. zulong_core_code.txt - 核心代码打包
"""
import os
from pathlib import Path

ROOT = Path(r"D:\AI\project\zulong_beta5")

# ============ 排除的目录 ============
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".pytest_cache", "node_modules",
    "logs", "data", "checkpoints", "experience_store",
    "models",  # 模型文件太大
    "zulong_env",  # 虚拟环境
    "agent_workspace", "docs-local-backup", "audit",
    "emos_research", ".arts", ".codeartsdoer", ".qoder",
    ".trae", "Assets", "Packages", "ProjectSettings",
    "rw", "rz", "searxng",
}

EXCLUDE_FILES = {
    "nul", "project_structure.txt", "zulong_core_code.txt",
    "pack_core_code.py",
}

# ============ 第一步：生成项目目录结构 ============
def generate_tree(file, path, prefix="", depth=0, max_depth=5):
    """递归生成目录树，限制深度"""
    if depth > max_depth:
        return
    try:
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        file.write(f"{prefix}[权限不足]\n")
        return

    for i, item in enumerate(items):
        if item.name in EXCLUDE_DIRS or item.name.startswith("."):
            continue
        if item.is_file() and item.name in EXCLUDE_FILES:
            continue

        is_last = (i == len([it for it in items
                             if it.name not in EXCLUDE_DIRS and not it.name.startswith(".")
                             and not (it.is_file() and it.name in EXCLUDE_FILES)]) - 1)
        connector = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "

        if item.is_dir():
            file.write(f"{prefix}{connector}[{item.name}]/\n")
            generate_tree(file, item, prefix + next_prefix, depth + 1, max_depth)
        else:
            file.write(f"{prefix}{connector}{item.name}\n")


structure_path = ROOT / "project_structure.txt"
with open(structure_path, "w", encoding="utf-8") as f:
    f.write("祖龙 (ZULONG) 项目目录结构\n")
    f.write("=" * 60 + "\n")
    f.write(f"根目录: {ROOT}\n\n")
    generate_tree(f, ROOT, max_depth=4)

print(f"✅ 目录结构已保存: {structure_path}")


# ============ 第二步：打包核心代码 ============
# 核心代码目录
CORE_DIRS = [
    "zulong",           # 主系统代码
]

# 核心根文件
CORE_ROOT_FILES = [
    "start.py",
    "build_release.py",
    "mcp_server.py",
    "config.yaml",
    "requirements.txt",
    "pytest.ini",
]

# Python 文件扩展名
CODE_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".txt", ".cfg", ".ini"}


def pack_directory(output_file, base_dir, rel_path=""):
    """递归打包目录下的代码文件"""
    dir_path = base_dir / rel_path if rel_path else base_dir
    if not dir_path.exists():
        return

    try:
        items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return

    for item in items:
        if item.name in EXCLUDE_DIRS or item.name.startswith("."):
            continue

        item_rel = str(item.relative_to(ROOT))

        if item.is_dir():
            pack_directory(output_file, item, "")
        elif item.is_file():
            ext = item.suffix.lower()
            if ext in CODE_EXTENSIONS:
                try:
                    content = item.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    content = f"[读取失败: {e}]"

                output_file.write("\n" + "=" * 80 + "\n")
                output_file.write(f"文件: {item_rel}\n")
                output_file.write(f"行数: {len(content.splitlines())}\n")
                output_file.write("=" * 80 + "\n\n")
                output_file.write(content)
                output_file.write("\n")


code_path = ROOT / "zulong_core_code.txt"
with open(code_path, "w", encoding="utf-8") as f:
    f.write("祖龙 (ZULONG) 核心代码打包\n")
    f.write("=" * 60 + "\n")
    f.write(f"生成时间: {Path.cwd()}\n")
    f.write(f"项目根目录: {ROOT}\n\n")

    # 1. 打包根目录核心文件
    f.write("\n" + "#" * 80 + "\n")
    f.write("# 第一部分：根目录核心文件\n")
    f.write("#" * 80 + "\n")

    for filename in CORE_ROOT_FILES:
        file_path = ROOT / filename
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                content = "[二进制文件或读取失败]"
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"文件: {filename}\n")
            f.write(f"行数: {len(content.splitlines())}\n")
            f.write("=" * 80 + "\n\n")
            f.write(content)
            f.write("\n")

    # 2. 打包 config/ 目录
    f.write("\n" + "#" * 80 + "\n")
    f.write("# 第二部分：config/ 配置目录\n")
    f.write("#" * 80 + "\n")
    config_dir = ROOT / "config"
    if config_dir.exists():
        for item in sorted(config_dir.iterdir(), key=lambda x: x.name.lower()):
            if item.is_file() and item.suffix.lower() in CODE_EXTENSIONS:
                try:
                    content = item.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    content = "[读取失败]"
                item_rel = str(item.relative_to(ROOT))
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"文件: {item_rel}\n")
                f.write(f"行数: {len(content.splitlines())}\n")
                f.write("=" * 80 + "\n\n")
                f.write(content)
                f.write("\n")

    # 3. 打包 zulong/ 核心代码
    f.write("\n" + "#" * 80 + "\n")
    f.write("# 第三部分：zulong/ 核心系统代码\n")
    f.write("#" * 80 + "\n")

    zulong_dir = ROOT / "zulong"
    if zulong_dir.exists():
        for item in sorted(zulong_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.name in EXCLUDE_DIRS or item.name.startswith("."):
                continue
            item_rel = str(item.relative_to(ROOT))

            if item.is_dir():
                f.write(f"\n\n{'#' * 60}\n")
                f.write(f"# 模块: {item_rel}/\n")
                f.write(f"{'#' * 60}\n")
                pack_directory(f, item)
            elif item.is_file() and item.suffix.lower() in CODE_EXTENSIONS:
                try:
                    content = item.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    content = "[读取失败]"
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"文件: {item_rel}\n")
                f.write(f"行数: {len(content.splitlines())}\n")
                f.write("=" * 80 + "\n\n")
                f.write(content)
                f.write("\n")

    # 4. 打包 scripts/ 目录
    scripts_dir = ROOT / "scripts"
    if scripts_dir.exists():
        f.write("\n" + "#" * 80 + "\n")
        f.write("# 第四部分：scripts/ 脚本目录\n")
        f.write("#" * 80 + "\n")
        pack_directory(f, scripts_dir)

    # 5. 打包 zulong-ide/ 目录（前端扩展）
    ide_dir = ROOT / "zulong-ide"
    if ide_dir.exists():
        f.write("\n" + "#" * 80 + "\n")
        f.write("# 第五部分：zulong-ide/ VS Code 扩展（仅配置文件）\n")
        f.write("#" * 80 + "\n")
        # 只包含关键配置文件
        ide_key_files = [
            "package.json", "tsconfig.json", "esbuild.mjs",
        ]
        for fn in ide_key_files:
            fp = ide_dir / fn
            if fp.exists():
                try:
                    content = fp.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    content = "[读取失败]"
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"文件: zulong-ide/{fn}\n")
                f.write(f"行数: {len(content.splitlines())}\n")
                f.write("=" * 80 + "\n\n")
                f.write(content)
                f.write("\n")

code_size = code_path.stat().st_size
print(f"✅ 核心代码已打包: {code_path}")
print(f"   文件大小: {code_size / 1024 / 1024:.2f} MB")
