"""Runtime platform and shell detection helpers.

This module keeps OS/Shell-specific guidance out of L2 prompts and tools.
It does not change the TSD layer responsibilities; it only describes the
current host so L2 can choose commands that match the user's environment.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class RuntimeEnvironment:
    os_name: str
    os_family: str
    shell_name: str
    shell_family: str
    path_style: str
    preferred_commands: List[str]
    forbidden_command_markers: List[str]
    command_guidance: str

    def to_context(self) -> Dict[str, object]:
        return {
            "os_name": self.os_name,
            "os_family": self.os_family,
            "shell": self.shell_name,
            "shell_family": self.shell_family,
            "path_style": self.path_style,
            "preferred_commands": self.preferred_commands,
            "forbidden_command_markers": self.forbidden_command_markers,
            "command_guidance": self.command_guidance,
        }


def _detect_os_family(system_name: str) -> str:
    normalized = system_name.lower()
    if normalized.startswith("win"):
        return "windows"
    if normalized == "darwin":
        return "macos"
    if normalized == "linux":
        return "linux"
    return "posix" if os.name == "posix" else normalized or "unknown"


def _detect_shell_name() -> str:
    candidates = [
        os.environ.get("ZULONG_SHELL"),
        os.environ.get("SHELL"),
        os.environ.get("COMSPEC"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return "cmd.exe" if os.name == "nt" else "/bin/sh"


def _detect_shell_family(shell_name: str, os_family: str) -> str:
    lower = shell_name.lower()
    if "powershell" in lower or "pwsh" in lower:
        return "powershell"
    if lower.endswith("cmd.exe") or lower == "cmd" or "\\cmd.exe" in lower:
        return "cmd"
    if any(name in lower for name in ("bash", "zsh", "fish", "sh")):
        return "posix"
    if os_family == "windows":
        return "cmd"
    return "posix"


def get_runtime_environment() -> RuntimeEnvironment:
    system_name = platform.system() or ("Windows" if os.name == "nt" else "POSIX")
    os_family = _detect_os_family(system_name)
    shell_name = _detect_shell_name()
    shell_family = _detect_shell_family(shell_name, os_family)

    if os_family == "windows":
        preferred = [
            "Get-ChildItem",
            "Select-String",
            "Get-Content",
            "Set-Content",
            "New-Item",
            "rg",
            "python",
            "npm",
            "git",
        ]
        forbidden = [
            "find / -name",
            "ls -la",
            "pwd &&",
            "2>/dev/null",
            "| head",
            " head -",
            "grep ",
            "chmod ",
            "mkdir -p",
        ]
        guidance = (
            "当前环境是 Windows。优先使用 PowerShell/CMD 兼容命令，"
            "例如 Get-ChildItem、Select-String、Get-Content、rg、python、npm、git。"
        )
        path_style = "windows"
    else:
        preferred = [
            "ls",
            "find",
            "grep",
            "cat",
            "sed",
            "rg",
            "python3",
            "python",
            "npm",
            "git",
        ]
        forbidden = [
            "Get-ChildItem",
            "Select-String",
            "Get-Content",
            "Set-Content",
            "New-Item",
            "cmd.exe",
            "powershell.exe",
        ]
        shell_label = "zsh/bash" if os_family == "macos" else "bash/sh"
        guidance = (
            f"当前环境是 {system_name}，通常使用 {shell_label}。"
            "优先使用 POSIX 命令，例如 ls、find、grep、cat、sed、rg、python3、npm、git。"
        )
        path_style = "posix"

    return RuntimeEnvironment(
        os_name=system_name,
        os_family=os_family,
        shell_name=shell_name,
        shell_family=shell_family,
        path_style=path_style,
        preferred_commands=preferred,
        forbidden_command_markers=forbidden,
        command_guidance=guidance,
    )


def get_runtime_context() -> Dict[str, object]:
    return get_runtime_environment().to_context()


def is_windows() -> bool:
    return get_runtime_environment().os_family == "windows"


def is_macos() -> bool:
    return get_runtime_environment().os_family == "macos"


def is_linux() -> bool:
    return get_runtime_environment().os_family == "linux"
