#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-platform smoke checks for ZULONG.

The smoke suite intentionally avoids loading large models. It validates the
platform adaptation surface: config profiles, backend resolution, device
selectors, ffmpeg resolution, model asset paths, and lightweight tests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StepResult:
    name: str
    command: list[str]
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


SYNTAX_TARGETS = [
    "zulong/utils/runtime_env.py",
    "zulong/utils/device.py",
    "zulong/config/config_manager.py",
    "zulong/adapters/backend_resolver.py",
    "zulong/l0/devices/camera_backend.py",
    "zulong/l0/devices/audio_backend.py",
    "zulong/l0/audio/ffmpeg_resolver.py",
    "zulong/workspace/vscode_launcher.py",
    "zulong/ide/ide_server.py",
    "zulong/tools/vscode_tool.py",
    "zulong/tools/system_tools.py",
    "scripts/doctor.py",
    "scripts/doctor_audio.py",
    "scripts/doctor_camera.py",
    "scripts/doctor_models.py",
    "scripts/diagnose_system.py",
    "scripts/smoke_platform.py",
    "scripts/syntax_check.py",
    "zulong/ide/ide_prompt_handler.py",
    "zulong/tools/task_tools.py",
]


def run_step(name: str, command: list[str], *, quiet: bool = False) -> StepResult:
    print(f"\n== {name} ==")
    print(" ".join(command))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
        check=False,
    )
    if quiet and completed.stdout:
        print(completed.stdout)
    print(f"[{'OK' if completed.returncode == 0 else 'FAIL'}] {name}")
    return StepResult(name=name, command=command, returncode=completed.returncode)


def build_steps(args: argparse.Namespace) -> Iterable[tuple[str, list[str]]]:
    py = sys.executable
    yield "syntax_check", [py, "scripts/syntax_check.py", *SYNTAX_TARGETS]
    yield "doctor", [py, "scripts/doctor.py"]
    yield "doctor_models", [py, "scripts/doctor_models.py"]
    if not args.skip_audio_devices:
        yield "doctor_audio", [py, "scripts/doctor_audio.py"]
    if args.with_camera:
        yield "doctor_camera", [
            py,
            "scripts/doctor_camera.py",
            "--scan-range",
            str(args.camera_scan_range),
            "--no-read",
        ]
    yield "platform_tests", [py, "-m", "pytest", "tests/platform", "-q"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ZULONG cross-platform smoke checks")
    parser.add_argument("--with-camera", action="store_true", help="Also probe camera device indexes without reading frames")
    parser.add_argument("--camera-scan-range", type=int, default=1, help="Camera indexes to scan when --with-camera is used")
    parser.add_argument("--skip-audio-devices", action="store_true", help="Skip live PyAudio device enumeration")
    parser.add_argument("--json", action="store_true", help="Print JSON summary at the end")
    parser.add_argument("--quiet", action="store_true", help="Capture step output and print it after each step")
    args = parser.parse_args()

    results = [run_step(name, command, quiet=args.quiet) for name, command in build_steps(args)]
    ok = all(result.ok for result in results)

    if args.json:
        print("")
        print(json.dumps({
            "ok": ok,
            "results": [
                {
                    "name": result.name,
                    "command": result.command,
                    "returncode": result.returncode,
                    "ok": result.ok,
                }
                for result in results
            ],
        }, ensure_ascii=False, indent=2))

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
