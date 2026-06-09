#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightweight ZULONG runtime doctor.

This script checks the cross-platform P0 runtime surface without loading large
models. It is safe to run on Windows, Linux, and macOS.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _add_repo_to_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _status(ok: bool) -> str:
    return "OK" if ok else "WARN"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _print_check(name: str, ok: bool, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[{_status(ok)}] {name}{suffix}")


def check_runtime() -> dict:
    from zulong.utils.runtime_env import get_runtime_context
    from zulong.utils.device import accelerator_info

    runtime = get_runtime_context()
    accelerator = accelerator_info()
    _print_check("runtime.os", True, f"{runtime['os_name']} ({runtime['os_family']})")
    _print_check("runtime.shell", True, f"{runtime['shell_family']} -> {runtime['shell']}")
    _print_check("accelerator.selected", True, str(accelerator.get("selected_device")))
    _print_check("accelerator.cuda", bool(accelerator.get("cuda_available")), str(accelerator.get("cuda_available")))
    _print_check("accelerator.mps", bool(accelerator.get("mps_available")), str(accelerator.get("mps_available")))
    return {"runtime": runtime, "accelerator": accelerator}


def check_config() -> dict:
    os.environ.setdefault("ZULONG_AUTO_INIT_CONFIG", "false")
    from zulong.config.config_manager import ConfigManager

    cm = ConfigManager()
    from zulong.adapters.backend_resolver import resolve_llm_backend

    llm_resolution = resolve_llm_backend(cm.config)
    profile = getattr(cm, "platform_profile", "")
    config_path = Path(cm.config_path)
    profile_path = config_path.resolve().parent / "profiles" / f"{profile}.yaml"
    _print_check("config.file", config_path.exists(), str(config_path))
    _print_check("config.profile", profile_path.exists(), f"{profile} -> {profile_path}")
    _print_check("config.llm.backend", bool(cm.get("llm.backend")), str(cm.get("llm.backend")))
    _print_check("config.llm.resolved", llm_resolution.compatible, f"{llm_resolution.backend} -> {llm_resolution.config.get('base_url')}")
    _print_check("config.llm.recommended", llm_resolution.recommended, str(llm_resolution.recommended))
    for warning in llm_resolution.warnings:
        _print_check("config.llm.warning", False, warning)
    _print_check("config.workspace.root", bool(cm.get("workspace.root")), str(cm.get("workspace.root")))
    return {
        "config_path": str(config_path),
        "platform_profile": profile,
        "profile_path": str(profile_path),
        "llm_backend": cm.get("llm.backend"),
        "llm_resolved": llm_resolution.to_config(),
        "workspace_root": cm.get("workspace.root"),
        "audio_asr_device": cm.get("audio.asr.device"),
    }


def check_audio_plan(asr_device: str | None = "auto") -> dict:
    from zulong.utils.device import resolve_audio_model_devices

    plan = resolve_audio_model_devices(asr_device, prefer_gpu=True)
    _print_check("audio.device.requested", True, str(plan.get("requested")))
    _print_check("audio.device.sensevoice", True, str(plan.get("sensevoice")))
    _print_check("audio.device.whisper", True, str(plan.get("whisper")))
    _print_check("audio.device.yamnet", True, str(plan.get("yamnet")))
    for note in plan.get("notes", []):
        _print_check("audio.device.note", True, note)
    return plan


def check_tts() -> dict:
    from zulong.tts.cosyvoice_config import get_external_runtime_config

    runtime = get_external_runtime_config()
    _print_check("tts.cosyvoice.python", bool(runtime.get("integrated_python_path")), runtime.get("integrated_python_path", ""))
    for key in ("code_path", "model_dir", "prompt_audio"):
        value = runtime.get(key, "")
        _print_check(f"tts.cosyvoice.{key}", Path(value).exists(), value or "not configured")
    _print_check("tts.cosyvoice.server_url", bool(runtime.get("server_url")), runtime.get("server_url", ""))
    return runtime


def check_model_assets() -> dict:
    try:
        import importlib.util

        module_path = ROOT / "scripts" / "doctor_models.py"
        spec = importlib.util.spec_from_file_location("zulong_doctor_models", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {module_path}")
        doctor_models = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = doctor_models
        spec.loader.exec_module(doctor_models)

        results = doctor_models.run_checks(doctor_models.build_asset_specs())
    except Exception as exc:
        _print_check("models.assets", False, f"doctor_models failed: {exc}")
        return {"available": False, "error": str(exc), "assets": []}

    warn_count = 0
    error_count = 0
    for item in results:
        if item["ok"]:
            continue
        if item["required"]:
            error_count += 1
        else:
            warn_count += 1

    _print_check("models.assets", error_count == 0, f"{len(results)} checked, {warn_count} optional warnings, {error_count} errors")
    return {
        "available": error_count == 0,
        "optional_warnings": warn_count,
        "errors": error_count,
        "assets": results,
    }


def check_commands() -> dict:
    from zulong.l0.audio.ffmpeg_resolver import ffmpeg_diagnostic
    from zulong.workspace.vscode_launcher import resolve_vscode_command

    commands = ["python", "python3", "node", "npm", "git", "ffmpeg"]
    found = {cmd: shutil.which(cmd) for cmd in commands}
    ffmpeg = ffmpeg_diagnostic()
    if ffmpeg.get("available"):
        found["ffmpeg"] = ffmpeg.get("path")
    vscode = resolve_vscode_command()
    found["code"] = vscode.get("command") if vscode.get("ok") else None
    for cmd, path in found.items():
        _print_check(f"command.{cmd}", path is not None, path or "not found")
    if not ffmpeg.get("available"):
        _print_check("command.ffmpeg.hint", False, str(ffmpeg.get("hint", "")))
    if not vscode.get("ok"):
        _print_check("command.code.hint", False, str(vscode.get("error", "")))
    return found


def check_modules(full: bool = False) -> dict:
    modules = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "yaml",
        "numpy",
        "torch",
        "cv2",
        "soundfile",
    ]
    if full:
        modules.extend([
            "transformers",
            "sentence_transformers",
            "whisper",
            "sherpa_onnx",
            "kokoro",
            "pyaudio",
            "onnxruntime",
        ])
    availability = {name: _module_available(name) for name in modules}
    for name, ok in availability.items():
        _print_check(f"module.{name}", ok)
    return availability


def check_paths() -> dict:
    paths = {
        "root": ROOT,
        "config": ROOT / "config" / "zulong_config.yaml",
        "profiles": ROOT / "config" / "profiles",
        "data": ROOT / "data",
        "models": ROOT / "models",
        "logs": ROOT / "logs",
    }
    for name, path in paths.items():
        _print_check(f"path.{name}", path.exists(), str(path))
    return {name: str(path) for name, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="ZULONG cross-platform runtime doctor")
    parser.add_argument("--full", action="store_true", help="Check optional model/audio packages too")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args()

    _add_repo_to_path()

    print("ZULONG Doctor")
    print(f"Root: {ROOT}")
    print("")

    summary = {}
    summary.update(check_runtime())
    summary["config"] = check_config()
    summary["audio_devices"] = check_audio_plan(summary["config"].get("audio_asr_device", "auto"))
    summary["tts"] = check_tts()
    summary["models"] = check_model_assets()
    summary["commands"] = check_commands()
    summary["modules"] = check_modules(full=args.full)
    summary["paths"] = check_paths()

    if args.json:
        print("")
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
