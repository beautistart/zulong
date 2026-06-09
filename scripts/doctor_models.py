#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model asset diagnostics for ZULONG cross-platform setup.

This script only checks paths and key files. It does not load large models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class AssetSpec:
    name: str
    path: str
    required_files: list[str] = field(default_factory=list)
    required: bool = False
    note: str = ""


def resolve_asset_path(path: str, root: Path = ROOT) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path or ""))
    candidate = Path(expanded)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def check_asset(spec: AssetSpec, root: Path = ROOT) -> dict:
    path = resolve_asset_path(spec.path, root)
    exists = path.exists()
    missing_files = []
    for file_name in spec.required_files:
        if not (path / file_name).exists():
            missing_files.append(file_name)

    ok = exists and not missing_files
    return {
        "name": spec.name,
        "path": str(path),
        "exists": exists,
        "required": spec.required,
        "required_files": spec.required_files,
        "missing_files": missing_files,
        "ok": ok,
        "status": "OK" if ok else ("ERROR" if spec.required else "WARN"),
        "note": spec.note,
    }


def build_asset_specs() -> list[AssetSpec]:
    os.environ.setdefault("ZULONG_AUTO_INIT_CONFIG", "false")
    from zulong.config.config_manager import ConfigManager, get_llm_config
    from zulong.tts.cosyvoice_config import get_external_runtime_config, get_cosyvoice3_model_path

    cm = ConfigManager()
    cosy_runtime = get_external_runtime_config()
    specs = [
        AssetSpec(
            name="SenseVoice ONNX",
            path=cm.get("audio.asr.model_path", "./models/OpenASR/sensevoice-small-onnx"),
            required_files=["model.int8.onnx", "tokens.txt"],
            required=False,
            note="Primary ASR when sherpa-onnx is enabled; Whisper can fallback.",
        ),
        AssetSpec(
            name="Kokoro TTS",
            path=cm.get("audio.tts.model_path", "./models/hexgrad/Kokoro-82M"),
            required=False,
            note="Primary local TTS; package cache or edge-tts may fallback.",
        ),
        AssetSpec(
            name="CosyVoice3",
            path=str(get_cosyvoice3_model_path()),
            required=False,
            note="Optional L3 TTS expert.",
        ),
        AssetSpec(
            name="CosyVoice2 external model",
            path=cosy_runtime.get("model_dir", ""),
            required=False,
            note="Optional external CosyVoice direct/server runtime.",
        ),
        AssetSpec(
            name="CosyVoice prompt audio",
            path=cosy_runtime.get("prompt_audio", ""),
            required=False,
            note="Optional zero-shot prompt audio.",
        ),
        AssetSpec(
            name="YOLO vision model",
            path=cm.get("vision.yolo.model_path", "models/yolov10n.pt"),
            required=False,
            note="L1-C visual attention; camera/text paths can run without it.",
        ),
        AssetSpec(
            name="ALBERT intent model",
            path=cm.get("voice_intent_classification.albert.model_path", "./models/albert-tiny-chinese"),
            required_files=["config.json"],
            required=False,
            note="L1-B fine-grained intent classifier; rules can fallback.",
        ),
        AssetSpec(
            name="BGE embedding model",
            path="./models/BAAI/bge-small-zh-v1.5",
            required_files=["config.json"],
            required=False,
            note="Memory/RAG embedding model; remote model id may be used if local files are absent.",
        ),
        AssetSpec(
            name="MediaPipe hand landmarker",
            path="hand_landmarker.task",
            required=False,
            note="Optional gesture recognition asset.",
        ),
    ]

    for backend in ("vllm", "sglang"):
        llm = get_llm_config(backend)
        model_id = str(llm.get("model_id", ""))
        if _looks_like_local_path(model_id):
            specs.append(AssetSpec(
                name=f"{backend} local model",
                path=model_id,
                required=False,
                note=f"Only required when using {backend} locally.",
            ))

    return specs


def _looks_like_local_path(value: str) -> bool:
    if not value:
        return False
    return value.startswith((".", "/", "~")) or "\\" in value or value.startswith("models/")


def run_checks(specs: Iterable[AssetSpec]) -> list[dict]:
    return [check_asset(spec) for spec in specs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ZULONG model assets without loading them")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    results = run_checks(build_asset_specs())

    print("ZULONG Model Doctor")
    for item in results:
        print(f"[{item['status']}] {item['name']} - {item['path']}")
        if item["missing_files"]:
            print(f"  missing: {', '.join(item['missing_files'])}")
        if item["note"]:
            print(f"  note: {item['note']}")

    ok = all(item["ok"] or not item["required"] for item in results)

    if args.json:
        print("")
        print(json.dumps({"ok": ok, "assets": results}, ensure_ascii=False, indent=2))

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
