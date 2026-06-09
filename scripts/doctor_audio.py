#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio device diagnostics for ZULONG P1 cross-platform adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe PyAudio input/output devices")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    from zulong.l0.devices.audio_backend import list_audio_devices
    from zulong.l0.audio.ffmpeg_resolver import ffmpeg_diagnostic
    from zulong.utils.runtime_env import get_runtime_context

    runtime = get_runtime_context()
    devices = list_audio_devices()
    ffmpeg = ffmpeg_diagnostic()

    summary = {
        "runtime": runtime,
        "ffmpeg": ffmpeg,
        "audio": devices,
    }

    print("ZULONG Audio Doctor")
    print(f"Platform: {runtime['os_name']} ({runtime['os_family']})")
    print(f"ffmpeg: {ffmpeg.get('path') or 'not found'}")

    if not devices["available"]:
        print(f"[WARN] audio unavailable: {devices['error']}")
        print(f"Hint: {devices['hint']}")
    else:
        print(f"[OK] input devices: {len(devices['input_devices'])}")
        for dev in devices["input_devices"]:
            default = " default" if dev["index"] == devices["default_devices"].get("input") else ""
            print(
                f"  input.{dev['index']}{default}: {dev['name']} "
                f"channels={dev['max_input_channels']} sr={dev['default_sample_rate']}"
            )
        print(f"[OK] output devices: {len(devices['output_devices'])}")
        for dev in devices["output_devices"]:
            default = " default" if dev["index"] == devices["default_devices"].get("output") else ""
            print(
                f"  output.{dev['index']}{default}: {dev['name']} "
                f"channels={dev['max_output_channels']} sr={dev['default_sample_rate']}"
            )

    if not ffmpeg.get("available"):
        print(f"[WARN] {ffmpeg.get('hint')}")

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
