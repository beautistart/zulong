#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Camera diagnostics for ZULONG P1 cross-platform adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OpenCV camera backends")
    parser.add_argument("--scan-range", type=int, default=3, help="Number of device indexes to probe")
    parser.add_argument("--backend", default="auto", help="Preferred backend: auto/dshow/msmf/v4l2/avfoundation")
    parser.add_argument("--no-read", action="store_true", help="Only open devices, do not read a frame")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    from zulong.l0.devices.camera_backend import get_camera_backends, probe_cameras
    from zulong.utils.runtime_env import get_runtime_context

    runtime = get_runtime_context()
    backends = get_camera_backends(args.backend)
    cameras = probe_cameras(
        scan_range=args.scan_range,
        preferred_backend=args.backend,
        read_frame=not args.no_read,
    )

    summary = {
        "runtime": runtime,
        "backends": [{"id": b.id, "name": b.name} for b in backends],
        "cameras": cameras,
    }

    print("ZULONG Camera Doctor")
    print(f"Platform: {runtime['os_name']} ({runtime['os_family']})")
    print("Backends: " + ", ".join(b.name for b in backends))
    if cameras:
        for cam in cameras:
            print(
                f"[OK] camera.{cam['index']} backend={cam['backend']} "
                f"size={cam['width']}x{cam['height']} fps={cam['fps']} "
                f"frame_read={cam['frame_read']}"
            )
    else:
        print("[WARN] no camera detected")
        if runtime["os_family"] == "macos":
            print("Hint: allow camera permission in System Settings > Privacy & Security > Camera.")
        elif runtime["os_family"] == "linux":
            print("Hint: check /dev/video* permissions and V4L2 support.")
        elif runtime["os_family"] == "windows":
            print("Hint: check Windows camera privacy settings and device drivers.")

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
