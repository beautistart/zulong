"""Cross-platform OpenCV camera backend selection and probing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2

from zulong.utils.runtime_env import get_runtime_environment


@dataclass(frozen=True)
class CameraBackend:
    id: int
    name: str


def get_camera_backends(preferred: str = "auto") -> List[CameraBackend]:
    """Return OpenCV camera backends in platform-friendly priority order."""
    preferred = (preferred or "auto").lower()
    env = get_runtime_environment()

    all_backends: Dict[str, CameraBackend] = {
        "auto": CameraBackend(cv2.CAP_ANY, "Auto"),
        "dshow": CameraBackend(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY), "DirectShow"),
        "msmf": CameraBackend(getattr(cv2, "CAP_MSMF", cv2.CAP_ANY), "Media Foundation"),
        "v4l2": CameraBackend(getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), "V4L2"),
        "avfoundation": CameraBackend(getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY), "AVFoundation"),
    }

    if preferred != "auto" and preferred in all_backends:
        return [all_backends[preferred], all_backends["auto"]]

    if env.os_family == "windows":
        keys = ["dshow", "msmf", "auto"]
    elif env.os_family == "linux":
        keys = ["v4l2", "auto"]
    elif env.os_family == "macos":
        keys = ["avfoundation", "auto"]
    else:
        keys = ["auto"]

    backends: List[CameraBackend] = []
    seen = set()
    for key in keys:
        backend = all_backends[key]
        if backend.id in seen and backend.name != "Auto":
            continue
        seen.add(backend.id)
        backends.append(backend)
    return backends


def open_camera(device_index: int, preferred_backend: str = "auto"):
    """Try platform backends and return (capture, backend_name)."""
    for backend in get_camera_backends(preferred_backend):
        cap = cv2.VideoCapture(device_index, backend.id)
        if cap.isOpened():
            return cap, backend.name
        cap.release()
    return None, ""


def safe_set_camera_property(cap, prop: int, value: float) -> bool:
    """Best-effort camera property set. Unsupported properties are non-fatal."""
    try:
        return bool(cap.set(prop, value))
    except Exception:
        return False


def probe_cameras(
    scan_range: int = 10,
    preferred_backend: str = "auto",
    read_frame: bool = True,
) -> List[dict]:
    """Probe available cameras without starting the runtime capture loop."""
    cameras = []
    for device_index in range(max(0, int(scan_range))):
        cap, backend_name = open_camera(device_index, preferred_backend)
        if cap is None:
            continue
        try:
            ret = False
            frame = None
            if read_frame:
                ret, frame = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cameras.append({
                "index": device_index,
                "backend": backend_name,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_read": bool(ret),
                "frame_shape": list(frame.shape) if ret and frame is not None else None,
            })
        finally:
            cap.release()
    return cameras
