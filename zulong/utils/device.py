"""Cross-platform accelerator selection helpers."""

from __future__ import annotations

import platform
from typing import Any, Dict


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def is_mps_available() -> bool:
    torch = _torch()
    if torch is None:
        return False
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if not mps:
        return False
    try:
        return bool(mps.is_available() and mps.is_built())
    except Exception:
        return False


def resolve_device(requested: str | None = "auto", *, prefer_gpu: bool = True) -> str:
    """Resolve cuda/mps/cpu in a Windows/Linux/macOS friendly order."""
    requested = (requested or "auto").lower()
    torch = _torch()
    if requested in ("cpu", "cuda", "mps"):
        if requested == "cuda" and not (torch and torch.cuda.is_available()):
            return "mps" if prefer_gpu and is_mps_available() else "cpu"
        if requested == "mps" and not is_mps_available():
            return "cuda" if prefer_gpu and torch and torch.cuda.is_available() else "cpu"
        return requested

    if not prefer_gpu:
        return "cpu"
    if torch and torch.cuda.is_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def torch_device(requested: str | None = "auto", *, prefer_gpu: bool = True):
    torch = _torch()
    resolved = resolve_device(requested, prefer_gpu=prefer_gpu)
    return torch.device(resolved) if torch else resolved


def accelerator_info() -> Dict[str, Any]:
    torch = _torch()
    info: Dict[str, Any] = {
        "platform": platform.system().lower(),
        "machine": platform.machine(),
        "selected_device": resolve_device("auto"),
        "cuda_available": False,
        "mps_available": is_mps_available(),
    }
    if torch and torch.cuda.is_available():
        info["cuda_available"] = True
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_gb"] = total_mem / (1024 ** 3)
            info["gpu_free_gb"] = free_mem / (1024 ** 3)
        except Exception:
            pass
    if info["mps_available"]:
        info["gpu_name"] = info.get("gpu_name") or "Apple Metal Performance Shaders"
    return info


def empty_accelerator_cache() -> None:
    torch = _torch()
    if not torch:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if is_mps_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
