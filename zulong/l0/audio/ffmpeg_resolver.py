"""Cross-platform ffmpeg resolver for audio decode/transcode paths."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional


def find_ffmpeg() -> Optional[str]:
    """Find an ffmpeg executable from bundled paths, PATH, or imageio-ffmpeg."""
    for candidate in _bundled_candidates():
        if candidate.exists():
            return str(candidate)

    system = shutil.which("ffmpeg")
    if system:
        return system

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    return None


def ffmpeg_diagnostic() -> Dict[str, object]:
    exe = find_ffmpeg()
    return {
        "available": bool(exe),
        "path": exe,
        "hint": "" if exe else (
            "Install ffmpeg or install the Python package imageio-ffmpeg. "
            "Windows can also place ffmpeg.exe under the project bin directory."
        ),
    }


def _bundled_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[3]
    names = ["ffmpeg.exe"] if os.name == "nt" else ["ffmpeg"]
    return [root / "bin" / name for name in names]
