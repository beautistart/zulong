"""Cross-platform PyAudio device inspection helpers."""

from __future__ import annotations

from typing import Dict, List

from zulong.utils.runtime_env import get_runtime_environment


def audio_platform_hint(error: Exception | None = None) -> str:
    env = get_runtime_environment()
    base = f"当前平台: {env.os_name}。"
    if env.os_family == "windows":
        hint = "请确认系统麦克风/扬声器权限已开启，设备驱动正常。"
    elif env.os_family == "macos":
        hint = "请在 System Settings > Privacy & Security 中允许终端或应用访问 Microphone。"
    elif env.os_family == "linux":
        hint = "请确认已安装 PortAudio，并检查 ALSA/PulseAudio/PipeWire 设备可见性。"
    else:
        hint = "请确认 PortAudio/PyAudio 可用且音频设备已连接。"
    if error:
        return f"{base}{hint} 原始错误: {error}"
    return f"{base}{hint}"


def list_audio_devices() -> Dict[str, object]:
    """List input/output audio devices using PyAudio.

    The function returns structured diagnostics instead of raising so callers
    can degrade gracefully when PyAudio, PortAudio, permissions, or devices are
    unavailable.
    """
    try:
        import pyaudio
    except Exception as exc:
        return {
            "available": False,
            "error": f"PyAudio unavailable: {exc}",
            "hint": audio_platform_hint(exc),
            "input_devices": [],
            "output_devices": [],
        }

    audio = None
    try:
        audio = pyaudio.PyAudio()
        input_devices: List[dict] = []
        output_devices: List[dict] = []
        for index in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(index)
                device = {
                    "index": index,
                    "name": info.get("name", ""),
                    "host_api": info.get("hostApi"),
                    "max_input_channels": int(info.get("maxInputChannels", 0)),
                    "max_output_channels": int(info.get("maxOutputChannels", 0)),
                    "default_sample_rate": int(info.get("defaultSampleRate", 0)),
                }
                if device["max_input_channels"] > 0:
                    input_devices.append(device)
                if device["max_output_channels"] > 0:
                    output_devices.append(device)
            except Exception:
                continue

        defaults = {}
        try:
            defaults["input"] = audio.get_default_input_device_info().get("index")
        except Exception:
            defaults["input"] = None
        try:
            defaults["output"] = audio.get_default_output_device_info().get("index")
        except Exception:
            defaults["output"] = None

        return {
            "available": True,
            "error": "",
            "hint": audio_platform_hint(),
            "input_devices": input_devices,
            "output_devices": output_devices,
            "default_devices": defaults,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "hint": audio_platform_hint(exc),
            "input_devices": [],
            "output_devices": [],
        }
    finally:
        if audio is not None:
            try:
                audio.terminate()
            except Exception:
                pass
