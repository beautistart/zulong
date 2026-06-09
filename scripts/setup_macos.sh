#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PATH="${ZULONG_VENV:-zulong_env}"
SKIP_INSTALL="${ZULONG_SKIP_INSTALL:-false}"

echo "ZULONG macOS setup"
echo "Root: $ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is intended for macOS." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Install Python 3.10-3.12 first." >&2
  exit 1
fi

if command -v brew >/dev/null 2>&1; then
  echo "Checking PortAudio via Homebrew..."
  brew list portaudio >/dev/null 2>&1 || brew install portaudio
else
  echo "Homebrew not found. PyAudio may need PortAudio installed manually." >&2
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Creating virtual environment: $VENV_PATH"
  python3 -m venv "$VENV_PATH"
fi

PYTHON="$VENV_PATH/bin/python"
if [[ "$SKIP_INSTALL" != "true" ]]; then
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
  echo "Installing macOS requirements..."
  "$PYTHON" -m pip install -r requirements-macos.txt
fi

echo "Running doctor..."
"$PYTHON" scripts/doctor.py

echo "macOS setup complete."
