#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Syntax-only Python source checker.

This intentionally avoids py_compile so Windows runs are not affected by
locked __pycache__ files while still catching syntax errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def check_file(raw_path: str) -> None:
    path = resolve_path(raw_path)
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Python source syntax without writing bytecode")
    parser.add_argument("paths", nargs="+", help="Python files to check")
    parser.add_argument("--quiet", action="store_true", help="Only print failures")
    args = parser.parse_args()

    failed = False
    for raw_path in args.paths:
        try:
            check_file(raw_path)
        except Exception as exc:  # noqa: BLE001 - CLI should report any syntax/read failure.
            failed = True
            print(f"[FAIL] {raw_path}: {exc}", file=sys.stderr)
        else:
            if not args.quiet:
                print(f"[OK] {raw_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
