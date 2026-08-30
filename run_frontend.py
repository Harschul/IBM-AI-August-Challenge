#!/usr/bin/env python3
"""Convenience launcher for the Streamlit frontend."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "src/frontend/app.py"])


if __name__ == "__main__":
    raise SystemExit(main())
