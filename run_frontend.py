#!/usr/bin/env python3
"""Convenience launcher for the Streamlit frontend."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit is not installed. From the repository root run:\n"
            "  python -m pip install -r requirements-frontend.txt\n\n"
            "For true RL checkpoint execution also run:\n"
            "  python -m pip install -r requirements-frontend-rl.txt",
            file=sys.stderr,
        )
        return 2

    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", "src/frontend/app.py"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
