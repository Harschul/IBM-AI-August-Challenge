#!/usr/bin/env python3
"""Launch the final Streamlit demo from any working directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "src" / "frontend" / "app.py"


def main() -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit is not installed. Run:\n"
            "  python -m pip install -r requirements-frontend.txt\n\n"
            "For the reported PPO demo also install:\n"
            "  python -m pip install -r requirements-frontend-rl.txt",
            file=sys.stderr,
        )
        return 2
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP)], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
