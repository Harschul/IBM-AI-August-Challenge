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
            "Streamlit / PPO runtime dependencies are not installed. Run:\n"
            "  python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP)], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
