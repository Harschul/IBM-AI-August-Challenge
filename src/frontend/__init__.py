"""Integrated orbital/network frontend package.

The Streamlit app is intentionally not imported at package import time so replay,
figure and test code can be used without forcing the UI runtime dependency.
"""


def main():
    from .app import main as app_main

    return app_main()


__all__ = ["main"]
