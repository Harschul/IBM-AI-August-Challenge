#!/usr/bin/env python3
"""Backward-compatible wrapper for the locked final benchmark."""

from src.experiment.runner import run_algorithm  # compatibility for existing tests/imports
from run_final_benchmark import main

__all__ = ["run_algorithm", "main"]

if __name__ == "__main__":
    main()
