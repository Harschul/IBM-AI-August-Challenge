"""Shared visual theme for the integrated frontend.

The aim is to stay visually aligned with the repository's existing muted,
minimal render language: light background, restrained palette, crisp links,
and clean annotation rather than a dashboard-heavy aesthetic.
"""

from __future__ import annotations

BACKGROUND = "#f8fafc"
PANEL = "#ffffff"
TEXT = "#111827"
MUTED = "#475569"
GRID = "#e2e8f0"
EARTH_SURFACE = "#e7edf4"
EARTH_EDGE = "#94a3b8"
GROUND = "#10b981"
SCIENCE = "#2563eb"
LEO = "#7c3aed"
GEO = "#f59e0b"
LINK = "#0f172a"
LINK_SOFT = "rgba(15, 23, 42, 0.28)"
GROUND_LINK = "rgba(16, 185, 129, 0.38)"
PACKET = "#ef4444"
PACKET_URGENT = "#dc2626"
SELECTED = "#06b6d4"
SUCCESS = "#16a34a"
WARNING = "#d97706"
FAIL = "#dc2626"
TEMPORAL = "#0ea5e9"
RL = "#8b5cf6"
FALLBACK = "#ef4444"

ROLE_COLOURS = {
    "SCIENCE": SCIENCE,
    "LEO": LEO,
    "GEO": GEO,
    "GROUND": GROUND,
}
