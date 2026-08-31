"""Lightweight 2D Plotly fallbacks for tests and static exports.

The live Streamlit application uses the browser-side canvas in ``client_ui``
for smooth animation.  These helpers deliberately avoid 3D rendering.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from src.frontend.replay import ReplayData, node_label
from src.integration.config import GEO_IDS, GROUND_IDS, LEO_IDS, SCIENCE_IDS, node_role

ROLE_COLOURS = {
    "SCIENCE": "#111827",
    "LEO": "#155eef",
    "GEO": "#c25b00",
    "GROUND": "#087f5b",
}
DISPLAY_RADII = {"SCIENCE": 1.18, "LEO": 1.34, "GEO": 1.82, "GROUND": 0.94}


def _display_xy(replay: ReplayData, node_id: int, time_s: float) -> tuple[float, float]:
    xyz = np.asarray(replay.node_position_3d(node_id, time_s), dtype=float)
    norm = float(np.linalg.norm(xyz)) or 1.0
    scale = DISPLAY_RADII[node_role(node_id)]
    return scale * float(xyz[0]) / norm, scale * float(xyz[1]) / norm


def build_orbital_figure(
    replay: ReplayData,
    time_s: float,
    *,
    selected_bundle_id: str | None = None,
    **_: object,
) -> go.Figure:
    """Return a flat orthographic orbital projection; no WebGL 3D scene."""
    fig = go.Figure()
    theta = np.linspace(0.0, 2.0 * math.pi, 180)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode="lines",
        fill="toself", fillcolor="#e8f0f8", line={"color": "#b9c9d9", "width": 1.5},
        hoverinfo="skip", showlegend=False,
    ))
    for radius in (1.18, 1.34, 1.82):
        fig.add_trace(go.Scatter(
            x=radius * np.cos(theta), y=0.78 * radius * np.sin(theta), mode="lines",
            line={"color": "#d7e1eb", "width": 1, "dash": "dot"},
            hoverinfo="skip", showlegend=False,
        ))

    for role, ids in (
        ("SCIENCE", SCIENCE_IDS), ("LEO", LEO_IDS), ("GEO", GEO_IDS), ("GROUND", GROUND_IDS)
    ):
        xy = [_display_xy(replay, node_id, time_s) for node_id in ids]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in xy], y=[p[1] for p in xy], mode="markers+text",
            text=[node_label(node_id, replay.config) if role in {"SCIENCE", "GEO", "GROUND"} else "" for node_id in ids],
            textposition="top center",
            marker={"size": 9 if role != "LEO" else 7, "color": ROLE_COLOURS[role], "line": {"color": "white", "width": 1}},
            hovertemplate="%{text}<extra></extra>", showlegend=False,
        ))

    selected = replay.selected_bundle(selected_bundle_id)
    if selected is not None:
        active = next((p for p in replay.active_packets(time_s) if p.bundle_id == selected.bundle_id), None)
        if active is not None:
            src = _display_xy(replay, active.source_id, time_s)
            dst = _display_xy(replay, active.destination_id, time_s)
            alpha = max(0.0, min(1.0, float(active.progress)))
            fig.add_trace(go.Scatter(
                x=[src[0] + alpha * (dst[0] - src[0])],
                y=[src[1] + alpha * (dst[1] - src[1])],
                mode="markers", marker={"size": 10, "color": "#ffb000"},
                hovertemplate=selected.bundle_id + "<extra></extra>", showlegend=False,
            ))

    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis={"visible": False, "range": [-2.05, 2.05], "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False, "range": [-2.05, 2.05]},
        showlegend=False,
    )
    return fig


def build_topology_figure(
    replay: ReplayData,
    time_s: float,
    *,
    selected_bundle_id: str | None = None,
    **_: object,
) -> go.Figure:
    """Show fixed nodes and only contacts that are available at ``time_s``."""
    topo = replay.topology_positions
    fig = go.Figure()
    for contact in replay.active_contacts(time_s):
        a, b = topo[contact.source_id], topo[contact.destination_id]
        fig.add_trace(go.Scatter(
            x=[a[0], b[0]], y=[a[1], b[1]], mode="lines",
            line={"color": "rgba(100,116,139,.28)", "width": 1},
            hoverinfo="skip", showlegend=False,
        ))

    selected = replay.selected_bundle(selected_bundle_id)
    if selected is not None:
        for i in range(len(selected.path) - 1):
            a, b = topo[selected.path[i]], topo[selected.path[i + 1]]
            fig.add_trace(go.Scatter(
                x=[a[0], b[0]], y=[a[1], b[1]], mode="lines",
                line={"color": "#155eef", "width": 3, "dash": "dot"},
                hoverinfo="skip", showlegend=False,
            ))
        for attempt in selected.attempts:
            if attempt.success:
                continue
            a, b = topo[attempt.holder_id], topo[attempt.destination_id]
            fig.add_trace(go.Scatter(
                x=[a[0], b[0]], y=[a[1], b[1]], mode="lines",
                line={"color": "#c1123f", "width": 3, "dash": "dash"},
                hoverinfo="skip", showlegend=False,
            ))

    for role, ids in (
        ("SCIENCE", SCIENCE_IDS), ("LEO", LEO_IDS), ("GEO", GEO_IDS), ("GROUND", GROUND_IDS)
    ):
        fig.add_trace(go.Scatter(
            x=[topo[node_id][0] for node_id in ids],
            y=[topo[node_id][1] for node_id in ids],
            mode="markers+text",
            text=[node_label(node_id, replay.config) for node_id in ids],
            textposition="top center",
            marker={"size": 11, "color": "white", "line": {"color": ROLE_COLOURS[role], "width": 2}},
            hovertemplate="%{text}<extra></extra>", showlegend=False,
        ))
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10}, paper_bgcolor="white", plot_bgcolor="white",
        xaxis={"visible": False, "range": [-2.8, 2.8]}, yaxis={"visible": False, "range": [-2.8, 2.8]}, showlegend=False,
    )
    return fig
