"""Plotly figure builders for the integrated frontend."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from src.frontend.theme import (
    BACKGROUND,
    EARTH_EDGE,
    EARTH_SURFACE,
    FALLBACK,
    FAIL,
    GEO,
    GROUND,
    GROUND_LINK,
    GRID,
    LEO,
    LINK_SOFT,
    MUTED,
    PANEL,
    PACKET,
    PACKET_URGENT,
    RL,
    ROLE_COLOURS,
    SCIENCE,
    SELECTED,
    TEMPORAL,
    TEXT,
)
from src.integration.config import GEO_IDS, GROUND_IDS, LEO_IDS, SCIENCE_IDS, node_role
from src.frontend.replay import ReplayData, node_label


def _earth_surface(radius_km: float) -> go.Surface:
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 18)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    surface = np.zeros_like(x)
    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=surface,
        colorscale=[[0.0, EARTH_SURFACE], [1.0, EARTH_SURFACE]],
        showscale=False,
        opacity=0.82,
        hoverinfo="skip",
        contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
        lighting={"ambient": 0.8, "diffuse": 0.2, "specular": 0.05, "roughness": 0.95},
        name="Earth",
    )


def _contact_segments_3d(replay: ReplayData, time_s: float):
    sat_x, sat_y, sat_z = [], [], []
    grd_x, grd_y, grd_z = [], [], []
    for contact in replay.active_contacts(time_s):
        a = replay.node_position_3d(contact.source_id, time_s)
        b = replay.node_position_3d(contact.destination_id, time_s)
        xs = [float(a[0]), float(b[0]), None]
        ys = [float(a[1]), float(b[1]), None]
        zs = [float(a[2]), float(b[2]), None]
        if contact.destination_id in GROUND_IDS or contact.source_id in GROUND_IDS:
            grd_x.extend(xs)
            grd_y.extend(ys)
            grd_z.extend(zs)
        else:
            sat_x.extend(xs)
            sat_y.extend(ys)
            sat_z.extend(zs)
    return (sat_x, sat_y, sat_z), (grd_x, grd_y, grd_z)


def _contact_segments_2d(replay: ReplayData, time_s: float):
    topo = replay.topology_positions
    sat_x, sat_y = [], []
    grd_x, grd_y = [], []
    for contact in replay.active_contacts(time_s):
        a = topo[contact.source_id]
        b = topo[contact.destination_id]
        xs = [a[0], b[0], None]
        ys = [a[1], b[1], None]
        if contact.destination_id in GROUND_IDS or contact.source_id in GROUND_IDS:
            grd_x.extend(xs)
            grd_y.extend(ys)
        else:
            sat_x.extend(xs)
            sat_y.extend(ys)
    return (sat_x, sat_y), (grd_x, grd_y)


def _selected_path_segments(replay: ReplayData, bundle_id: str | None, time_s: float):
    bundle = replay.selected_bundle(bundle_id)
    if bundle is None:
        return [], []
    xyz_segments = []
    xy_segments = []
    topo = replay.topology_positions
    for hop in bundle.hops:
        a3 = replay.node_position_3d(hop.source_id, time_s)
        b3 = replay.node_position_3d(hop.destination_id, time_s)
        xyz_segments.append(((float(a3[0]), float(b3[0])), (float(a3[1]), float(b3[1])), (float(a3[2]), float(b3[2]))))
        a2 = topo[hop.source_id]
        b2 = topo[hop.destination_id]
        xy_segments.append(((a2[0], b2[0]), (a2[1], b2[1])))
    return xyz_segments, xy_segments


def build_orbital_figure(
    replay: ReplayData,
    time_s: float,
    *,
    selected_bundle_id: str | None = None,
    show_trails: bool = True,
    trail_seconds: float = 180.0,
) -> go.Figure:
    frame_index = replay.frame_index_at(time_s)
    snapshot = replay.snapshots[frame_index]
    fig = go.Figure()
    fig.add_trace(_earth_surface(replay.config.earth_radius_km))

    positions = snapshot.positions
    limit = float(np.linalg.norm(np.asarray(positions), axis=1).max() * 1.18)

    if show_trails:
        start_index = replay.frame_index_at(max(0.0, time_s - trail_seconds))
        for node_id in [*SCIENCE_IDS, *LEO_IDS, *GEO_IDS]:
            trail = np.stack([replay.snapshots[i].positions[node_id] for i in range(start_index, frame_index + 1)])
            fig.add_trace(
                go.Scatter3d(
                    x=trail[:, 0], y=trail[:, 1], z=trail[:, 2],
                    mode="lines",
                    line={"color": ROLE_COLOURS[node_role(node_id)], "width": 3 if node_id in SCIENCE_IDS else 1},
                    opacity=0.25,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Active links.
    sat_seg, grd_seg = _contact_segments_3d(replay, time_s)
    fig.add_trace(
        go.Scatter3d(
            x=sat_seg[0], y=sat_seg[1], z=sat_seg[2],
            mode="lines",
            line={"color": LINK_SOFT, "width": 3},
            hoverinfo="skip",
            name="Active sat links",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=grd_seg[0], y=grd_seg[1], z=grd_seg[2],
            mode="lines",
            line={"color": GROUND_LINK, "width": 4},
            hoverinfo="skip",
            name="Ground links",
            showlegend=False,
        )
    )

    # Highlight selected bundle path.
    xyz_segments, _ = _selected_path_segments(replay, selected_bundle_id, time_s)
    for xs, ys, zs in xyz_segments:
        fig.add_trace(
            go.Scatter3d(
                x=list(xs), y=list(ys), z=list(zs),
                mode="lines",
                line={"color": SELECTED, "width": 6},
                opacity=0.55,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Node markers by role.
    node_ids = [*SCIENCE_IDS, *LEO_IDS, *GEO_IDS]
    for role, ids in [("SCIENCE", list(SCIENCE_IDS)), ("LEO", list(LEO_IDS)), ("GEO", list(GEO_IDS))]:
        xyz = np.stack([positions[node_id] for node_id in ids])
        labels = [node_label(node_id, replay.config) for node_id in ids]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers+text" if role in ("SCIENCE", "GEO") else "markers",
                text=labels if role in ("SCIENCE", "GEO") else None,
                textposition="top center",
                marker={
                    "size": 8 if role == "LEO" else 11,
                    "color": ROLE_COLOURS[role],
                    "line": {"color": PANEL, "width": 2},
                    "symbol": "circle",
                },
                hovertemplate="%{text}<extra></extra>" if role in ("SCIENCE", "GEO") else None,
                name=role,
                showlegend=False,
            )
        )

    ground_xyz = np.stack([replay.node_position_3d(node_id, time_s) for node_id in GROUND_IDS])
    fig.add_trace(
        go.Scatter3d(
            x=ground_xyz[:, 0], y=ground_xyz[:, 1], z=ground_xyz[:, 2],
            mode="markers+text",
            text=[node_label(node_id, replay.config) for node_id in GROUND_IDS],
            textposition="bottom center",
            marker={"size": 7, "color": GROUND, "line": {"color": PANEL, "width": 1}},
            name="Ground",
            showlegend=False,
        )
    )

    # Packets in flight.
    packets = replay.active_packets(time_s)
    if packets:
        fig.add_trace(
            go.Scatter3d(
                x=[packet.xyz[0] for packet in packets],
                y=[packet.xyz[1] for packet in packets],
                z=[packet.xyz[2] for packet in packets],
                mode="markers+text",
                text=[packet.bundle_id for packet in packets],
                textposition="top center",
                marker={
                    "size": [8 + 4 * packet.priority for packet in packets],
                    "color": [FAIL if not packet.will_succeed else (FALLBACK if packet.fallback_used else (RL if packet.actual_algorithm == "rl" else TEMPORAL)) for packet in packets],
                    "symbol": "diamond",
                    "line": {"color": PANEL, "width": 1},
                },
                customdata=[[packet.requested_algorithm, packet.actual_algorithm, packet.fallback_reason or "", packet.will_succeed, packet.failure_probability, packet.success_draw] for packet in packets],
                hovertemplate="bundle=%{text}<br>requested=%{customdata[0]}<br>actual=%{customdata[1]}<br>fallback=%{customdata[2]}<br>success=%{customdata[3]}<br>p_fail=%{customdata[4]:.3f}<br>draw=%{customdata[5]:.3f}<extra></extra>",
                name="Packets",
                showlegend=False,
            )
        )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "bgcolor": BACKGROUND,
            "camera": {"eye": {"x": 1.55, "y": -1.45, "z": 0.75}},
            "aspectmode": "cube",
            "xaxis_range": [-limit, limit],
            "yaxis_range": [-limit, limit],
            "zaxis_range": [-limit, limit],
        },
        title={
            "text": f"Orbital view · t = {time_s:,.0f}s · requested = {replay.requested_algorithm_at(time_s)} · actual = {replay.actual_algorithm_at(time_s)}",
            "x": 0.02,
            "font": {"size": 16, "color": TEXT},
        },
        height=620,
    )
    return fig


def build_topology_figure(
    replay: ReplayData,
    time_s: float,
    *,
    selected_bundle_id: str | None = None,
) -> go.Figure:
    topo = replay.topology_positions
    fig = go.Figure()

    sat_seg, grd_seg = _contact_segments_2d(replay, time_s)
    fig.add_trace(
        go.Scatter(
            x=sat_seg[0], y=sat_seg[1], mode="lines",
            line={"color": LINK_SOFT, "width": 2.5},
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grd_seg[0], y=grd_seg[1], mode="lines",
            line={"color": GROUND_LINK, "width": 3.5},
            hoverinfo="skip", showlegend=False,
        )
    )

    _, xy_segments = _selected_path_segments(replay, selected_bundle_id, time_s)
    for xs, ys in xy_segments:
        fig.add_trace(
            go.Scatter(
                x=list(xs), y=list(ys), mode="lines",
                line={"color": SELECTED, "width": 5},
                opacity=0.6,
                hoverinfo="skip", showlegend=False,
            )
        )

    grouped = {
        "SCIENCE": list(SCIENCE_IDS),
        "LEO": list(LEO_IDS),
        "GEO": list(GEO_IDS),
        "GROUND": list(GROUND_IDS),
    }
    for role, ids in grouped.items():
        fig.add_trace(
            go.Scatter(
                x=[topo[node_id][0] for node_id in ids],
                y=[topo[node_id][1] for node_id in ids],
                mode="markers+text",
                text=[node_label(node_id, replay.config) for node_id in ids],
                textposition="top center",
                marker={
                    "size": 24 if role != "LEO" else 18,
                    "color": ROLE_COLOURS[role],
                    "line": {"color": PANEL, "width": 2},
                },
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    packets = replay.active_packets(time_s)
    if packets:
        fig.add_trace(
            go.Scatter(
                x=[packet.topology_xy[0] for packet in packets],
                y=[packet.topology_xy[1] for packet in packets],
                mode="markers+text",
                text=[packet.bundle_id for packet in packets],
                textposition="top center",
                marker={
                    "size": [11 + 8 * packet.priority for packet in packets],
                    "color": [FAIL if not packet.will_succeed else (FALLBACK if packet.fallback_used else (RL if packet.actual_algorithm == "rl" else TEMPORAL)) for packet in packets],
                    "symbol": "diamond",
                    "line": {"color": PANEL, "width": 1},
                },
                customdata=[[packet.requested_algorithm, packet.actual_algorithm, packet.fallback_reason or "", packet.will_succeed, packet.failure_probability, packet.success_draw] for packet in packets],
                hovertemplate="bundle=%{text}<br>requested=%{customdata[0]}<br>actual=%{customdata[1]}<br>fallback=%{customdata[2]}<br>success=%{customdata[3]}<br>p_fail=%{customdata[4]:.3f}<br>draw=%{customdata[5]:.3f}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        margin={"l": 8, "r": 8, "t": 42, "b": 8},
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,
        xaxis={"visible": False, "range": [-2.8, 2.8]},
        yaxis={"visible": False, "range": [-2.8, 2.8], "scaleanchor": "x", "scaleratio": 1},
        title={
            "text": f"Network topology · requested={replay.requested_algorithm_at(time_s)} · actual={replay.actual_algorithm_at(time_s)} · {len(replay.active_contacts(time_s))} links",
            "x": 0.02,
            "font": {"size": 16, "color": TEXT},
        },
        height=620,
        annotations=[
            {"x": -2.45, "y": 2.62, "xref": "x", "yref": "y", "text": ("Interactive switch mode — not part of reported benchmark" if replay.mode == "interactive_switch" else f"Reported experiment seed offset {replay.seed_offset}"), "showarrow": False, "font": {"size": 11, "color": MUTED}, "align": "left"}
        ],
    )
    return fig
