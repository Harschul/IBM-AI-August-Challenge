"""Visualization-only front end for the satellite network.

Orbital propagation and connectivity are deliberately delegated to nodes.py and
network.py. This module only turns immutable NetworkSnapshot objects into pixels.
"""

from pathlib import Path
import logging

import matplotlib
matplotlib.use("Agg")  # Faster/stabler for video-only rendering.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection

from network import Network
from satellite_generator import generate_satellites


BACKGROUND = "#f8fafc"
EARTH_FACE = "#e7edf4"
EARTH_EDGE = "#64748b"
LINK_COLOUR = "#111827"
MUTED_TEXT = "#475569"


def camera_basis(azimuth_deg, elevation_deg):
    """Return a 3x3 orthographic projection basis for rendering."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)

    view = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(view, right)
    return np.column_stack((right, up, view))


def project(points, basis, earth_radius_sq):
    """Project 3D model output to the 2D screen and apply visual occlusion."""
    projected = points @ basis
    xy = projected[..., :2]
    visible = (projected[..., 2] >= 0.0) | (
        np.sum(xy * xy, axis=-1) >= earth_radius_sq
    )
    return xy, visible


def visible_segments(polylines, basis, earth_radius_sq):
    """Project drawable polylines and hide screen segments behind Earth."""
    if polylines.size == 0 or polylines.shape[1] < 2:
        return np.empty((0, 2, 2)), np.empty(0, dtype=np.int32)

    xy, visible = project(polylines, basis, earth_radius_sq)
    keep = visible[:, :-1] & visible[:, 1:]
    if not np.any(keep):
        return np.empty((0, 2, 2)), np.empty(0, dtype=np.int32)

    segments = np.stack((xy[:, :-1], xy[:, 1:]), axis=2)
    owners = np.broadcast_to(np.arange(len(polylines))[:, None], keep.shape)
    return segments[keep], owners[keep]


def render_video(
    satellites,
    snapshots,
    output="satellite_network.mp4",
    fps=60,
    dpi=120,
    earth_radius=6371.0,
    trail_frames=180,
    trail_stride=2,
    azimuth=-55.0,
    elevation=22.0,
    camera_rotation=90.0,
    link_samples=12,
    crf=16,
    preset="slow",
):
    """Render already-simulated snapshots. No propagation/connectivity occurs here."""
    snapshots = tuple(snapshots)
    if not snapshots:
        raise ValueError("snapshots cannot be empty")

    frame_count = len(snapshots)
    satellite_count = len(satellites)
    earth_radius_sq = earth_radius * earth_radius

    # Read-only model output copied into one rendering-friendly array.
    positions = np.stack([snapshot.positions for snapshot in snapshots])

    colours = plt.get_cmap("tab20")(
        np.linspace(0.0, 1.0, satellite_count, endpoint=False)
    )
    limit = np.linalg.norm(positions, axis=2).max() * 1.08

    fig = plt.figure(figsize=(16, 9), facecolor=BACKGROUND)
    gs = fig.add_gridspec(2, 1, height_ratios=(2.2, 1.0), hspace=0.18)
    ax_orbit = fig.add_subplot(gs[0])
    ax_net = fig.add_subplot(gs[1])

    ax_orbit.set_facecolor(BACKGROUND)
    ax_orbit.set_aspect("equal")
    ax_orbit.set_xlim(-limit, limit)
    ax_orbit.set_ylim(-limit, limit)
    ax_orbit.axis("off")
    ax_orbit.set_title("ORBITAL VIEW", fontsize=12, fontweight="bold", color=MUTED_TEXT)
    ax_orbit.add_patch(
        plt.Circle(
            (0.0, 0.0),
            earth_radius,
            facecolor=EARTH_FACE,
            edgecolor=EARTH_EDGE,
            linewidth=1.2,
            zorder=1,
        )
    )

    trails = LineCollection([], linewidths=1.0, alpha=0.60, zorder=3)
    orbit_links = LineCollection(
        [], colors=LINK_COLOUR, linewidths=0.9, alpha=0.32, zorder=4
    )
    satellites_artist = ax_orbit.scatter(
        [], [], s=42, edgecolors="white", linewidths=0.7, zorder=5
    )
    ax_orbit.add_collection(trails)
    ax_orbit.add_collection(orbit_links)

    cols = int(np.ceil(np.sqrt(satellite_count)))
    rows = int(np.ceil(satellite_count / cols))
    idx = np.arange(satellite_count)
    grid_row = idx // cols
    grid_col = idx % cols
    net_pos = np.column_stack((grid_col.astype(float), -grid_row.astype(float)))

    last_row_count = satellite_count % cols
    if last_row_count:
        last = grid_row == grid_row.max()
        net_pos[last, 0] += (cols - last_row_count) / 2.0

    ax_net.set_facecolor(BACKGROUND)
    ax_net.set_xlim(-0.75, cols - 0.25)
    ax_net.set_ylim(-rows + 0.10, 0.75)
    ax_net.set_aspect("equal")
    ax_net.axis("off")
    ax_net.set_title("NETWORK TOPOLOGY", fontsize=12, fontweight="bold", color=MUTED_TEXT)

    topology_links = LineCollection(
        [], colors=LINK_COLOUR, linewidths=1.45, alpha=0.42, zorder=1
    )
    topology_nodes = ax_net.scatter(
        net_pos[:, 0],
        net_pos[:, 1],
        s=135,
        c=colours,
        edgecolors="#1f2937",
        linewidths=1.0,
        zorder=3,
    )
    ax_net.add_collection(topology_links)

    sample_t = np.linspace(0.0, 1.0, link_samples)[None, :, None]
    camera_angles = np.linspace(azimuth, azimuth + camera_rotation, frame_count)

    def draw_frame(frame):
        """Draw one immutable NetworkSnapshot."""
        snapshot = snapshots[frame]
        current = snapshot.positions
        basis = camera_basis(camera_angles[frame], elevation)

        current_xy, current_visible = project(current, basis, earth_radius_sq)
        satellites_artist.set_offsets(current_xy[current_visible])
        satellites_artist.set_facecolors(colours[current_visible])

        start = max(0, frame - trail_frames + 1)
        trail_poly = np.swapaxes(positions[start : frame + 1 : trail_stride], 0, 1)
        trail_seg, owners = visible_segments(trail_poly, basis, earth_radius_sq)
        trails.set_segments(trail_seg)
        if len(trail_seg):
            trails.set_color(colours[owners])

        # Connection indices come directly from NetworkSnapshot: renderer does not
        # decide which satellites are connected.
        if snapshot.connection_indices:
            active = np.asarray(snapshot.connection_indices, dtype=np.int32)
            active_i = active[:, 0]
            active_j = active[:, 1]

            a = current[active_i]
            links_3d = a[:, None, :] + sample_t * (current[active_j] - a)[:, None, :]
            link_seg, _ = visible_segments(links_3d, basis, earth_radius_sq)
            orbit_links.set_segments(link_seg)
            topology_links.set_segments(
                np.stack((net_pos[active_i], net_pos[active_j]), axis=1)
            )
        else:
            active_i = np.empty(0, dtype=np.int32)
            active_j = np.empty(0, dtype=np.int32)
            orbit_links.set_segments([])
            topology_links.set_segments([])

        connected = np.zeros(satellite_count, dtype=bool)
        connected[active_i] = True
        connected[active_j] = True
        node_colours = colours.copy()
        node_colours[:, 3] = np.where(connected, 1.0, 0.28)
        topology_nodes.set_facecolors(node_colours)

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-crf", str(crf),
            "-preset", preset,
            "-tune", "animation",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
    )

    fig.subplots_adjust(top=0.96, bottom=0.04, left=0.03, right=0.97)
    with writer.saving(fig, str(output), dpi=dpi):
        for frame in range(frame_count):
            draw_frame(frame)
            writer.grab_frame(facecolor=BACKGROUND)

    plt.close(fig)
    print(f"Saved: {output}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    fps = 60
    video_seconds = 30
    simulation_seconds = 20_000
    frame_count = fps * video_seconds
    earth_radius = 6371.0

    # Orchestration only: generation and simulation work live in their own modules.
    satellites = generate_satellites(number=30, seed=42)
    network = Network(
        satellites,
        earth_radius=earth_radius,
        require_line_of_sight=True,
    )
    snapshots = network.simulate(simulation_seconds, frame_count)

    render_video(
        satellites,
        snapshots,
        output="satellite_network.mp4",
        fps=fps,
        dpi=120,
        earth_radius=earth_radius,
        trail_frames=180,
        trail_stride=2,
        camera_rotation=90.0,
        link_samples=12,
        crf=16,
        preset="slow",
    )


if __name__ == "__main__":
    main()
