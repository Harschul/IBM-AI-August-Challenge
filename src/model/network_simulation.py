"""Run, score, inspect and render a satellite-network constellation.

By default this creates a reproducible random constellation.  Pass
``--constellation optimized_constellation.json`` to replay an optimized one.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection

from network import Network
from ground_receivers import fibonacci_sphere
from satellite_generator import generate_satellites, load_satellites, save_satellites
from scoring import (
    NetworkScore,
    print_frame_samples,
    print_score_summary,
    score_network,
    write_score_csv,
)


BACKGROUND = "#f8fafc"
EARTH_FACE = "#e7edf4"
EARTH_EDGE = "#64748b"
LINK_COLOUR = "#111827"
MUTED_TEXT = "#475569"
GROUND_POINT_COLOUR = "#60a5fa"


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


def _metric_frame(frame: int, fps: int, update_seconds: float) -> int:
    """Select which metric snapshot to display in the video overlay.

    ``update_seconds <= 0`` means update metrics on every rendered frame.
    Otherwise metrics are held for that many seconds of *video time*.
    """
    if update_seconds <= 0:
        return frame
    hold_frames = max(1, int(round(float(update_seconds) * fps)))
    return (frame // hold_frames) * hold_frames


def render_video(
    satellites,
    snapshots,
    score_results: NetworkScore,
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
    metrics_update_seconds=1.0,
    ground_points=None,
):
    """Render immutable snapshots plus the score diagnostics for each frame."""
    snapshots = tuple(snapshots)
    if not snapshots:
        raise ValueError("snapshots cannot be empty")
    if len(score_results.frame_scores) != len(snapshots):
        raise ValueError("score_results must contain exactly one row per snapshot")

    frame_count = len(snapshots)
    satellite_count = len(satellites)
    earth_radius_sq = earth_radius * earth_radius

    positions = np.stack([snapshot.positions for snapshot in snapshots])
    colours = plt.get_cmap("tab20")(
        np.linspace(0.0, 1.0, satellite_count, endpoint=False)
    )
    limit = np.linalg.norm(positions, axis=2).max() * 1.08

    # Keep the two main views perfectly centred/aligned.  The only live
    # diagnostics are two compact figure-level values in the top margin, so
    # nothing covers or squeezes the orbital simulation.
    fig = plt.figure(figsize=(16, 9), facecolor=BACKGROUND)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=(2.2, 1.0),
        hspace=0.18,
    )
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
    ground_artist = ax_orbit.scatter(
        [], [], s=12, marker="o", color=GROUND_POINT_COLOUR, alpha=0.28, zorder=2.5
    )
    ax_orbit.add_collection(trails)
    ax_orbit.add_collection(orbit_links)

    # Two compact live values only.  They sit in the figure margin rather than
    # inside either axis, preserving the clean orbital/topology composition.
    coverage_artist = fig.text(
        0.035,
        0.965,
        "EARTH COVERAGE  ---%",
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=MUTED_TEXT,
    )
    links_artist = fig.text(
        0.965,
        0.965,
        "TOTAL LINKS  ---",
        va="top",
        ha="right",
        fontsize=11,
        fontweight="bold",
        color=MUTED_TEXT,
    )

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
    ground_points = (
        np.asarray(ground_points, dtype=float)
        if ground_points is not None
        else np.empty((0, 3), dtype=float)
    )

    def draw_frame(frame):
        snapshot = snapshots[frame]
        current = snapshot.positions
        basis = camera_basis(camera_angles[frame], elevation)

        if len(ground_points):
            ground_xy, ground_visible = project(ground_points, basis, earth_radius_sq)
            ground_artist.set_offsets(ground_xy[ground_visible])
        else:
            ground_artist.set_offsets(np.empty((0, 2)))

        current_xy, current_visible = project(current, basis, earth_radius_sq)
        satellites_artist.set_offsets(current_xy[current_visible])
        satellites_artist.set_facecolors(colours[current_visible])

        start = max(0, frame - trail_frames + 1)
        trail_poly = np.swapaxes(positions[start : frame + 1 : trail_stride], 0, 1)
        trail_seg, owners = visible_segments(trail_poly, basis, earth_radius_sq)
        trails.set_segments(trail_seg)
        if len(trail_seg):
            trails.set_color(colours[owners])

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

        metric_index = min(
            _metric_frame(frame, fps=fps, update_seconds=metrics_update_seconds),
            frame_count - 1,
        )
        metric = score_results.frame_scores[metric_index]
        coverage_artist.set_text(
            f"EARTH COVERAGE  {100.0 * metric.coverage_fraction:5.1f}%"
        )
        links_artist.set_text(
            f"TOTAL LINKS  {metric.active_links:d}"
        )

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

    fig.subplots_adjust(top=0.925, bottom=0.04, left=0.035, right=0.975)
    with writer.saving(fig, str(output), dpi=dpi):
        for frame in range(frame_count):
            draw_frame(frame)
            writer.grab_frame(facecolor=BACKGROUND)

    plt.close(fig)
    print(f"Saved video: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--constellation",
        type=Path,
        help="Replay a constellation JSON file instead of generating a random one.",
    )
    source.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random constellation (default: 42).",
    )

    parser.add_argument("--satellites", type=int, default=30)
    parser.add_argument("--simulation-seconds", type=float, default=20_000.0)
    parser.add_argument("--video-seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--earth-radius", type=float, default=6371.0)
    parser.add_argument("--ground-points", type=int, default=100)
    parser.add_argument("--ground-distance-scale", type=float, default=5000.0)
    parser.add_argument(
        "--worst-distance-weight",
        type=float,
        default=0.5,
        help=(
            "Weight assigned to the worst receiver-distance penalty; the "
            "remaining weight is assigned to the mean penalty."
        ),
    )
    parser.add_argument("--coverage-tolerance", type=float, default=1e-12)
    parser.add_argument("--min-elevation-deg", type=float, default=0.0)
    parser.add_argument("--output", default="satellite_network.mp4")
    parser.add_argument("--metrics-csv", default="satellite_network_metrics.csv")
    parser.add_argument(
        "--save-constellation",
        default=None,
        help=(
            "Optional path for the exact t=0 constellation JSON. Random runs "
            "automatically save random_constellation.json when this is omitted; "
            "replayed JSON constellations are not re-saved unless a path is given."
        ),
    )
    parser.add_argument(
        "--metrics-update-seconds",
        type=float,
        default=1.0,
        help=(
            "How often the on-video coverage/link indicators change in video seconds. "
            "Use 0 to update it on every frame."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Simulate, score and export metrics without rendering MP4.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.fps < 1:
        raise ValueError("fps must be at least 1")
    if args.video_seconds <= 0:
        raise ValueError("video_seconds must be positive")
    if args.simulation_seconds < 0:
        raise ValueError("simulation_seconds cannot be negative")
    if args.ground_points < 1:
        raise ValueError("ground_points must be at least 1")
    if args.ground_distance_scale <= 0:
        raise ValueError("ground-distance-scale must be positive")
    if not 0.0 <= args.worst_distance_weight <= 1.0:
        raise ValueError("worst-distance-weight must be between 0 and 1")
    if args.coverage_tolerance < 0:
        raise ValueError("coverage-tolerance cannot be negative")
    if not -90.0 <= args.min_elevation_deg < 90.0:
        raise ValueError("min-elevation-deg must be in [-90, 90)")

    frame_count = max(1, int(round(args.fps * args.video_seconds)))

    if args.constellation:
        satellites = load_satellites(args.constellation)
        print(f"Loaded constellation: {args.constellation.resolve()}")
    else:
        satellites = generate_satellites(number=args.satellites, seed=args.seed)

    # Save before simulation so the JSON always represents the exact t=0 state.
    save_path = args.save_constellation
    if save_path is None and not args.constellation:
        save_path = "random_constellation.json"
    if save_path:
        saved = save_satellites(satellites, save_path)
        print(f"Saved constellation: {saved}")

    network = Network(
        satellites,
        earth_radius=args.earth_radius,
        require_line_of_sight=True,
    )
    snapshots = network.simulate(args.simulation_seconds, frame_count)

    ground_points = fibonacci_sphere(args.ground_points, radius=args.earth_radius)
    results = score_network(
        snapshots,
        satellites,
        ground_points=ground_points,
        earth_radius=args.earth_radius,
        ground_distance_scale=args.ground_distance_scale,
        worst_distance_weight=args.worst_distance_weight,
        coverage_tolerance=args.coverage_tolerance,
        min_elevation_deg=args.min_elevation_deg,
    )
    print_score_summary(results, satellite_count=len(satellites))

    # The simulation and video have the same number of frames, so sampling every
    # fps frames gives a readable one-row-per-video-second console view.
    print_frame_samples(results, every_frames=args.fps)

    metrics_path = write_score_csv(results, args.metrics_csv)
    print(f"Saved per-snapshot metrics: {metrics_path}")

    if not args.no_video:
        render_video(
            satellites,
            snapshots,
            results,
            output=args.output,
            fps=args.fps,
            dpi=120,
            earth_radius=args.earth_radius,
            trail_frames=180,
            trail_stride=2,
            camera_rotation=90.0,
            link_samples=12,
            crf=16,
            preset="slow",
            metrics_update_seconds=args.metrics_update_seconds,
            ground_points=ground_points,
        )


if __name__ == "__main__":
    main()
