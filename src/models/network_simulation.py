"""Satellite network visualization with synchronized orbital and topology views."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from satellite_generator import generate_satellites


def simulate_positions(satellites, steps=10_000, dt=1.0):
    """Return satellite positions with shape (frames, satellites, 3)."""
    if not satellites:
        raise ValueError("At least one satellite is required.")
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    if dt <= 0:
        raise ValueError("dt must be > 0.")

    radii = np.asarray([sat.radius() for sat in satellites], dtype=float)
    longitude0 = np.asarray([sat.longitude() for sat in satellites], dtype=float)
    inclination0 = np.asarray([sat.inclination() for sat in satellites], dtype=float)
    velocity = np.asarray([sat.angular_velocity() for sat in satellites], dtype=float)

    times = np.arange(steps + 1, dtype=float) * dt
    longitude = (longitude0[None, :] + times[:, None] * velocity[None, :, 0]) % (2.0 * np.pi)
    inclination = inclination0[None, :] + times[:, None] * velocity[None, :, 1]

    r = radii[None, :]
    x = r * np.cos(longitude)
    y = r * np.sin(longitude) * np.cos(inclination)
    z = r * np.sin(longitude) * np.sin(inclination)

    return np.stack((x, y, z), axis=-1)


def view_basis(azimuth_degrees, elevation_degrees):
    """Build an orthographic camera basis."""
    azimuth = np.deg2rad(azimuth_degrees)
    elevation = np.deg2rad(elevation_degrees)

    view = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=float,
    )
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0], dtype=float)
    up = np.cross(view, right)
    return view, right, up


def project(points, view, right, up, earth_radius):
    """Orthographically project 3D points and mark points hidden by Earth."""
    points = np.asarray(points, dtype=float)
    screen_x = points @ right
    screen_y = points @ up
    depth = points @ view

    screen_xy = np.column_stack((screen_x, screen_y))
    projected_radius_squared = screen_x**2 + screen_y**2
    visible = (depth >= 0.0) | (projected_radius_squared >= earth_radius**2)
    return screen_xy, visible


def visible_segments(points_3d, view, right, up, earth_radius):
    """Project a sampled 3D polyline and remove portions hidden by Earth."""
    points_3d = np.asarray(points_3d, dtype=float)
    if len(points_3d) < 2:
        return np.empty((0, 2, 2), dtype=float)

    screen_xy, visible = project(points_3d, view, right, up, earth_radius)
    keep = visible[:-1] & visible[1:]
    if not np.any(keep):
        return np.empty((0, 2, 2), dtype=float)

    return np.stack((screen_xy[:-1][keep], screen_xy[1:][keep]), axis=1)


def active_links(positions, pair_i, pair_j, source_range_squared, earth_radius):
    """Return physically active links: in range and with clear Earth line-of-sight."""
    point_a = positions[pair_i]
    point_b = positions[pair_j]
    delta = point_b - point_a
    distance_squared = np.einsum("ij,ij->i", delta, delta)

    # Filter by range first so the more expensive line-of-sight work only
    # runs for pairs that could actually connect.
    candidate_mask = distance_squared <= source_range_squared
    if not np.any(candidate_mask):
        empty = np.empty(0, dtype=int)
        return empty, empty

    candidate_indices = np.flatnonzero(candidate_mask)
    point_a = point_a[candidate_mask]
    delta = delta[candidate_mask]
    distance_squared = distance_squared[candidate_mask]

    denominator = np.maximum(distance_squared, 1e-12)
    t = -np.einsum("ij,ij->i", point_a, delta) / denominator
    t = np.clip(t, 0.0, 1.0)
    closest = point_a + t[:, None] * delta
    closest_distance_squared = np.einsum("ij,ij->i", closest, closest)

    clear = closest_distance_squared > earth_radius**2
    active_indices = candidate_indices[clear]
    return pair_i[active_indices], pair_j[active_indices]


def network_degree(satellite_count, active_i, active_j):
    """Return the number of active links connected to each satellite."""
    degree = np.zeros(satellite_count, dtype=int)
    np.add.at(degree, active_i, 1)
    np.add.at(degree, active_j, 1)
    return degree


def make_animation(
    satellites,
    positions,
    earth_radius=6371.0,
    trail_length=450,
    trail_stride=5,
    frame_stride=10,
    interval_ms=20,
    view_azimuth=-55.0,
    view_elevation=22.0,
    view_rotation_per_frame=0.05,
    link_samples=24,
    show_satellite_labels=False,
    repeat=True,
):
    """Create synchronized front, back, and network-topology views."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (frames, satellites, 3).")

    frame_count, satellite_count, _ = positions.shape
    if satellite_count != len(satellites):
        raise ValueError("positions satellite dimension does not match satellites.")
    if frame_stride < 1 or trail_stride < 1:
        raise ValueError("frame_stride and trail_stride must be >= 1.")
    if link_samples < 2:
        raise ValueError("link_samples must be >= 2.")

    displayed_frames = np.arange(0, frame_count, frame_stride, dtype=int)
    if displayed_frames[-1] != frame_count - 1:
        displayed_frames = np.append(displayed_frames, frame_count - 1)

    pair_i, pair_j = np.triu_indices(satellite_count, k=1)
    connection_ranges = np.asarray([sat.connection_range() for sat in satellites], dtype=float)
    source_range_squared = connection_ranges[pair_i] ** 2

    colour_map = plt.get_cmap("tab20")
    satellite_colours = colour_map(np.linspace(0.0, 1.0, satellite_count, endpoint=False))

    maximum_orbit_radius = np.linalg.norm(positions, axis=2).max()
    limit = maximum_orbit_radius * 1.08

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#f8fafc")
    grid_spec = fig.add_gridspec(
        2,
        2,
        height_ratios=(2.0, 1.15),
        hspace=0.22,
        wspace=0.08,
    )

    ax_front = fig.add_subplot(grid_spec[0, 0])
    ax_back = fig.add_subplot(grid_spec[0, 1])
    ax_network = fig.add_subplot(grid_spec[1, :])

    for ax, title in ((ax_front, "FRONT"), (ax_back, "BACK")):
        ax.set_facecolor("#f8fafc")
        ax.set_aspect("equal")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        earth = plt.Circle(
            (0.0, 0.0),
            earth_radius,
            facecolor="#e7edf4",
            edgecolor="#64748b",
            linewidth=1.2,
            zorder=1,
        )
        ax.add_patch(earth)

    def create_orbit_artists(ax):
        trails = LineCollection([], linewidths=1.0, alpha=0.58, zorder=3)
        links = LineCollection([], colors="#111827", linewidths=0.9, alpha=0.30, zorder=4)
        satellites_artist = ax.scatter(
            [],
            [],
            s=40,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        ax.add_collection(trails)
        ax.add_collection(links)
        return trails, links, satellites_artist

    front_trails, front_links, front_satellites = create_orbit_artists(ax_front)
    back_trails, back_links, back_satellites = create_orbit_artists(ax_back)

    ax_network.set_facecolor("#f8fafc")
    ax_network.axis("off")

    grid_columns = int(np.ceil(np.sqrt(satellite_count)))
    grid_rows = int(np.ceil(satellite_count / grid_columns))
    network_positions = np.zeros((satellite_count, 2), dtype=float)

    for satellite_index in range(satellite_count):
        row = satellite_index // grid_columns
        column = satellite_index % grid_columns
        network_positions[satellite_index] = (column, -row)

    final_row_count = satellite_count % grid_columns
    if final_row_count:
        final_row = grid_rows - 1
        offset = (grid_columns - final_row_count) / 2.0
        first_final_index = final_row * grid_columns
        for local_index in range(final_row_count):
            network_positions[first_final_index + local_index, 0] = local_index + offset

    ax_network.set_xlim(-0.75, grid_columns - 0.25)
    ax_network.set_ylim(-grid_rows + 0.15, 0.75)
    ax_network.set_aspect("equal")

    topology_links = LineCollection(
        [],
        colors="#111827",
        linewidths=1.5,
        alpha=0.42,
        zorder=1,
    )
    ax_network.add_collection(topology_links)

    topology_nodes = ax_network.scatter(
        network_positions[:, 0],
        network_positions[:, 1],
        s=135,
        c=satellite_colours,
        edgecolors="#1f2937",
        linewidths=1.0,
        zorder=3,
    )

    if show_satellite_labels:
        for satellite_index, satellite in enumerate(satellites):
            x, y = network_positions[satellite_index]
            try:
                satellite_name = satellite.name()
            except Exception:
                satellite_name = f"Sat {satellite_index}"
            ax_network.text(
                x,
                y - 0.24,
                satellite_name,
                fontsize=7,
                ha="center",
                va="top",
                color="#334155",
                zorder=4,
            )

    status = ax_network.set_title(
        f"NETWORK  |  Connected satellites: 0/{satellite_count}",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )

    sample_t = np.linspace(0.0, 1.0, link_samples)

    def update_orbit_view(
        current,
        frame,
        view,
        right,
        up,
        trail_artist,
        link_artist,
        satellite_artist,
        active_i,
        active_j,
    ):
        current_xy, current_visible = project(current, view, right, up, earth_radius)
        satellite_artist.set_offsets(current_xy[current_visible])
        satellite_artist.set_facecolors(satellite_colours[current_visible])

        start = 0 if trail_length is None else max(0, frame - trail_length + 1)
        trail_frames = np.arange(start, frame + 1, trail_stride, dtype=int)
        if trail_frames[-1] != frame:
            trail_frames = np.append(trail_frames, frame)

        all_trail_segments = []
        all_trail_colours = []
        for satellite_index in range(satellite_count):
            trail_points = positions[trail_frames, satellite_index, :]
            segments = visible_segments(trail_points, view, right, up, earth_radius)
            if len(segments):
                all_trail_segments.extend(segments)
                all_trail_colours.extend([satellite_colours[satellite_index]] * len(segments))

        trail_artist.set_segments(all_trail_segments)
        if all_trail_colours:
            trail_artist.set_color(all_trail_colours)

        visible_link_segments = []
        for satellite_i, satellite_j in zip(active_i, active_j):
            point_a = current[satellite_i]
            point_b = current[satellite_j]
            sampled = point_a + sample_t[:, None] * (point_b - point_a)
            segments = visible_segments(sampled, view, right, up, earth_radius)
            if len(segments):
                visible_link_segments.extend(segments)

        link_artist.set_segments(visible_link_segments)

    def update_topology(active_i, active_j):
        if len(active_i):
            topology_links.set_segments(
                np.stack((network_positions[active_i], network_positions[active_j]), axis=1)
            )
        else:
            topology_links.set_segments([])

        degree = network_degree(satellite_count, active_i, active_j)
        connected_count = int(np.count_nonzero(degree))

        # Keep node size fixed for readability; only isolated satellites fade.
        node_colours = satellite_colours.copy()
        node_colours[:, 3] = np.where(degree > 0, 1.0, 0.28)
        topology_nodes.set_facecolors(node_colours)

        status.set_text(
            f"NETWORK  |  Connected satellites: {connected_count}/{satellite_count}"
        )

    def update(display_index):
        frame = int(displayed_frames[display_index])
        current = positions[frame]

        # Calculate the physical network once and reuse it in every view.
        active_i, active_j = active_links(
            current,
            pair_i,
            pair_j,
            source_range_squared,
            earth_radius,
        )

        azimuth = view_azimuth + display_index * view_rotation_per_frame
        front_view, front_right, front_up = view_basis(azimuth, view_elevation)
        back_view = -front_view
        back_right = -front_right
        back_up = front_up

        update_orbit_view(
            current,
            frame,
            front_view,
            front_right,
            front_up,
            front_trails,
            front_links,
            front_satellites,
            active_i,
            active_j,
        )
        update_orbit_view(
            current,
            frame,
            back_view,
            back_right,
            back_up,
            back_trails,
            back_links,
            back_satellites,
            active_i,
            active_j,
        )
        update_topology(active_i, active_j)

        return (
            front_satellites,
            front_trails,
            front_links,
            back_satellites,
            back_trails,
            back_links,
            topology_nodes,
            topology_links,
            status,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(displayed_frames),
        interval=interval_ms,
        blit=True,
        repeat=repeat,
        cache_frame_data=False,
    )

    fig.subplots_adjust(top=0.92, bottom=0.05, left=0.04, right=0.96)
    return fig, animation


def save_animation(animation, filename="satellite_network.mp4", fps=30, dpi=120):
    """Save the animation as MP4. Requires ffmpeg."""
    animation.save(filename, writer="ffmpeg", fps=fps, dpi=dpi)


if __name__ == "__main__":
    satellites = generate_satellites(number=20, seed=42)
    positions = simulate_positions(satellites, steps=20_000, dt=1.0)

    fig, animation = make_animation(
        satellites,
        positions,
        earth_radius=6371.0,
        trail_length=450,
        trail_stride=5,
        frame_stride=10,
        interval_ms=20,
        view_azimuth=-55.0,
        view_elevation=22.0,
        view_rotation_per_frame=0.05,
        link_samples=24,
        show_satellite_labels=False,
        repeat=True,
    )

    plt.show()
    # save_animation(animation, filename="satellite_network.mp4", fps=30, dpi=120)