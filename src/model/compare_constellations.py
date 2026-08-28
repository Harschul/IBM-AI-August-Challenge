"""Automatic side-by-side comparison of two optimized constellations.

Loads both saved constellation JSONs, runs the same network simulation and
packet routing on each, then prints a comparison table covering:

  - Coverage and receiver-distance metrics  (from score_network)
  - Packet delivery success rate            (from delivery_success_rate)
  - End-to-end delay statistics             (from end_to_end_delay)

Usage
-----
python compare_constellations.py
python compare_constellations.py \\
    --constellation-a coverage_optimized_constellation.json \\
    --constellation-b distance_optimized_constellation.json \\
    --label-a "Coverage-First" --label-b "Distance-Only"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ground_receivers import fibonacci_sphere
from network import Network
from packet import delivery_success_rate, end_to_end_delay, simulate_packet_routing
from satellite_generator import load_satellites
from scoring import score_network


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--constellation-a",
        type=Path,
        default=Path("../../coverage_optimized_constellation.json"),
    )
    p.add_argument(
        "--constellation-b",
        type=Path,
        default=Path("../../distance_optimized_constellation.json"),
    )
    p.add_argument("--label-a", default="Coverage-First")
    p.add_argument("--label-b", default="Distance-Only")
    p.add_argument("--simulation-seconds", type=float, default=20_000.0)
    p.add_argument("--frames", type=int, default=360)
    p.add_argument("--ground-points", type=int, default=100)
    p.add_argument("--earth-radius", type=float, default=6371.0)
    p.add_argument("--ground-distance-scale", type=float, default=5000.0)
    p.add_argument("--worst-distance-weight", type=float, default=0.5)
    p.add_argument("--route-strategy", default="bfs",
                   choices=["bfs", "greedy", "least_congested"])
    p.add_argument("--receiver-capacity", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _run(constellation_path, args):
    """Simulate and score one constellation. Returns (NetworkScore, routing stats)."""
    satellites = load_satellites(constellation_path)
    network = Network(satellites, earth_radius=args.earth_radius, require_line_of_sight=True)
    snapshots = network.simulate(args.simulation_seconds, args.frames)

    ground_points = fibonacci_sphere(args.ground_points, radius=args.earth_radius)

    net_score = score_network(
        snapshots,
        satellites,
        ground_points=ground_points,
        earth_radius=args.earth_radius,
        ground_distance_scale=args.ground_distance_scale,
        worst_distance_weight=args.worst_distance_weight,
    )

    delivered, dropped, queues = simulate_packet_routing(
        snapshots,
        satellites,
        ground_points,
        args.earth_radius,
        route_strategy=args.route_strategy,
        receiver_capacity=args.receiver_capacity,
        seed=args.seed,
    )

    rate = delivery_success_rate(delivered, dropped, queues)
    delay = end_to_end_delay(delivered)

    return net_score, rate, delay, len(delivered), len(dropped)


def _row(label, a_val, b_val):
    print(f"  {label:<36} {a_val:>18}  {b_val:>18}")


def main():
    args = parse_args()

    print(f"\nLoading A: {args.constellation_a}")
    score_a, rate_a, delay_a, delivered_a, dropped_a = _run(args.constellation_a, args)

    print(f"Loading B: {args.constellation_b}")
    score_b, rate_b, delay_b, delivered_b, dropped_b = _run(args.constellation_b, args)

    col_a = args.label_a[:18].center(18)
    col_b = args.label_b[:18].center(18)

    print(f"\n{'=' * 76}")
    print(f"  {'METRIC':<36} {col_a}  {col_b}")
    print(f"{'=' * 76}")

    print("  -- Coverage --")
    _row("Fitness score",              f"{score_a.score:.6f}",                       f"{score_b.score:.6f}")
    _row("Average coverage",           f"{100*score_a.average_coverage:.2f}%",       f"{100*score_b.average_coverage:.2f}%")
    _row("Worst-frame coverage",       f"{100*score_a.worst_coverage:.2f}%",         f"{100*score_b.worst_coverage:.2f}%")
    _row("Avg nearest recv distance",  f"{score_a.average_nearest_receiver_distance:.1f} km", f"{score_b.average_nearest_receiver_distance:.1f} km")
    _row("Avg active links/frame",     f"{score_a.average_active_links:.1f}",        f"{score_b.average_active_links:.1f}")

    print("  -- Packet Routing --")
    _row("Packets delivered",          str(delivered_a),                             str(delivered_b))
    _row("Packets dropped",            str(dropped_a),                               str(dropped_b))
    _row("Delivery success rate",      f"{100*rate_a:.2f}%",                         f"{100*rate_b:.2f}%")
    _row("Delay min (frames)",         str(delay_a["min"]),                          str(delay_b["min"]))
    _row("Delay max (frames)",         str(delay_a["max"]),                          str(delay_b["max"]))
    _row("Delay mean (frames)",        f"{delay_a['mean']:.2f}",                     f"{delay_b['mean']:.2f}")

    print(f"{'=' * 76}\n")


if __name__ == "__main__":
    main()
