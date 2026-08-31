"""Minimal single-view Streamlit demo for the locked final experiment."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.experiment.spec import load_final_spec
from src.frontend.figures import build_orbital_figure, build_topology_figure
from src.frontend.replay import build_replay, node_label

st.set_page_config(
    page_title="Multi-Orbit Scientific Data Relay",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #ffffff; }
    [data-testid="stSidebar"] { background: #f6f7f9; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
    h1 { font-size: 1.65rem !important; margin-bottom: 0.1rem !important; }
    div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.55rem 0.7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Building the locked physical/stochastic replay…")
def _cached_replay(mode: str, seed_offset: int, before: str, after: str, switch_time: float | None):
    return build_replay(
        mode=mode,
        seed_offset=seed_offset,
        before_policy=before,
        after_policy=after,
        switch_time_s=switch_time,
    )


def _bundle_frame(replay) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "bundle_id": row.bundle_id,
            "source": node_label(row.source_id, replay.config),
            "priority": round(row.science_priority, 3),
            "data_type": row.data_type,
            "created_s": round(row.created_s, 2),
            "deadline_s": None if row.deadline_s is None else round(row.deadline_s, 2),
            "delivered": row.delivered,
            "on_time": row.on_time,
            "arrival_s": None if row.arrival_s is None else round(row.arrival_s, 2),
            "hops": len(row.hops),
            "attempts": len(row.attempts),
            "failures": row.transfer_failures,
            "wasted_mb": round(row.wasted_capacity_bytes / 1e6, 3),
            "path": " → ".join(str(node) for node in row.path),
            "reason": row.reason,
        }
        for row in replay.bundle_runs
    ])


def _sidebar(spec):
    st.sidebar.header("Controls")
    mode_label = st.sidebar.selectbox(
        "Routing algorithm",
        ["Reported PPO", "Reported Temporal", "Interactive switch"],
        index=0,
    )
    mode = {
        "Reported PPO": "reported_rl",
        "Reported Temporal": "reported_temporal",
        "Interactive switch": "interactive_switch",
    }[mode_label]

    seed_offset = st.sidebar.slider(
        "Scenario",
        0,
        spec.benchmark.num_seeds - 1,
        spec.demo.default_seed_offset,
        1,
        help="One of the exact held-out scenarios used by the final benchmark.",
    )

    before, after, switch_time = "temporal", "rl", None
    if mode == "interactive_switch":
        st.sidebar.caption("Interactive switching is for demonstration only; it is not part of the reported benchmark.")
        before = st.sidebar.selectbox("Start with", ["temporal", "rl"], index=0)
        after = st.sidebar.selectbox("Switch to", ["temporal", "rl"], index=1)
        switch_time = float(st.sidebar.slider("Switch at (s)", 0, 1800, 900, 15))

    st.sidebar.divider()
    show_trails = st.sidebar.checkbox("Orbit trails", value=True)
    trail_seconds = float(st.sidebar.slider("Trail length (s)", 30, 600, 180, 15))
    show_topology = st.sidebar.checkbox("Show network topology", value=False)
    show_details = st.sidebar.checkbox("Show detailed tables", value=False)
    return mode, seed_offset, before, after, switch_time, show_trails, trail_seconds, show_topology, show_details


def _timeline(replay) -> float:
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False

    frames = len(replay.times)
    index = max(0, min(frames - 1, int(st.session_state.frame_index)))
    c1, c2, c3, c4, c5 = st.columns([0.8, 0.8, 1.1, 5.2, 1.4])
    if c1.button("◀", help="Previous frame"):
        st.session_state.playing = False
        index = max(0, index - 1)
    if c2.button("▶", help="Next frame"):
        st.session_state.playing = False
        index = min(frames - 1, index + 1)
    if c3.button("Pause" if st.session_state.playing else "Play"):
        st.session_state.playing = not st.session_state.playing
    index = c4.slider("Simulation time", 0, frames - 1, index, 1, label_visibility="collapsed")
    speed = c5.select_slider("Speed", [1, 2, 4, 8], value=4, label_visibility="collapsed")
    st.session_state.frame_index = index
    time_s = float(replay.times[index])
    if st.session_state.playing:
        time.sleep(max(0.02, 0.28 / speed))
        st.session_state.frame_index = (index + 1) % frames
        st.rerun()
    return time_s


def _metrics(replay, time_s: float):
    summary = replay.summary()
    active_packets = replay.active_packets(time_s)
    cols = st.columns(5)
    cols[0].metric("Algorithm", replay.actual_algorithm_at(time_s).upper())
    cols[1].metric("Delivered", f"{summary.delivered}/{summary.bundles}")
    cols[2].metric("On time", f"{100 * summary.deadline_success:.1f}%")
    cols[3].metric("Packets in flight", len(active_packets))
    cols[4].metric("Transfer failures", f"{100 * summary.transfer_failure_rate:.1f}%")


def _benchmark_summary(spec):
    path = spec.benchmark.output_dir / "summary.json"
    if not path.exists():
        st.info("Final benchmark evidence is not present. Run `python run_final_benchmark.py`.")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for algorithm in ("temporal", "rl_pure"):
        metrics = payload.get("algorithms", {}).get(algorithm, {})
        if not metrics:
            continue
        rows.append({
            "algorithm": "Temporal" if algorithm == "temporal" else "PPO",
            "delivery": metrics["delivery_ratio"]["mean"],
            "on_time": metrics["deadline_success"]["mean"],
            "priority_weighted": metrics["priority_weighted_timely"]["mean"],
            "latency_s": metrics["mean_latency_s"]["mean"],
            "hops": metrics["mean_hops"]["mean"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main():
    spec = load_final_spec()
    st.title("Multi-Orbit Scientific Data Relay Network")
    st.caption("Research satellites → LEO/GEO relays → three operational ground receivers. Paths are generated by the same locked runner used for the reported benchmark.")

    controls = _sidebar(spec)
    mode, seed_offset, before, after, switch_time, show_trails, trail_seconds, show_topology, show_details = controls

    try:
        replay = _cached_replay(mode, seed_offset, before, after, switch_time)
    except Exception as exc:
        st.error(str(exc))
        if mode == "reported_rl":
            st.caption("Reported PPO never silently falls back to Temporal or an older checkpoint.")
        st.stop()

    traffic_seed, stochastic_seed = spec.benchmark.seeds(seed_offset)
    if not replay.reported_experiment:
        st.info("Interactive switch mode is a visualization mode, not a reported benchmark result.")

    time_s = _timeline(replay)
    _metrics(replay, time_s)

    choices = [bundle.bundle_id for bundle in replay.bundle_runs]
    default = choices.index(replay.selected_bundle_id) if replay.selected_bundle_id in choices else 0
    selected = st.selectbox("Highlight packet route", choices, index=default)

    # The orbital simulation is intentionally the dominant view.
    st.plotly_chart(
        build_orbital_figure(
            replay,
            time_s,
            selected_bundle_id=selected,
            show_trails=show_trails,
            trail_seconds=trail_seconds,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if show_topology:
        with st.expander("Network topology", expanded=True):
            st.plotly_chart(
                build_topology_figure(replay, time_s, selected_bundle_id=selected),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    bundle_df = _bundle_frame(replay)
    event_df = pd.DataFrame(replay.event_rows())
    selected_row = bundle_df[bundle_df.bundle_id == selected]
    if not selected_row.empty:
        row = selected_row.iloc[0]
        st.caption(
            f"{row['source']} · priority {row['priority']:.2f} · path {row['path']} · "
            f"delivered={row['delivered']} · on-time={row['on_time']} · failures={row['failures']}"
        )

    if show_details:
        with st.expander("Experiment details", expanded=True):
            tabs = st.tabs(["Selected packet", "Events", "Benchmark", "Identity"])
            with tabs[0]:
                if not selected_row.empty:
                    st.write(selected_row.iloc[0].to_dict())
            with tabs[1]:
                current = event_df[event_df.time_s <= round(time_s, 3)].tail(50)
                st.dataframe(current, use_container_width=True, hide_index=True)
            with tabs[2]:
                _benchmark_summary(spec)
            with tabs[3]:
                st.json({
                    "experiment": spec.name,
                    "reported": replay.reported_experiment,
                    "config_sha256": spec.scenario_config_sha256,
                    "model": str(spec.model),
                    "seed_offset": seed_offset,
                    "traffic_seed": traffic_seed,
                    "stochastic_seed": stochastic_seed,
                    "summary": asdict(replay.summary()),
                })


if __name__ == "__main__":
    main()
