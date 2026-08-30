"""Streamlit frontend for the integrated orbital + topology routing demo."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.frontend.figures import build_orbital_figure, build_topology_figure
from src.frontend.replay import build_replay, node_label


st.set_page_config(
    page_title="IBM AI August Challenge · Integrated Frontend",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def _cached_replay(
    config_path: str,
    bundles: int,
    traffic_seed: int,
    before_policy: str,
    after_policy: str,
    switch_time_s: float | None,
    model_path: str,
):
    return build_replay(
        config_path=config_path,
        bundles=bundles,
        traffic_seed=traffic_seed,
        before_policy=before_policy,
        after_policy=after_policy,
        switch_time_s=switch_time_s,
        model_path=model_path,
        allow_missing_model=True,
    )


def _event_frame(replay):
    return pd.DataFrame(replay.event_rows())


def _bundle_frame(replay):
    rows = []
    for bundle in replay.bundle_runs:
        rows.append(
            {
                "bundle_id": bundle.bundle_id,
                "source": node_label(bundle.source_id, replay.config),
                "priority": round(bundle.science_priority, 3),
                "data_type": bundle.data_type,
                "created_s": round(bundle.created_s, 2),
                "deadline_s": None if bundle.deadline_s is None else round(bundle.deadline_s, 2),
                "delivered": bundle.delivered,
                "on_time": bundle.on_time,
                "arrival_s": None if bundle.arrival_s is None else round(bundle.arrival_s, 2),
                "hops": len(bundle.hops),
                "requested_algorithms": " + ".join(bundle.requested_algorithms) or "none",
                "actual_algorithms": " + ".join(bundle.actual_algorithms) or "none",
                "fallbacks": bundle.fallbacks,
                "path": " -> ".join(str(n) for n in bundle.path),
                "reason": bundle.reason,
            }
        )
    return pd.DataFrame(rows)


def _metric_row(replay, time_s: float):
    summary = replay.summary()
    active_contacts = replay.active_contacts(time_s)
    active_packets = replay.active_packets(time_s)
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    requested = replay.requested_algorithm_at(time_s).upper()
    actual = replay.actual_algorithm_at(time_s).upper()
    m1.metric("Requested", requested)
    m2.metric("Actually executing", actual)
    m3.metric("Active links", len(active_contacts))
    m4.metric("Packets in flight", len(active_packets))
    m5.metric("Delivered", f"{summary.delivered}/{summary.bundles}")
    m6.metric("On-time", f"{100 * summary.deadline_success:.1f}%")
    m7.metric("RL → temporal fallbacks", summary.total_fallbacks)


def _sidebar_controls():
    st.sidebar.title("Frontend controls")
    st.sidebar.caption("Minimalist synchronized orbital + topology playback")

    config_path = st.sidebar.text_input("Config path", value="config/prototype.yaml")
    model_path = st.sidebar.text_input("RL model path", value="RL/rl_env_v0/models/rl_agent_seed_42.zip")

    bundles = st.sidebar.slider("Visible science bundles", 4, 80, 24, 1)
    traffic_seed = st.sidebar.number_input("Traffic seed", value=20260830, step=1)

    before_policy = st.sidebar.selectbox("Policy before switch", ["temporal", "rl"], index=0)
    switch_enabled = st.sidebar.checkbox("Enable mid-run switch", value=True)
    after_policy = before_policy
    switch_time_s = None
    if switch_enabled:
        after_policy = st.sidebar.selectbox("Policy after switch", ["temporal", "rl"], index=1 if before_policy == "temporal" else 0)
        switch_time_s = st.sidebar.slider("Switch time (s)", 0, 1800, 900, 15)

    show_trails = st.sidebar.checkbox("Show orbital trails", value=True)
    trail_seconds = st.sidebar.slider("Trail length (s)", 30, 600, 180, 15)

    return {
        "config_path": config_path,
        "model_path": model_path,
        "bundles": bundles,
        "traffic_seed": int(traffic_seed),
        "before_policy": before_policy,
        "after_policy": after_policy,
        "switch_time_s": float(switch_time_s) if switch_enabled else None,
        "show_trails": show_trails,
        "trail_seconds": float(trail_seconds),
    }


def _timeline_controls(replay):
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False

    frames = len(replay.times)
    current_index = max(0, min(frames - 1, int(st.session_state.frame_index)))

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 4, 2])
    if c1.button("◀ Step"):
        st.session_state.playing = False
        current_index = max(0, current_index - 1)
    if c2.button("▶ Step"):
        st.session_state.playing = False
        current_index = min(frames - 1, current_index + 1)
    if c3.button("Play / Pause"):
        st.session_state.playing = not st.session_state.playing

    current_index = c4.slider("Timeline", 0, frames - 1, current_index, 1)
    speed = c5.select_slider("Playback", options=[1, 2, 4, 8], value=4)

    st.session_state.frame_index = current_index
    time_s = replay.times[current_index]

    if st.session_state.playing:
        time.sleep(max(0.02, 0.28 / speed))
        st.session_state.frame_index = (current_index + 1) % frames
        st.rerun()

    return float(time_s)


def main():
    st.title("Integrated constellation frontend")
    st.caption(
        "Side-by-side orbital 3D render and network topology graph, with animated packet motion and a mid-run pathfinding switch between temporal and RL routing."
    )

    controls = _sidebar_controls()
    replay = _cached_replay(
        controls["config_path"],
        controls["bundles"],
        controls["traffic_seed"],
        controls["before_policy"],
        controls["after_policy"],
        controls["switch_time_s"],
        controls["model_path"],
    )

    if not replay.model_loaded:
        st.warning(
            "RL checkpoint is unavailable or not loadable. If RL is requested, the replay records the requested algorithm as RL but the actual algorithm as TEMPORAL fallback. It will not be reported as RL execution."
        )

    time_s = _timeline_controls(replay)
    _metric_row(replay, time_s)

    bundle_choices = [bundle.bundle_id for bundle in replay.bundle_runs]
    selected_bundle_id = st.selectbox(
        "Highlight bundle route",
        bundle_choices,
        index=bundle_choices.index(replay.selected_bundle_id) if replay.selected_bundle_id in bundle_choices else 0,
    )

    left, right = st.columns((1.6, 1.15), gap="medium")
    with left:
        st.plotly_chart(
            build_orbital_figure(
                replay,
                time_s,
                selected_bundle_id=selected_bundle_id,
                show_trails=controls["show_trails"],
                trail_seconds=controls["trail_seconds"],
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        st.plotly_chart(
            build_topology_figure(replay, time_s, selected_bundle_id=selected_bundle_id),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    tabs = st.tabs(["Bundle inspector", "Event log", "Scenario", "Downloads"])

    with tabs[0]:
        bundle_df = _bundle_frame(replay)
        selected = bundle_df[bundle_df["bundle_id"] == selected_bundle_id]
        if not selected.empty:
            row = selected.iloc[0]
            st.markdown(
                f"**{selected_bundle_id}** · source **{row['source']}** · {row['data_type']} · priority **{row['priority']:.2f}** · path **{row['path']}**"
            )
            st.markdown(
                f"Created at **{row['created_s']}s**, deadline **{row['deadline_s']}s**, requested **{row['requested_algorithms']}**, actually used **{row['actual_algorithms']}**, delivered **{row['delivered']}**, on-time **{row['on_time']}**, RL fallbacks **{row['fallbacks']}**"
            )
        st.dataframe(bundle_df, use_container_width=True, hide_index=True)

    with tabs[1]:
        event_df = _event_frame(replay)
        current_events = event_df[event_df["time_s"] <= round(time_s, 2)].tail(24)
        st.markdown(f"Showing the last 24 events up to **t={time_s:.0f}s**")
        st.dataframe(current_events, use_container_width=True, hide_index=True)

    with tabs[2]:
        summary = replay.summary()
        st.markdown("### Scenario summary")
        st.write(
            {
                "contacts": replay.diagnostics.contacts,
                "satellite_contacts": replay.diagnostics.satellite_contacts,
                "ground_contacts": replay.diagnostics.ground_contacts,
                "direct_science_to_ground": replay.diagnostics.direct_to_ground_contacts,
                "delivery_ratio": round(summary.delivery_ratio, 3),
                "deadline_success": round(summary.deadline_success, 3),
                "priority_weighted_timely": round(summary.priority_weighted_timely, 3),
                "mean_latency_s": None if summary.mean_latency_s is None else round(summary.mean_latency_s, 2),
                "mean_hops": round(summary.mean_hops, 2),
                "rl_requested_hops": summary.rl_requested_hops,
                "rl_executed_hops": summary.rl_executed_hops,
                "temporal_executed_hops": summary.temporal_executed_hops,
            }
        )
        st.markdown("### Features included in this frontend bundle")
        st.markdown(
            "- synchronized 3D orbital render and topology graph\n"
            "- animated packet motion on active links\n"
            "- mid-run temporal ↔ RL switch control\n"
            "- selected-bundle route highlighting\n"
            "- bundle/event tables for debugging and demos\n"
            "- requested-vs-actual routing labels on every hop and packet\n"
            "- graceful fallback when the RL checkpoint is unavailable"
        )


    with tabs[3]:
        bundle_df = _bundle_frame(replay)
        event_df = _event_frame(replay)
        st.download_button(
            "Download bundle table CSV",
            bundle_df.to_csv(index=False).encode("utf-8"),
            file_name="frontend_bundle_runs.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download event log CSV",
            event_df.to_csv(index=False).encode("utf-8"),
            file_name="frontend_event_log.csv",
            mime="text/csv",
        )
        replay_json = json.dumps(
            {
                "before_policy": replay.before_policy,
                "after_policy": replay.after_policy,
                "switch_time_s": replay.switch_time_s,
                "model_loaded": replay.model_loaded,
                "summary": asdict(replay.summary()),
            },
            indent=2,
        )
        st.download_button(
            "Download replay summary JSON",
            replay_json.encode("utf-8"),
            file_name="frontend_replay_summary.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
