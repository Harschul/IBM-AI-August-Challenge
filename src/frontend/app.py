"""Final Streamlit demo for the locked stochastic physical experiment."""

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
    page_title="IBM AI August Challenge · Final Physical Routing Demo",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Running the locked physical/stochastic replay…")
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
            "fallbacks": row.fallbacks,
            "path": " -> ".join(str(node) for node in row.path),
            "reason": row.reason,
        }
        for row in replay.bundle_runs
    ])


def _sidebar(spec):
    st.sidebar.title("Final experiment")
    st.sidebar.caption("All reported modes are locked to the same config, PPO checkpoint, traffic family and stochastic model as the benchmark.")

    mode_label = st.sidebar.selectbox(
        "Simulation mode",
        ["Reported PPO", "Reported Temporal", "Interactive Temporal ↔ RL"],
        index=0,
    )
    mode = {
        "Reported PPO": "reported_rl",
        "Reported Temporal": "reported_temporal",
        "Interactive Temporal ↔ RL": "interactive_switch",
    }[mode_label]

    seed_offset = st.sidebar.slider(
        "Benchmark seed offset",
        0,
        spec.benchmark.num_seeds - 1,
        spec.demo.default_seed_offset,
        1,
        help="Offset into the exact held-out seed family used by the final benchmark.",
    )

    before = "temporal"
    after = "rl"
    switch_time = None
    if mode == "interactive_switch":
        st.sidebar.warning("Interactive switch mode uses the same physical/stochastic experiment, but it is NOT part of the reported Temporal-vs-pure-PPO benchmark.")
        before = st.sidebar.selectbox("Before switch", ["temporal", "rl"], index=0)
        after = st.sidebar.selectbox("After switch", ["temporal", "rl"], index=1)
        switch_time = float(st.sidebar.slider("Switch time (s)", 0, 1800, 900, 15))

    show_trails = st.sidebar.checkbox("Show orbital trails", value=True)
    trail_seconds = float(st.sidebar.slider("Trail length (s)", 30, 600, 180, 15))
    return mode, seed_offset, before, after, switch_time, show_trails, trail_seconds


def _timeline(replay) -> float:
    if "final_frame_index" not in st.session_state:
        st.session_state.final_frame_index = 0
    if "final_playing" not in st.session_state:
        st.session_state.final_playing = False

    frames = len(replay.times)
    index = max(0, min(frames - 1, int(st.session_state.final_frame_index)))
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.2, 4, 1.5])
    if c1.button("◀ Step"):
        st.session_state.final_playing = False
        index = max(0, index - 1)
    if c2.button("▶ Step"):
        st.session_state.final_playing = False
        index = min(frames - 1, index + 1)
    if c3.button("Play / Pause"):
        st.session_state.final_playing = not st.session_state.final_playing
    index = c4.slider("Timeline", 0, frames - 1, index, 1)
    speed = c5.select_slider("Playback", [1, 2, 4, 8], value=4)
    st.session_state.final_frame_index = index
    time_s = float(replay.times[index])
    if st.session_state.final_playing:
        time.sleep(max(0.02, 0.28 / speed))
        st.session_state.final_frame_index = (index + 1) % frames
        st.rerun()
    return time_s


def _metrics(replay, time_s: float):
    summary = replay.summary()
    cols = st.columns(7)
    cols[0].metric("Requested", replay.requested_algorithm_at(time_s).upper())
    cols[1].metric("Actual", replay.actual_algorithm_at(time_s).upper())
    cols[2].metric("Delivered", f"{summary.delivered}/{summary.bundles}")
    cols[3].metric("On-time", f"{100 * summary.deadline_success:.1f}%")
    cols[4].metric("Transfer failures", f"{100 * summary.transfer_failure_rate:.1f}%")
    cols[5].metric("Mean wasted", f"{summary.mean_wasted_mb:.2f} MB")
    cols[6].metric("Fallbacks", summary.total_fallbacks)


def _benchmark_summary(spec):
    path = spec.benchmark.output_dir / "summary.json"
    if not path.exists():
        st.info("Final benchmark summary not found yet. Run `python run_final_benchmark.py` to populate this tab.")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    algorithms = payload.get("algorithms", {})
    rows = []
    for algorithm in ("temporal", "rl_pure"):
        metrics = algorithms.get(algorithm, {})
        if not metrics:
            continue
        rows.append({
            "algorithm": algorithm,
            "delivery_ratio": metrics["delivery_ratio"]["mean"],
            "deadline_success": metrics["deadline_success"]["mean"],
            "priority_weighted_timely": metrics["priority_weighted_timely"]["mean"],
            "mean_latency_s": metrics["mean_latency_s"]["mean"],
            "mean_hops": metrics["mean_hops"]["mean"],
            "transfer_failure_rate": metrics["transfer_failure_rate"]["mean"],
            "mean_wasted_mb": metrics["mean_wasted_mb"]["mean"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Loaded from {path}")


def main():
    spec = load_final_spec()
    st.title("Final physical routing experiment")
    st.caption(
        "The reported PPO and Temporal visualizations are generated by the exact same stochastic runner used by the final benchmark. "
        "The PPO mode refuses to run if the final checkpoint metadata does not match the locked scenario config."
    )

    mode, seed_offset, before, after, switch_time, show_trails, trail_seconds = _sidebar(spec)

    try:
        replay = _cached_replay(mode, seed_offset, before, after, switch_time)
    except Exception as exc:
        st.error(str(exc))
        if mode == "reported_rl":
            st.code(str(spec.model), language=None)
            st.caption("The final PPO demo intentionally does not fall back to an old checkpoint or to Temporal routing.")
        st.stop()

    traffic_seed, stochastic_seed = spec.benchmark.seeds(seed_offset)
    status = "REPORTED EXPERIMENT" if replay.reported_experiment else "INTERACTIVE / NOT REPORTED"
    st.markdown(f"**{status}** · seed offset `{seed_offset}` · traffic `{traffic_seed}` · stochastic `{stochastic_seed}` · bundles `{spec.benchmark.bundles_per_seed}`")
    if replay.model_loaded:
        st.caption(f"PPO checkpoint: `{spec.model}` · locked config SHA `{spec.scenario_config_sha256[:12]}…`")

    time_s = _timeline(replay)
    _metrics(replay, time_s)

    choices = [bundle.bundle_id for bundle in replay.bundle_runs]
    selected = st.selectbox(
        "Highlight bundle",
        choices,
        index=choices.index(replay.selected_bundle_id) if replay.selected_bundle_id in choices else 0,
    )

    left, right = st.columns((1.55, 1.15), gap="medium")
    with left:
        st.plotly_chart(
            build_orbital_figure(replay, time_s, selected_bundle_id=selected, show_trails=show_trails, trail_seconds=trail_seconds),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        st.plotly_chart(
            build_topology_figure(replay, time_s, selected_bundle_id=selected),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    tabs = st.tabs(["Bundle inspector", "Transfer/event log", "Reported benchmark", "Experiment identity", "Downloads"])
    bundle_df = _bundle_frame(replay)
    event_df = pd.DataFrame(replay.event_rows())

    with tabs[0]:
        row = bundle_df[bundle_df.bundle_id == selected]
        if not row.empty:
            st.write(row.iloc[0].to_dict())
        st.dataframe(bundle_df, use_container_width=True, hide_index=True)

    with tabs[1]:
        current = event_df[event_df.time_s <= round(time_s, 3)].tail(50)
        st.dataframe(current, use_container_width=True, hide_index=True)

    with tabs[2]:
        _benchmark_summary(spec)

    with tabs[3]:
        st.json({
            "experiment": spec.name,
            "reported_mode": replay.reported_experiment,
            "scenario_config": str(spec.scenario_config),
            "config_sha256": spec.scenario_config_sha256,
            "ppo_model": str(spec.model),
            "seed_offset": seed_offset,
            "traffic_seed": traffic_seed,
            "stochastic_seed": stochastic_seed,
            "bundles": spec.benchmark.bundles_per_seed,
            "max_hops": spec.benchmark.max_hops,
            "max_attempts": spec.benchmark.max_attempts,
        })

    with tabs[4]:
        st.download_button("Download replay bundles CSV", bundle_df.to_csv(index=False).encode(), "demo_bundle_results.csv", "text/csv")
        st.download_button("Download replay events CSV", event_df.to_csv(index=False).encode(), "demo_event_log.csv", "text/csv")
        st.download_button("Download replay identity JSON", json.dumps({
            "mode": replay.mode,
            "reported_experiment": replay.reported_experiment,
            "seed_offset": seed_offset,
            "traffic_seed": traffic_seed,
            "stochastic_seed": stochastic_seed,
            "model": str(spec.model),
            "config_sha256": spec.scenario_config_sha256,
            "summary": asdict(replay.summary()),
        }, indent=2).encode(), "demo_replay_identity.json", "application/json")


if __name__ == "__main__":
    main()
