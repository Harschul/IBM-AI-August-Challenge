"""Smooth, progressive-disclosure frontend for the final locked experiment."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from src.experiment.spec import load_final_spec
from src.frontend.client_ui import render_client_html
from src.frontend.replay import build_replay
from src.frontend.scenarios import build_scenario_profiles
from src.integration.config import load_config

st.set_page_config(
    page_title="Constellation Routing Simulator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stHeader"] { display:none; }
    #MainMenu, footer { visibility:hidden; }
    .block-container { max-width: 1840px; padding: .55rem .75rem 1rem; }
    [data-testid="stVerticalBlock"] { gap: .45rem; }
    [data-testid="stHorizontalBlock"] { gap: .55rem; }
    .stApp { background:#fff; }
    .project-title { font: 900 1rem ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; color:#172033; letter-spacing:-.03em; padding-top:.45rem; white-space:nowrap; }
    .project-subtitle { font: .68rem ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; color:#718198; margin-top:.1rem; }
    div[data-baseweb="select"] > div { min-height:2.35rem; border-color:#d7e0ea; }
    div[role="radiogroup"] { gap:.15rem; }
    div[role="radiogroup"] label { padding:.2rem .45rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _profile_rows(spec_path: str):
    spec = load_final_spec(spec_path)
    config = load_config(spec.scenario_config)
    return build_scenario_profiles(spec, config)


@st.cache_resource(show_spinner="Computing the locked replay…")
def _cached_replay(
    mode: str,
    seed_offset: int,
    before_policy: str,
    after_policy: str,
    switch_time_s: float | None,
):
    return build_replay(
        mode=mode,
        seed_offset=seed_offset,
        before_policy=before_policy,
        after_policy=after_policy,
        switch_time_s=switch_time_s,
    )


def main() -> None:
    spec = load_final_spec()
    config = load_config(spec.scenario_config)
    profiles = _profile_rows("config/final_experiment.json")
    # load_final_spec does not preserve its source path in raw; use the canonical file.
    if not profiles:
        st.error("No held-out traffic profiles are available.")
        st.stop()

    title_col, route_col, scenario_col = st.columns([3.2, 2.0, 7.0], vertical_alignment="center")
    with title_col:
        st.markdown('<div class="project-title">CONSTELLATION ROUTING SIMULATOR</div>', unsafe_allow_html=True)
        st.markdown('<div class="project-subtitle">physical contacts · stochastic transfers · three operational ground receivers</div>', unsafe_allow_html=True)
    with route_col:
        routing_label = st.radio(
            "Routing",
            ["Temporal", "PPO", "Switch demo"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with scenario_col:
        labels = [profile.label for profile in profiles]
        selected_label = st.selectbox(
            "Held-out traffic run",
            labels,
            index=min(spec.demo.default_seed_offset, len(labels) - 1),
            label_visibility="collapsed",
            help="Each option is one fixed held-out benchmark traffic + stochastic seed pair. The label summarizes the real generated workload; it does not create a new scenario.",
        )

    selected_profile = next(profile for profile in profiles if profile.label == selected_label)
    st.caption(selected_profile.description)
    before_policy, after_policy, switch_time_s = "temporal", "rl", 900.0
    if routing_label == "Temporal":
        mode = "reported_temporal"
        before_policy = after_policy = "temporal"
        switch_time_s = None
    elif routing_label == "PPO":
        mode = "reported_rl"
        before_policy = after_policy = "rl"
        switch_time_s = None
    else:
        mode = "interactive_switch"
        with st.popover("Switch settings"):
            before_name = st.radio("Start with", ["Temporal", "PPO"], horizontal=True)
            after_name = st.radio("Then use", ["PPO", "Temporal"], horizontal=True)
            switch_time_s = float(st.slider("Switch at", 60, max(60, int(config.horizon_s) - 60), min(900, int(config.horizon_s) - 60), 30, format="%d s"))
            before_policy = "temporal" if before_name == "Temporal" else "rl"
            after_policy = "temporal" if after_name == "Temporal" else "rl"
            if before_policy == after_policy:
                st.caption("Choose different policies to demonstrate a real switch.")

    try:
        replay = _cached_replay(
            mode,
            selected_profile.offset,
            before_policy,
            after_policy,
            switch_time_s,
        )
    except Exception as exc:
        st.error(str(exc))
        if mode == "reported_rl":
            st.caption("Reported PPO never silently substitutes Temporal or an older checkpoint.")
        st.stop()

    html = render_client_html(
        replay,
        scenario_label=selected_profile.label,
        scenario_description=selected_profile.description,
    )
    components.html(html, height=1210, scrolling=False)


if __name__ == "__main__":
    main()
