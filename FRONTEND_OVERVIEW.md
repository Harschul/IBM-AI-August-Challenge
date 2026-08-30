# Frontend bundle overview

This bundle adds a cohesive frontend for the integrated demo with the following
focus areas:

- **3D orbital render** of the physical constellation
- **2D network topology graph** shown side-by-side with the orbital scene
- **Clearly rendered links** for both inter-satellite and satellite-ground contacts
- **Animated packet transfer** markers that move along the active route
- **Mid-run switching** between **temporal routing** and **RL routing**
- **Minimalist visual design** aligned to the repository's earlier video renders

## Main files

- `src/frontend/app.py` — Streamlit UI and interaction controls
- `src/frontend/replay.py` — replay generation, packet hop playback, policy switching
- `src/frontend/figures.py` — Plotly figure builders for orbital and topology views
- `src/frontend/layout.py` — fixed 2D topology layout for the 14-node scenario
- `src/frontend/theme.py` — shared style constants
- `run_frontend.py` — one-command launcher

## Included quality-of-life features

- Bundle selector to highlight one science bundle's full route
- Timeline scrubber plus step/play controls
- Delivery metrics and fallback counts
- Event log table for demo/debugging
- Download buttons for bundle table, event log and summary JSON
- Graceful fallback when an RL checkpoint is not available

## How policy switching works

The replay accepts:

- a **policy before switch** (`temporal` or `rl`)
- a **policy after switch** (`temporal` or `rl`)
- an optional **switch time in seconds**

At each routing decision the replay engine chooses the active policy from those
controls. This means a bundle can begin under RL and complete under the temporal
router, or vice versa.
