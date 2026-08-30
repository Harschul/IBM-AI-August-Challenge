# Integrated frontend v2

The frontend keeps the minimalist orbital-render style while making the routing
state auditable rather than decorative.

## Views

- synchronized **3D orbital constellation**
- synchronized **2D network topology**
- active satellite and ground links
- moving data-bundle markers
- selected route highlighting
- timeline scrub / step / playback controls

## Multi-source science traffic

The 14-node interface now contains three science spacecraft. New bundles are
randomly (but reproducibly) assigned to SCI-0, SCI-1 or SCI-2.

## Routing transparency

The operator can request temporal or RL routing before and after a selected
switch time. For every hop the replay separately stores the requested algorithm
and the actual algorithm that chose the next hop.

Examples:

```text
requested=temporal  actual=temporal  fallback=false
requested=rl        actual=rl        fallback=false
requested=rl        actual=temporal  fallback=true
```

The last case is colored and labeled as fallback throughout the UI. It is never
counted as executed RL.

## Main implementation files

- `src/frontend/app.py` — UI and metrics
- `src/frontend/replay.py` — routing replay and requested/actual audit state
- `src/frontend/figures.py` — synchronized Plotly views
- `src/frontend/layout.py` — fixed 14-node topology layout
- `src/integration/scenario.py` — 3 science / 6 LEO / 2 GEO physical scenario
- `src/integration/traffic.py` — multi-source bundle generation
