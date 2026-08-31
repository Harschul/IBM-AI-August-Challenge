# Decluttered frontend drop-in

This bundle replaces only the presentation layer of the final project.
It does not change routing, PPO, contact planning, stochastic transfers,
benchmark evidence, model files, or experiment configuration.

## What changed

- removed architecture dropdown and playback speed control
- removed 3D orbit rendering
- added a lightweight 2D orbital projection with a rotating Earth grid
- topology base view contains only nodes + contacts available at the current time
- active/non-delivered packets live in one indexed table
- delivered packets move into a collapsed archive table
- clicking a row selects that packet
- selected packet overlays its successful route and failed attempts on topology
- packet metadata, raw attempt history, node/satellite metadata, queues and full experiment/config metadata are available under Metadata & diagnostics

## Install

From the root of the existing final Git repository:

```bash
rsync -av /path/to/IBM-AI-August-Challenge-DECLUTTERED-FRONTEND-DROPIN/ ./
```

Do not use --delete.

Then:

```bash
python -m pytest -q
python scripts/final/verify_release.py
python run_frontend.py
```

Expected validation: 29 passed and release verifier PASS.
