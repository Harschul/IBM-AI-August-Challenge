# Satellite network scoring + optimization bundle

Drop all files in this folder together. The Python imports assume these exact
filenames.

## Files

- `nodes.py` — satellite physics, propagation, range and line-of-sight checks.
- `network.py` — network stepping and immutable snapshots.
- `satellite_generator.py` — random constellation generation + JSON save/load.
- `scoring.py` — normalized 0–1 fitness and per-snapshot diagnostics.
- `network_simulation.py` — random/replayed simulation, CSV export and MP4 render.
- `optimize_constellation.py` — SciPy differential-evolution optimizer.

## Score definition

For every snapshot and every possible satellite pair:

- invalid/disconnected pair: contribution = `0`
- valid link: contribution = `distance / pair_connection_range`

The snapshot score is the mean contribution over all possible pairs. The final
network score is the mean of all snapshot scores. Therefore the score is always
between `0` and `1` and naturally rewards both **more links** and **longer links**.

## Install

```bash
python -m pip install -r requirements.txt
```

MP4 rendering also needs `ffmpeg` installed and available on your PATH.

## Run the random baseline

```bash
python network_simulation.py
```

Default outputs:

- `satellite_network.mp4`
- `satellite_network_metrics.csv` — one row for every snapshot
- `random_constellation.json` — exact initial conditions for reproducibility

The video contains a snapshot panel with active links, mean valid-link distance,
total valid-link distance, snapshot score and running score. By default the panel
changes once per second of video so it remains readable.

Update the panel every video frame instead:

```bash
python network_simulation.py --metrics-update-seconds 0
```

Score only, with no MP4 render:

```bash
python network_simulation.py --no-video
```

## Optimize initial conditions

The optimizer defaults to phase + inclination + RAAN for every satellite while
keeping radius, angular velocity and radio range fixed from the random baseline.

```bash
python optimize_constellation.py
```

For a much faster first test, optimize phase only:

```bash
python optimize_constellation.py --parameters phase --maxiter 5 --popsize 4 --frames 100
```

The optimizer writes:

- `optimized_constellation.json`
- `optimized_constellation_metrics.csv`

Render the optimized result at full video resolution:

```bash
python network_simulation.py --constellation optimized_constellation.json
```

## Useful switches

```bash
python network_simulation.py --help
python optimize_constellation.py --help
```

Optimization can be computationally expensive. Use fewer scoring frames and/or
phase-only optimization while testing, then increase the search size once the
pipeline is behaving as expected.
