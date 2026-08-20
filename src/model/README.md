# Satellite Network Coverage-First Optimizer

This bundle contains the complete satellite-network simulation, high-speed SciPy optimizer, Earth receiver model, scoring code, CSV diagnostics and MP4 renderer.

The current optimizer implements three important improvements:

1. **Worst-frame Earth coverage is optimized, not average coverage alone.** A temporary coverage hole cannot hide inside a good average.
2. **Worst receiver distance is included as well as mean receiver distance.** A badly served Earth location can no longer disappear inside a strong average.
3. **Full sampled Earth coverage is treated as a feasibility requirement.** Incomplete candidates remain below `0.5` fitness. Candidates whose worst sampled frame reaches 100% coverage enter the upper half of the score and are then ranked by satellite-link and receiver-distance quality.

The renderer is deliberately simple: the orbital view and network topology remain centred on the same vertical axis. The only live text is **EARTH COVERAGE** at the top-left and **TOTAL LINKS** at the top-right. The 100 Earth receiver dots are rendered as faint light-blue points.

---

## Files

- `nodes.py` — `Satellite` domain model, circular-orbit propagation, inter-satellite distance and Earth line-of-sight checks.
- `network.py` — readable object-oriented reference simulation and immutable per-frame snapshots.
- `satellite_generator.py` — random constellation generation and JSON save/load.
- `ground_receivers.py` — Fibonacci-distributed Earth receiver points and satellite-above-horizon geometry.
- `fitness.py` — shared final normalized fitness equations used by both scoring implementations.
- `scoring.py` — readable/reference scoring, diagnostics and CSV export.
- `fast_scoring.py` — fully vectorized NumPy evaluator used inside SciPy optimization.
- `optimize_constellation.py` — vectorized Differential Evolution search, benchmarking, Ctrl+C checkpointing and reference validation.
- `network_simulation.py` — random/replay simulation, score report, CSV export and MP4 rendering.
- `benchmark_optimizer.py` — checks that the fast and reference implementations agree and estimates optimization time on your computer.
- `requirements.txt` — Python dependencies.
- `BENCHMARK_RESULTS.txt` — validation and timing results from the build environment.

---

# 1. Current fitness function

## 1.1 Satellite link score

At every frame, every possible satellite pair is considered.

For a valid satellite-to-satellite link:

```text
link contribution = actual link distance / pair connection range
```

A disconnected pair contributes `0`.

The sum is divided by the total number of possible satellite pairs:

```text
0 <= link_score <= 1
```

This automatically rewards both:

- more simultaneous valid links;
- longer valid links.

For 30 satellites there are:

```text
30 * 29 / 2 = 435 possible links
```

---

## 1.2 Earth receivers

By default, `100` virtual receiver points are distributed approximately evenly across Earth using a Fibonacci sphere.

At every frame, every receiver:

1. checks which satellites are above its local horizon / minimum elevation angle;
2. finds the nearest visible satellite;
3. converts that distance to a normalized penalty using `ground_distance_scale`;
4. receives penalty `1` if it has no visible satellite.

For a covered receiver:

```text
receiver penalty = clip(nearest satellite distance / ground_distance_scale, 0, 1)
```

Default:

```text
ground_distance_scale = 5000 km
```

---

## 1.3 Mean + worst receiver distance

The optimizer no longer uses only the mean receiver penalty.

It calculates:

```text
mean_receiver_penalty
worst_receiver_penalty
```

and blends them:

```text
receiver_penalty =
    (1 - worst_distance_weight) * mean_receiver_penalty
    + worst_distance_weight * worst_receiver_penalty
```

Default:

```text
worst_distance_weight = 0.5
```

so mean and worst receiver distance receive equal weight.

If any sampled receiver is uncovered at any sampled frame, its normalized penalty is `1`, so the global worst receiver penalty is also `1`.

---

## 1.4 Link-distance minus receiver-distance quality

Once the mean/worst receiver penalty has been combined, the normalized network quality is:

```text
quality = (1 + average_link_score - blended_receiver_penalty) / 2
```

Therefore:

```text
0 <= quality <= 1
```

This preserves the original design idea:

```text
maximize satellite link quality
-
receiver-to-satellite distance
```

without mixing raw kilometre totals with unrelated scales.

---

## 1.5 Worst-frame coverage comes first

Coverage is deliberately **not averaged away** for the final feasibility decision.

The optimizer calculates:

```text
worst_coverage = minimum Earth coverage over every sampled frame
```

If worst-frame coverage is incomplete:

```text
fitness = 0.5 * worst_coverage + tiny quality tie-break
```

The tie-break is deliberately smaller than the value of covering one additional receiver, so a lower-coverage candidate cannot beat a higher-coverage candidate merely by having better satellite links.

If worst-frame coverage reaches 100%:

```text
fitness = 0.5 + 0.5 * quality
```

This creates two regions:

```text
0.0 <= fitness < 0.5   -> incomplete sampled Earth coverage
0.5 <= fitness <= 1.0  -> full sampled Earth coverage
```

So the optimizer behaves approximately lexicographically:

```text
FIRST: maximize worst-frame Earth coverage to 100%
THEN : maximize long/many satellite links while minimizing mean + worst ground distance
```

This is much stronger than the previous `coverage ** 10` soft penalty.

---

# 2. Installation

Open a terminal in this folder.

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The required Python packages are NumPy, SciPy and Matplotlib.

FFmpeg is required only for MP4 rendering. Check whether it is available with:

```bash
ffmpeg -version
```

Optimization and numerical validation still work without FFmpeg.

---

# 3. Complete operating pipeline

This is the recommended full before/after workflow.

## Step A — create and render a random baseline

Run:

```bash
python network_simulation.py --save-constellation random_baseline.json --output random_baseline.mp4 --metrics-csv random_baseline_metrics.csv
```

This will:

1. generate the default 30-satellite random constellation using seed 42;
2. save its exact initial state to `random_baseline.json`;
3. run the readable/reference OOP simulation;
4. evaluate satellite links and the 100 Earth receiver points;
5. print the coverage-first score summary;
6. write one diagnostics row per simulation frame;
7. render `random_baseline.mp4`.

The video shows only two live values:

```text
EARTH COVERAGE                            TOTAL LINKS
```

The orbital and topology plots remain vertically centred and aligned.

If you only want numerical results and do not need the baseline video:

```bash
python network_simulation.py --save-constellation random_baseline.json --metrics-csv random_baseline_metrics.csv --no-video
```

---

## Step B — optimize that exact baseline

Run:

```bash
python optimize_constellation.py --baseline-constellation random_baseline.json
```

By default the optimizer changes, for every satellite:

```text
phase
inclination
RAAN
```

These remain fixed from the baseline:

```text
radius
angular velocity
connection range
```

Changing phase, inclination and RAAN changes the initial 3D satellite positions and orbital-plane orientations while keeping the circular-orbit model intact.

Default search configuration:

```text
30 satellites
90 optimization variables
180 sampled simulation frames
100 Earth receivers
10 Differential Evolution generations
popsize = 5
```

The optimizer writes:

```text
optimized_constellation.json
optimizer_checkpoint.json
optimized_constellation_metrics.csv
```

Before the search starts, it benchmarks the vectorized evaluator on your own machine and prints an estimated wall time.

---

## Step C — render the optimized result

Run:

```bash
python network_simulation.py --constellation optimized_constellation.json --output optimized_constellation.mp4 --metrics-csv optimized_render_metrics.csv
```

You can now compare:

```text
random_baseline.mp4
random_baseline_metrics.csv

vs.

optimized_constellation.mp4
optimized_render_metrics.csv
```

---

## Step D — high-resolution Earth coverage validation

The optimizer intentionally uses only 100 Earth points to stay fast. A solution with 100% coverage on those 100 points is **100% sampled coverage**, not a mathematical proof of continuous global coverage.

Validate the winner using far more points:

```bash
python network_simulation.py --constellation optimized_constellation.json --ground-points 5000 --no-video --metrics-csv coverage_validation_5000.csv
```

For a stricter test:

```bash
python network_simulation.py --constellation optimized_constellation.json --ground-points 20000 --no-video --metrics-csv coverage_validation_20000.csv
```

The most important console value is:

```text
Worst-frame Earth coverage
```

For a strong sampled global-coverage result, this should remain:

```text
100.00%
```

at the higher receiver count as well.

---

# 4. Copy/paste normal workflow

```bash
python -m pip install -r requirements.txt
python network_simulation.py --save-constellation random_baseline.json --output random_baseline.mp4 --metrics-csv random_baseline_metrics.csv
python optimize_constellation.py --baseline-constellation random_baseline.json
python network_simulation.py --constellation optimized_constellation.json --output optimized_constellation.mp4 --metrics-csv optimized_render_metrics.csv
python network_simulation.py --constellation optimized_constellation.json --ground-points 5000 --no-video --metrics-csv coverage_validation_5000.csv
```

---

# 5. Benchmark before a long optimization

Run:

```bash
python benchmark_optimizer.py
```

It performs two checks:

1. evaluates the same constellation with the vectorized optimizer model and the readable OOP/reference model;
2. measures candidate evaluations per second and estimates the default Differential Evolution runtime.

The fast and reference scores should agree to floating-point precision.

---

# 6. Optimization controls

## Very quick test

```bash
python optimize_constellation.py --baseline-constellation random_baseline.json --frames 60 --maxiter 2 --popsize 3
```

Use this to verify the workflow before a larger search.

## Default search

```bash
python optimize_constellation.py --baseline-constellation random_baseline.json
```

## Deeper search

```bash
python optimize_constellation.py --baseline-constellation random_baseline.json --maxiter 30 --popsize 6
```

The approximate maximum number of candidate constellations is:

```text
(maxiter + 1) * popsize * number_of_variables
```

For 30 satellites with phase/inclination/RAAN:

```text
number_of_variables = 90
```

---

## Change mean-vs-worst receiver-distance importance

Default equal weighting:

```bash
python optimize_constellation.py --baseline-constellation random_baseline.json --worst-distance-weight 0.5
```

Emphasize the worst-served ground location more strongly:

```bash
python optimize_constellation.py --baseline-constellation random_baseline.json --worst-distance-weight 0.75
```

A value of `1.0` ignores the mean receiver distance and optimizes the worst receiver penalty only after full sampled coverage has been reached.

Use the same flag when re-scoring/rendering if you changed it during optimization:

```bash
python network_simulation.py --constellation optimized_constellation.json --worst-distance-weight 0.75
```

---

## Require a minimum ground elevation angle

Default horizon visibility:

```text
min elevation = 0 degrees
```

For a stricter 10-degree minimum elevation:

```bash
python optimize_constellation.py --baseline-constellation random_baseline.json --min-elevation-deg 10
```

Render/validate with the same setting:

```bash
python network_simulation.py --constellation optimized_constellation.json --min-elevation-deg 10
```

---

## Optimize angular velocity as well

```bash
python optimize_constellation.py --baseline-constellation random_baseline.json --parameters phase inclination raan angular_velocity
```

This increases the problem from 90 to 120 variables for 30 satellites.

---

# 7. Stop and resume

Press:

```text
Ctrl+C
```

The optimizer keeps the best candidate from the last completed vectorized batch and writes the best constellation found so far.

The checkpoint is normally:

```text
optimizer_checkpoint.json
```

Resume from it with:

```bash
python optimize_constellation.py --baseline-constellation random_baseline.json --resume-checkpoint optimizer_checkpoint.json
```

Checkpoints from the previous soft-coverage objective are accepted as starting vectors, but their old stored scores are not comparable and are recalculated using the new coverage-first objective.

---

# 8. Rendering controls

The normal render is:

```bash
python network_simulation.py --constellation optimized_constellation.json
```

The top-line indicators update once per second of video by default.

Update them every visual frame with:

```bash
python network_simulation.py --constellation optimized_constellation.json --metrics-update-seconds 0
```

The faint light-blue dots are the virtual Earth receiver/sample locations.

---

# 9. Important interpretation

The optimizer currently addresses:

- satellite-to-satellite range and Earth line of sight;
- many/long valid satellite links;
- nearest visible satellite distance for Earth receivers;
- worst sampled receiver distance;
- worst-frame sampled Earth coverage.

It **does not yet enforce graph-wide routing connectivity**. A constellation can have full ground coverage and many satellite links without guaranteeing that every satellite belongs to one connected communication component at every frame. If the engineering requirement is literally "a signal from any satellite can route to any point on Earth," graph connectivity/reachability should be added as a separate feasibility requirement in a future revision.
