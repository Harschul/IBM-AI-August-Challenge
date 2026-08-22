# Satellite Network Optimizers: Coverage-First vs Distance-Only

This bundle contains the complete satellite-network model, the existing **coverage-first** optimizer, a new **distance-only maximin-separation** optimizer, and the renderer used to replay either result.

The intended workflow is to start both optimizers from the exact same constellation, render both outputs, and keep the results separate for later comparison.

**No new code in this bundle automatically decides which optimizer is better at delivering data.** There is no new end-to-end data routing, packet queue, storage, latency, throughput, or receiver-delivery comparison layer.

The existing simulator still computes its existing link and Earth-receiver diagnostics when a constellation is rendered. Those diagnostics are not used by the new distance-only optimizer.

---

# 1. What is new

## 1.1 New optimizer: `optimize_separation_constellation.py`

The new optimizer has exactly one search objective:

```text
maximize the minimum satellite-to-satellite distance
observed over every unique satellite pair
and every sampled simulation frame
```

Equivalently:

```text
score = min over frames t and pairs (i,j) of distance(satellite_i(t), satellite_j(t))
```

Differential Evolution maximizes that value.

This is a **maximin** objective. A candidate is only as good as its closest satellite pair at its worst sampled instant.

The distance-only optimizer does **not** use any of these quantities in its objective:

- Earth coverage;
- the 100 Earth receivers;
- receiver-to-satellite distance;
- satellite connection range;
- number of active satellite links;
- Earth line of sight;
- route existence;
- data storage;
- data transfer rate;
- latency;
- throughput;
- the existing coverage-first fitness score.

Those items therefore cannot influence which constellation the new optimizer selects.

## 1.2 Renderer labeling

`network_simulation.py` now accepts:

```bash
--video-label "YOUR LABEL"
```

The label is embedded at the top center of every rendered frame. This makes it easy to distinguish the two optimizer outputs visually.

---

# 2. Files in this bundle

- `nodes.py` — satellite state, circular-orbit propagation, distance, line-of-sight, and connection checks.
- `network.py` — reference network simulation and immutable frame snapshots.
- `satellite_generator.py` — seeded random constellation generation plus JSON save/load.
- `ground_receivers.py` — Fibonacci-distributed Earth receiver points and visibility geometry.
- `fitness.py` — shared existing coverage-first fitness equations.
- `scoring.py` — readable/reference Earth coverage and network scoring.
- `fast_scoring.py` — vectorized evaluator used by the existing coverage-first optimizer.
- `optimize_constellation.py` — existing coverage-first Differential Evolution optimizer.
- `optimize_separation_constellation.py` — **new distance-only maximin-separation optimizer**.
- `network_simulation.py` — simulation, existing diagnostics, CSV export, and MP4 rendering; now also supports `--video-label`.
- `benchmark_optimizer.py` — existing fast/reference coverage-first scoring benchmark.
- `requirements.txt` — Python dependencies.

---

# 3. Installation

Open a terminal in this folder.

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The Python requirements are:

```text
NumPy
Matplotlib
SciPy
```

MP4 rendering also requires FFmpeg. Check it with:

```bash
ffmpeg -version
```

If FFmpeg is missing, optimization and non-video simulation still work. Rendering MP4 files does not.

---

# 4. Recommended controlled workflow

Use one saved baseline for both optimizers. This ensures both searches begin from the same satellite radii, angular velocities, connection ranges, and initial random constellation.

## Step A — create the shared baseline

Run:

```bash
python network_simulation.py \
  --save-constellation shared_baseline.json \
  --output VIDEO_0_shared_baseline.mp4 \
  --metrics-csv shared_baseline_metrics.csv \
  --video-label "0 - SHARED RANDOM BASELINE"
```

With the default settings this creates 30 satellites using seed 42 and saves the exact initial constellation to:

```text
shared_baseline.json
```

The MP4 is clearly labeled:

```text
0 - SHARED RANDOM BASELINE
```

If you do not need a baseline video, use:

```bash
python network_simulation.py \
  --save-constellation shared_baseline.json \
  --metrics-csv shared_baseline_metrics.csv \
  --no-video
```

---

## Step B — run the existing coverage-first optimizer

Run:

```bash
python optimize_constellation.py \
  --baseline-constellation shared_baseline.json \
  --output coverage_optimized_constellation.json \
  --checkpoint coverage_optimizer_checkpoint.json \
  --metrics-csv coverage_optimized_metrics.csv
```

By default it optimizes these parameters for each satellite:

```text
phase
inclination
RAAN
```

It keeps these baseline values fixed unless you explicitly change the optimized parameter list:

```text
radius
angular velocity
connection range
```

The coverage-first optimizer prioritizes worst-frame sampled Earth coverage and then its existing link/receiver-distance quality function.

Its primary saved constellation is:

```text
coverage_optimized_constellation.json
```

---

## Step C — run the new distance-only optimizer

Use the same baseline:

```bash
python optimize_separation_constellation.py \
  --baseline-constellation shared_baseline.json \
  --output distance_optimized_constellation.json \
  --checkpoint distance_optimizer_checkpoint.json
```

By default it optimizes the same three orbital parameters:

```text
phase
inclination
RAAN
```

and leaves these baseline values fixed:

```text
radius
angular velocity
connection range
```

Its optimization score is only:

```text
minimum pairwise satellite distance across all sampled frames
```

The 100 Earth receiver locations are not generated or evaluated inside this optimizer.

The primary output is:

```text
distance_optimized_constellation.json
```

---

## Step D — render the coverage-first result

Run:

```bash
python network_simulation.py \
  --constellation coverage_optimized_constellation.json \
  --output VIDEO_A_coverage_first.mp4 \
  --metrics-csv VIDEO_A_coverage_first_metrics.csv \
  --video-label "A - COVERAGE-FIRST OPTIMIZER"
```

The video itself is labeled:

```text
A - COVERAGE-FIRST OPTIMIZER
```

---

## Step E — render the distance-only result

Run:

```bash
python network_simulation.py \
  --constellation distance_optimized_constellation.json \
  --output VIDEO_B_distance_only.mp4 \
  --metrics-csv VIDEO_B_distance_only_metrics.csv \
  --video-label "B - DISTANCE-ONLY MAXIMIN SEPARATION"
```

The video itself is labeled:

```text
B - DISTANCE-ONLY MAXIMIN SEPARATION
```

The renderer will still display its existing Earth coverage and total-link indicators. Those values are observations from the replay; they were **not** part of the distance-only optimization objective.

---

# 5. Copy/paste full workflow

Linux/macOS shell form:

```bash
python -m pip install -r requirements.txt

python network_simulation.py --save-constellation shared_baseline.json --output VIDEO_0_shared_baseline.mp4 --metrics-csv shared_baseline_metrics.csv --video-label "0 - SHARED RANDOM BASELINE"

python optimize_constellation.py --baseline-constellation shared_baseline.json --output coverage_optimized_constellation.json --checkpoint coverage_optimizer_checkpoint.json --metrics-csv coverage_optimized_metrics.csv

python optimize_separation_constellation.py --baseline-constellation shared_baseline.json --output distance_optimized_constellation.json --checkpoint distance_optimizer_checkpoint.json

python network_simulation.py --constellation coverage_optimized_constellation.json --output VIDEO_A_coverage_first.mp4 --metrics-csv VIDEO_A_coverage_first_metrics.csv --video-label "A - COVERAGE-FIRST OPTIMIZER"

python network_simulation.py --constellation distance_optimized_constellation.json --output VIDEO_B_distance_only.mp4 --metrics-csv VIDEO_B_distance_only_metrics.csv --video-label "B - DISTANCE-ONLY MAXIMIN SEPARATION"
```

Expected key outputs:

```text
shared_baseline.json
coverage_optimized_constellation.json
distance_optimized_constellation.json

VIDEO_0_shared_baseline.mp4
VIDEO_A_coverage_first.mp4
VIDEO_B_distance_only.mp4
```

---

# 6. Distance-only optimizer details

## 6.1 Default settings

The new script defaults to:

```text
30 satellites
20,000 seconds simulated per candidate
180 sampled frames per candidate
phase + inclination + RAAN optimized
10 Differential Evolution generations
popsize = 5
candidate chunk size = 8
chunk workers = 1
```

For 30 satellites there are:

```text
30 * 29 / 2 = 435 unique satellite pairs
```

At every sampled frame the optimizer computes all 435 pair distances. The candidate score is the smallest value anywhere in that frame/pair grid.

## 6.2 Why the minimum is used

Optimizing the average pair distance could allow two satellites to get very close as long as the other satellites stay far apart.

The maximin objective prevents that averaging effect:

```text
candidate score = its single closest sampled pair encounter
```

Increasing the score therefore pushes up the worst sampled separation.

## 6.3 Diagnostics printed by the new optimizer

The new optimizer reports:

```text
minimum pair distance
mean frame minimum distance
mean all-pair distance
closest-approach frame
closest-approach time
closest satellite pair
```

Only `minimum pair distance` is optimized. The other values are diagnostics and do not affect selection.

---

# 7. Quick test runs

These commands are useful for checking that everything launches correctly before a longer optimization.

Coverage-first quick run:

```bash
python optimize_constellation.py \
  --baseline-constellation shared_baseline.json \
  --frames 30 \
  --maxiter 1 \
  --popsize 2 \
  --benchmark-candidates 0 \
  --output coverage_quick.json \
  --skip-reference-validation
```

Distance-only quick run:

```bash
python optimize_separation_constellation.py \
  --baseline-constellation shared_baseline.json \
  --frames 30 \
  --maxiter 1 \
  --popsize 2 \
  --benchmark-candidates 0 \
  --output distance_quick.json
```

These are workflow checks, not meaningful optimization runs.

---

# 8. Longer distance-only searches

Default:

```bash
python optimize_separation_constellation.py \
  --baseline-constellation shared_baseline.json
```

Deeper search:

```bash
python optimize_separation_constellation.py \
  --baseline-constellation shared_baseline.json \
  --maxiter 30 \
  --popsize 6
```

Increase temporal sampling:

```bash
python optimize_separation_constellation.py \
  --baseline-constellation shared_baseline.json \
  --frames 720
```

A larger frame count makes the maximin separation objective inspect more times during the same simulation window. This can catch close approaches that a coarser sampling schedule misses.

---

# 9. Optimize angular velocity too

The default distance-only search changes only:

```text
phase inclination raan
```

To also optimize angular velocity:

```bash
python optimize_separation_constellation.py \
  --baseline-constellation shared_baseline.json \
  --parameters phase inclination raan angular_velocity
```

The allowed angular velocity bound matches the existing optimizer:

```text
0.0005 to 0.0015 rad/s
```

For 30 satellites this increases the number of Differential Evolution variables from 90 to 120.

---

# 10. Stop and resume the distance-only optimizer

The default checkpoint is:

```text
distance_optimizer_checkpoint.json
```

You can stop with:

```text
Ctrl+C
```

The best candidate from the last completed vectorized batch is retained and saved.

Resume with:

```bash
python optimize_separation_constellation.py \
  --baseline-constellation shared_baseline.json \
  --resume-checkpoint distance_optimizer_checkpoint.json
```

The checkpoint stores the optimized variable vector and its minimum-separation score in kilometres.

Coverage-first and distance-only checkpoints are intentionally different formats and are not interchangeable.

---

# 11. Rendering options

Render any saved constellation:

```bash
python network_simulation.py --constellation FILE.json --output OUTPUT.mp4
```

Add a permanent label:

```bash
python network_simulation.py \
  --constellation FILE.json \
  --output OUTPUT.mp4 \
  --video-label "MY EXPERIMENT LABEL"
```

Change video duration:

```bash
python network_simulation.py --constellation FILE.json --video-seconds 60
```

Change frame rate:

```bash
python network_simulation.py --constellation FILE.json --fps 30
```

Update the existing on-video Earth-coverage/link indicators every rendered frame:

```bash
python network_simulation.py \
  --constellation FILE.json \
  --metrics-update-seconds 0
```

Run the simulation and export existing diagnostics without creating an MP4:

```bash
python network_simulation.py \
  --constellation FILE.json \
  --no-video \
  --metrics-csv metrics.csv
```

---

# 12. About the 100 Earth receivers

The existing renderer/scorer uses 100 virtual Earth receiver points by default:

```text
--ground-points 100
```

They are approximately evenly distributed with a Fibonacci sphere.

The coverage-first optimizer uses those receiver samples in its objective.

The new distance-only optimizer does not use them at all.

When you replay either saved constellation with `network_simulation.py`, the renderer/scorer evaluates the selected constellation against the receiver set for its normal display and CSV diagnostics.

You can render or inspect with more receiver points, for example:

```bash
python network_simulation.py \
  --constellation distance_optimized_constellation.json \
  --ground-points 5000 \
  --no-video \
  --metrics-csv distance_render_5000_receivers.csv
```

This does not retroactively change what the distance-only optimizer optimized.

---

# 13. Data transmission and storage scope

The satellite model currently provides physical satellite positions and pairwise connection checks based on range and Earth line of sight.

This bundle does **not** add a new model for:

<!-- Done! -->
<!-- Check packet.py -->
- generated data packets;

<!-- Done! -->
<!-- Added storage_capacity to Satellite (nodes.py), enforced per-satellite packet queue capacity in
     simulate_packet_routing() (packet.py), and added storage_capacity in satellite_generator.py -->
- per-satellite storage capacity;


- store-and-forward queues;
- scheduling transmissions;
- route selection;
- link bandwidth;
- receiver capacity;
- end-to-end delivery success;
- end-to-end delay;
- automatic comparison of the two optimized constellations.

That omission is intentional for this revision. The two optimizers now produce two clearly separate constellation candidates that can be rendered and saved, but the requested future question—how effectively data originating anywhere in orbit reaches one of the Earth receivers, including store-and-forward behavior—has not been implemented here.

---

# 14. Important interpretation of the distance-only result

A larger minimum inter-satellite distance is not automatically the same thing as a better communications network.

Because the distance-only objective ignores connection range and routing, it may spread satellites far enough apart that some inter-satellite links disappear. It may also produce excellent geometric separation while giving poor Earth coverage.

That is expected behavior for this experiment: the optimizer is deliberately isolated to one objective so its result can later be evaluated against the coverage-first approach without contaminating the search with communication-quality terms.

---

# 15. Existing coverage-first benchmark

The existing benchmark still checks the agreement between the vectorized coverage-first evaluator and the readable/reference scorer:

```bash
python benchmark_optimizer.py
```

It does not benchmark or compare the new distance-only constellation against the coverage-first constellation.

---

# 16. Suggested filenames for clean experiments

Use these names consistently:

```text
Input baseline:
  shared_baseline.json

Coverage-first output:
  coverage_optimized_constellation.json
  coverage_optimizer_checkpoint.json
  VIDEO_A_coverage_first.mp4

Distance-only output:
  distance_optimized_constellation.json
  distance_optimizer_checkpoint.json
  VIDEO_B_distance_only.mp4
```

This keeps optimizer A and optimizer B visually and operationally distinct without adding any automatic decision logic.
