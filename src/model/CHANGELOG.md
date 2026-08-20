# Changes in this bundle

- Added `optimize_separation_constellation.py` with a distance-only maximin objective: maximize the minimum pairwise satellite separation over all sampled frames.
- Kept the existing `optimize_constellation.py` coverage-first optimizer unchanged.
- Added `--video-label` to `network_simulation.py` so rendered outputs can be visibly marked as baseline, coverage-first, or distance-only.
- Replaced the README with a controlled A/B workflow using one shared baseline and clearly named output videos.
- Did not add any automatic effectiveness comparison, packet routing, store-and-forward, storage queue, throughput, or latency model.
