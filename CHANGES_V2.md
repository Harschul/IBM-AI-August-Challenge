# v2 changes

## 1. Frontend dependency / Actions fix

The old frontend workflow installed `requirements.txt` under Python 3.11. The
root file pins NumPy 2.5.2, which requires Python 3.12+, so Actions failed during
installation before tests ran.

v2 adds a minimal `requirements-frontend.txt` containing only what the physical
frontend needs and makes the RL stack optional via
`requirements-frontend-rl.txt`. The frontend workflow now installs that minimal
file, smoke-imports it, compiles the frontend, and runs the routing/integration/
frontend tests.

## 2. Multiple science sources

The fixed 14-node interface is now:

- `SCIENCE_IDS = (0, 1, 2)`
- `LEO_IDS = (3, 4, 5, 6, 7, 8)`
- `GEO_IDS = (9, 10)`
- `GROUND_IDS = (11, 12, 13)`

Three physically distinct science spacecraft are generated with separated RAAN
and phase. `generate_bundles()` randomly chooses a source from `SCIENCE_IDS`
while remaining deterministic for a given traffic seed.

## 3. Requested vs actual routing

Every replay hop now stores:

- `requested_algorithm`
- `actual_algorithm`
- `fallback_used`
- `fallback_reason`

The UI shows requested and actual execution separately in the metric row, plot
titles, packet hover state, event log, bundle inspector and exported tables.
An unavailable/invalid RL decision therefore renders as temporal fallback, not
as RL.
