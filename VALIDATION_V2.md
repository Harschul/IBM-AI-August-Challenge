# v2 validation

## GitHub Actions failure identified

The current frontend workflow fails in its dependency-install step because it
uses Python 3.11 and installs the repository root `requirements.txt`, which pins
`numpy==2.5.2`. The GitHub runner reports that NumPy 2.5.x requires Python 3.12+.

v2 removes that coupling: frontend CI installs only `requirements-frontend.txt`,
which depends on the lightweight integration requirements and does not consume
the root NumPy pin.

## Local validation performed

- Python-compiled all v2 integration/frontend source and test files successfully.
- Ran the v2 integration + frontend tests against an API-compatible local harness
  for the repository's `Satellite`, `Network`, `Contact`, `DataBundle` and
  temporal-router interfaces: **13 tests passed**.
- The tests cover:
  - 3 distinct science spacecraft
  - random/reproducible bundle source selection across all `SCIENCE_IDS`
  - preservation of the 14-action / 158-feature RL interface
  - direct science-to-ground contacts from all science sources
  - capacity reservation
  - requested-RL vs actual-temporal fallback labeling
  - requested algorithm switching
  - 3D/topology Plotly figure construction

A new remote GitHub Actions run can only occur after these files are committed
and pushed to the repository.
