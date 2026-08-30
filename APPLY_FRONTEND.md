# Apply and run the frontend (v2)

Copy this bundle over the root of your cloned `IBM-AI-August-Challenge` repo.
It includes both the integration layer and the frontend changes needed by this
version.

## 1. Copy the bundle

```bash
cp -a /path/to/ibm_full_bundle_v2/. /home/harshul/IBM-AI-August-Challenge/
cd /home/harshul/IBM-AI-August-Challenge
```

## 2. Install the frontend

For temporal routing and the full visualization UI:

```bash
python -m pip install -r requirements-frontend.txt
```

This is intentionally a **minimal** install and does not consume the root
`requirements.txt`, whose current NumPy pin requires Python 3.12+.

To execute the trained RL checkpoint as well:

```bash
python -m pip install -r requirements-frontend-rl.txt
```

If the optional RL stack or checkpoint is unavailable, the UI remains usable.
Every affected hop is explicitly recorded as:

```text
requested_algorithm = rl
actual_algorithm    = temporal
fallback            = true
```

so fallback can never be displayed as real RL execution.

## 3. Run

```bash
python run_frontend.py
```

or:

```bash
python -m streamlit run src/frontend/app.py
```

## 4. Tests

```bash
python -m pytest \
  tests/test_temporal_router.py \
  tests/test_integration.py \
  tests/test_frontend_bundle.py \
  -q
```

## 5. Multi-source node map

The PPO action space stays at 14 nodes:

```text
0..2    science spacecraft (SCI-0, SCI-1, SCI-2)
3..8    LEO relays
9..10   GEO relays
11..13  ground stations
```

Bundles are reproducibly randomized across `SCIENCE_IDS=(0,1,2)` using the
traffic seed.

The existing checkpoint remains shape-compatible (14 actions / 158 observation
features), but it was trained on the old single-source distribution. Treat its
multi-source score as transfer evaluation until it is retrained on the new
physical contact-plan distribution.
