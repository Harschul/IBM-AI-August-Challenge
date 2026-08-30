# Apply the frontend bundle

Copy the contents of this directory into the root of your cloned
`IBM-AI-August-Challenge` repository.

## 1) Copy files

```bash
cp -a /path/to/ibm_frontend_bundle/. /path/to/IBM-AI-August-Challenge/
```

## 2) Install dependencies

Install the repo's dependencies first, then add the frontend extras.

```bash
pip install -r requirements.txt
pip install -r requirements-integration.txt
pip install -r requirements-frontend.txt
```

## 3) Run the frontend

```bash
python run_frontend.py
```

or directly:

```bash
streamlit run src/frontend/app.py
```

## 4) Run the tests

```bash
python -m pytest tests/test_temporal_router.py tests/test_integration.py tests/test_frontend_bundle.py -q
```

## Notes

- The frontend reuses the integrated physical contact-plan pipeline from the
  earlier bundle.
- If the RL checkpoint cannot be loaded, the frontend stays usable and labels
  the replay as temporal fallback where relevant.
- The app is intentionally minimalist: one synchronized timeline drives both the
  orbital 3D render and the topology graph.
