
Run: `pip install gymnasium numpy matplotlib`. numpy/matplotlib you likely already have; gymnasium is the only new package this needs.

cd into the folder where `src/rl/` and `run_demo.py` sit side by side, then run:

```bash
python run_demo.py
```

It must be run from there so the `from src.rl.env import ...` import line resolves.

You'll see **unmasked random** and **masked random**, each with delivered / on_time / invalid_action percentages. This is the actual proof of correctness, not the chart.

`masked random`'s `invalid_action` **MUST read 0%**. If it's anything above 0%, the action-masking logic is broken and shouldn't be trusted yet. `delivered` / `on_time` being clearly higher for masked than unmasked is the second sanity signal.

It writes to `rl_env_sanity_check.png` (change that path if running outside this environment).

* **Left panel:** delivered/on-time bars
* **Right panel:** invalid-action rate; masked should be a flat zero bar.
