# Install the final bundle

Extract the ZIP, then copy its contents into the repository root:

```bash
cp -a /path/to/final_release/. /home/harshul/IBM-AI-August-Challenge/
cd /home/harshul/IBM-AI-August-Challenge
```

Archive the old bundle-specific docs:

```bash
bash scripts/final/cleanup_legacy_docs.sh
```

Install runtime + PPO inference dependencies:

```bash
python -m pip install -r requirements-frontend-rl.txt
```

Verify the final checkpoint/config identity:

```bash
python scripts/final/verify_release.py
```

If your completed 2M checkpoint already lives at
`RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip`, nothing needs moving.
The companion `.metadata.json` must sit next to it.

Run the visual demo:

```bash
python run_frontend.py
```

Run the final benchmark:

```bash
python run_final_benchmark.py
```
