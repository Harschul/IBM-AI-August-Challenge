# Stochastic transfer layer

This bundle changes the physical routing problem from:

> choose a feasible link -> the transfer always succeeds

to:

> choose a feasible link -> sample a physical transmission outcome -> either
> deliver the whole bundle across the hop or pay the time/capacity cost of a
> failed attempt and replan from the same sender.

It is intended to be applied on top of the frozen **3 SCI / 6 LEO / 2 GEO /
3 ground** integration bundle and the physical PPO retraining bundle.

## 1. Failure probability

For each attempted physical contact, the default probability is:

```text
p_fail = base
       + 0.50 * weather_risk
       + 0.30 * (1 - node_health)
       + 0.20 * (1 - link_reliability)
```

and then clamped to `0 .. max_failure_probability`.

The exact defaults are now explicit in `config/prototype.yaml`:

```json
"stochastic_transfer": {
  "enabled": true,
  "base_failure_probability": 0.01,
  "weather_weight": 0.5,
  "health_weight": 0.3,
  "reliability_weight": 0.2,
  "max_failure_probability": 0.9,
  "min_failure_progress": 0.25,
  "max_failure_progress": 0.95,
  "risk_shaping_weight": 2.0,
  "failure_penalty_base": 8.0,
  "failure_penalty_priority": 12.0
}
```

Ground contacts inherit both the configured link weather risk and the specific
ground station's weather risk from the physical contact-plan adapter, so they
can naturally be riskier than space-space links.

## 2. What happens on success

When the Bernoulli draw succeeds:

1. The full bundle size is removed from the contact's residual capacity.
2. Transmission time and propagation delay elapse.
3. The bundle moves to the selected next-hop node.
4. The route history records the successful hop.
5. PPO receives the usual delivery/deadline/priority rewards.

## 3. What happens on failure

A failure is deliberately costly rather than cosmetic.

The layer samples a failure-detection point between 25% and 95% of the attempted
transmission. If a 100 MB bundle fails at 60% progress:

- approximately 60 MB of that contact's capacity is consumed;
- the corresponding 60% of transmission time elapses;
- **zero bundle bytes are considered delivered**;
- the complete 100 MB bundle stays at the sender;
- the next policy decision can retry the contact if it still has enough time and
  residual capacity, or choose a different route.

The bundle uses atomic hop delivery. This is conservative and avoids pretending
that partial fragments can be resumed when the repository has no fragment
reassembly/FEC layer yet.

## 4. Data importance

Physical failure probability is **not** changed by science priority. A valuable
observation does not make a radio link physically more reliable.

Instead, importance changes the *cost of choosing risky links*:

```text
failed-attempt reward penalty
  = failure_penalty_base
  + failure_penalty_priority * science_priority
```

With the defaults this is:

```text
8 + 12 * science_priority
```

A priority 1.0 transient therefore loses 20 reward points on a failed attempt,
while a priority 0.2 housekeeping bundle loses 10.4. High-priority bundles also
retain the existing large priority-weighted on-time delivery bonus and tighter
deadlines.

This lets PPO learn a meaningful policy such as:

> use a slightly slower high-reliability path for an urgent transient, while
> accepting an opportunistic riskier path for low-value housekeeping data.

## 5. Common random numbers for a fair Temporal-vs-RL comparison

A normal RNG consumed sequentially would be unfair: Temporal and PPO choose
different paths, so each would consume random numbers in a different order and
one could get luckier by accident.

`TransferOracle` instead derives every draw from a stable hash of:

```text
stochastic_seed
bundle_id
contact identity
attempt ordinal on that bundle/contact
```

Therefore two algorithms that attempt the same bundle over the same contact for
the same attempt ordinal receive the **exact same random draw**.

They still remain free to choose different routes. This is a common-random-
numbers experimental design: matched choices see matched uncertainty, while
algorithmic choices remain independent.

## 6. Training PPO

The environment file `src/integration/physical_rl_env.py` is replaced with the
stochastic version. Do not reuse a PPO checkpoint trained under deterministic
transfers as the final result: stochastic failures materially change the MDP.

Install:

```bash
python -m pip install -r requirements-stochastic-training.txt
```

Smoke train first:

```bash
python train_physical_ppo_stochastic.py \
  --timesteps 100000 \
  --n-envs 2 \
  --bundles-per-episode 24 \
  --seed 42 \
  --out RL/rl_env_v0/models/physical_multisource_stochastic_smoke.zip
```

Then a serious run:

```bash
python train_physical_ppo_stochastic.py \
  --timesteps 2000000 \
  --n-envs 4 \
  --bundles-per-episode 32 \
  --max-attempts 24 \
  --seed 42 \
  --out RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip
```

The model metadata records both the full config SHA-256 and the stochastic
settings, so an accidental environment/config change is visible later.

## 7. Final stochastic Temporal-vs-RL benchmark

Use held-out traffic and stochastic seeds:

```bash
python benchmark_temporal_vs_rl_stochastic.py \
  --model RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip \
  --num-seeds 20 \
  --seed-base 900000000 \
  --stochastic-seed-base 700000000 \
  --bundles 500 \
  --out benchmark_results/final_stochastic
```

The primary comparison is:

```text
Temporal router
vs
Pure PPO
```

No Temporal rescue is included in the pure PPO score.

For an operational third result you can additionally run:

```bash
python benchmark_temporal_vs_rl_stochastic.py \
  --model RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip \
  --num-seeds 20 \
  --bundles 500 \
  --include-operational-fallback \
  --out benchmark_results/final_stochastic_with_fallback
```

That third result is explicitly named `rl_with_temporal_fallback` and keeps the
actual algorithm used per attempt in the audit log.

## 8. Benchmark outputs

The benchmark writes:

```text
bundle_results.csv
attempt_log.csv
seed_metrics.csv
summary.json
```

### `attempt_log.csv`

This is the most useful stochastic audit file. Every attempted transmission
contains:

- traffic seed
- stochastic seed
- bundle ID and science source
- holder and attempted destination
- actual routing algorithm that selected the hop
- contact start/end
- calculated failure probability
- exact uniform random draw
- success/failure result
- sampled transfer progress at failure
- capacity bytes consumed
- departure, failure/event and arrival times

This makes the experiment reproducible and inspectable rather than treating
"random failure" as a black box.

### Additional benchmark metrics

Alongside delivery/deadline/priority/latency metrics, the stochastic benchmark
reports:

- mean transfer attempts per bundle
- transfer failure rate
- mean failed attempts per bundle
- mean wasted capacity in MB
- fallback rate for the optional operational mode
- 95% confidence intervals across held-out seeds
- paired `RL - Temporal` deltas

## 9. Important modeling assumptions

This is a controlled stochastic link model, not a calibrated RF channel model.
The weights should be treated as explicit experiment parameters until you have
real telemetry or a channel/weather model to fit them against.

Current assumptions are:

- failure draws are conditionally independent between attempts;
- risk is constant within each temporal contact window;
- bundles transfer atomically across a hop;
- failed partial bytes cannot be resumed;
- failed bytes consume link capacity but not bundle progress;
- failure detection occurs during transmission, before propagation;
- routing replans after every failed attempt.

Those assumptions are simple enough to explain in a demonstration and strong
enough to make reliability, weather, retries, deadlines, congestion and science
importance interact in the learned policy.
