# Integration gap review and v2 resolution

The original integration work joined orbital physics, temporal contacts, routing,
capacity reservation and the frozen 14-action RL interface. v2 closes three
additional gaps exposed by the frontend rollout.

## Resolved in v2

1. **Frontend install was coupled to the full root dependency stack.**
   The old workflow installed `requirements.txt` under Python 3.11, but that file
   pins NumPy 2.5.2, which requires Python 3.12+. v2 gives integration/frontend
   their own minimal dependency files and keeps RL dependencies optional.

2. **Only one physical science source existed.**
   v2 keeps the checkpoint's 14-action shape while changing the mission layout
   to 3 science + 6 LEO + 2 GEO + 3 ground. Bundle generation chooses among
   `SCIENCE_IDS=(0,1,2)` using the reproducible traffic RNG.

3. **Requested RL could visually look like executed RL after fallback.**
   Every frontend hop now records requested and actual algorithms separately,
   plus a fallback reason. UI metrics, packet hover state, event logs and exports
   all use the actual algorithm for execution claims.

## Remaining empirical limitation

The existing PPO checkpoint was trained on the old single-source mock topology.
It remains shape-compatible because the action space is still 14 and the
observation remains 158 floats, but the new multi-source layout is a distribution
shift. Retraining/fine-tuning on v2 physical contact plans remains the correct
next model-quality step.
