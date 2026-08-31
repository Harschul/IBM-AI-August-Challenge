from src.experiment.spec import load_final_spec


def test_final_experiment_locks_config_model_and_benchmark_family():
    spec = load_final_spec()
    assert spec.verify_config() == "cf9f11e8ac81dc066fad151d751b9207201c6e5354dbd21a26440da192ce3004"
    assert spec.model.name == "physical_multisource_stochastic_ppo.zip"
    assert spec.benchmark.algorithms == ("temporal", "rl_pure")
    assert spec.benchmark.num_seeds == 20
    assert spec.benchmark.bundles_per_seed == 500
    assert spec.benchmark.seeds(0) == (900_000_000, 700_000_000)
