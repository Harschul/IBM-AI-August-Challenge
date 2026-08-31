# Release validation

Validation performed while assembling this final bundle:

- locked scenario SHA verified as `cf9f11e8ac81dc066fad151d751b9207201c6e5354dbd21a26440da192ce3004`
- all Python source compiled with `py_compile`
- API-compatible local repository harness executed the final reported Temporal replay with all 500 bundles
- synchronized orbital and topology Plotly figures rendered successfully
- integration/frontend/stochastic/spec suite: **18 tests passed**

The locally trained final PPO binary was not uploaded into this conversation, so this
assembly environment could not execute that binary. The release intentionally verifies
its companion metadata at runtime and refuses "Reported PPO" if the checkpoint is absent
or was trained against a different config hash.
