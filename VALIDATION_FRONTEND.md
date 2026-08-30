# Frontend bundle validation

Validation performed while assembling this bundle:

- Python syntax compilation (`py_compile`) for all frontend source files
- Bundle manifest generated and checked
- Imports and file paths aligned to the earlier integration bundle layout

Because the live repository code and binary RL checkpoint are not materialized in
this assembly environment, the full Streamlit runtime was not executed here.
The frontend is written to degrade gracefully when the RL model is absent by
falling back to the temporal router.
