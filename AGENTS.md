# Workspace Guidelines — SDSS Galaxy Morphology & Scaling Relations Atlas

## Documentation Notebooks for Major Changes (Required)

Whenever you make significant changes to this project — such as completing a project phase, adding a new pipeline stage, training a model, introducing new analysis methods, or making major architectural updates — you **must create a brand-new Jupyter notebook file** inside `notebooks/`. **Do NOT overwrite or upgrade previous notebooks** to document new changes; preserve existing notebooks as historical snapshots and create a new file each time.

- **Naming**: Use sequential zero-padded numbering matching the project timeline (for example: `01_sdss_sql_pipeline.ipynb`, `02_phase1_project_guide.ipynb`, `03_galaxy_image_acquisition.ipynb`, `04_morphology_classification.ipynb`, `05_dashboard_export.ipynb`).
- **Clear Markdown Explanations**: Explain every step in plain, accessible language using Markdown cells — cover *what* was built or changed, *why* it matters scientifically and technically, and *how* the pieces connect. Define astrophysics and engineering terms concisely so someone new to the codebase can follow along effortlessly.
- **Complete Runnable Code**: Include all Python code cells needed to configure, run, inspect, verify, and visualize the feature end-to-end using the modules in `src/`.
- **Safe Defaults**: Provide a clear toggle (e.g., `RUN_LIVE_FETCH = False`) when a step makes slow network calls or heavy downloads, so the notebook can also be explored immediately against local data in `data/`.

## Environment & Engineering Conventions

- Always run Python commands and package installs with the project virtual environment: `.\.venv\Scripts\python.exe`.
- Store all physical numeric quantities in SQLite strictly as unitless `REAL` columns (document units in `docs/ARCHITECTURE.md` and notebook Markdown).
- Keep SQLite DDL idempotent (`CREATE ... IF NOT EXISTS`) and use parameterized SQL queries (`?`).
- Keep `src/` modular, typed, and covered by `pytest` tests in `tests/`.
