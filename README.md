# SDSS Galaxy Morphology & Scaling Relations Atlas

An open-source computational astrophysics pipeline, dual-track machine learning / deep learning classification suite, and zero-backend interactive web atlas studying galaxy morphology and empirical scaling relations using **Sloan Digital Sky Survey Data Release 17 (SDSS DR17)**.

---

## Architecture Overview (All 4 Phases Complete)

| Phase | Scope | Status |
| :--- | :--- | :--- |
| **Phase 1** | Modular SDSS DR17 SQL acquisition (`astroquery.sdss`), cosmological & physical parameter derivation (`astropy` `FlatLambdaCDM`), normalized SQLite atlas (`data/galaxy_atlas.db`), SQL-to-Pandas statistical analysis & 6-panel dashboard. | **Complete** (`2,489` clean galaxies) |
| **Phase 2** | Resumable multi-threaded `128×128` SDSS RGB cutout downloader + calibrated multi-band FITS (`g, r, i`) showcase pipeline (`Cutout2D`, sigma-clipped NaN masking & sky subtraction, `make_lupton_rgb`). | **Complete** (`284` RGB + `6` FITS showcase) |
| **Phase 3** | Dual-track galaxy morphology classifiers in **separate folders**: `classifiers/astronn_and_astroml/` (`scikit-learn` Random Forest photometric baseline + `astroNN` Galaxy10CNN with Monte Carlo Dropout uncertainty) and `classifiers/tensorflow_cnn/` (custom 4-block TensorFlow/Keras CNN with 360° rotational invariance). | **Complete** (predictions stored in SQLite) |
| **Phase 4** | Static JSON/JS exporter (`src/export_dashboard.py`) powering a zero-backend interactive HTML/CSS/JS **Plotly.js** dashboard (`web/index.html`) with live cross-filtering, galaxy image inspector, FITS showcase, and ML/CNN benchmark viewer. | **Complete** (`web/index.html`) |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full technical architecture, physical unit conventions, and normalized SQLite schema.

---

## Repository Structure

```text
.
├── classifiers/
│   ├── astronn_and_astroml/             # Track 1: Referenced astronomy packages
│   │   ├── tabular_baseline.py          # scikit-learn Random Forest / GBDT photometric baseline
│   │   └── astronn_classifier.py        # astroNN Galaxy10 SDSS CNN with MC Dropout uncertainty
│   └── tensorflow_cnn/                  # Track 2: Custom TensorFlow / Keras CNN
│       ├── hubble_taxonomy.py           # 10-class Galaxy10 SDSS -> 5-class Hubble Sequence mapping
│       ├── cnn_model.py                 # 4-block Conv2D + BatchNorm + 360° astro augmentation
│       └── train_and_predict.py         # Training, evaluation, and SQLite catalog inference
├── docs/
│   └── ARCHITECTURE.md                  # System blueprint, physics contracts & SQLite schema
├── notebooks/
│   ├── 01_sdss_sql_pipeline.ipynb
│   ├── 02_phase1_project_guide.ipynb
│   ├── 03_scaled_2500_galaxy_analysis.ipynb
│   ├── 04_galaxy_image_acquisition.ipynb
│   ├── 05_statistical_scaling_relations_dashboard.ipynb
│   ├── 06_astronn_and_tabular_classification.ipynb
│   ├── 07_tensorflow_cnn_hubble_classifier.ipynb
│   └── 08_interactive_dashboard_export.ipynb
├── src/
│   ├── config.py                        # Paths, FlatLambdaCDM cosmology, DR17 cuts & imaging config
│   ├── acquisition.py                   # SDSS DR17 SQL fetcher, cleaning & cosmological physics
│   ├── database.py                      # GalaxyDatabase context manager & normalized SQLite schema
│   ├── analysis.py                      # Power-law fits, CMD separation, correlations & 6-panel dashboard
│   ├── imaging.py                       # SkyServer/Legacy RGB cutouts & FITS Cutout2D + Lupton RGB
│   └── export_dashboard.py              # SQLite -> web/data/atlas_data.{json,js} exporter
├── tests/
│   ├── test_database.py
│   ├── test_imaging.py
│   └── test_classifiers_and_export.py
├── web/                                 # Zero-backend interactive Plotly.js web atlas
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   ├── data/                            # Exported atlas_data.json & atlas_data.js
│   └── assets/                          # Self-contained RGB cutouts & FITS Lupton showcase PNGs
├── data/galaxy_atlas.db                 # Normalized SQLite database
├── outputs/figures/                     # Publication-quality static figures & PDFs
└── requirements.txt
```

---

## Setup & Usage

Install all dependencies inside the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run Any Pipeline Stage

```powershell
# Phase 1 — Query SDSS DR17 & populate data/galaxy_atlas.db
.\.venv\Scripts\python.exe -m src.acquisition --top-n 2500

# Phase 2 — Download SDSS RGB cutouts & process calibrated FITS (g, r, i) showcase
.\.venv\Scripts\python.exe -m src.imaging --rgb-limit 250 --fits-showcase 6

# Phase 3 (Track 1) — Tabular photometric Random Forest & astroNN Galaxy10CNN
.\.venv\Scripts\python.exe -m classifiers.astronn_and_astroml.tabular_baseline
.\.venv\Scripts\python.exe -m classifiers.astronn_and_astroml.astronn_classifier

# Phase 3 (Track 2) — Custom TensorFlow / Keras Hubble Sequence CNN
.\.venv\Scripts\python.exe -m classifiers.tensorflow_cnn.train_and_predict

# Phase 4 — Export SQLite views & metrics to web/data/
.\.venv\Scripts\python.exe -m src.export_dashboard

# Run full unit test suite
.\.venv\Scripts\python.exe -m pytest -v
```

### Launch the Interactive Web Dashboard
- **Zero-backend instant open**: Double-click `web/index.html` in your browser (uses `web/data/atlas_data.js` to avoid browser `file://` CORS issues).
- **Or serve locally**:
  ```powershell
  .\.venv\Scripts\python.exe -m http.server 8000 --directory web
  ```
  and navigate to `http://localhost:8000`.
