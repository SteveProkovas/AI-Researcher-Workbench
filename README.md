# AI Researcher Workbench


This document defines a research-friendly repository layout for **AI Researcher Workbench**. It includes a recommended directory tree, file templates, CI, contribution guidelines, and notes on how each piece supports reproducible research, experiment tracking, and collaborative development.

---

## Goals of this repository

* Serve as a single unified workbench for experiments, utilities, and research notes.
* Make experiments reproducible (random seeds, env, data manifests, notebooks).
* Offer clean API for core utilities (linear algebra, integration problems, experiment helpers).
* Provide examples and notebooks that are easy to run locally or via Binder/GitHub Codespaces.
* Be collaboration-ready with clear contribution docs, license, and CI.

---

---


## Recommended package layout (`src/airesearcher`)

**`math/integrals.py`** — responsibilities:

* Provide closed-form integrals from the 30-day AI list when analytic (Gaussian, Gamma, Beta, simple KLs).
* Provide numeric fallback wrappers using `scipy.integrate.quad` and Monte Carlo for expectations without closed form.
* Provide testable functions with clear docstrings and examples.

**`math/linear_algebra.py`** — responsibilities:

* PCA helpers: `center_data`, `compute_svd`, `project`, `reconstruct`, `variance_explained`.
* Small stable wrappers for SVD/eigs that detect shape (use `scipy.sparse.linalg` when needed).
* Utility functions: `whiten`, `mahalanobis_distance`, `rank`, `pseudoinverse`.

**`experiments/runner.py`** — responsibilities:

* Read `configs/` (JSON/YAML) describing experiments.
* Seed RNGs, set up logging, checkpointing, and deterministic behavior.
* Save minimal `run_manifest.json` with parameters + git commit SHA for reproducibility.

**`utils/rng.py`**

* Provide `set_global_seed(seed)` to seed numpy, python random, and torch (if present).

**`data/dataset.py`**

* Small local dataset utilities and data manifest loader. Keep external dataset downloads in `scripts/` if large.


## License & Contact

Apache-2.0 — Open for academic/industrial collaboration.
Contact: **Steve Stavros Prokovas**
Email: sprokovas@gmail.com
