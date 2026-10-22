# Reproducibility

Install uv using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```bash
uv python install
uv python pin 3.11
uv sync --locked --extra dev
uv run --locked pytest
uv run --locked python scripts/demo.py
uv run --locked python scripts/demo.py --task regression
```

`uv sync --locked` verifies that the lock agrees with project metadata and installs that resolution. `--frozen` skips the consistency check; it is only appropriate in deliberately partial container build layers. This repository uses `--locked` in Docker and CI because the full metadata is available. `uv run` also synchronizes the environment; add `--locked` for repeatable execution.

Maintain dependencies with `uv add pandas`, `uv add --optional dev pytest`, and `uv lock --upgrade`. The project uses the `dev` optional extra consistently rather than mixing development groups and extras. Commit changes to both project metadata and the lockfile. XGBoost and CatBoost are core dependencies, included in ordinary syncs, Docker and CI. The empty `boosting` extra preserves compatibility with older setup commands.

Run metadata records the seed, configuration, Python/platform/package versions, data SHA-256 fingerprints and exact split indices. Model randomness uses the same seed; CPU threading and library/platform differences may still change floating-point results. Resuming verifies file contents rather than trusting filenames. An interrupted Optuna sampler's RNG stream is not checkpointed byte-for-byte; already queued trials remain fixed, while newly sampled trials after a crash can differ from an uninterrupted run.

The demonstrated environment uses a workspace-local uv cache/runtime because the execution sandbox restricts the default Windows directories. On a similarly restricted PowerShell environment:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.uv-python'
$env:TEMP = Join-Path $PWD '.tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
uv sync --locked --extra dev
uv run --locked pytest --basetemp=.test-tmp
```

These paths are ignored by Git and Ruff. Source checks explicitly target `src tests dashboard scripts examples` so dependency runtimes are never linted or formatted.

`artifacts/` and `data/` are private local outputs. Curated `evidence/` contains only generated synthetic examples, reports and test results. No Kaggle data, credentials or large checkpoints should be committed.
