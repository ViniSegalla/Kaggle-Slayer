# Reproducibility

Install uv using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```bash
uv python install
uv python pin 3.11
uv sync --locked --extra dev
uv run --locked pytest
uv run --locked python -m examples.scripted_swarm --output artifacts/swarm-example
uv run --locked python -m examples.scripted_swarm --task regression --output artifacts/swarm-regression
```

These integration examples use scripted agent responses and real training, without provider credentials or network requests. For deterministic baselines and the legacy development scripts, see [offline baselines](offline-baselines.md); those workflows require an explicit provider override when `.env` is configured for live agentic execution.

`uv sync --locked` verifies that the lock agrees with project metadata and installs that resolution. `--frozen` skips the consistency check. This repository uses `--locked` in CI to enforce consistency. `uv run` also synchronizes the environment; add `--locked` for repeatable execution.

Maintain dependencies with `uv add pandas`, `uv add --optional dev pytest`, and `uv lock --upgrade`. The project uses the `dev` optional extra consistently rather than mixing development groups and extras. Commit changes to both project metadata and the lockfile. XGBoost and CatBoost are core dependencies, included in ordinary syncs and CI. The empty `boosting` extra preserves compatibility with older setup commands.

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

`artifacts/` and `data/` are private local outputs. The `evidence/` index links to the best-run audit package and historical engineering-verification reports. The selected `demo/` package includes its trained checkpoints through a scoped Git exception; raw competition inputs, credentials and per-row training labels remain excluded.
