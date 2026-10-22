# Contributing

Use uv exclusively. Create an environment with `uv sync --extra dev`, implement changes under `src/kaggle_slayer`, and run `uv run pytest`, `uv run ruff check src tests dashboard scripts examples` and `uv run ruff format --check src tests dashboard scripts examples`.

Add meaningful behavior tests for model/metric/configuration changes. Keep API tests mocked by default. Preserve fold-fit preprocessing, dataset fingerprint checks, explicit upload authorization, bounded experiments and no-write retry semantics. Do not add credentials, competition data or unverified leaderboard claims. Keep generated model binaries in ignored artifact directories; the six preserved checkpoints in the [primary audit demo](demo/home-data-for-ml-course-8ed561cbed5e/README.md) use a narrowly scoped Git exception.

Maintain dependencies with:

```bash
uv add NAME
uv add --optional dev NAME
uv sync --extra dev
```

Use `uv add NAME` for runtime dependencies and `uv add --optional dev NAME` for development dependencies. `uv sync --extra dev` installs the development extra. For an intentional dependency upgrade, optionally run:

```bash
uv lock --upgrade
uv sync --extra dev
```

Review and commit the updated `pyproject.toml` and `uv.lock`. New competitions generally require a YAML profile and documentation rather than core-code changes.
