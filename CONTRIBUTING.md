# Contributing

Use uv exclusively. Create an environment with `uv sync --extra dev`, implement changes under `src/kaggle_slayer`, and run `uv run pytest`, `uv run ruff check src tests dashboard scripts examples` and `uv run ruff format --check src tests dashboard scripts examples`.

Add meaningful behavior tests for model/metric/configuration changes. Keep API tests mocked by default. Preserve fold-fit preprocessing, dataset fingerprint checks, explicit upload authorization, bounded experiments and no-write retry semantics. Do not add credentials, competition data, model binaries or unverified leaderboard claims.

When adding dependencies use `uv add NAME` or `uv add --optional dev NAME`, then commit the updated `pyproject.toml` and `uv.lock`. New competitions generally require a YAML profile and documentation rather than core-code changes.
