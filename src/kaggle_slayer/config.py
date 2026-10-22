"""Configuration precedence: defaults < competition profile < explicit CLI flags."""

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kaggle_slayer.schemas import Competition, RunConfig

MAX_LLM_OUTPUT_TOKENS = 32768


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KS_", env_file=".env", extra="ignore")
    artifact_root: str = "artifacts"
    llm_provider: str = "deterministic"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout: int = 30
    llm_max_tokens: int = Field(default=700, ge=128, le=MAX_LLM_OUTPUT_TOKENS)
    llm_native_tools: bool = False


def read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def cli_run_profile(execution_mode: str | None = None) -> Path:
    """Resolve the CLI default in checkouts and installed wheels."""
    name = "default.yaml" if execution_mode == "deterministic" else "agentic.yaml"
    roots = [
        Path.cwd() / "configs",
        Path(__file__).resolve().parents[2] / "configs",
        Path(__file__).resolve().parent / "resources/configs",
    ]
    path = next((root / name for root in roots if (root / name).is_file()), None)
    if path is None:
        raise ValueError(f"Default CLI configuration not found: {name}")
    return path


def load_config(
    slug: str, profile: Path | None = None, run_profile: Path | None = None, **overrides
) -> tuple[Competition, RunConfig]:
    roots = [
        Path.cwd() / "configs",
        Path(__file__).resolve().parents[2] / "configs",
        Path(__file__).resolve().parent / "resources/configs",
    ]
    defaults = next(
        (root / "default.yaml" for root in roots if (root / "default.yaml").exists()), None
    )
    values = read_yaml(defaults) if defaults else {}
    values["artifact_root"] = Settings().artifact_root
    if run_profile:
        values.update(read_yaml(run_profile))
    values.update({k: v for k, v in overrides.items() if v is not None})
    profile_path = profile or next(
        (
            root / "competitions" / f"{slug}.yaml"
            for root in roots
            if (root / "competitions" / f"{slug}.yaml").exists()
        ),
        roots[0] / "competitions" / f"{slug}.yaml",
    )
    competition = read_yaml(profile_path) if profile_path.exists() else {}
    if profile and not profile.exists():
        raise ValueError(f"Competition profile not found: {profile}")
    if competition.get("competition", slug) != slug:
        raise ValueError("Profile competition does not match --competition")
    competition["competition"] = slug
    return Competition.model_validate(competition), RunConfig.model_validate(values)
