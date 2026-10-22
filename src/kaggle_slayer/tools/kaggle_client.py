"""Official Kaggle API adapter. Writes are never retried automatically."""

import os
import time
import zipfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from dotenv import load_dotenv


class KaggleError(RuntimeError):
    pass


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "to_dict"):
        return plain(value.to_dict())
    if hasattr(value, "__dict__"):
        return {k: plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def credentials_present() -> bool:
    load_dotenv()
    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    return bool(
        os.getenv("KAGGLE_API_TOKEN")
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        or (config_dir / "kaggle.json").exists()
        or (config_dir / "access_token").exists()
    )


class KaggleClient:
    def __init__(self, api=None):
        self._api = api

    @property
    def api(self):
        if self._api is None:
            if not credentials_present():
                raise KaggleError(
                    "Kaggle credentials unavailable. Configure KAGGLE_API_TOKEN, KAGGLE_USERNAME/KAGGLE_KEY, or ~/.kaggle/kaggle.json; accept competition rules in your browser. Use --data-dir for local data."
                )
            try:
                with redirect_stdout(StringIO()):
                    from kaggle.api.kaggle_api_extended import KaggleApi

                    self._api = KaggleApi()
                    self._api.authenticate()
            except (Exception, SystemExit) as exc:
                self._api = None
                raise KaggleError(
                    "Kaggle authentication failed. Check your official Kaggle credential configuration."
                ) from exc
        return self._api

    def read(self, method: str, *args, **kwargs):
        for attempt in range(3):
            try:
                with redirect_stdout(StringIO()):
                    return getattr(self.api, method)(*args, **kwargs)
            except KaggleError:
                raise
            except Exception as exc:
                status = getattr(
                    exc, "status", getattr(getattr(exc, "response", None), "status_code", None)
                )
                if status in {401, 403}:
                    raise KaggleError(
                        "Kaggle denied access. Check credentials and manually accept the competition rules; access controls cannot be bypassed."
                    ) from exc
                if status == 429:
                    raise KaggleError(
                        "Kaggle rate/submission quota reached. Wait for the account quota to reset."
                    ) from exc
                if attempt == 2 or status not in {None, 408, 500, 502, 503, 504}:
                    raise KaggleError(
                        f"Kaggle {method} failed ({type(exc).__name__}); check network and competition access."
                    ) from exc
                time.sleep(2**attempt)

    def metadata(self, slug: str) -> dict:
        response = plain(self.read("competitions_list", search=slug))
        competitions = response.get("competitions", []) if isinstance(response, dict) else response
        rows = [plain(row) for row in competitions or []]
        return next(
            (row for row in rows if str(row.get("ref", "")).rstrip("/").split("/")[-1] == slug), {}
        )

    def download(self, slug: str, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        self.read("competition_download_files", slug, path=str(destination), quiet=True)
        archive = destination / f"{slug}.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as bundle:
                total = 0
                for member in bundle.infolist():
                    target = (destination / member.filename).resolve()
                    if not target.is_relative_to(destination.resolve()):
                        raise KaggleError("Unsafe dataset archive path")
                    total += member.file_size
                    if total > 5_000_000_000 or member.file_size > 2_000_000_000:
                        raise KaggleError(
                            "Dataset archive exceeds tabular runner extraction limit (5 GB)"
                        )
                bundle.extractall(destination)
        if not list(destination.rglob("*.csv")):
            raise KaggleError(
                "No CSV files downloaded. This runner supports tabular CSV competitions only."
            )
        return destination

    def submit(self, slug: str, file: Path, message: str, *, authorized: bool) -> dict:
        if not authorized:
            raise KaggleError("Submission requires explicit --submit authorization")
        try:
            # No retry: the server might accept an upload before a connection drops.
            with redirect_stdout(StringIO()):
                return plain(self.api.competition_submit(str(file), message, slug, quiet=True))
        except Exception as exc:
            raise KaggleError(
                "Submission failed or its outcome is uncertain. Inspect Kaggle submission history before any further submission."
            ) from exc

    def submissions(self, slug: str) -> list[dict]:
        return plain(self.read("competition_submissions", slug)) or []

    def leaderboard(self, slug: str) -> list[dict]:
        return plain(self.read("competition_leaderboard_view", slug)) or []
