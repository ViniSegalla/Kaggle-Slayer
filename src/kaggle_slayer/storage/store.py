import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from kaggle_slayer.schemas import Decision, RunState
from kaggle_slayer.schemas.models import now


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "metadata.sqlite"
        with self.connection() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, competition TEXT, status TEXT, phase TEXT,
                    created_at TEXT, updated_at TEXT, state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY, run_id TEXT, timestamp TEXT, agent TEXT,
                    action TEXT, level TEXT, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS decisions_run ON decisions(run_id);
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT, experiment_id TEXT, model TEXT, status TEXT,
                    score REAL, duration REAL, payload TEXT NOT NULL,
                    PRIMARY KEY(run_id, experiment_id)
                );
                CREATE TABLE IF NOT EXISTS submissions (
                    run_id TEXT, attempt INTEGER, status TEXT, payload TEXT NOT NULL,
                    PRIMARY KEY(run_id, attempt)
                );
            """)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.db, timeout=20)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def run_dir(self, run_id: str) -> Path:
        if not run_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in run_id
        ):
            raise ValueError("Invalid run ID")
        path = self.root / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, state: RunState) -> None:
        state.updated_at = now()
        serialized = state.model_dump_json()
        # SQLite is the authoritative checkpoint. JSON is an inspectable mirror.
        with self.connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
                (
                    state.run_id,
                    state.competition.competition,
                    state.status,
                    state.phase,
                    state.created_at,
                    state.updated_at,
                    serialized,
                ),
            )
            for experiment in state.experiments:
                connection.execute(
                    "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?)",
                    (
                        state.run_id,
                        experiment.id,
                        experiment.model,
                        experiment.status,
                        experiment.score,
                        experiment.duration,
                        experiment.model_dump_json(),
                    ),
                )
            for index, submission in enumerate(state.submissions):
                connection.execute(
                    "INSERT OR REPLACE INTO submissions VALUES (?,?,?,?)",
                    (state.run_id, index, submission.get("status"), json.dumps(submission)),
                )
        write_json(self.run_dir(state.run_id) / "state.json", state.model_dump(mode="json"))

    def load(self, run_id: str) -> RunState:
        with self.connection() as connection:
            row = connection.execute("SELECT state FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown run: {run_id}")
        return RunState.model_validate_json(row[0])

    def record(self, run_id: str, event: Decision) -> None:
        payload = event.model_dump_json()
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO decisions(run_id,timestamp,agent,action,level,payload) VALUES(?,?,?,?,?,?)",
                (run_id, event.timestamp, event.agent, event.selected_action, event.level, payload),
            )
        with (self.run_dir(run_id) / "decisions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(payload + "\n")

    def rows(self, table: str, run_id: str | None = None) -> list[dict]:
        if table not in {"runs", "decisions", "experiments", "submissions"}:
            raise ValueError("Unknown metadata table")
        query = f"SELECT * FROM {table}"
        with self.connection() as connection:
            rows = connection.execute(
                query + (" WHERE run_id=?" if run_id else ""), (run_id,) if run_id else ()
            ).fetchall()
        return [dict(row) for row in rows]

    @contextmanager
    def lock(self, run_id: str):
        """Prevent concurrent resume processes from racing on one run."""
        path = self.run_dir(run_id) / ".running.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ValueError(
                f"Run is locked: {path}. After a hard crash, verify no runner is active before removing this file."
            ) from exc
        try:
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            yield
        finally:
            path.unlink(missing_ok=True)
