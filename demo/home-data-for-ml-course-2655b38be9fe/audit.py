"""Verify and display this archived run without training or contacting Kaggle."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
RUN_ID = ROOT.name
RUN_DIRECTORY = ROOT / "runs" / RUN_ID


class AuditHandler(SimpleHTTPRequestHandler):
    """Serve the report and shareable evidence, excluding private local outputs."""

    def send_head(self):
        path = Path(self.translate_path(self.path)).resolve()
        if path == ROOT:
            self.path = "/index.html"
            path = ROOT / "index.html"
        manifest = read_json(ROOT / "manifest.json")
        allowed = {
            "index.html",
            "README.md",
            "metrics.json",
            "manifest.json",
            "audit.py",
            *(entry["path"] for entry in manifest["files"] if entry["shareable"]),
        }
        if not path.is_relative_to(ROOT) or path.relative_to(ROOT).as_posix() not in allowed:
            self.send_error(404, "Artifact is not part of the shareable audit package")
            return None
        return super().send_head()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(full=False):
    manifest = read_json(ROOT / "manifest.json")
    checked, byte_count, local_only = 0, 0, 0
    for entry in manifest["files"]:
        if not entry["shareable"] and not full:
            local_only += 1
            continue
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Manifest path leaves the demo directory")
        if not path.is_file():
            raise ValueError(f"Missing archived artifact: {entry['path']}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {entry['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {entry['path']}")
        checked += 1
        byte_count += entry["bytes"]
    state = read_json(RUN_DIRECTORY / "state.json")
    submission = RUN_DIRECTORY / "submissions" / "submission.csv"
    if (
        hashlib.sha256(submission.read_bytes()).hexdigest()
        != state["metadata"]["submission_sha256"]
    ):
        raise ValueError("Submission differs from the validated uploaded artifact")
    result = {
        "run_id": RUN_ID,
        "verified_files": checked,
        "verified_bytes": byte_count,
        "local_only_files_skipped": local_only,
        "status": state["status"],
        "local_cv_mae": state["feedback"]["local_score"],
        "public_mae": state["feedback"]["leaderboard"]["public_score"],
        "submission_reference": state["submissions"][-1]["reference"],
    }
    print(json.dumps(result, indent=2))
    return state


def dashboard_snapshot(source_state):
    """Create a derived, one-run UI store; original evidence is never rewritten."""
    sys.path.insert(0, str(REPOSITORY / "src"))
    from kaggle_slayer.storage.store import Store

    manifest = read_json(ROOT / "manifest.json")
    original_run = manifest["original_run_root"].replace("\\", "/")
    original_workspace = original_run.rsplit("/artifacts/runs/", 1)[0]

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            for old, new in (
                (original_run, RUN_DIRECTORY.as_posix()),
                (original_workspace, REPOSITORY.as_posix()),
            ):
                if normalized == old or normalized.startswith(old + "/"):
                    return new + normalized[len(old) :]
            return value
        return value

    state = relocate(source_state)
    state["config"]["artifact_root"] = ROOT.as_posix()
    store = Store(ROOT)
    with store.connection() as connection:
        other = connection.execute(
            "SELECT run_id FROM runs WHERE run_id != ?", (RUN_ID,)
        ).fetchone()
        if other is not None:
            raise ValueError("The demo UI store unexpectedly contains another run")
        connection.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
            (
                RUN_ID,
                state["competition"]["competition"],
                state["status"],
                state["phase"],
                state["created_at"],
                state["updated_at"],
                json.dumps(state),
            ),
        )
        connection.execute("DELETE FROM decisions WHERE run_id=?", (RUN_ID,))
        for line in (RUN_DIRECTORY / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
            event = relocate(json.loads(line))
            connection.execute(
                "INSERT INTO decisions(run_id,timestamp,agent,action,level,payload) "
                "VALUES(?,?,?,?,?,?)",
                (
                    RUN_ID,
                    event["timestamp"],
                    event["agent"],
                    event["selected_action"],
                    event["level"],
                    json.dumps(event),
                ),
            )
        for experiment in state["experiments"]:
            connection.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?)",
                (
                    RUN_ID,
                    experiment["id"],
                    experiment["model"],
                    experiment["status"],
                    experiment["score"],
                    experiment["duration"],
                    json.dumps(experiment),
                ),
            )
        for index, submission in enumerate(state["submissions"]):
            connection.execute(
                "INSERT OR REPLACE INTO submissions VALUES (?,?,?,?)",
                (RUN_ID, index, submission["status"], json.dumps(submission)),
            )
    return store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="Check archived file hashes")
    check.add_argument("--full", action="store_true", help="Also verify local-only label outputs")
    dashboard = commands.add_parser("dashboard", help="Show only this run in Mission Control")
    dashboard.add_argument("--port", type=int, default=8502)
    serve = commands.add_parser("serve", help="Serve the offline audit report on localhost")
    serve.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    state = verify(getattr(args, "full", False))
    if args.command == "dashboard":
        dashboard_snapshot(state)
        environment = dict(os.environ, KS_ARTIFACT_ROOT=str(ROOT))
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(REPOSITORY / "dashboard" / "app.py"),
                    "--server.address=127.0.0.1",
                    f"--server.port={args.port}",
                    "--server.headless=true",
                    "--browser.gatherUsageStats=false",
                ],
                cwd=REPOSITORY,
                env=environment,
            )
        )
    elif args.command == "serve":
        handler = partial(AuditHandler, directory=str(ROOT))
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        print(f"Audit report: http://127.0.0.1:{args.port}/index.html", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
