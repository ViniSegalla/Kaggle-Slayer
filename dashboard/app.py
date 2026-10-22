"""Operations console. Every statistic comes from the persisted run store."""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from kaggle_slayer.config import Settings
from kaggle_slayer.storage.store import Store

st.set_page_config(page_title="KaggleSlayer · Mission Control", page_icon="⚡", layout="wide")
st.markdown(
    """<style>
.block-container {padding-top: 2rem; max-width: 1550px;}
h1 {letter-spacing: -1.5px; font-weight: 750;}
[data-testid="stMetric"] {background:linear-gradient(135deg,#16243a,#141d32);
 border:1px solid #263956;border-radius:14px;padding:18px;}
[data-testid="stMetricValue"] {color:#68f0cd;}
.eyebrow {color:#41e2ba;letter-spacing:3px;font-size:12px;font-weight:700;}
.hero {padding:22px 28px;border:1px solid #29425a;border-radius:18px;
 background:linear-gradient(110deg,#142c3a,#111b31 70%);margin-bottom:24px;}
.hero p {color:#a7b7ce;margin-bottom:0;}
div[data-baseweb="tab-list"] {gap:22px;}
footer {visibility:hidden;}
</style>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div class="hero"><span class="eyebrow">AUTONOMOUS ML OPERATIONS</span>
<h1>⚡ KaggleSlayer</h1><p>Five specialized agents. Measured experiments. Observable decisions.</p></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Mission control")
    st.caption("Local workspace · persistent execution history")
    root = st.text_input(
        "Artifact directory", os.getenv("KS_ARTIFACT_ROOT", Settings().artifact_root)
    )
    auto_refresh = st.toggle("Refresh every 5 seconds", value=False)
    st.button("Refresh now", use_container_width=True)
    st.divider()
    st.caption(
        "Scores reflect recorded validation. Kaggle results appear only after an actual upload and API feedback."
    )


@st.fragment(run_every=5 if auto_refresh else None)
def console():
    store = Store(root)
    raw_runs = store.rows("runs")
    if not raw_runs:
        st.info(
            "No runs yet. Execute `uv run python scripts/demo.py` to populate this console with a synthetic local demonstration."
        )
        return
    states = [
        json.loads(row["state"])
        for row in sorted(raw_runs, key=lambda row: row["updated_at"], reverse=True)
    ]
    all_experiments = [e for state in states for e in state["experiments"]]
    totals = st.columns(4)
    totals[0].metric(
        "Competitions attempted", len({s["competition"]["competition"] for s in states})
    )
    totals[1].metric("Runs recorded", len(states))
    totals[2].metric("Validated models", sum(e["status"] == "completed" for e in all_experiments))
    totals[3].metric("Kaggle uploads recorded", sum(bool(s["submissions"]) for s in states))
    by_id = {state["run_id"]: state for state in states}
    selected = st.selectbox(
        "Competition run",
        list(by_id),
        format_func=lambda run_id: (
            f"{by_id[run_id]['competition']['competition']} · {by_id[run_id]['status']} · {run_id[-12:]}"
        ),
    )
    state = by_id[selected]
    c = state["competition"]
    experiments = state["experiments"]
    completed = [e for e in experiments if e["status"] == "completed"]
    best = next((e for e in experiments if e["id"] == state["best_id"]), None)
    events = [json.loads(row["payload"]) for row in store.rows("decisions", selected)]
    tabs = st.tabs(
        ["Overview", "Agent Activity", "Experiments", "Competition Performance", "Artifacts"]
    )

    with tabs[0]:
        left, right = st.columns([2, 1])
        with left:
            st.subheader(c["competition"])
            st.caption(
                f"{c['task'] or 'Task pending inspection'} · {c['metric'] or 'Metric pending'} · Seed {state['config']['seed']} · {state['created_at']}"
            )
            cards = st.columns(3)
            cards[0].metric("Best validation", f"{best['score']:.5f}" if best else "—")
            cards[1].metric("Current phase", state["phase"])
            cards[2].metric("Elapsed", f"{state['elapsed_seconds']:.1f}s")
            st.progress(
                min(len(experiments) / state["config"]["max_experiments"], 1),
                text=f"{len(experiments)} / {state['config']['max_experiments']} experiment budget used",
            )
            if state["plan"]:
                st.markdown("#### Modeling strategy")
                st.write(state["plan"]["strategy"])
                st.caption(
                    f"Planner: {state['plan']['source']} · validation: {state['plan']['validation']} · {state['plan']['direction']}"
                )
            provenance = state["metadata"].get("provenance")
            if provenance:
                st.info(
                    f"Data provenance: {provenance.get('kind')}. {provenance.get('description', '')}"
                )
            if state["error"]:
                st.error(state["error"])
        with right:
            st.markdown("#### Dataset")
            report = state["report"]
            st.metric(
                "Training / test rows",
                f"{report.get('train_rows', '—')} / {report.get('test_rows', '—')}",
            )
            st.metric("Input features", len(report.get("features", [])))
            st.markdown("#### Latest activity")
            for event in events[-4:][::-1]:
                st.caption(f"{event['timestamp'][11:19]} · {event['agent']}")
                st.write(event["selected_action"].replace("_", " "))
        if state["warnings"]:
            with st.expander(f"Warnings ({len(state['warnings'])})"):
                for warning in state["warnings"]:
                    st.warning(warning)

    with tabs[1]:
        st.subheader("Agent collaboration & tool traces")
        st.caption(
            f"Current agent: {events[-1]['agent'] if events else 'pending'} · phase: {state['phase']}"
        )
        if events:
            activity = pd.DataFrame(events)
            activity["time"] = pd.to_datetime(activity["timestamp"], utc=True)
            st.plotly_chart(
                px.scatter(
                    activity,
                    x="time",
                    y="agent",
                    color="agent",
                    hover_data=["selected_action"],
                    template="plotly_dark",
                    title="Workflow timeline",
                ),
                use_container_width=True,
            )
            agents = st.multiselect(
                "Filter agents", activity.agent.unique(), default=list(activity.agent.unique())
            )
            st.dataframe(
                activity.loc[
                    activity.agent.isin(agents),
                    ["timestamp", "agent", "selected_action", "tool_invoked", "level"],
                ],
                hide_index=True,
                use_container_width=True,
            )
            for event in events[::-1]:
                if event["agent"] in agents:
                    with st.expander(
                        f"{event['agent']} · {event['selected_action']} · {event['timestamp'][11:19]}"
                    ):
                        st.write(event["rationale"])
                        st.json(event)

    with tabs[2]:
        st.subheader("Experiment registry")
        if experiments:
            frame = pd.DataFrame(experiments)
            frame["best"] = frame.id == state["best_id"]
            st.dataframe(
                frame[["id", "model", "status", "score", "train_score", "std", "duration", "best"]],
                hide_index=True,
                use_container_width=True,
            )
            if completed:
                st.plotly_chart(
                    px.bar(
                        pd.DataFrame(completed),
                        x="id",
                        y="score",
                        error_y="std",
                        color="model",
                        template="plotly_dark",
                        title=f"Cross-validation · {c['metric']} ({state['plan']['direction']})",
                    ),
                    use_container_width=True,
                )
                st.plotly_chart(
                    px.scatter(
                        pd.DataFrame(completed),
                        x="duration",
                        y="score",
                        color="model",
                        hover_name="id",
                        template="plotly_dark",
                        title="Validation quality vs training time",
                    ),
                    use_container_width=True,
                )
            chosen = st.selectbox("Inspect experiment", [e["id"] for e in experiments])
            experiment = next(e for e in experiments if e["id"] == chosen)
            st.json(experiment)
            if experiment.get("importance_path") and Path(experiment["importance_path"]).exists():
                importance = pd.read_csv(experiment["importance_path"]).head(20)
                st.plotly_chart(
                    px.bar(
                        importance,
                        x="importance",
                        y="feature",
                        orientation="h",
                        template="plotly_dark",
                        title="Top fitted-model feature importances",
                    ),
                    use_container_width=True,
                )

    with tabs[3]:
        st.subheader("Local validation & Kaggle feedback")
        feedback = state["feedback"].get("leaderboard", {})
        cards = st.columns(3)
        cards[0].metric("Local CV", f"{best['score']:.5f}" if best else "—")
        public = feedback.get("public_score")
        cards[1].metric(
            "Kaggle public score", f"{public:.5f}" if public is not None else "Unavailable"
        )
        cards[2].metric(
            "Leaderboard percentile",
            "Unknown"
            if feedback.get("top_20_percent") is None
            else str(feedback["top_20_percent"]),
        )
        st.caption(
            "Top 20% is an aspiration. A partial leaderboard page cannot establish a percentile."
        )
        if completed:
            progress = pd.DataFrame(completed)
            progress["step"] = range(1, len(progress) + 1)
            progress["best_so_far"] = (
                progress.score.cummin()
                if state["plan"]["direction"] == "minimize"
                else progress.score.cummax()
            )
            st.plotly_chart(
                px.line(
                    progress,
                    x="step",
                    y=["score", "best_so_far"],
                    markers=True,
                    template="plotly_dark",
                    title="Experiment progress on fixed folds",
                ),
                use_container_width=True,
            )
        if state["submissions"]:
            st.dataframe(
                pd.DataFrame(state["submissions"]).astype(str),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No Kaggle submission was made for this run.")
        st.json(feedback)

    with tabs[4]:
        st.subheader("Run artifacts")
        directory = store.run_dir(selected)
        files = [p for p in directory.rglob("*") if p.is_file() and not p.name.startswith(".")]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "artifact": str(p.relative_to(directory)),
                        "size_kb": round(p.stat().st_size / 1024, 1),
                    }
                    for p in files
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        if files:
            selected_file = st.selectbox(
                "Preview or download", files, format_func=lambda p: str(p.relative_to(directory))
            )
            if selected_file.stat().st_size < 20_000_000:
                st.download_button(
                    "Download artifact", selected_file.read_bytes(), file_name=selected_file.name
                )
            if selected_file.suffix == ".html":
                components.html(
                    selected_file.read_text(encoding="utf-8"), height=550, scrolling=True
                )
            elif selected_file.suffix == ".csv":
                st.dataframe(pd.read_csv(selected_file).head(500), hide_index=True)
            elif selected_file.suffix in {".json", ".jsonl", ".log"}:
                st.code(
                    selected_file.read_text(encoding="utf-8")[:40000],
                    language="json" if selected_file.suffix == ".json" else None,
                )
        st.caption(
            "Checkpoints contain Python objects. Load only trusted, locally generated joblib files."
        )


console()
