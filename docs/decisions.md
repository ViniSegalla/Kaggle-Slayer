# Technical decisions

1. **Typed state machine over LangGraph.** Five specialized agents, explicit phases and durable state provide inspectable collaboration without adding a graph runtime. The orchestrator is intentionally local and sequential.
2. **Deterministic autonomy by default.** Free, offline use trains and optimizes real models. Optional OpenAI-compatible advice changes approved experiment priorities while policy controls validation and uploads.
3. **SQLite plus structured files.** SQLite serves the dashboard and resume logic; JSON snapshots and JSONL make runs inspectable without the app. Atomic file replacement avoids partial JSON reads.
4. **One isolated worker per experiment.** A real wall timeout can stop a stuck model. Estimator thread counts are capped. This is isolation for reliability, not a sandbox for executing untrusted code.
5. **Dense capped categorical features.** A shared sklearn pipeline makes all model families and persisted checkpoints consistent. Category limits and an estimated-cell memory guard trade capacity for bounded resource use. Specialized native categorical handling is future work.
6. **Full-data refit per completed experiment.** More fitting work than refitting only the winner, but reliable final output after later timeouts. Only training data enters the refit.
7. **Conservative integration scope.** Single train/test CSV relationships, one target, label/value or class-probability templates. Ambiguities require YAML. Images, text-model tasks, notebook-only competitions, arbitrary relational joins and ranking tasks are unsupported.
8. **One upload per run.** Explicit CLI consent, persisted intent and no retry prevent duplicate uploads after uncertain network responses. Follow-up feedback is read-only. Kaggle enforces account quotas and competition eligibility; no mechanism bypasses rules.
9. **uv-only dependency and runtime workflow.** `pyproject.toml`, `.python-version` and `uv.lock` are the source of truth. Optional boosting keeps the standard install smaller.
