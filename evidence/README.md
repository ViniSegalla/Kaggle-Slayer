# Evidence index

The submission demonstration is the [Home Data personal-best audit package](../demo/home-data-for-ml-course-8ed561cbed5e/README.md), recorded on October 22, 2026.

- Run: **8ed561cbed5e**; live agentic execution, six completed experiments.
- Selected model: **LightGBM + CatBoost ensemble**.
- Five-fold CV MAE: **15,257.82778**.
- Kaggle public MAE: **14,252.33305**; submission **57535510**, status **COMPLETE**.
- [Report](../demo/home-data-for-ml-course-8ed561cbed5e/index.html), [exact metrics](../demo/home-data-for-ml-course-8ed561cbed5e/metrics.json), and [SHA-256 manifest](../demo/home-data-for-ml-course-8ed561cbed5e/manifest.json).

The package contains the original decisions, worker logs, six trained checkpoints, predictions, validation reports and official upload/feedback receipts. Its README provides commands to verify hashes and display the one-run dashboard. [index.json](index.json) lists this single competition demonstration.

## Engineering verification

The retained reports below are historical test evidence. Their counts and timestamps describe those executions, not a fresh test of every later commit. Scripted providers and fixtures demonstrate integration behavior; their scores are not Kaggle results.

| Evidence | Purpose |
|---|---|
| [Test log](tests.log) and [JUnit results](test-results.xml) | Original offline regression suite |
| [Wheel smoke log](wheel-smoke.log) | Installed-package profile and SDK-adapter checks |
| [Offline swarm verification](swarm-offline.md), [report](swarm-verification.json), [JUnit results](swarm-test-results.xml) | Scripted agent handoffs and real numerical training |
| [Coordination report](swarm-coordination-verification.json) and [trace](swarm-coordination-trace.json) | Delegation, dependencies and recovery |
| [Identity verification](swarm-identity-verification.json) | Task/action identity behavior |
| [Repair verification](swarm-repair-verification.json) | Invalid-response recovery |
| [Role guidance](swarm-role-guidance-verification.json) | Role-specific action guidance |
| [Submission protocol](swarm-submission-protocol-verification.json) | Prediction validation and upload safeguards |

Raw competition input CSVs and credentials are excluded. Per-row training labels in the audit package remain local-only and Git-ignored.
