# Leaderboard standing

The reviewer retrieves the authenticated account's `userRank` and `teamCount`
from official Kaggle competition metadata during submission review and refresh.
The dashboard displays the rank and top percentage (`100 × rank / team_count`,
lower is better). Top 20% means `rank / team_count <= 0.2`.

This is the account's current competition standing, potentially based on a
different run's best submission. It is separate from the selected run's public
score and is not a historical rank attributed to that submission. The stored
report includes the source, scope and retrieval timestamp.

Missing, zero or inconsistent rank/count data cannot establish a percentage.
The dashboard reports Unknown and, when available, the reason. It never estimates
rank from rounded scores or treats a partial leaderboard page as the full field.
Failure to retrieve standing does not discard successfully retrieved scores.

Update an existing submitted run, without uploading again:

```powershell
uv run kaggle-slayer refresh YOUR_RUN_ID
uv run streamlit run dashboard/app.py
```

Reload the dashboard after refreshing. Previously persisted runs need a refresh
to populate the new fields. Kaggle metadata can lag behind a scored submission.
