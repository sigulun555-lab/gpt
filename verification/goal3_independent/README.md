# totoGOAL3 model comparison — independent verification (Claude)

Independent re-implementation of `model.js`'s `backtestGoal3()` comparison
(leagueMeanPoisson / empiricalLaplaceBaseline / the simple last-12-games
ratio "teamAdjustedPoisson"), plus one addition: the shrinkage-MLE Poisson
attack/defense model (lambda=10, L-BFGS-B, jointly fit across all teams)
that Claude actually used for round 1666's saved totoGOAL3 predictions.

**Why the addition matters:** `model.js`'s `teamAdjustedPoisson` (via
`teamStats()`) is a simple ratio of each team's last-12-games scored/allowed
average to the league base rate, with no regularization. That is a
different model from the shrinkage-MLE model Claude used for round 1666.
GPT's finding that "team-adjusted hurts" was true for that simple ratio
model, but this script shows it does not hold for the regularized model —
see `results.json` for the numbers.

## Dependencies

- Python 3, `numpy`, `scipy` (`pip install numpy scipy`)

## Input data

- `data/history.json` and `data/team_aliases.json` are read from the repo
  root by default (two directories up from this script).
- The 2025 season is **not** part of this repo. It comes from the public
  community mirror `k-addtimes/jleague_open_data`:

  ```
  git clone https://github.com/k-addtimes/jleague_open_data
  cp jleague_open_data/processed/2025_schedule_with_ids.csv \
     verification/goal3_independent/2025.csv
  ```

  A copy is already bundled here (`2025.csv`) for convenience as of this
  writing; re-fetch it if you want the freshest snapshot.

## Run

```
cd verification/goal3_independent
python3 bench.py
```

Flags: `--history2026`, `--aliases`, `--csv2025`, `--out` to point at
different files. Output is written to `results.json` (same shape as
printed to stderr: per mode, per model, accuracy/brier/logloss/
exactBothSidesAccuracy).

## What it does NOT do

- It does not touch `model.js`, `scripts/auto_pipeline.js`, or any
  production prediction/log file. It is read-only against the repo's data
  files and writes only inside this `verification/` folder.
- It does not decide anything about adopting 2025+2026 training for
  production — that's a separate decision once both sides agree the
  comparison methodology has no leakage (tuning params on the same window
  used for final evaluation, etc.).
