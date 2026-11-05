# GOAL3 all-k selection and calibration uncertainty audit

## Scope

This is a research-only retrospective audit of the frozen 2026-only outer cohort.
It does not alter production forecasts, model code, market weights, prediction logs,
or any existing prospective shadow artifact. It does not make a production
adoption recommendation.

Input: the immutable goal3_outer_allk_reproduction.json generated
from the fixed 236-match / 472-team-side / 17-date cohort. All 16 k candidates
are read from the same per-match predictions.

## k-selection stability

- Candidates: 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200.
- Resampling unit: sorted target dates, circular moving blocks of length 2.
- Replicates: 50,000; NumPy PCG64; seed 20261105.
- Each sampled date retains all matches and both home/away team-side targets.
- Metrics are target-count weighted across sampled dates.
- Report the bootstrap frequency with which each k is best for Brier and LogLoss.
  Accuracy-best frequency is descriptive only.
- Selection-adjusted simultaneous intervals compare each of the 15 non-reference
  candidates with k=10 for both primary metrics (30 contrasts), using a
  studentized max-absolute-t critical value. The k=6 interval is therefore
  adjusted for searching across the full candidate set and both primary scores.

## Four-category calibration uncertainty

For each k in {6, 10}, report the predicted-minus-observed category rate gap
for 0 / 1 / 2 / 3+ goals, overall and by J1/J2/J3.

- Resampling unit: target date, with all matches and home/away team-sides on a
  sampled date retained together.
- Replicates: 50,000; NumPy PCG64; seed 20261105.
- Within each scope, gaps are target-count weighted across sampled dates.
- Report nominal percentile intervals and simultaneous studentized max-|t|
  bands for separate families: k6 by league (12 intervals), k10 by league
  (12), k6 overall (4), k10 overall (4), and one global family (32).

The separate 'k6_by_league' family reproduces the previously reported
12-comparison J1/J2/J3 x four-category analysis. The global family is a more
conservative sensitivity analysis across all scopes and both k values.

## Reproduction

~~~bash
python verification/goal3_independent/selection_calibration_bootstrap_audit.py \
  --input artifacts/goal3_outer_allk_reproduction.json \
  --out artifacts/goal3_selection_calibration_bootstrap_audit.json \
  --replicates 50000 --seed 20261105
~~~

The workflow uploads the resulting JSON artifact and runs deterministic unit
tests. The artifact records the exact input JSON SHA-256 and script SHA-256.

## Interpretation limits

Only 17 target-date clusters are available. Circular moving blocks treat
adjacent sorted target dates as adjacent blocks despite unequal calendar gaps;
repeated teams across different dates are not separately clustered. The
historical history snapshot has one fetchedAt timestamp and cannot establish
the first time each historical result became available. These results are
descriptive uncertainty diagnostics only; selection frequency, calibration
bands, and residual diagnostics do not establish prospective superiority.
