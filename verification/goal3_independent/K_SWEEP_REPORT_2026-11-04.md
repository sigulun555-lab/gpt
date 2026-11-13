# totoGOAL3 regularization k-sweep — research report

Date: 2026-11-04

## Scope and parameter definition

In the existing GOAL3 shrinkage-MLE implementation, the L2 penalty coefficient is named `lambda`. This study calls that same single coefficient `k` for convenience. It does **not** add a second penalty on top of lambda.

No production model, prediction file, or prediction log was changed.

## Data and protocol

- 2025 history: 1,140 match rows (community mirror source).
- 2026 history: 429 match rows across **37 distinct dates**.
- Outer evaluation: 3 chronological folds, 17 total outer-test dates (2026-09-19 through 2026-11-03), 472 home/away goal-category targets.
- Candidate penalty strengths: 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200.
- Training windows compared: 2026-only and 2025+2026.
- Inner selection remains chronological and uses pooled Brier, then Log Loss, accuracy, and lower penalty as tie-breaks.
- Same-day/future leakage guard retained.
- 1,088 inner optimizer fits completed with zero failures; all 17 selected outer fits completed with zero failures.
- Additional fixed-k outer diagnostics were recorded for every candidate/window. These are diagnostic only and were not used to retune the same outer blocks.

**Date-count correction (2026-11-10):** the earlier “34 distinct dates” value was a documentation error. The pinned `data/history.json` blob `f5cc8c044758e45b1e679e4ce796e68b3f27a854` contains 429 rows; applying the filters in `nested_validation.load_data()` retains all 429 rows and yields 37 distinct dates (2026-08-07 through 2026-11-03). The 17 outer-test dates and 472 targets are unchanged. This correction changes no fitted values, scores, or bootstrap results.

## Fixed-k outer results: 2026-only training

| k | Pooled outer Brier | Pooled outer Log Loss | Accuracy |
|---:|---:|---:|---:|
| 6 | 0.716248 | 1.310399 | 0.3750 |
| 8 | 0.716364 | 1.311110 | 0.3686 |
| 10 | 0.716865 | 1.312558 | 0.3665 |
| 4 | 0.717169 | 1.311772 | 0.3792 |
| 15 | 0.718498 | 1.316769 | 0.3581 |

Across the outer folds, k=8 had the best Brier in fold 1; k=6 was best in folds 2 and 3. k=6 ranked first on pooled Brier and Log Loss, but the margins over k=8 and k=10 are small.

## Paired date-block bootstrap: k=6 minus k=10

Paired circular moving-block bootstrap over 17 outer-test dates, block length 2, 50,000 resamples, seed 20261104; date-level losses were weighted by their target counts.

| Metric | Observed difference (k6 − k10) | 95% interval |
|---|---:|---:|
| Brier | -0.000617 | [-0.002585, 0.001353] |
| Log Loss | -0.002159 | [-0.006107, 0.001709] |

Both intervals include zero. Therefore the current evidence does **not** establish a statistically reliable improvement for k=6 over k=10. The same is true for k=8 versus k=10.

## Interpretation

- **Provisional candidate:** k=6, with 2026-only training, because it has the best pooled outer Brier and Log Loss in this run.
- **Not approved for replacement:** the paired confidence intervals cross zero, only 17 test dates are available, and inner-fold selection favored k=10 in two of three outer folds (the first fold selected k=150 with 2025+2026 history).
- The best inner pooled 2026-only candidate was k=10 (Brier 0.716677; Log Loss 1.316061); k=6 was 0.718152 / 1.317677.
- Adding 2025 history did not help the fixed-k outer comparisons in this dataset. Its provenance is a community mirror, so this should not be interpreted as a definitive proof that all prior-season information is harmful.
- This is a rolling-origin estimate, not proof of future performance. More prospective rounds/dates are needed before adoption.

## CI note

The benchmark, npm tests, nested-validation unit test, and full nested GOAL3 run passed. The overall GitHub Actions job is red only because the separate live-page regression test still hardcodes GOAL3 round 1668; the current official page returned round 1670. That stale assertion is unrelated to the k-sweep and was not modified here.

## Decision

Keep the current GOAL3 research reference at lambda=10 (same coefficient as k=10). Record k=6 as the leading research candidate, but do not change production or start GOAL3 shadow from this result alone.
