# totoGOAL3 regularization k-sweep — research report

Date: 2026-11-04

## Scope and parameter definition

In the existing GOAL3 shrinkage-MLE implementation, the L2 penalty coefficient is named `lambda`. This study calls that same single coefficient `k` for convenience. It does **not** add a second penalty on top of lambda.

No production model, prediction file, or prediction log was changed.

## Data and protocol

- 2025 history: 1,140 match rows (community mirror source).
- 2026 history: 429 match rows across **37 distinct dates** (corrected from the earlier 34-date statement; see snapshot reconciliation below).
- Outer evaluation: 3 chronological folds, 17 total outer-test dates (2026-09-19 through 2026-11-03), 472 home/away goal-category targets.
- Candidate penalty strengths: 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200.
- Training windows compared: 2026-only and 2025+2026.
- Inner selection remains chronological and uses pooled Brier, then Log Loss, accuracy, and lower penalty as tie-breaks.
- Same-day/future leakage guard retained.
- 1,088 inner optimizer fits completed with zero failures; all 17 selected outer fits completed with zero failures.
- Additional fixed-k outer diagnostics were recorded for every candidate/window. These are diagnostic only and were not used to retune the same outer blocks.

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

## Snapshot reconciliation (2026-11-06)

The immutable files and the latest CI artifact were cross-checked to prevent older nested/backtest outputs from being mistaken for the current 429-row snapshot.

- `data/history.json`, Git blob `f5cc8c044758e45b1e679e4ce796e68b3f27a854`: 429 rows, 37 distinct dates, 2026-08-07 through 2026-11-03. All rows declare source `community_jleague_csv` and the same `fetchedAt=2026-11-03T23:21:25.123Z`.
- `verification/goal3_independent/goal3_outer_matches.csv`, Git blob `56892251679b8774a9e1b7ccc4b5beb42e660e36`: 472 team-side rows / 236 matches / 17 target dates, 2026-09-19 through 2026-11-03. Its probability columns are rounded to six decimal places; full-precision probability reconstruction is in the all-k JSON below.
- GOAL3 workflow run #106 all-k artifact ID `13445387327`: ZIP SHA-256 `130b17b53336dd6bc4421c40126a9dad3a1a497bdeac1ff65f4c8ab68b0b857a`; inner JSON SHA-256 `23c503e3cb4e5c6155678a0d5aa58a51d84200a4b8f46e2a1fb55f8006094bdd`.
- Run #106 selection/calibration artifact ID `13444982923`: ZIP SHA-256 `d529cbf9dbb11797b8696cec88c50185844d5c718be0af30eeec208f8477f342`; its `inputSha256` equals the all-k JSON SHA-256 above.
- The committed `nested_results.json` blob `9b37443e6e4bd52054c1d19b6d799d33ea140545` and `results.json` blob `4132caebe7b1b3c584cd4cb929ad48b60b774e22` are older outputs: each records 380 2026 history rows / 34 dates through 2026-10-25. They are retained unchanged for provenance and must not be cited as the current 429-row nested run.
- The current nested-validation result is Validate #269 artifact ID `13443839995`: ZIP SHA-256 `65c27f47966034521523f33ffea2ef2cfb550f788bbcb6d6f5df575c594d4828`; inner JSON SHA-256 `8afe912d40543d46207402b8a8572dc7fbc099c4f9d3427f4e346a534573060f`. It records 429 rows / 37 dates and outer-fold target counts 120 / 134 / 218, selected windows/lambdas `2025_plus_2026 / 150`, `2026_only / 10`, `2026_only / 10`, 1,088 inner fits, 17 selected outer fits, and zero optimizer failures.
- The 34-date statement above is corrected to 37. This metadata correction does not change the fixed-k 236-match / 472-target scores or their bootstrap results.

The shared `fetchedAt` value and community-maintained source do **not** establish first-known availability at each historical forecast date or full official-source reconciliation. This remains a retrospective limitation, not prospective adoption evidence.

## CI note

The benchmark, npm tests, nested-validation unit test, and full nested GOAL3 run passed. The live official HTML test (`tests/test_live_official.js`) intentionally retains standard toto round 1668 as a fixed regression fixture. Its separate GOAL3 path explicitly selects `commodityId=02` and checks that the current official GOAL3 round is parsed dynamically as a positive integer, with a valid deadline and exactly three fixtures. This prevents standard toto round 1668 from being mistaken for the current GOAL3 round. The current PR #24 validation run #273 passed; GOAL3 research workflow run #107 passed its static/unit and retrospective audit steps, while its prospective shadow job was skipped. No prospective forecast was generated by these PR checks.

## Decision

Keep the current GOAL3 research reference at lambda=10 (same coefficient as k=10). Record k=6 as the leading research candidate, but do not change production or start GOAL3 shadow from this result alone.
