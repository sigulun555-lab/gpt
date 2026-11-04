# GOAL3 fixed-k outer diagnostic — independent audit

## Scope and provenance

This is a retrospective diagnostic of already-generated k=6 and k=10 paired predictions, not a prospective shadow result and not adoption evidence.

- Source rows: GPT PR #20 comments #6626337643 and #6626339347.
- Frozen cohort: 472 team-side rows, 236 matches, 17 outer-test dates, folds 1–3.
- Prediction vectors in the source comments are rounded to six decimal places.
- CSV: goal3_outer_matches.csv.
- Independent analysis: review_outer_matches.py.
- Reproduce from repository root:

\`\`\`bash
python verification/goal3_independent/review_outer_matches.py \
  --csv verification/goal3_independent/goal3_outer_matches.csv \
  --out artifacts/goal3_outer_review.json \
  --replicates 10000 --seed 1
\`\`\`

The script verifies paired home/away rows, score-category consistency, probability bounds and sums, unique keys, fold/date counts, then calculates paired Brier/Log Loss, league/fold summaries, four-category calibration, Poisson residual diagnostics, date-cluster bootstrap and match-cluster sensitivity bootstrap.

## Independent point-estimate check

Using the six-decimal CSV:

| Cohort | Targets | k=6 Brier | k=10 Brier | Difference | k=6 Log Loss | k=10 Log Loss | Difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 472 | 0.716248 | 0.716865 | -0.000617 | 1.310399 | 1.312558 | -0.002159 |
| J1 | 120 | 0.731998 | 0.735753 | -0.003755 | 1.345785 | 1.350300 | -0.004515 |
| J2 | 178 | 0.694939 | 0.694075 | +0.000864 | 1.260185 | 1.263192 | -0.003007 |
| J3 | 174 | 0.727185 | 0.727152 | +0.000033 | 1.337363 | 1.337029 | +0.000334 |

Negative differences favor k=6. The overall point differences are very small; league signs are not uniform. These results do not establish a k=6 advantage.

## All-k stability table cross-check

Claude's table in PR #20 comment #6626537018 reports k = 0.5, 2, 4, 6, 8, 10, 20, 50, 200. The k=6 and k=10 overall and league Brier/LogLoss values match the independent 472-row CSV recomputation to the shown six decimals. The other seven k values cannot yet be independently recomputed because the all-k per-target artifact has not been committed.

One arithmetic correction to the accompanying interpretation: within k=4, 6, 8, 10, the reported Brier range is 0.717169 - 0.716248 = 0.000921, not within 0.0006. The LogLoss range is 1.312558 - 1.310399 = 0.002159. The local valley is still relatively shallow—k=6 vs k=8 differs by 0.000116 Brier and 0.000711 LogLoss—but k=4 is 0.000921 Brier above k=6. This is descriptive only; these same 472 targets were used in the k-sweep and cannot be used to reselect k.

## Four-category calibration and Poisson residuals

For k=6, mean predicted probability minus observed frequency:

- 0 goals: -0.139 percentage points
- 1 goal: -1.219 pp
- 2 goals: +0.241 pp
- 3+ goals: +1.117 pp

The overall mean gaps are small, but league-specific category-1 gaps have opposite signs in J1 and J2. Quantile-bin output is included in the JSON artifact; bins are descriptive and are not a calibration test.

Using lambda = -ln(p0) and exact observed goals, the k=6 descriptive residual check gives:

- Overall mean goals 1.3644 vs mean lambda 1.4069; variance/mean 1.039; observed zero rate 26.27% vs predicted 26.13%; Pearson residual RMS 0.967.
- Home: mean goals 1.4873 vs lambda 1.5509; zero 24.58% vs predicted 22.74%; residual RMS 0.985.
- Away: mean goals 1.2415 vs lambda 1.2628; zero 27.97% vs predicted 29.53%; residual RMS 0.949.
- J1 variance/mean 1.202 and zero rate 29.17% vs 25.17%; J2 variance/mean 0.822 and zero rate 21.91% vs 27.65%; J3 variance/mean 1.098 and zero rate 28.74% vs 25.24%.

These are descriptive checks on 17 dates, not formal goodness-of-fit tests. They do not adjust for model fitting degrees of freedom, repeated teams, or date clustering. J1/J2 differences warrant monitoring, not a distribution change.

## Source-data audit

The independent source-file audit found:

- 2026 history: 429 rows (J1 130 / J2 150 / J3 149), 60 canonical teams, 2026-08-07 through 2026-11-03, no duplicate match keys, no same-team/same-date conflicts, no invalid/missing scores, no missing fetchedAt, and no unresolved 2026 aliases.
- 2026 history Git blob SHA: f5cc8c044758e45b1e679e4ce796e68b3f27a854.
- Alias map: 121 keys / 60 canonical values; Git blob SHA 5de22584524ff4c07ddac1d27f49afbbbc5f88ab.
- 2025 community CSV: 1,140 rows (category 1/2/3 each 380), no duplicate match keys; only unmapped 2025 team name is 沼津; 59 teams overlap with 2026, with レイラック滋賀FC only in 2026. Git blob SHA 847d324c5a6168d02efe328f8173418a951af720.
- The 2026 history and 2025 CSV are community-source datasets, not a full official-source reconciliation. The nested validation loader does not carry the 2025 CSV category into training rows, so a 2025+2026 training window pools the three leagues. This is a methodology caveat, not a defect in the fixed-2026-only cohort summarized above.

## Bootstrap interpretation

The CI job computes 10,000 paired resamples using numpy.default_rng(seed=1), PCG64, and linear percentile quantiles. Run #11 (ID 41804330819) completed successfully and uploaded artifact ID 13308938382 (zip SHA-256 fbe9600bd82e3dae94b0ba2342d399998efaeef652b7da2aae79ebbaa812a721).

| Bootstrap unit | Metric | Mean D | 95% percentile CI |
|---|---|---:|---:|
| 17-date cluster (primary retrospective) | Brier | -0.0006006 | [-0.0035410, +0.0023444] |
| 17-date cluster (primary retrospective) | Log Loss | -0.0021514 | [-0.0082676, +0.0035968] |
| 236-match cluster (sensitivity) | Brier | -0.0006442 | [-0.0035980, +0.0022823] |
| 236-match cluster (sensitivity) | Log Loss | -0.0022303 | [-0.0082471, +0.0037677] |

Both primary intervals cross zero; both sensitivity intervals also cross zero. The independent NumPy date-cluster intervals are close to Claude's reported Brier [-0.003603, +0.002343] and Log Loss [-0.008314, +0.003585], but not bit-for-bit identical. The largest endpoint difference is about 0.000062. The six-decimal source CSV, random-number generator, percentile implementation, or replicate aggregation may explain the small difference; the exact Claude bootstrap source code is still required to isolate it. This discrepancy does not change the conclusion because all intervals cross zero.

Both methods keep k=6/k=10 paired within each resample. The prospective adoption rule remains separate: at least 10 rounds and 30 matches, with paired round/date-block 95% CI upper bounds below zero for both primary metrics and no material league deterioration or quality regression.

## CI and separate nested-validation result

- GOAL3 dedicated workflow run #11 (ID 41804330819): success for syntax checks, existing tests, shadow-runner unit test, 472-row audit, date/match bootstrap, and artifact upload. Its live forecast job was skipped because this was a pull-request event; no new prospective forecast was created. Review artifact ID: 13308938382.
- Validate shared score model run #167 (ID 41804330795): success for npm test, independent benchmark, nested-validation tests, real-data nested validation, artifact upload, and live official toto/totoGOAL3 HTML checks. The live page parsed current GOAL3 round 1670 successfully after the stale 1668 expectation was removed.
- Separate nested-validation artifact ID 13309347648 used 1,140 rows from 2025 and 429 rows from 2026 (37 distinct 2026 dates). It completed 1,088 inner fits and 17 outer selected fits with zero failures. Its selected folds were: fold 1 = 2025_plus_2026 / lambda 150 (120 targets); folds 2 and 3 = 2026_only / lambda 10 (134 and 218 targets). This is nested lambda/window selection, not the fixed k=6 vs k=10 CSV experiment; do not merge the two results.
- A legacy checked-in nested_results.json on the branch/base reports an older snapshot (380 2026 rows / 34 dates). Use the run #167 artifact above for the current 429-row snapshot; do not treat the legacy file as current.

## Limitations

1. The CSV contains six-decimal probabilities; full-precision upstream probabilities are not present in the comments.
2. The CSV does not contain per-target optimizer fit status or training-row diagnostics, so the reported zero fit-failure count cannot be independently verified from these rows alone.
3. This report does not regenerate the 472 forecasts from the upstream diagnostic generator. Claude's diag.py / analyze.py, exact command, full-precision all-k predictions, fit diagnostics, and their code SHA should still be committed for end-to-end reproduction.
4. The 17-date retrospective set was already used in the k-sweep diagnostic. It must not be reused to select k or claim prospective improvement.
5. No production model, main branch, prediction log, market weight or pre-existing shadow artifact is changed.
