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

The CI job computes 10,000 paired resamples using numpy.default_rng(seed=1) and percentile intervals:
- date-cluster bootstrap: resample 17 dates, retaining every match and both team-sides within each sampled date; primary retrospective interval;
- match-cluster bootstrap: resample 236 matches, retaining both team-sides; sensitivity interval.

Both methods keep k=6/k=10 paired within each resample. Exact results are in the CI artifact named goal3-outer-review-472-targets. The prospective adoption rule remains separate: at least 10 rounds and 30 matches, with paired round/date-block 95% CI upper bounds below zero for both primary metrics and no material league deterioration or quality regression.

## Limitations

1. The CSV contains six-decimal probabilities; full-precision upstream probabilities are not present in the comments.
2. This report does not regenerate the 472 forecasts from the upstream diagnostic generator. Claude's diag.py / analyze.py, exact command and their own code SHA should still be committed for end-to-end reproduction.
3. The 17-date retrospective set was already used in the k-sweep diagnostic. It must not be reused to select k or claim prospective improvement.
4. No production model, main branch, prediction log, market weight or pre-existing shadow artifact is changed.
