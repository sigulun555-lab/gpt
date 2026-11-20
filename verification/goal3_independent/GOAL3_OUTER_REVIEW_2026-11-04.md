# GOAL3 fixed-k and all-k outer diagnostics — independent audit

## Scope and provenance

This is a retrospective audit of the frozen GOAL3 outer cohort. It is not a prospective shadow score, not independent selection evidence, and not a basis for production adoption.

- Frozen cohort: 472 team-side targets, 236 matches, 17 target dates, folds 1–3 (2026-09-19 through 2026-11-03).
- Fixed k=6/k=10 source: `goal3_outer_matches.csv`, six-decimal probabilities from GPT PR #20 comments #6626337643 and #6626339347.
- Fixed-k audit code Git blob SHA: `e3bdf49727fb70aa94cbb6af8b1cad134d4f5078`.
- All-k reconstruction code Git blob SHA: `f2e8fbbcae2023355d50b83d99e7bb743b2a2256`.
- Reference optimizer code Git blob SHA: `7ac8f83da420fa577b45e9c5c021ce46387f584e`; PR #19 reference SHA: `5f73df9e6fdb43d5cf333c04036f87ae16c20d1f`. The PR #20 version differs only by the explicit 2026-only date guard.
- Reproduction commands from repository root:

```bash
python verification/goal3_independent/review_outer_matches.py \
  --csv verification/goal3_independent/goal3_outer_matches.csv \
  --out artifacts/goal3_outer_review.json \
  --replicates 10000 --seed 1

python verification/goal3_independent/reproduce_allk_outer.py \
  --frozen-csv verification/goal3_independent/goal3_outer_matches.csv \
  --history2026 data/history.json \
  --aliases data/team_aliases.json \
  --csv2025 verification/goal3_independent/2025.csv \
  --out artifacts/goal3_outer_allk_reproduction.json \
  --bootstrap-replicates 10000 --bootstrap-seed 20261104
```

The all-k JSON is retained as a GitHub Actions artifact, not committed as a repository data file:
- Workflow run #27 / ID `41821374374`, head commit `37bf242ba40886075de292d79ba7dacc8586b686`.
- Artifact #13316862747, ZIP SHA-256 `835538454ffd6b3b8712ad6de075493bbf54a8dd1acdb3ef0e3c3e689b7c7f34`.
- It contains 236 per-match records, all 16 k probability arms, 272 fit diagnostics, and both RNG variants of the all-k moving-block bootstrap.

The all-k reconstruction independently rebuilds the strict earlier-date training cutoffs, per-target predictions, aggregation, and scores, but calls the same `fit_shrinkage` / `predict_shrinkage` optimizer functions as the PR #19 reference. It is not a second independently coded optimizer.

## Fixed k=6 vs k=10: independent score and bootstrap

Negative k6-minus-k10 differences favor k=6.

| Cohort | Targets | k=6 Brier | k=10 Brier | Δ Brier | k=6 LogLoss | k=10 LogLoss | Δ LogLoss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 472 | 0.716248 | 0.716865 | -0.000617 | 1.310399 | 1.312558 | -0.002159 |
| J1 | 120 | 0.731998 | 0.735753 | -0.003755 | 1.345785 | 1.350300 | -0.004515 |
| J2 | 178 | 0.694939 | 0.694075 | +0.000864 | 1.260185 | 1.263192 | -0.003007 |
| J3 | 174 | 0.727185 | 0.727152 | +0.000033 | 1.337363 | 1.337029 | +0.000334 |

Independent paired bootstrap from the frozen six-decimal CSV: 10,000 replicates, NumPy default_rng / PCG64, seed 1, linear percentile intervals. Date-cluster is primary; match-cluster is sensitivity analysis.

| Bootstrap unit | Metric | Mean Δ | 95% CI |
|---|---|---:|---:|
| 17-date cluster | Brier | -0.0006006 | [-0.0035410, +0.0023444] |
| 17-date cluster | LogLoss | -0.0021514 | [-0.0082676, +0.0035968] |
| 236-match cluster | Brier | -0.0006442 | [-0.0035980, +0.0022823] |
| 236-match cluster | LogLoss | -0.0022303 | [-0.0082471, +0.0037677] |

All four intervals cross zero. This cohort does not establish k=6 superiority. Claude's previously reported date-cluster endpoints are close but not identical (largest endpoint difference about 0.000062); the exact Claude RNG, cluster sampling, aggregation, and quantile implementation is unavailable, so bit-for-bit source reproduction is not possible.

## All-k stability reconstruction

The GPT-side reconstruction ran the 16-penalty grid over the same 17 dates: 272 fits, zero optimizer failures, 193–410 prior matches per date, maximum 41 optimizer iterations. k=6 and k=10 each passed the frozen-CSV comparison for 1,888 probability cells, with maximum absolute errors (4.9929\times10^{-7}) and (4.9966\times10^{-7}), respectively (tolerance (2\times10^{-6})).

| k | Brier | LogLoss | Accuracy |
|---:|---:|---:|---:|
| 0.5 | 0.731056 | 1.340518 | 34.32% |
| 1 | 0.726422 | 1.330512 | 34.96% |
| 2 | 0.721163 | 1.319550 | 35.17% |
| 3 | 0.718526 | 1.314307 | 36.02% |
| 4 | 0.717169 | 1.311772 | 37.92% |
| 6 | **0.716248** | **1.310399** | 37.50% |
| 8 | 0.716364 | 1.311110 | 36.86% |
| 10 | 0.716865 | 1.312558 | 36.65% |
| 15 | 0.718498 | 1.316769 | 35.81% |
| 20 | 0.720013 | 1.320537 | 35.81% |
| 30 | 0.722292 | 1.326159 | 34.96% |
| 50 | 0.724926 | 1.332667 | 34.75% |
| 75 | 0.726616 | 1.336867 | 34.75% |
| 100 | 0.727576 | 1.339265 | 34.75% |
| 150 | 0.728625 | 1.341893 | 34.75% |
| 200 | 0.729185 | 1.343302 | 34.75% |

k=6 is the best overall point estimate for Brier and LogLoss in this retrospective cohort, but the valley is shallow and not statistically established. The k=4-to-k=6 Brier gap is 0.000921; k=6-to-k=8 is 0.000116 Brier / 0.000711 LogLoss; k=6-to-k=10 is 0.000617 Brier / 0.002159 LogLoss. Accuracy is not a primary selection metric (k=4 has the highest accuracy in this table).

The post-hoc subgroup split is not stable across sides: k6−k10 is -0.003140 Brier / -0.005464 LogLoss on home targets, but +0.001907 / +0.001146 on away targets. League signs are also mixed (J1 favors k=6; J2 Brier favors k=10; J3 differences are near zero). Fold-level point differences are descriptive only. These same outer dates were used in the original sweep, so none of these slices may be used to retune k.

### All-k paired moving-block bootstrap

The all-k artifact contains 10,000 paired resamples for each candidate versus k=10, with two consecutive ordered target dates per circular block, 17 sampled dates per replicate, target-count-weighted aggregation, seed 20261104, and linear 95% percentile intervals. Both PCG64 and MT19937 were run.

| Candidate − k=10 | RNG | Δ Brier mean [95% CI] | Δ LogLoss mean [95% CI] |
|---|---|---|---|
| k=4 | PCG64 | +0.000333 [-0.003357, +0.003950] | -0.000752 [-0.008058, +0.006156] |
| k=4 | MT19937 | +0.000329 [-0.003298, +0.003973] | -0.000811 [-0.007895, +0.006151] |
| k=6 | PCG64 | -0.000603 [-0.002623, +0.001345] | -0.002139 [-0.006215, +0.001727] |
| k=6 | MT19937 | -0.000606 [-0.002576, +0.001359] | -0.002171 [-0.006081, +0.001741] |
| k=8 | PCG64 | -0.000495 [-0.001356, +0.000327] | -0.001438 [-0.003204, +0.000232] |
| k=8 | MT19937 | -0.000496 [-0.001335, +0.000330] | -0.001451 [-0.003141, +0.000233] |

The intervals for k=6 and k=8 cross zero under both RNGs. This moving-block analysis is a sensitivity analysis on the same retrospective cohort, not prospective adoption evidence.

Claude's aggregate nine-k table and GPT's reconstruction agree to the reported six-decimal precision, but Claude's original per-target `goal3_outer_allk.json` is unavailable. Therefore the seven non-k6/k10 arms have not been compared against Claude's original full-precision per-target predictions.

## Four-category calibration

Mean predicted probability minus observed frequency, in percentage points (positive = overprediction):

| Model / group | 0 | 1 | 2 | 3+ |
|---|---:|---:|---:|---:|
| k6 overall | -0.14 | -1.22 | +0.24 | +1.12 |
| k10 overall | -0.70 | -0.89 | +0.69 | +0.89 |
| k6 J1 | -3.99 | +9.61 | -2.64 | -2.98 |
| k6 J2 | +5.74 | -10.61 | -0.02 | +4.89 |
| k6 J3 | -3.50 | +0.92 | +2.50 | +0.08 |
| k10 J1 | -4.42 | +10.07 | -2.20 | -3.44 |
| k10 J2 | +4.99 | -10.43 | +0.44 | +5.00 |
| k10 J3 | -3.94 | +1.33 | +2.95 | -0.34 |

Overall pooled gaps are small, but hide a J1/J2 split: category 1 is overpredicted in J1 by about 9.6–10.1 pp and underpredicted in J2 by about 10.4–10.6 pp. J3 gaps are smaller. The largest pooled home/away gap is 3.31 pp (k6 away, category 1). The five-quantile-bin maximum absolute gap is 7.32 pp for k6 and 8.21 pp for k10; each bin contains 94–95 targets. These are noisy descriptive diagnostics, not calibration tests or grounds for retrospective correction.

## Poisson residual diagnostics (k=6)

Using (lambda=-\ln(p_0)) and exact observed goals:

- Overall: mean goals 1.3644, mean lambda 1.4069, variance/mean 1.039, observed zero 26.27% vs predicted 26.13%, Pearson residual RMS 0.967.
- Home: mean goals 1.4873 vs lambda 1.5509, zero 24.58% vs predicted 22.74%, residual RMS 0.985.
- Away: mean goals 1.2415 vs lambda 1.2628, zero 27.97% vs predicted 29.53%, residual RMS 0.949.
- J1: variance/mean 1.202; observed/predicted zero 29.17% / 25.17%.
- J2: variance/mean 0.822; observed/predicted zero 21.91% / 27.65%.
- J3: variance/mean 1.098; observed/predicted zero 28.74% / 25.24%.

These are descriptive residual summaries over 17 dates, not formal goodness-of-fit tests. They do not adjust for parameter estimation, repeated teams, date clustering, or between-target variation in lambda. In particular, pooled variance/mean need not equal one when the expected rates vary. No distributional correction is proposed from these diagnostics.

## Source-data quality and temporal caveat

The independent source audit found:

- 2026 history: 429 rows (J1 130 / J2 150 / J3 149), 60 canonical teams, 37 distinct dates from 2026-08-07 through 2026-11-03; duplicate match keys 0, same-team/same-date conflicts 0, invalid scores 0, missing fetchedAt 0, unresolved 2026 aliases 0.
- Alias map: 121 keys / 60 canonical values; every canonical value has a self-key, with no within-season source-name collision in the checked 2025/2026 data. Git blob SHA `5de22584524ff4c07ddac1d27f49afbbbc5f88ab`.
- 2025 community CSV: 1,140 rows (categories 1/2/3 each 380), duplicate keys 0, score/date parse failures 0, invalid team IDs 0, distinct canonical teams 60 and distinct team IDs 60. ID→canonical conflicts 0 and canonical→ID conflicts 0. The only unmapped 2025 name is 沼津; 59 teams overlap with 2026, and レイラック滋賀FC appears only in 2026. No automatic alias was added between those names.
- 2026 history Git blob SHA `f5cc8c044758e45b1e679e4ce796e68b3f27a854`; alias SHA `5de22584524ff4c07ddac1d27f49afbbbc5f88ab`; 2025 CSV SHA `847d324c5a6168d02efe328f8173418a951af720`.

The all-k artifact records the exact input SHA-256 values: frozen CSV `5cc8960c10f3554d44ccc0d5388d0ceeb4c8a39dec4cc8b28bd713f0e3755678`; 2026 history `efca00ea491fcfea6a1267af403cb5ba80ae540fca97a076def25815fc318201`; aliases `92e6377068df8dd63935e0c003adfd7b355e7adac23f7aace4f947be4f5f027e`; 2025 CSV `6175674bc28d4311b219eed5ea2aa5e478fe0f17b8e6b8653f37718057da4288`; nested validation source `cf7ce0b89f2d2c6cfd48f30e1ce72666ad112251121242dbd81b9c467b4bb05d`.

The 2026 history snapshot used by the all-k run has 429 rows / 37 dates, updatedAt `2026-11-03T23:21:25.123Z`, and one shared fetchedAt value across all rows. Strict earlier-date cutoffs prevent future-dated and same-day match leakage, but this snapshot cannot establish when each historical result first became available. This remains retrospective walk-forward evidence only. The 2025 CSV is ancillary; its category is not carried into nested_validation.py training rows, and the 2025+2026 option pools leagues. The fixed-2026-only cohort does not use 2025 rows.

## Latest CI and artifact verification

For head commit `37bf242ba40886075de292d79ba7dacc8586b686`:

- GOAL3 workflow #27 / run ID `41821374374`: success. Syntax, npm tests, shadow-runner unit test, fixed 472-target audit/bootstrap, all-k reconstruction, and both artifact uploads succeeded. The shadow job was skipped because this was a pull-request event; no live forecast was created.
- Validate shared score model #183 / run ID `41821374184`: success. npm test, independent benchmark, nested-validation tests, real-data nested validation, and live official toto/GOAL3 HTML check all succeeded.
- Fixed-k review artifact #13317196907, ZIP SHA-256 `c26bbbb55f8ec4c221c040f58977f94418b4de7b7522c7240381f6eb60286894`.
- All-k artifact #13316862747, ZIP SHA-256 `835538454ffd6b3b8712ad6de075493bbf54a8dd1acdb3ef0e3c3e689b7c7f34`.

The latest PR CI did not create a new prospective forecast. The previously saved round-1670 paired GOAL3 shadow artifact remains untouched and awaits official result-based scoring.

## Remaining limitations and decision rule

1. Claude's comment #6629250071 (2026-11-04 11:18:40 UTC) confirms that its original `goal3_outer_allk.json`, generator, fit diagnostics, and bootstrap implementation were not present in its current workspace. No source-to-source comparison against those originals is possible until they are supplied.
2. GPT's all-k generation uses the shared reference optimizer, so this is not independent optimizer implementation validation. The independent portions are cohort reconstruction, strict cutoff checks, k6/k10 frozen-CSV probability comparison, scoring, subgroup aggregation, calibration, source audit, and bootstrap reimplementation.
3. The fixed cohort contains six-decimal probabilities, not the original full-precision upstream values.
4. The 17-date retrospective set was already used in the original k sweep and must not be reused to choose k.
5. Prospective evidence remains separate from this retrospective audit. At least 10 completed rounds / 30 matches is only an interim descriptive floor. Adoption review requires at least 20 completed rounds / 60 matches, complete paired targets, both overall paired round-cluster 95% upper bounds below zero (four-category Brier and natural-log LogLoss), J1/J2/J3 one-sided 90% upper bounds below the predeclared non-inferiority margins (+0.01 Brier / +0.02 LogLoss), and no increase in k=6 fit failures or missing forecasts. Results from 10–19 rounds remain descriptive and cannot trigger adoption. Until all conditions are met, no adoption decision is justified.

PR #20 remains open and unmerged. No production model, main branch, prediction log, market weight, or pre-existing shadow artifact was changed.
