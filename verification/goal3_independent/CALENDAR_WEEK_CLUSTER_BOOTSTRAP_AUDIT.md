# GOAL3 calendar-aware ISO-week bootstrap sensitivity

## Purpose

The existing retrospective audit resamples sorted target-date clusters. Its
limitations note that adjacent entries in the sorted date list can be separated
by unequal calendar gaps. This supplement checks whether the k=6 vs k=10
conclusions change when the resampling unit is a real ISO calendar week.

This is sensitivity analysis only. It does not replace the prospective
round-cluster adoption protocol and cannot authorize production adoption.

## Cohort and data

- 236 matches / 472 home-away team-side targets
- 17 target dates from 2026-09-19 through 2026-11-03
- 8 ISO-week clusters
- all 16 k candidates retained for selection-frequency and max-t comparisons
- k=6 vs k=10 remains the focal contrast

All matches and both team-side targets in a sampled week receive the same
cluster multiplicity. Scores are target-count weighted by aggregating score
sums and dividing by the sampled target count.

## Resampling designs

1. `iid_iso_week`: sample eight ISO-week clusters with replacement.
2. `circular_two_week_blocks`: sample four circular blocks, each containing
   two consecutive ISO weeks, then truncate to eight week slots.

Both use NumPy PCG64. The two designs use separate deterministic seeds
(`seed` and `seed + 1`). The circular block design preserves one adjacent-week
lag but wraps the final week to the first; this is a limitation, not a claim
that the endpoints are temporally adjacent.

For selection uncertainty, the family is 15 non-reference k contrasts against
k=10 across Brier and natural-log LogLoss (30 contrasts), with a studentized
max-absolute-t 95% band.

## Calibration and residual diagnostics

Four-category calibration is category-wise marginal calibration-in-the-large:
mean predicted category probability minus empirical category frequency, in
percentage points. The family contains 32 intervals (ALL/J1/J2/J3 x k=6/10 x
categories 0/1/2/3+). A common-valid replicate must contain at least one target
in every scope. If the common-valid rate is below 95%, no simultaneous bands
are emitted.

The Poisson diagnostic uses `lambda = -log(P(goal=0))`, conditional Pearson
residual `(goals-lambda)/sqrt(lambda)`, and dispersion `mean(residual**2)`.
This is not raw-goal VMR. It is descriptive, not a formal goodness-of-fit test.

## Known limitations

- Only eight ISO-week clusters are available, so bootstrap tails are uncertain.
- The iid-week design assumes exchangeable weeks and does not preserve serial
  dependence between adjacent weeks.
- The two-week block design preserves one adjacent-week lag but uses circular
  wraparound.
- Neither design separately clusters repeated teams across different weeks.
- The 17 target dates are sparse and not evenly spaced.
- Marginal calibration does not assess probability-bin reliability, calibration
  slope/intercept, or joint multinomial calibration.
- Retrospective source-row availability does not prove forecasts were generated
  before historical deadlines.

## Reproduction

Run from the repository root:

    python verification/goal3_independent/calendar_week_cluster_bootstrap_audit.py --input artifacts/goal3_outer_allk_reproduction.json --out artifacts/goal3_calendar_week_cluster_bootstrap_audit.json --replicates 50000 --seed 20261108

The JSON records input and code SHA-256 values, environment versions, cluster
counts, valid-replicate rates, selection bands, calibration bands, and
conditional Poisson residual bands.
