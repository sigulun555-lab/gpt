# GOAL3 probability-bin reliability and one-vs-rest calibration audit

## Purpose and scope

This is an exploratory, research-only extension of the marginal four-category
calibration-in-the-large audit. It evaluates probability-conditional reliability
descriptively for the frozen retrospective cohort. It does not refit the football
model, change forecasts, alter production, or authorize adoption.

Input is the all-k outer-reconstruction JSON for 236 matches / 472 team-side
targets / 17 target dates (2026-09-19 through 2026-11-03). The analysis compares
k=6 and k=10 and treats the four goal categories (0, 1, 2, 3+) as separate
one-vs-rest outcomes.

## Probability-bin reliability

Bins are fixed in advance at [0.0,0.1), [0.1,0.2), ..., [0.5,0.6), [0.6,1.0].
They are not quantile bins. Each non-empty bin reports target count, mean forecast
probability, observed frequency, and predicted-minus-observed gap in percentage
points. Empty bins are retained explicitly. Bins with fewer than 20 targets
receive point estimates only and no confidence interval.

For bins with at least 20 targets, uncertainty uses 5,000 NumPy PCG64 bootstrap
replicates (seed 20261112), resampling the 17 target dates with replacement and
retaining all matches and both team-sides on each sampled date. Individual
percentile intervals are reported alongside a studentized max-|t| simultaneous
95% band across all eligible non-empty bins for both k values and all categories.

## One-vs-rest logistic recalibration

For each k/category, fit:

    logit Pr(Y=category) = intercept + slope * logit(predicted category probability)

Ideal calibration is intercept 0 and slope 1. The fit is a separate binary
one-vs-rest diagnostic for each category; it is not a joint multinomial
recalibration. The same 5,000 date-cluster bootstrap draws are used across both
k arms and all categories. Percentile intervals are accompanied by a
studentized max-|t| simultaneous 95% band across the 16 arm/category/parameter
estimates. Paired k=6 minus k=10 differences have a separate simultaneous family
of 8 contrasts.

## Reproduction

The PR workflow first rebuilds the fixed-2026-only all-k outer reconstruction,
then runs:

    python verification/goal3_independent/calibration_reliability_audit.py \
      --input artifacts/goal3_outer_allk_reproduction.json \
      --out artifacts/goal3_calibration_reliability_audit.json \
      --replicates 5000 --seed 20261112
    python tests/test_goal3_calibration_reliability_audit.py -v

The output records input JSON SHA-256, script SHA-256, Python/NumPy versions,
all bootstrap counts, and the calibration estimands.

## Interpretation limits

Only 17 retrospective date clusters are available. Teams recur across dates and
are not separately clustered in this sensitivity analysis. Bins are sparse at
the probability extremes; low-count bins are not interpreted. One-vs-rest
logistic calibration is not a joint multiclass calibration model. The analysis
does not assess calibration conditional on match covariates, sharpness or
resolution, and it contains no prospective official-round outcomes. The
retrospective history snapshot has a single fetchedAt timestamp and does not
prove first-known availability for each target date. All results are exploratory
and cannot satisfy the prospective adoption rule.
