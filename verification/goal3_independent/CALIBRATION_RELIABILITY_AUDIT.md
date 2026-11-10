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


## Independent interpretation check (2026-11-10; run #263)

An independent review of the run #263 reliability artifact re-fit all eight
one-vs-rest logistic point models with a separate statsmodels GLM implementation.
All eight intercept/slope pairs matched the artifact to a maximum absolute
difference below 7.5e-11. A separate vectorized date-cluster bootstrap
reconstruction (5,000 PCG64 replicates, seed 20261112) reproduced all 16
intercept/slope coefficient percentile intervals and the paired 8-contrast
studentized max-|t| bands; the maximum paired-band endpoint difference was
below 8e-11.

The paired k=6 minus k=10 simultaneous bands exclude zero for category 0
intercept and slope, and for category 3+ intercept and slope. The category 1
and category 2 paired bands include zero. These are **relative parameter
differences**, not evidence that either arm is absolutely miscalibrated or
that one arm is globally better calibrated: all 16 arm/category absolute
simultaneous bands still contain their ideal values (intercept 0, slope 1).
The direction is mixed: category 0 point estimates are closer to ideal under
k=10, while category 3+ point estimates are closer under k=6. No recalibration
or k adoption is justified by these results.

The same independent reconstruction of fixed probability-bin reliability
reproduced the 23 eligible bins and their simultaneous bands exactly:
23/23 bands contain zero; 10 non-empty bins below 20 targets have point
estimates only; 23 bins are empty. The family max-|t| critical value was
2.9845353653225204, matching the saved artifact.

This supplements, but does not replace, the separate 32-family
marginal-calibration audit. All results remain exploratory because the cohort
contains only 17 retrospective target dates and repeated teams are not
separately clustered in this date-only calibration analysis.
