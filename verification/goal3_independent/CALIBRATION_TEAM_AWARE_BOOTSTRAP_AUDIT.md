# GOAL3 team-aware, match-pair-preserving calibration sensitivity

## Purpose and scope

This is a research-only sensitivity analysis for the frozen retrospective
GOAL3 all-k reconstruction. It does not refit the football model, create
forecasts, change production, or authorize adoption.

Input cohort: 236 matches, 472 home/away team-side targets, 17 target dates
(2026-09-19 through 2026-11-03), with 20 teams in each of J1/J2/J3. The analysis
compares k=6 and k=10 for one-vs-rest outcomes 0, 1, 2, and 3+.

## Resampling design

For each replicate:

1. Draw 17 target dates with replacement.
2. Independently draw 20 team identities with replacement within each league.
3. Assign each match the product

   `date multiplicity * home-team multiplicity * away-team multiplicity`.

4. Apply that single match weight to both the home and away target rows.

This is a multiway pigeonhole weighting scheme. It preserves the home/away
match pair and introduces repeated-team sensitivity, but it is not an exact
randomization distribution and cannot remove all schedule/network dependence.
The same replicate weights are used for both k arms and all four categories.

## Calibration fits and fail-closed gates

For each k/category, fit the unpenalized one-vs-rest logistic diagnostic:

`logit Pr(Y=category) = intercept + slope * logit(predicted category probability)`.

Ideal values are intercept 0 and slope 1. Bootstrap fits are accepted only if
all are finite, gradient infinity norm <= 1e-6, minimum weighted Hessian
eigenvalue >= 1e-8, and maximum absolute coefficient <= 50. Near-singular,
separated, non-finite, or non-converged fits are marked invalid. Simultaneous
families use the intersection of valid replicates across all eight fits and
require at least 95% common validity; otherwise the audit fails rather than
silently regularizing.

The report includes:
- a 16-parameter max-|t| simultaneous family for the eight arm/category fits;
- an 8-contrast max-|t| simultaneous family for paired k=6 minus k=10
  intercept/slope differences;
- fixed probability-bin reliability for bins with at least 20 targets, with
  a simultaneous family across all eligible bins;
- date, team, and match effective sample size summaries; and
- invalid-fit counts and Hessian/gradient/coefficient diagnostics.

## Pre-CI local validation (not a GitHub Actions result)

A local run against run #205's all-k JSON used:

- input SHA-256:
  `b8f495870355407fdf219f173a4b1597d07c700b538e2ad98483b55889ae8ffa`
- NumPy PCG64 seed: `20261113`
- replicates: 5,000
- batch size: 5,000

Results:
- common valid calibration replicates: 5,000 / 5,000;
- the original bin implementation required only a non-empty weighted denominator. An independent re-audit found that this was insufficient: the raw-eligible bins did not retain adequate Kish effective sample size under the team/date weights;
- the previous "23/23 simultaneous bands include zero" statement is withdrawn as inferential evidence for the team-aware bin family; the underlying point gaps remain descriptive only;
- all 16 per-arm logistic simultaneous bands include their ideal values;
- all 8 paired k=6 minus k=10 logistic simultaneous bands include zero.

Paired k=6 minus k=10 simultaneous 95% bands:

| Category | Intercept difference band | Slope difference band |
|---|---:|---:|
| 0 | [-0.7039, +0.2881] | [-0.6176, +0.2609] |
| 1 | [-1.0270, +0.6442] | [-1.6536, +1.0284] |
| 2 | [-1.5343, +1.2911] | [-1.3703, +1.1534] |
| 3+ | [-1.1409, +0.2802] | [-0.7427, +0.1872] |

The nominal percentile intervals for category 3+ intercept/slope differences
exclude zero in this 5,000-replicate run, but the predeclared simultaneous
8-contrast bands include zero. They are not treated as a family-wise finding.

The median effective sample sizes were 8.76 dates, 10.53 teams per league,
and 34.75 matches. Median zero-weight match fraction was 74.15%. This confirms
that the team-aware product weights create sparse effective samples and wide
uncertainty; the nominal cohort size must not be confused with bootstrap ESS.

### Independent bin-ESS re-audit and reporting gate (2026-11-08)

The previous bin-CI code counted a replicate as valid whenever every eligible
bin had a positive weighted denominator. That did not protect against a bin
being represented by only a handful of effective target rows. Using the exact
PCG64 seed 20261113 and 5,000 date/team draws from the pinned all-k JSON, an
independent implementation computed each bin's Kish ESS as:

    (sum_i w_i)^2 / sum_i w_i^2

over target rows, where both target rows from one match share the same match
weight. Results:

- 23 bins had at least 20 raw targets, but zero of 5,000 replicates had
  Kish ESS >=20 simultaneously in all 23 bins.
- Only 6/23 bins individually met ESS >=20 in at least 95% of replicates.
- The intersection across those six bins was 4,744/5,000 = 94.88%, below the
  predeclared 95% common-validity gate.
- The sparsest eligible bins had median ESS between 4.30 and 10.91; several
  had ESS <5 in more than half of replicates.

Therefore the old 4,982/5,000 "all bins non-empty" count was not a valid
effective-sample-size gate. The updated script requires per-bin target-row Kish ESS >=20 in every
replicate used for the simultaneous family. A further independent review
found that target-row Kish ESS alone can still overstate independent
information when both home and away targets from one match fall in the same
bin: those two rows share one match bootstrap weight and are dependent.
The audit now also requires match-cluster contribution Kish ESS >=20, where
each match contributes its match weight multiplied by the number of its
targets in the bin, and that aggregated contribution is squared only after
pairing the two target rows. Both ESS gates must pass in every replicate.
If the common family does not reach 95% validity, it emits no simultaneous
probability-bin bands and records both ESS distributions instead of
presenting underpowered CIs.

Using the exact same 5,000 PCG64 draws (seed 20261113) as the existing
run #228 artifact, an independent reconstruction found:
- target-row ESS gate: common valid replicates 0/5,000;
- match-cluster contribution ESS gate: common valid replicates 0/5,000;
- no eligible bin individually met match-cluster ESS >=20 in 95% of draws
  (best bin: 4,700/5,000 = 94.0%);
- median match-cluster ESS across eligible bins ranged from 4.28 to 33.18.
The stricter gate therefore does not change the current conclusion (no
simultaneous bin CI), but prevents future bin inference from treating two
paired targets as two independent clusters.

This is a conservative reporting correction, not a change to the adoption
rule or a claim that calibration improved. The one-vs-rest logistic model
families use the full weighted target set and retain their own independent
fit-validity gates; they are not the same as the sparse fixed-bin family.

A prior uncommitted 50,000-replicate exploration reported one near-singular
k=10/category-3+ fit (minimum Hessian eigenvalue about 5.86e-10 and a very
large coefficient). That exploratory implementation was not portable or in
CI. The present 5,000-replicate run did not encounter an invalid fit; this
does not disprove a rare failure in a different 50,000-draw sequence. The
committed script now records and rejects such fits explicitly, and future
larger runs can quantify their frequency.

## Reproduction

After rebuilding `artifacts/goal3_outer_allk_reproduction.json`, run:

```bash
python tests/test_goal3_calibration_team_aware_bootstrap.py -v
python verification/goal3_independent/calibration_team_aware_bootstrap_audit.py \
  --input artifacts/goal3_outer_allk_reproduction.json \
  --out artifacts/goal3_calibration_team_aware_bootstrap.json \
  --replicates 5000 --seed 20261113 --batch-size 5000
```

The JSON records the input and script SHA-256 values, Python/NumPy versions,
seed, batch size, common-valid counts, effective sample sizes, and all
simultaneous intervals. GitHub Actions uploads it as a run-specific artifact.

## Interpretation limits

- Only 17 dates and 60 teams are available; this is exploratory retrospective
  sensitivity, not prospective official-round evidence.
- Product multiplicity is a multiway pigeonhole weighting scheme, not an
  exact randomization distribution.
- Pair preservation does not remove all fixture-network dependence.
- One-vs-rest calibration is not joint multinomial calibration.
- Bins below 20 raw targets receive no confidence interval.
- Team-aware probability-bin intervals additionally require both
  target-row Kish ESS >=20 and match-cluster contribution Kish ESS >=20 in
  every replicate across the simultaneous family; otherwise no simultaneous
  bin intervals are emitted.
- The historical `fetchedAt` timestamp does not prove first-known availability
  for each target date.
- No prospective GOAL3 shadow outcomes are included.
- This analysis does not change the k=10 comparison baseline or any production
  model, market weight, prediction log, or existing shadow artifact.
