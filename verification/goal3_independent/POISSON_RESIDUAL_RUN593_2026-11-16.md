# GOAL3 Poisson residual audit — Run #593 (2026-11-16)

## Source and scope

- Workflow: GOAL3 Prospective Shadow Run #593
- Run ID: `43766237639`
- Artifact ID: `14181889868`
- Artifact ZIP SHA-256: `b7ca9afe242d993dea5fc58d6f8953d18cd18ada92f088a424dd25ccf915b60c`
- Inner all-k JSON SHA-256: `3baf2baa9b6dc9fe948acd0588777245de05982cc29d3d759124073342faa477`
- Cohort: 236 matches, 472 team-side targets, 17 dates
- Script SHA-256: `170d0d17e9c8b3ff4bd6f6fd3b919830cb208d5a687189a07af435aaf458d36d`

The audit derives `lambda = -log(P(goal=0))` and computes the Pearson
residual `(observedGoals-lambda)/sqrt(lambda)`. Dispersion is the mean squared
Pearson residual. Uncertainty uses 50,000 NumPy PCG64 date-cluster bootstrap
replicates (seed 20261105), retaining both team-sides and every match on a
sampled date. This is descriptive, not a formal goodness-of-fit test.

## Dispersion by competition

| Scope | k | n | Mean squared Pearson residual | 95% percentile CI | Joint max-|t| 95% band |
|---|---:|---:|---:|---:|---:|
| All | 6 | 472 | 0.93580 | [0.79125, 1.07378] | [0.75948, 1.11211] |
| All | 10 | 472 | 0.92768 | [0.78598, 1.06224] | [0.75514, 1.10023] |
| J1 | 6 | 120 | 1.16910 | [0.91022, 1.39958] | [0.86026, 1.47795] |
| J1 | 10 | 120 | 1.16192 | [0.91847, 1.37507] | [0.87408, 1.44976] |
| J2 | 6 | 178 | 0.70877 | [0.53818, 0.90312] | [0.48009, 0.93746] |
| J2 | 10 | 178 | 0.70487 | [0.53366, 0.89768] | [0.47716, 0.93259] |
| J3 | 6 | 174 | 1.00713 | [0.79553, 1.20193] | [0.75175, 1.26251] |
| J3 | 10 | 174 | 0.99407 | [0.78366, 1.18848] | [0.74047, 1.24768] |

The theoretical reference is 1.0. Overall, J1, and J3 intervals include 1.0.
J2 is below 1.0 in this retrospective cohort for both k values; this is a
league-specific under-dispersion signal to investigate, not a reason to change
the model. Paired k=6 minus k=10 dispersion intervals include zero in every
scope:

| Scope | Difference | 95% percentile CI |
|---|---:|---:|
| All | +0.00811 | [-0.00838, +0.02353] |
| J1 | +0.00718 | [-0.03897, +0.05147] |
| J2 | +0.00390 | [-0.01319, +0.02032] |
| J3 | +0.01306 | [-0.01177, +0.03977] |

An independent reconstruction from the four-category vectors confirmed that
the implied Poisson probabilities (P0, P1, P2, P3+) match the stored vectors
to a maximum absolute difference of (1.11\times10^{-16}). This verifies the
residual transformation; it does not establish first-known data availability.

## Limitations and decision

There are only 17 date clusters, and recurring teams are not separately
clustered in this audit. The 429-row history snapshot has one shared
`fetchedAt=2026-11-03T23:21:25.123Z`; first-known availability at each target
date is not established. The J2 signal is descriptive and may reflect sample,
league, or model-fit effects. No k adoption, production model change, or market
weight change is authorized. k=10 remains the research baseline and k=6 remains
a candidate.
