# GOAL3 Prospective Shadow: k=6 vs k=10

## Purpose and status

Research-only prospective comparison of the shrinkage-MLE penalty coefficient. The single L2 penalty currently named `lambda` is called `k` in this experiment; it is not a second penalty.

This experiment does not alter production forecasts, model code, market weights, or published prediction logs. The runner is manually dispatched and stores one immutable workflow artifact per run.

## Frozen comparison

- Arm A: k=10, current research reference.
- Arm B: k=6, leading candidate from the 2026-only nested outer-fold sweep.
- Both arms use the same shrinkage-MLE implementation, same official fixtures, same 2026-only result history, same team aliases, and same forecast cutoff.
- No k=8 sensitivity arm in the primary test.
- No market blend, Dixon-Coles correction, draw boost, or other simultaneous model change.
- Each fixture is trained only on matches with a date strictly earlier than its fixture date.
- The existing snapshot pipeline is run first and writes the input files in the workflow workspace. Its `generatedAt` and `lastFetchedAt` values are captured before its HTTP requests; they are retained as declared pipeline metadata and are **not** treated as actual fetch-completion time.
- The research runner starts only after that fetch step completes and the files are available. It records `snapshotAvailableBy` after reading the inputs; every eligible history row must have a `fetchedAt` timestamp no later than this conservative post-fetch bound. Missing timestamps fail closed. This bound is not claimed to be the exact HTTP completion timestamp.
- `forecastAsOf` is recorded after both k arms finish fitting. Both `snapshotAvailableBy` and `forecastAsOf` must be strictly before the official deadline; a snapshot or completed forecast at/after the deadline is rejected.
- The official GOAL3 round, three fixtures, six team targets, and deadline must be confirmed.
- The artifact records SHA-256 hashes for history, official target snapshot, aliases, runner, and model implementation.

## Outcomes and adoption rule

The target is each team's goal category: 0, 1, 2, or 3+.

Primary metrics:
- Multiclass Brier score (lower is better).
- Log Loss (lower is better).

Minimum evidence before adoption review: at least 10 rounds and 30 matches, with paired predictions for the same team targets. Compare k=6 minus k=10 using a paired round/date-block bootstrap. Adopt k=6 only if the 95% confidence-interval upper bound is below zero for both Brier and Log Loss, with no material deterioration in J1/J2/J3 and no increase in fit failures or missing forecasts. Otherwise retain k=10.

## Operational safeguards

- Actual paired shadow forecast generation runs only on `workflow_dispatch`; pull-request actions, including `ready_for_review`, never launch it.
- Pull-request CI may run syntax/unit tests and read-only retrospective diagnostics, but it does not create or publish prospective shadow forecasts.
- Each run artifact name includes the official round and GitHub run ID. Existing output files are never overwritten.
- The runner does not write to `data/`, production files, or the default branch.
- Production `scripts/auto_pipeline.js` is not modified by this experiment; timestamp handling specific to the shadow is implemented in the research runner.
- A shadow artifact is evidence of forecast generation, not a claim of improved accuracy.
