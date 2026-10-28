# Nested time-series validation for totoGOAL3

Run from the repository root:

\`\`\`bash
python3 -m unittest verification.goal3_independent.test_nested_validation
python3 verification/goal3_independent/nested_validation.py
\`\`\`

The script writes \`nested_results.json\` and prints a compact outer-fold summary.
It does not modify production prediction code or prediction logs.

## Pre-declared validation protocol

- Target outcomes: 2026 J1/J2/J3 fixtures, scored separately for home and away
  goals in categories 0, 1, 2, and 3+.
- Outer evaluation: three expanding/rolling-origin blocks starting at 55%, 70%,
  and 85% of distinct 2026 match dates. Each fold's hyperparameters are chosen
  using only earlier dates, then evaluated on that fold.
- Inner selection: three contiguous time-ordered validation blocks drawn from
  the earlier 40%-to-outer-cutoff date range.
- Candidate penalty strengths: lambda = 1, 3, 10, 30, 100.
- Candidate training windows: 2026 history only; or all 2025 history plus 2026
  matches dated strictly before the target date.
- Selection rule: lowest pooled inner Brier score, then lower Log Loss, then
  higher Accuracy, then lower lambda. This rule is fixed before viewing outer
  fold results.
- Baselines: league-mean Poisson, empirical Laplace category frequencies, and
  the existing simple last-12-match team-ratio model, all using the same
  selected training window for each outer fold.
- Leakage guard: for every scored date, all training rows must have a date
  strictly earlier than the target date. Same-day fixtures are excluded from
  that day's fit, including fixtures with earlier kickoff times.

## Interpretation cautions

This is a nested rolling-origin estimate, not proof of future performance.
The number of outer folds is small, and the available 2026 window is limited.
Lambda=10 was inherited from 1X2 work rather than selected specifically for
GOAL3; this script evaluates whether inner time-series selection improves
outer-fold performance. Do not change production settings based only on one
metric or one fold. Compare pooled outer Brier and Log Loss against baselines,
inspect fold-to-fold stability, and preserve the outer results from tuning.
