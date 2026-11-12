# Independent exact-Newton sensitivity for nested GOAL3 validation

This audit is research-only and does not change the pinned L-BFGS-B nested-validation
protocol or its existing artifacts.

## What changes

The original \`nested_validation.py\` is imported for the fixed data loading,
strict date cutoff, inner/outer date blocks, candidate grid, selection rule,
baselines, prediction mapping, and scoring. Only the optimizer function is
temporarily replaced in-memory during this separate sensitivity run.

The replacement minimizes the same clipped penalized-Poisson negative
log-likelihood:

\[
\sum_m \{e^{\eta_{h,m}}-y_{h,m}\eta_{h,m}
          +e^{\eta_{a,m}}-y_{a,m}\eta_{a,m}\}
+\lambda(\|attack\|_2^2+\|defense\|_2^2)
\]

with the same log-rate clipping at [-20, 20] and the same chain-rule mask used
by the reference gradient. Its Hessian is computed analytically as
\(X^T W X + 2\lambda I\) on attack/defense parameters. A damped linear solve
and Armijo backtracking are used first. If the manual Newton path stalls or reaches
its iteration limit, SciPy `trust-exact` polishes the same objective using the
same analytic Hessian. If the trust-region model cannot improve (for example at
a clipping kink), full BFGS may polish the same objective using its analytic
gradient. The final max absolute gradient is always recomputed from the original
objective and must be at most 1e-6 (the same max-absolute-gradient threshold as
the protocol diagnostic gate) or the run fails. Trust-exact and BFGS fallback
attempts, statuses, and selected candidate are recorded per fit; the threshold
is not relaxed.

## Two grids remain separate

- Predeclared protocol: [1, 3, 10, 30, 100].
- Exploratory sensitivity: [0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200].

The outputs are compared fold-by-fold with the original SciPy L-BFGS-B outputs.
The original outputs are read-only. The expanded grid must not replace the
predeclared result after seeing outer-fold scores.

## Interpretation limits

A successful exact-Newton run establishes that this independent solver reached
a stationary point of the stated objective for the recorded fits. It does not
by itself establish historical first-known data availability, prospective
performance, model adoption, or production suitability. Any difference in
selected windows/lambdas or outer scores is a solver-sensitivity finding, not
a new tuning decision. The 5-value grid remains protocol-conforming; the
16-value grid remains exploratory.

Run from the repository root:

\`\`\`bash
python -m unittest verification.goal3_independent.test_nested_newton_sensitivity -v
python verification/goal3_independent/nested_newton_sensitivity.py \\
  --predeclared-baseline artifacts/goal3_nested_validation_predeclared_5lambda.json \\
  --expanded-baseline artifacts/goal3_nested_validation_expanded_16lambda.json \\
  --out-dir artifacts/nested-newton
\`\`\`
