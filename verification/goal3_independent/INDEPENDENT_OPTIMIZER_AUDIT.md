# Independent GOAL3 optimizer audit (research only)

## Purpose

This audit independently reconstructs the frozen 2026-only outer predictions for k=6 and k=10. It is separate from the shared reference implementation and does not change production forecasts, market weights, prediction logs, main, or existing prospective-shadow artifacts.

## Independent implementation

`independent_optimizer_audit.py` does not import `nested_validation.py` or `reproduce_allk_outer.py`. It builds explicit dense home/away design matrices and a separate objective/gradient for the penalized Poisson model:

- Home log-rate: `mu + home_advantage + attack[home] - defense[away]`
- Away log-rate: `mu + attack[away] - defense[home]`
- Poisson negative log likelihood, omitting score-only constants, plus `k * (sum(attack^2) + sum(defense^2))`
- Log-rates are clipped to [-20, 20], matching the documented numerical guard.
- Optimization uses SciPy L-BFGS-B with the same initial intercept (log(1.3)) and max-iteration cap (300) as the reference fit, while the objective and dense design-matrix gradient are implemented separately. The gradient is checked against central finite differences in unit tests. This aligns stopping behavior for a like-for-like probability reconstruction; it does not claim a different optimizer algorithm.

The frozen cohort is 236 matches / 472 team-side targets / 17 target dates. Each fit uses only 2026 history rows strictly earlier than the target date. Same-day matches are excluded. The 2025 CSV is not used.

## Input identity guards

The script fails closed unless these SHA-256 values match the all-k artifact's frozen input manifest:

- history2026: `efca00ea491fcfea6a1267af403cb5ba80ae540fca97a076def25815fc318201`
- aliases: `92e6377068df8dd63935e0c003adfd7b355e7adac23f7aace4f947be4f5f027e`
- frozen cohort CSV: `5cc8960c10f3554d44ccc0d5388d0ceeb4c8a39dec4cc8b28bd713f0e3755678`

The script also cross-checks every frozen match key and raw score against the canonicalized 2026 history. It compares reconstructed probabilities against the frozen six-decimal CSV with a maximum absolute tolerance of 2e-6.

## Reproduction

```bash
python -m pip install numpy scipy
python tests/test_goal3_independent_optimizer.py -v
python verification/goal3_independent/independent_optimizer_audit.py \
  --history data/history.json \
  --aliases data/team_aliases.json \
  --frozen-csv verification/goal3_independent/goal3_outer_matches.csv \
  --out artifacts/goal3_independent_optimizer_audit.json \
  --bootstrap-replicates 50000 --seed 20261105
```

The report includes per-date/per-k fit diagnostics, independent probability reconstruction error, Brier / LogLoss / Accuracy, paired circular moving-block bootstrap intervals, four-category calibration, and descriptive Poisson Pearson residual diagnostics.

## Interpretation limits

This is a retrospective independent optimizer reconstruction, not prospective evidence. The history snapshot's common `fetchedAt` does not establish the original first-known time of every historical result. The 17-date cohort is small, and k=6 is not adopted based on this audit. Existing prospective shadow artifacts remain immutable and separate.
