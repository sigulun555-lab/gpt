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

Scoring uses the prospective protocol v1.1 convention: four-category Brier is
the sum of squared category errors per team-side target, and natural-log
LogLoss uses a fixed probability floor of `1e-12`.

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


## 2026-11-07 solver-stationarity sensitivity (same frozen cohort)

The PR #20 workflow run `42328643873` executed an additional solver sensitivity test over the same 17 target dates and both penalties (34 fits total). This test is read-only and is not used to create or overwrite any forecast artifact.

### Why a second solver was needed

The reference-style L-BFGS-B run (default stopping controls, `maxiter=300`) reported SciPy success for 34/34 fits, but the separately calculated infinity norm of the analytic gradient reached 0.0359705. A stricter L-BFGS-B sensitivity run (`maxiter=2000, ftol=1e-12, gtol=1e-6, maxls=50, maxcor=20`) still met the requested gradient tolerance in only 1/34 fits (maximum norm 0.000471082). Therefore SciPy's `success` flag and the requested stationarity tolerance are reported separately.

### Independent exact-Hessian check

The unit-test suite also solves all 34 date/k fits with a separately implemented damped full-Newton method, exact Hessian, and Armijo backtracking:

- Requested gradient infinity-norm tolerance: `1e-7`.
- Converged fits: 34/34.
- Maximum gradient infinity norm: `7.1546155e-9`.
- Median gradient infinity norm: `2.7067415e-10`.
- Maximum Newton iterations: 4.
- Minimum Hessian eigenvalue across fits: `11.5921286` (positive).

The exact-Hessian fits are a numerical sensitivity solution to the same clipped penalized-Poisson objective, not a replacement for the frozen historical outputs. Relative to the frozen six-decimal probabilities, maximum probability drift was `4.11258e-5` for k=6 and `5.35220e-5` for k=10. The resulting point estimates were:

| Metric | k=6 exact Newton | k=10 exact Newton | k6-k10 |
|---|---:|---:|---:|
| Four-category Brier | 0.716246640 | 0.716864843 | -0.000618204 |
| Natural LogLoss | 1.310395659 | 1.312557595 | -0.002161937 |
| Accuracy | 37.5000% | 36.6525% | +0.8475 pp |

The 10,000-replicate circular date moving-block bootstrap remained inconclusive: Brier 95% interval `[-0.0025415, +0.0013501]`; LogLoss `[-0.0060867, +0.0016913]`. Both include zero. The point-estimate direction is unchanged, but this solver sensitivity does not establish k=6 superiority.

The test output now reports `successCount` separately from `gradientToleranceMetCount` and `stricter1e8Count`, avoiding the previous ambiguity where a 1e-8 count was labelled as generic solver success. The exact-Hessian result is a research-only diagnostic; it changes no production model, market weight, prediction log, main branch, or existing shadow artifact.


## 2026-11-08 machine-readable strict stationarity sensitivity

The optimizer audit now writes the exact-Hessian sensitivity into the same
goal3-independent-optimizer-k6-k10 workflow artifact under
strictStationarityAudit. It reruns all 17 target dates for both k=6 and
k=10 on the exact same frozen 2026 history, aliases, and 472-target cohort.

The predeclared diagnostic criterion is identical for both arms:
gradient infinity norm <= 1e-7, damped full-Newton with the exact Hessian,
Armijo c1=1e-4, at most 100 Newton iterations and 60 backtracking halvings.
A failed fit aborts the audit. This is a numerical convergence diagnostic,
not a change to the prospective adoption rule or to the frozen prediction
outputs.

The default L-BFGS-B reference fits and strict Newton sensitivity fits are
both reported per date/k with status, objective, gradient norm, and objective
gap. The artifact also reports strict-solver probabilities, score metrics,
a same-seed circular date-block bootstrap, and probability drift versus both
the frozen CSV and default L-BFGS-B.

Important interpretation: the <=2e-6 frozen-CSV reconstruction tolerance
applies to the independent reproduction of the frozen default optimizer.
The strict Newton result is a different stopping-solution sensitivity; its
probability drift versus default may exceed 2e-6 and is reported rather than
misclassified as a reconstruction failure. Neither solution changes the
frozen predictions or the prospective k=6/k=10 protocol.
