# P1 λ comparison and warm-start audit — 2026-10-28

## Scope and safety
- Audit-only branch: `audit/lambda-out-of-time-2025`
- No production code, `main`, prediction logs, or Round 1666 prediction changed.
- Experiment source: workflow run [40518879108](https://github.com/sigulun555-lab/gpt/actions/runs/40518879108), artifact `p1-2026-experiment-results`.
- Warm-start sample audit: [workflow run 40598051408](https://github.com/sigulun555-lab/gpt/actions/runs/40598051408), artifact `p1-warmstart-sample-20-results`.

## Warm-start sample audit (20 fits)
- Sample size: 20; deterministic stratified selection; seed 20261028.
- Warm = SciPy L-BFGS-B initialized from recorded JS fit parameters.
- Cold = independent zero parameters with intercept log(1.3).
- Warm-start vs cold-start maximum absolute probability difference: `8.356120567709624e-08`.
- Warm vs recorded JS maximum absolute probability difference: `1.0684261370075276e-07`.
- Cold vs recorded JS maximum absolute probability difference: `1.1991125109522827e-07`.
- Target argmax mismatches: 0 for warm and 0 for cold.
- Audit errors: 0; all 20 cases below `1e-4` probability-difference threshold.
- All 20 warm and all 20 cold SciPy fits terminated with `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`.
- Important limitation: this tests sensitivity to optimizer initialization on 20 sampled fits; it does **not** replay the entire sequential JS warm-start chain. Relative-objective termination is not the same as satisfying a gradient-based convergence condition.

## λ comparison
Values below are unweighted means over the 15 metric rows for each training mode (3 start counts × 5 models). They are exploratory comparisons, not a final production selection.

| Training mode | λ | Mean accuracy | Mean Brier (lower better) | Mean log loss (lower better) |
|---|---:|---:|---:|---:|
| 2025 + 2026 | 10 | 0.419626 | 0.650160 | 1.075407 |
| 2025 + 2026 | 30 | 0.432799 | 0.645114 | 1.067238 |
| 2026 only | 10 | 0.444088 | 0.639304 | 1.058743 |
| 2026 only | 30 | 0.451837 | 0.643205 | 1.064271 |

### Interpretation
- For **2025 + 2026**, λ=30 is better on all three aggregate means (higher accuracy, lower Brier, lower log loss).
- For **2026 only**, λ=30 has higher mean accuracy, while λ=10 has better mean Brier and log loss. Choose based on the intended objective; if calibrated probabilistic quality is primary, λ=10 is the more conservative provisional choice. If hit-rate is the primary objective, λ=30 is the provisional choice.
- Therefore, do not force one universal λ across both training regimes based on these results alone. A mode-specific provisional choice is better supported.

## Blocking caveat before production adoption
The experiment manifest reports 16,600/16,600 fits as `precision_stall` and `all_fits_converged=false`. The close SciPy probability comparisons and zero argmax mismatches in the 20-fit sample support numerical stability of the sampled predictions, but they do **not** erase the convergence-status issue or prove global optimality. Keep the λ conclusion provisional until the precision-stall handling/termination criterion is reviewed and the decision is validated out-of-time.

## Suggested message to Claude
The 20-fit independent SciPy warm-start sample audit has completed successfully: no audit errors, no target-argmax mismatches, and max warm-vs-cold probability difference (8.36e-8), well below the (1e-4) threshold. This is a 20-fit initialization-sensitivity sample, not a replay of the entire sequential JS warm-start chain. All sampled SciPy runs terminated by relative objective reduction, so that must not be described as all fits passing a gradient convergence test.

For λ, the 15-row means show:
- 2025+2026: λ=30 wins on accuracy (0.432799 vs 0.419626), Brier (0.645114 vs 0.650160), and log loss (1.067238 vs 1.075407).
- 2026-only: λ=30 wins on accuracy (0.451837 vs 0.444088), while λ=10 wins on Brier (0.639304 vs 0.643205) and log loss (1.058743 vs 1.064271).

Recommendation: move the λ decision forward without waiting for another sample run, but keep it provisional and mode-specific: λ=30 for the combined training regime; for 2026-only use λ=10 if probabilistic scoring/calibration is the primary objective, or λ=30 if accuracy alone is primary. Do not production-adopt until the manifest's 16,600 precision_stall statuses are investigated. No main/production files or prediction logs were changed.
