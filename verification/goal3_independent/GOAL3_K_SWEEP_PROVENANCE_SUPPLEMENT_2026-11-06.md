# GOAL3 k-sweep provenance and independent recheck supplement — 2026-11-06

Status: research-only supplement for PR #20. This file clarifies the stale data-count statement in the 2026-11-04 report and records the current, immutable CI evidence. It does not replace historical reports or overwrite any legacy result.

## 1. Scope and immutable references

- Repository: `sigulun555-lab/gpt`
- PR: #20, still OPEN / UNMERGED
- Branch: `research/goal3-prospective-shadow-20261104-v2`
- Audited head before this supplement: `8298f63d756ca471d789f379cfb6ee55c6a7c6cb`
- Fixed retrospective comparison: 2026-only, 236 matches / 472 team-side targets / 17 target dates (2026-09-19 through 2026-11-03).
- Primary comparison: shrinkage-MLE k=6 minus k=10; negative differences favor k=6.
- This is not toto13, not the Issue #23 candidate-3 experiment, and not prospective scoring.

Input Git blob SHAs at the audited head:

| File | Git blob SHA |
|---|---|
| `data/history.json` | `f5cc8c044758e45b1e679e4ce796e68b3f27a854` |
| `data/team_aliases.json` | `5de22584524ff4c07ddac1d27f49afbbbc5f88ab` |
| `verification/goal3_independent/goal3_outer_matches.csv` | `56892251679b8774a9e1b7ccc4b5beb42e660e36` |
| `verification/goal3_independent/nested_results.json` (legacy tracked file) | `9b37443e6e4bd52054c1d19b6d799d33ea140545` |
| `verification/goal3_independent/2025.csv` | `847d324c5a6168d02efe328f8173418a951af720` |
| `verification/goal3_independent/nested_validation.py` | `7ac8f83da420fa577b45e9c5c021ce46387f584e` |

Independent audit scripts:

| Script | Git blob SHA |
|---|---|
| `review_outer_matches.py` | `c6036e816168103200ac6d2dff6c03e50d56c940` |
| `reproduce_allk_outer.py` | `f054c6ba73b7f9f7ed1bfc9608ac338ee6641233` |
| `selection_calibration_bootstrap_audit.py` | `89b4154beed963daa6d0050fc483ca89ecc05fee` |
| `poisson_residual_cluster_audit.py` | `7620970922cb06ec9f6a667e0e48e4ee57080e77` |

## 2. Correction to historical data-count text

The historical `K_SWEEP_REPORT_2026-11-04.md` says “2026 history: 429 match rows across 34 distinct dates.” That pair is not supported by the current frozen input or current CI output and must not be treated as a verified snapshot description.

The current PR #20 head input `data/history.json` is:

- 429 rows / 37 distinct dates, 2026-08-07 through 2026-11-03.
- 60 canonical teams; 429 unique match keys; duplicate match keys 0.
- Same-team/same-date conflicts 0; unmapped 2026 names 0; missing fetchedAt 0.
- All 429 rows use source `community_jleague_csv` and the same `fetchedAt=2026-11-03T23:21:25.123Z`.

The tracked legacy `nested_results.json` records 380 rows / 34 dates through 2026-10-25. It is an older result and is not the current Validate output.

The current Validate artifact independently records 2026 history 429 / 37, 2025 history 1,140 rows, and outer folds of 120 / 134 / 218 targets (5 / 6 / 6 dates). The older tracked file and the 2026-11-04 report are retained unchanged; this supplement is the versioned correction.

The frozen outer CSV itself is internally consistent: 472 rows, 236 unique fixtures, 17 dates, exactly one home and one away target per fixture, no duplicate target keys, all probabilities finite, maximum probability-simplex absolute error about 1e-6, and no mismatch between the stored 0/1/2/3+ category and the raw goal count capped at 3.

Data limitations:
- The 2026 history is community-source data, not an independently reconciled official-results archive.
- A common retrospective fetchedAt timestamp does not establish first-known result availability at each historical forecast cutoff.
- The 2025 audit has one unmapped source name, “沼津”; 2025-only and 2026-only name sets include “沼津” and “レイラック滋賀FC” respectively. They are not silently equated.
- The nested-validation implementation pools leagues in the 2025+2026 window; this limitation applies to that separate window-selection analysis, not to the fixed 2026-only k=6/k=10 cohort.

## 3. Fixed k=6 versus k=10: independent score audit

The fixed six-decimal outer CSV was fetched from the audited PR head and scored independently with the four-category Brier sum per team-side and natural-log loss (floor 1e-12).

| Metric | k=6 | k=10 | Difference k6−k10 |
|---|---:|---:|---:|
| Brier | 0.716248204 | 0.716864911 | -0.000616707 |
| Natural-log loss | 1.310398858 | 1.312557691 | -0.002158833 |
| Argmax accuracy | 37.5000% | 36.6525% | +0.8475 pp |

Fold-level paired differences:

| Fold | Targets | Brier difference | LogLoss difference |
|---|---:|---:|---:|
| 1 | 120 | -0.00006767 | -0.00400762 |
| 2 | 134 | -0.00067440 | -0.00268141 |
| 3 | 218 | -0.00088346 | -0.00081994 |

k=6 is better than k=10 on both loss metrics in all three folds. This is a fixed two-arm directional result, not proof that k=6 is the stable best choice among all 16 candidates.

All-k fold winners from the full-precision reconstruction:

| Fold | Best Brier k | Best LogLoss k |
|---|---:|---:|
| 1 | 8 | 4 |
| 2 | 6 | 6 |
| 3 | 6 | 8 |

Leave-one-date-out fixed two-arm direction: Brier favors k=6 in 16/17 omissions; LogLoss in 17/17. Leave-one-team-out direction: Brier in 58/60 omissions; LogLoss in 60/60. These are sensitivity counts for k=6 versus k=10, not “best among 16 k” selection frequencies.

## 4. Selection-adjusted bootstrap and k stability

The current 50,000-replicate selection/calibration artifact was independently downloaded and its JSON was re-analysed from the all-k per-match artifact. The selection procedure was independently reimplemented with the recorded NumPy PCG64 moving-block scheme; point estimates, best-candidate frequencies, max-|t| critical value, and k=6 simultaneous bands reproduced exactly.

- 16 candidate k values.
- 17 sorted target dates; circular moving blocks of length 2.
- 50,000 replicates; NumPy PCG64; seed 20261105.
- Target-count-weighted aggregation.
- Simultaneous family: 15 non-reference k contrasts × 2 primary metrics against k=10.
- Studentized max-|t| critical value: 2.3785723885.

| Candidate | Brier best-candidate frequency | LogLoss best-candidate frequency |
|---|---:|---:|
| k=6 | 43.288% | 52.582% |
| k=8 | 30.840% | 22.650% |
| k=4 | 12.478% | 18.872% |
| k=10 | 11.300% | 4.542% |

Selection-adjusted simultaneous 95% bands for k=6 minus k=10:

| Metric | Point difference | Simultaneous 95% band |
|---|---:|---:|
| Brier | -0.000616668 | [-0.003001311, +0.001767974] |
| LogLoss | -0.002158787 | [-0.006951580, +0.002634005] |

Both bands include zero. The shallow k=6/k=8 region and the non-identical fold winners mean the retrospective evidence does not establish k=6 superiority.

A separate fixed two-arm bootstrap is stored in the outer-review artifact: 10,000 replicates, NumPy default_rng / PCG64, seed 1, percentile intervals. Date-cluster (17 clusters) Brier [-0.0035410, +0.0023444], LogLoss [-0.0082676, +0.0035968]. Match-cluster (236 clusters) Brier [-0.0035980, +0.0022823], LogLoss [-0.0082471, +0.0037677]. All cross zero. These are GPT-side fixed two-arm sensitivity outputs and use a different RNG sequence from Claude's Python bootstrap.

Separately, Claude's reported IID date-cluster bootstrap (#6678546873) was independently re-run from the full-precision all-k per-match artifact with Python `random.Random(1)`: 10,000 resamples of 17 dates with replacement, retaining all matches and both team-sides on each sampled date, and linear 2.5/97.5 percentiles. The resulting Brier interval [-0.00360258, +0.00234289] and LogLoss interval [-0.00830826, +0.00358537] reproduce Claude's reported endpoints to six decimals.

Claude's match-cluster sensitivity is not yet exactly reproduced. A separate Python `random.Random(1)` match-cluster rerun from the full-precision artifact gives Brier [-0.00354634, +0.00226183] and LogLoss [-0.00824636, +0.00402148], close but not identical to Claude's reported Brier [-0.003610, +0.002336] and LogLoss [-0.008262, +0.004021]. Without Claude's script, 944-row detail, cluster ordering, and exact RNG call sequence, this difference remains unresolved. The GPT-side match-cluster artifact does not substitute for those missing Claude inputs.

## 5. Four-category calibration

The full-precision all-k artifact was independently re-analysed with 50,000 date-cluster replicates (PCG64, seed 20261105), retaining all matches and both team-sides on each sampled date. The global family contains ALL/J1/J2/J3 × k=6/k=10 × categories 0/1/2/3+.

- Global 32-family max-|t| critical value: 2.8948762909.
- Simultaneous bands containing zero: 32/32.

k=6 overall predicted probability / observed frequency:

| Category | Predicted | Observed |
|---|---:|---:|
| 0 | 0.261321 | 0.262712 |
| 1 | 0.335268 | 0.347458 |
| 2 | 0.229107 | 0.226695 |
| 3+ | 0.174304 | 0.163136 |

The highest k=6 probability quintile has n=37, mean predicted probability 0.447761, observed frequency 0.378378. This is a small-bin descriptive signal only. No calibration correction is introduced.

## 6. Poisson raw-goal residual diagnostic

The 50,000-replicate residual artifact was independently re-computed from the all-k per-match JSON using raw home/away goal counts, `lambda=-ln(P(goal=0))`, and date-cluster resampling (PCG64, seed 20261105). The 8-scope joint max-|t| critical value reproduced as 2.4576076659.

| Scope | k=6 dispersion index | k=10 dispersion index |
|---|---:|---:|
| ALL | 0.935795 | 0.927684 |
| J1 | 1.169105 | 1.161922 |
| J2 | 0.708773 | 0.704874 |
| J3 | 1.007133 | 0.994073 |

J2's individual joint bands are below the Poisson reference value 1 for both k values, but the paired k6−k10 dispersion difference interval includes zero in every scope. This is a retrospective residual diagnostic, not a formal goodness-of-fit test or a reason to change the distribution. Raw-goal residuals are not interchangeable with a residual calculated from the capped 3+ category.

## 7. Immutable CI evidence on audited head 8298f63

Commit-associated PR workflow runs:

- GOAL3 workflow #119 / run ID `42193950575`: SUCCESS. Static/unit, 472-target audit, all-k reconstruction, league audit, Poisson residual audit, selection/calibration bootstrap, and independent optimizer steps all passed. The separate `shadow` job was SKIPPED.
- Validate #303 / run ID `42193950578`: SUCCESS. Syntax, npm test, independent benchmark, nested real-data validation, and live official HTML test all passed.

Downloaded ZIP SHA-256 values matched GitHub artifact digests for all seven artifacts:

| Artifact ID | Contents | ZIP SHA-256 | Inner JSON SHA-256 |
|---:|---|---|---|
| 13490528728 | fixed outer review | `d1e0b8084449b2b58087b2d9d2138727b98b13406e188e5730bfca58029f901a` | `d4673f292d34aa2c8d720ebdb5e81f1e6f2f1a10485bb04d12faafd760ad06bf` |
| 13490568464 | nested validation | `f1ac25fdcf06ae21c1edb6e605ea144f6cef9d5e2992460ed7ae5112fbb4521d` | `66d96c59ddb8d00524ad1fcab3e5a55b53042112d677aaa4ec28d28a9d017df7` |
| 13491081710 | Poisson residual cluster audit | `00f22123a3d1417b09b6a77529aebd445a883697be76f41240c2a4a438502f9a` | `9fdd3e12d5442a04a714faecd5ea2b1904d2a47658898fa01f3e0407c72fea32` |
| 13491091670 | independent optimizer audit | `a39a5f49f0abd3e79bddd47959bccda621dacb7524e9c42c3971b701e36ad846` | `b1af8670ed2e67f5c9e16f5064c76203f52a36eaa96641aac24a5294546b003f` |
| 13491281297 | league k6/k10 audit | `1ce046e41bfd5394363802a4e126873cd543dd527df49d49cc04857e7a3cfc03` | `42e7ed40e00797fa2ded1168f539d9782b2b42b147ed7fbce18a762d5feee5b5` |
| 13491366106 | all-k outer reconstruction | `e9217b5b390706d1b413aa2497037a6bb5321200cbed294b83057d41f5129a5e` | `23c503e3cb4e5c6155678a0d5aa58a51d84200a4b8f46e2a1fb55f8006094bdd` |
| 13491825964 | selection/calibration bootstrap | `9335134c9a337e20dc67a91ced36b155c63b70e17803375cc9af05f36ea9f4bb` | `59652088e018eba09a245d257b72ad29b5f99d4a7b610e1536657c964b01a3e0` |

The old 2026-11-04 report's statement that the overall Validate job was red because of a hard-coded round number is historical and superseded by Validate #303, where the live official HTML step passed. The older report is not edited in place.

## 8. Decision and limits

- k=6 remains a research candidate only; k=10 remains the comparison reference.
- Statistical superiority is not established: the selection-adjusted Brier and LogLoss bands both include zero.
- No calibration correction, home-effect correction, Poisson/negative-binomial change, market blend, or production adoption is justified by these retrospective diagnostics.
- Protocol v1.1 still requires at least 20 completed official rounds / 60 matches for adoption review, paired round-cluster scoring, both overall 95% upper bounds below zero, league non-inferiority guards, complete pairing, and auditable official-result provenance.
- The current PR #20 run #119 did not run the prospective shadow job. CI success is not a prospective forecast or outcome score.
- No official-result scoring or adoption claim is made from the existing forecast-only round-1670 artifact.

No production model, main branch, market weight, existing prediction log, or existing shadow artifact was modified. No PR was merged and no shadow dispatch was run.
