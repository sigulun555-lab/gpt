# GOAL3 run #567 artifact review and independent bootstrap reproduction

**Audit time:** 2026-11-14 08:25 JST  
**Repository head:** `f96ad88f39b1927cd5b31f5707f00847213bd7e7`  
**Workflow:** GOAL3 Prospective Shadow (k=6 vs k=10), run #567 / ID `43485559858`  
**Scope:** research-only retrospective diagnostics. No model, market weight, main branch, prediction log, or shadow artifact is modified.

## 1. Run and artifact integrity

Run #567 completed with:
- `static-and-unit-tests`: SUCCESS; 70 numbered steps completed successfully.
- `shadow`: SKIPPED.
- Workflow conclusion: SUCCESS.
- Therefore this run produced retrospective audit artifacts, **not** a new prospective shadow forecast.

The run listed 17 artifacts. Six selected ZIPs were downloaded and SHA-256 checked against the GitHub artifact API digest:

| Artifact | ID | ZIP SHA-256 |
|---|---:|---|
| all-k outer reconstruction | 14067621923 | `981b199baa1a1617af9827dab374ba345148c96f48843a4a667a3e75f2b58172` |
| outer source-data review | 14066438647 | `acb051559950135b6b28188916418c9950a4346febc18fe465061abfd3870d19` |
| Poisson residual date-cluster audit | 14066629501 | `659acd9b6c7fde1e3b4237eaa66883d51171b3eaa637ef27d16caf87921a4ae5` |
| selection/calibration bootstrap | 14066843976 | `dab169ca7e538a32257e239c4ae9416d13946c36c19ad33afa973822d4d5400e` |
| pair-preserving date/team bootstrap | 14067277588 | `68cfba8dbf0ead7aa4569d83aa2d3bafb54c98fb889ba164670fa5eb54765442` |
| k-stability influence audit | 14067582071 | `3ef43cd97cda3e819fdc5940a7636c4c4f77eb29b9b0feb5cde9d62f41c5125f` |

The all-k JSON inside artifact #14067621923 has SHA-256 `ed2829dfd1cc767c2a51ec935db5174f329351c404be85bf8377f7b2dd5f1a8c`, exactly matching the `inputSha256` recorded by the selection/calibration and pair-preserving audit outputs.

The pair-preserving audit records code SHA-256 `fe4435c18ac36f8e3313e430d28d6d48c9c5b08449f25fcef7a2bce5e1e34ff0`; the repository script blob is `00089a913a1df5ba3e7a4d227a6ae3667f4ad136`.

## 2. Frozen cohort and point metrics

- 236 matches / 472 team-side targets / 17 dates.
- Date range: 2026-09-19 through 2026-11-03.
- League targets: J1 120, J2 178, J3 174.
- k=6: Brier 0.7162482341217808; LogLoss 1.3103988552522547; accuracy 0.375000.
- k=10: Brier 0.7168649024420994; LogLoss 1.3125576426937850; accuracy 0.366525.

The point estimates favor k=6, but they are not sufficient for selection.

## 3. Selection-adjusted bootstrap

### Date-only primary sensitivity

The selection/calibration artifact specifies PCG64 seed 20261105, 50,000 replicates, circular moving blocks of two consecutive target dates, 17 sampled dates per replicate, and a simultaneous family of 15 non-reference k contrasts x two primary metrics.

For k=6 minus k=10:
- Brier: `[-0.0030013108567323, +0.0017679742160949]`
- LogLoss: `[-0.0069515798439498, +0.0026340049608888]`

Both simultaneous 95% bands include zero.

### Pair-preserving date/team sensitivity — independently reproduced

The pair-preserving artifact specifies:
- PCG64 seed 20261110; 50,000 replicates.
- Independently resample 17 date clusters and 60 team identities with replacement.
- Match weight = date multiplicity x home-team multiplicity x away-team multiplicity.
- Apply the same match weight to both linked home/away team-side targets.
- Redraw replicates with zero total weight in any required league scope.

I independently reimplemented this procedure from the downloaded all-k JSON. The run used NumPy 2.3.5 (the artifact environment reports NumPy 2.4.6). Point estimates, standard errors, simultaneous bands, calibration/residual critical values, and the 8 empty-scope redraws matched the stored artifact to floating-point tolerance:

| k=6 minus k=10 | Independent 95% simultaneous band | Contains 0 |
|---|---|---|
| Brier | `[-0.0097966556376166, +0.0085633189969794]` | Yes |
| LogLoss | `[-0.0206777783284440, +0.0163602034453830]` | Yes |

The exact repository reproduction command is:

```bash
python verification/goal3_independent/pair_preserving_date_team_bootstrap_audit.py \
  --input artifacts/goal3_outer_allk_reproduction.json \
  --out artifacts/goal3_pair_preserving_date_team_bootstrap_audit.json \
  --replicates 50000 --seed 20261110
```

This is a sensitivity analysis, not a prospective test. Both bootstrap designs leave the k=6 vs k=10 difference statistically unresolved.

## 4. k stability

The fixed-prediction leave-one-cluster-out audit covers 313 exclusions:

| Exclusion unit | Clusters | Brier sign flips | LogLoss sign flips |
|---|---:|---:|---:|
| Date | 17 | 1 | 0 |
| Team | 60 | 2 | 0 |
| Match | 236 | 0 | 0 |

This is influence sensitivity on fixed predictions, not refit stability.

## 5. Four-category calibration

The 32 global simultaneous intervals cover ALL/J1/J2/J3 x k=6/k=10 x categories 0/1/2/3+. **32/32 include zero.**

This is marginal category-prevalence calibration-in-the-large only. It does not establish probability-bin reliability, calibration slope/intercept, sharpness, resolution, or conditional calibration by match context. Sparse/empty reliability bins remain a separate limitation.

## 6. Poisson residual dispersion

Date-cluster bootstrap (seed 20261105, 50,000 replicates; 8 simultaneous intervals):
- 6/8 intervals contain theoretical dispersion 1.
- J2 k=6 and k=10 are both below 1 in the date-only simultaneous bands.
- The paired k6-k10 dispersion-difference intervals include zero in ALL, J1, J2, and J3.

Pair-preserving date/team sensitivity (seed 20261110, 50,000 replicates):
- 8/8 residual-dispersion intervals contain 1.
- Thus the apparent J2 under-dispersion is sensitive to the cluster design and is not a robust model-selection signal.

This is a conditional Pearson residual diagnostic based on `lambda=-log(P(goal=0))`, not raw-score toto13 VMR and not a formal goodness-of-fit test.

## 7. Data quality and provenance limitations

The outer-review artifact reports:
- 2026 community history: 429 rows, 37 dates, 60 canonical teams, 0 duplicate match keys, 0 same-team/same-date conflicts, 0 unmapped names, 0 missing fetchedAt values.
- Alias map: 121 keys / 60 canonical values; no missing canonical self-keys and no one-to-many mappings.
- 2025 community CSV: 1,140 rows, 0 duplicate match keys, 0 score/date/team-ID parse failures; however `沼津` is unmapped and appears only in 2025, while `レイラック滋賀FC` appears only in 2026. The nested validation does not carry the 2025 category into training rows and pools leagues.
- The 2026 history and 2025 CSV are community-source data, not a complete official-source reconciliation.
- A single fetchedAt snapshot does not prove original first-known availability for every historical result. The retrospective cohort is not a point-in-time prospective dataset.

## 8. Decision

Keep k=10 as the research baseline and k=6 as a research candidate. No statistical superiority is established. Do not change production model, market weights, or predictions. No new prospective shadow was generated by run #567 because the shadow job was SKIPPED.
