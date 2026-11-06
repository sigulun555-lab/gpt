# GOAL3 crossed date/team-cluster sensitivity (research only)

Date: 2026-11-06 09:05 JST

## Scope and immutable input

This is an exploratory sensitivity analysis of the fixed retrospective GOAL3 all-k outer reconstruction. It is **not** the pre-registered prospective round-cluster adoption analysis and does not change the primary conclusion.

- Repository ref reviewed: \`6010f4bf4801030d79768618c62184e28ba7ce4c\` (GPT PR #24 head at analysis time).
- Source artifact: run #108, artifact ID \`13455466305\`, \`goal3-all-k-outer-reconstruction-42112437917.zip\`.
- ZIP SHA-256: \`6fb90757176bf3848b96f7f1818bcb05a6b073b5fa2abf0e3f5a685a8c18ceb2\`.
- Inner JSON SHA-256: \`23c503e3cb4e5c6155678a0d5aa58a51d84200a4b8f46e2a1fb55f8006094bdd\`.
- Cohort: 236 matches, 472 team-side targets, 17 dates (2026-09-19 to 2026-11-03), 60 teams.

The raw all-k artifact was downloaded and its ZIP and inner-JSON SHA-256 values were recomputed locally. The inner JSON hash matches the artifact's recorded provenance and the previously reported selection-artifact input hash.

## Why this sensitivity was added

The existing retrospective primary sensitivity resamples adjacent sorted target dates in circular blocks of length 2. The target dates are irregularly spaced (1–6 calendar days), and teams recur across dates. The existing analysis already discloses that repeated teams are not separately clustered. This additional analysis tests whether the k=6 vs k=10 conclusion is sensitive to crossed date/team dependence.

## Method

- 50,000 replicates, NumPy PCG64, seed \`20261106\`, linear 2.5/97.5 percentiles.
- Variant A (\`date_x_target_team\`): target weight = sampled-date multiplicity × sampled own-team multiplicity. All fixtures/team-sides from a sampled date remain represented; repeated appearances by a team receive a shared team-cluster weight.
- Variant B (\`date_x_home_away_team_product\`): match weight = sampled-date multiplicity × home-team multiplicity × away-team multiplicity; both team-sides in a match share the same match weight. This dyadic product-weight variant is reported only for overall scores/selection because some league-only draws can be empty.
- For both variants, k=6 and k=10 are paired on the same target rows and same resampling weights. Selection-adjusted bands cover 15 non-reference k values × 2 primary metrics against k=10.
- Four-category calibration and Poisson residual sensitivity use Variant A only, with one global max-|t| family across ALL/J1/J2/J3 × k=6/k=10 × four categories (32 intervals), and one family across ALL/J1/J2/J3 × k=6/k=10 (8 Poisson intervals).

## Results

### k=6 minus k=10

Negative differences favor k=6.

| Crossed bootstrap | Metric | Point difference | Nominal percentile 95% CI | Selection-adjusted 95% band |
|---|---|---:|---:|---:|
| Date × own team | Brier | -0.00061667 | [-0.00547861, +0.00433793] | [-0.00643069, +0.00519735] |
| Date × own team | LogLoss | -0.00215879 | [-0.01187699, +0.00783999] | [-0.01384327, +0.00952569] |
| Date × home/away team product | Brier | -0.00061667 | [-0.00849826, +0.00686949] | [-0.00981372, +0.00858039] |
| Date × home/away team product | LogLoss | -0.00215879 | [-0.01746569, +0.01346699] | [-0.02073167, +0.01641410] |

Every interval includes zero. The point estimates are unchanged, but uncertainty widens materially once repeated team identity is resampled as an additional cluster dimension.

### Calendar-window moving-block sensitivity (46-day grid)

A second date-only sensitivity uses the complete calendar grid from 2026-09-19 through 2026-11-03 (46 days), including no-fixture days as zero-target days. Each replicate samples circular calendar-day blocks until 46 calendar positions are drawn, then retains all team-sides from every selected fixture date. This avoids treating irregularly spaced fixture dates as consecutive observations.

| Calendar block | Metric | Point difference | Nominal percentile 95% CI | Selection-adjusted 95% band |
|---|---|---:|---:|---:|
| 7 days | Brier | -0.00061667 | [-0.00257278, +0.00116326] | [-0.00285780, +0.00162447] |
| 7 days | LogLoss | -0.00215879 | [-0.00623644, +0.00139108] | [-0.00679617, +0.00247859] |
| 14 days | Brier | -0.00061667 | [-0.00212675, +0.00067409] | [-0.00233581, +0.00110247] |
| 14 days | LogLoss | -0.00215879 | [-0.00570347, +0.00121559] | [-0.00645413, +0.00213656] |

Both block lengths retain the same point estimates and all nominal and selection-adjusted intervals include zero. k=6 is best in 44.64%/51.96% of 7-day replicates (Brier/LogLoss) and 52.21%/53.58% of 14-day replicates. This remains exploratory date-only sensitivity and does not address repeated-team dependence.

The previously mentioned 7-day/14-day figures without a committed extraction implementation are superseded by these explicitly reproducible calendar-grid results; the older figures are not used.

### All-k selection frequency

| Variant | Metric | k=6 best | k=8 best | k=10 best |
|---|---|---:|---:|---:|
| Date × own team | Brier | 21.28% | 15.21% | 14.44% |
| Date × own team | LogLoss | 23.70% | 15.11% | 12.39% |
| Date × home/away team product | Brier | 14.15% | 10.23% | 10.61% |
| Date × home/away team product | LogLoss | 15.73% | 10.71% | 10.16% |

k=6 remains the most frequent winner among the displayed k values, but its frequency is much lower than in the date-only moving-block analysis and is far from decisive.

### Four-category calibration (date × own team)

- Global max-|t| critical value: \`3.1364688493\`.
- 32/32 simultaneous 95% bands include zero.
- No calibration correction is justified by this sensitivity.

### Poisson Pearson residual dispersion (date × own team)

- Global max-|t| critical value: \`2.6138047080\`.
- 4/8 simultaneous bands include the theoretical reference value 1.
- ALL and J2 remain below 1 in this exploratory crossed-cluster analysis; J1 and J3 bands include 1.
- This remains a descriptive residual diagnostic, not a formal goodness-of-fit test or a basis for changing the model distribution.

## Reproduction

\`\`\`bash
python verification/goal3_independent/crossed_date_team_bootstrap_sensitivity.py \\
  --input artifacts/goal3_outer_allk_reproduction.json \\
  --out artifacts/goal3_crossed_date_team_bootstrap_sensitivity.json \\
  --replicates 50000 --seed 20261106

python tests/test_goal3_crossed_date_team_bootstrap.py -v
\`\`\`

The unit test covers paired target expansion, probability-simplex rejection, duplicate-match rejection, and deterministic output under a fixed seed. Four tests passed locally; syntax compilation passed.

## Limitations and decision

- Only 17 target dates and 60 teams are available; tail behavior remains uncertain.
- The crossed date/team bootstrap is an exploratory sensitivity, not the pre-registered prospective round-cluster bootstrap.
- The dyadic product-weight variant does not support stable league-only resampling for every draw; league-specific calibration/Poisson results therefore use the date × own-team variant only.
- The 2026 history is community-source and all 429 rows share one \`fetchedAt\` timestamp (\`2026-11-03T23:21:25.123Z\`). Historical first-known availability is not established.
- No prospective shadow forecast was generated by this analysis.

**Decision:** k=6 remains a research candidate, but statistical superiority over k=10 is not established. Retain k=10 as the research comparison reference. No production adoption, main update, market-weight change, prediction-log write, existing-shadow modification, PR merge, or workflow dispatch is authorized by this result.
