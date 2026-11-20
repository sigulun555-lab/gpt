# GOAL3 date-cluster bootstrap — 2026-11-16

**Analysis generated:** 2026-11-16 04:10:24 JST  
**Scope:** retrospective sensitivity on the frozen 2026 outer predictions; not a prospective round-level primary analysis.

## Frozen inputs

- GPT PR #20 head: `0c33ca7c69597e2a72429602a4dfb7ddbd6f66f7`
- Source workflow run #569: [43527604151](https://github.com/sigulun555-lab/gpt/actions/runs/43527604151)
- Artifact #14085113748; ZIP SHA256 `1793be8afbe023e43c9bb5997df7760bb212e23f76d127bb9b4c5650c5c7b512`
- Inner all-k JSON SHA256 `ed2829dfd1cc767c2a51ec935db5174f329351c404be85bf8377f7b2dd5f1a8c`
- Cohort: 236 matches, 472 team-side targets, 17 dates (2026-09-19 through 2026-11-03)
- The companion `goal3_outer_matches.json` is a 472-row projection of the frozen CSV; its probabilities retain the CSV's 6-decimal precision. The bootstrap below uses the pinned raw all-k JSON artifact at full stored precision.

## Agreed bootstrap specification

- Resample 17 date clusters with replacement, 10,000 replicates, seed=1.
- RNG: Python `random.Random(1)` / MT19937; dates sorted before `choices`.
- Keep every match and both team-side targets on a sampled date together.
- Pair k=6 and k=10 on the same resampled dates.
- (D = score(k=6)-score(k=10)); negative values favor k=6.
- Brier: mean of the sum of four one-hot squared errors per team-side target.
- LogLoss: mean negative log probability of the observed 0/1/2/3+ category; floor (10^{-12}).
- 95% percentile intervals with linear interpolation.

## Point estimates

| Metric | k=6 | k=10 | k=6 minus k=10 |
|---|---:|---:|---:|
| Brier | 0.7162482341 | 0.7168649024 | -0.0006166683 |
| LogLoss | 1.3103988553 | 1.3125576427 | -0.0021587874 |
| Accuracy | 0.3750000000 | 0.3665254237 | +0.0084745763 |

## Paired date-cluster 95% intervals for D

| Metric | 95% interval | Includes zero |
|---|---:|---|
| Brier | [-0.0036798730, +0.0024272210] | Yes |
| LogLoss | [-0.0084825070, +0.0037774027] | Yes |
| Accuracy | [-0.0158109474, +0.0298165138] | Yes |

## Decision

The intervals all include zero. This analysis does **not** establish that k=6 is superior to k=10. Keep k=10 as the research baseline and k=6 as a candidate. No production model, market weight, or prediction pipeline change is justified.

## Limitations

- The 17 dates do not carry official toto round IDs; no round-cluster prospective primary analysis is claimed.
- The source 2026 history snapshot has 429 rows sharing one `fetchedAt` value (`2026-11-03T23:21:25.123Z`), so it is not match-specific first-observed provenance.
- The team alias map was updated 2026-10-22; 12/17 target dates precede that update.
- This calculation bootstraps frozen predictions. It does not rerun nested model fitting or establish prospective performance.
- The 472-target model regeneration requested in Claude's comment remains a separate task; this run computes the agreed bootstrap from the pinned raw artifact and does not claim that regeneration was performed.
