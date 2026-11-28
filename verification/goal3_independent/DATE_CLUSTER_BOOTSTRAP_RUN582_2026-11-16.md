# GOAL3 date-cluster bootstrap — Run #582 reproduction (2026-11-16)

**Analysis generated:** 2026-11-16 04:26 JST  
**Scope:** retrospective sensitivity on the frozen 2026 outer predictions; not a prospective round-level primary analysis.

## Pinned Run #582 input

- GPT PR #20 head: `105d4673ade10e13b1e2fa5a913efdbae859433b`
- Source workflow: GOAL3 Prospective Shadow Run #582, ID `43752228196`
- Artifact #14175799907; ZIP SHA256 `3283b0bb33f7e3c3194d6911415cc6942914d35305cf6cac044f739cbc034e2d`
- Inner all-k JSON SHA256 `3baf2baa9b6dc9fe948acd0588777245de05982cc29d3d759124073342faa477`
- Cohort: 236 matches, 472 team-side targets, 17 dates (2026-09-19 through 2026-11-03)
- Frozen inputs are unchanged: history SHA256 `efca00ea491fcfea6a1267af403cb5ba80ae540fca97a076def25815fc318201`, alias SHA256 `92e6377068df8dd63935e0c003adfd7b355e7adac23f7aace4f947be4f5f027e`, 2025 CSV SHA256 `6175674bc28d4311b219eed5ea2aa5e478fe0f17b8e6b8653f37718057da4288`, nested-validation Git blob `f84a97aeda80a4820f97fb9f21f5c55cf74f8d17`.

## Agreed bootstrap specification

- Resample 17 date clusters with replacement, 10,000 replicates, seed=1.
- RNG: Python `random.Random(1)` / MT19937; sorted dates before `choices`.
- Keep every match and both team-side targets on a sampled date together.
- Pair k=6 and k=10 on the same sampled dates.
- (D = score(k=6)-score(k=10)); negative values favor k=6.
- Brier is the mean sum of four one-hot squared errors; LogLoss is the mean negative log probability of the observed 0/1/2/3+ category (floor (10^{-12})).
- 95% percentile intervals use linear interpolation.

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

## Reproduction against Run #569

The prior Run #569 and this Run #582 use the same 236 match keys, actual scores, 17 dates, frozen history/alias/2025 inputs, and nested-validation blob. Across all 16 k values and 30,208 stored probability cells, maximum absolute probability difference is (9.0745\times10^{-10}); k=6 maximum difference is (2.1033\times10^{-12}), k=10 is (1.5200\times10^{-12}). The date-cluster Brier/LogLoss CI endpoints differ by less than (4.1\times10^{-14}) between the two runs. This is numerical reproducibility within tolerance, not bitwise identity.

## Decision and limitations

All three paired intervals include zero. No statistical superiority of k=6 is established; k=10 remains the research baseline and k=6 remains a candidate. No production model or market weight change.

The 17 dates lack official toto round IDs, so this is retrospective date-cluster sensitivity, not a prospective round-cluster primary analysis. The 429-row history snapshot has one shared `fetchedAt`; the alias map update is 2026-10-22 and 12/17 dates precede it. The result bootstraps frozen predictions and does not claim a nested-model refit or prospective forecast.
