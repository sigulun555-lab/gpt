# GOAL3 k=6 vs k=10: leave-one-cluster-out influence audit

This research-only audit complements the predeclared date-cluster bootstrap. It
reuses the fixed retrospective all-k predictions and removes one cluster at a
time, without refitting either model.

## Exclusions

- Date: remove both team-side targets for every match on one of the 17 dates.
- Team: remove both targets from every match involving one team in its league.
- Match: remove both paired team-side targets from one of the 236 matches.

For each remaining cohort, report the mean paired difference
`score(k=6) - score(k=10)`. Brier is the four-category squared-error sum
(range 0..2) averaged over targets. LogLoss uses natural logarithms and a
`1e-12` probability floor. Negative Brier/LogLoss differences favor
k=6; positive accuracy differences favor k=6.

## Interpretation limits

These are influence diagnostics only. They are not confidence intervals,
hypothesis tests, independent replications, or prospective evidence. Forecasts
are held fixed; the models are not refit after exclusions. A sign change means
the retrospective point estimate is sensitive to that exclusion, not that the
excluded cluster is erroneous. Historical first-known result availability
remains unproven.

The script records the exact input JSON SHA-256 and its own code SHA-256. It
refuses to overwrite an existing output file.


## Run #593 exact-cohort recheck (2026-11-16)

Source: GOAL3 Prospective Shadow Run #593 (`43766237639`), k-stability
artifact `14181964489`, input SHA-256
`3baf2baa9b6dc9fe948acd0588777245de05982cc29d3d759124073342faa477`.

Overall paired point estimates remain:

- Brier(k=6)-Brier(k=10): -0.0006166683
- LogLoss(k=6)-LogLoss(k=10): -0.0021587874
- Accuracy(k=6)-Accuracy(k=10): +0.0084745763

Leave-one-cluster-out sign changes:

| Exclusion | Clusters | Brier sign flips | LogLoss sign flips |
|---|---:|---:|---:|
| Date | 17 | 1 | 0 |
| Team (remove both targets from every match involving that team) | 60 | 2 | 0 |
| Match (remove both paired team-side targets) | 236 | 0 | 0 |

**Correction to an earlier ad-hoc check:** the authoritative team exclusion removes
both team-side rows for every match involving the excluded team. A prior
row-only exclusion undercounted sensitivity and reported 0/60 Brier sign flips;
the CI artifact correctly reports 2/60. The date and match results are unchanged.

The 16-candidate retrospective sweep selects k=6 for Brier and LogLoss, while
k=4 has the best accuracy. This is selection on the same 472-target retrospective
cohort and is not independent evidence. Leave-one-date/team/match diagnostics
are influence checks, not confidence intervals or prospective replications.
