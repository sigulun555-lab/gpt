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
