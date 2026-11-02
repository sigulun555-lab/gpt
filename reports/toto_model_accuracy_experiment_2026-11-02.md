# Toto 1X2 model accuracy experiment — 2026-11-02

## Scope and safeguards

- Research-only experiment on PR #16; no production model, scheduled prediction workflow, or saved prediction row was changed.
- Target period: checked-in 2026 matches beginning 2026-08-07 (410 target matches).
- Training history: 1,140 rows from the 2025 community J.League CSV mirror plus 410 checked-in 2026 rows (1,550 combined; J1 510, J2 520, J3 520).
- The 2025 source is a community-maintained mirror, not an official results feed. Results are diagnostic, not sufficient by themselves to authorize a production switch.
- Walk-forward rules: all same-day matches are excluded from training; outer test dates are later than the training period; inner selection uses only earlier dates.
- Outer evaluation: 220 target matches across 17 calendar-date blocks.
- Metrics: accuracy is top-probability outcome accuracy; Brier is the sum of three squared 1/X/2 errors averaged per match; Log Loss is averaged per match. Lower Brier/Log Loss is better.
- Uncertainty: paired bootstrap resamples calendar-date blocks (2,000 replicates) so matches on the same date remain together.

## Aggregate outer-fold results

| Method | Matches | Accuracy | Brier | Log Loss |
|---|---:|---:|---:|---:|
| Production baseline | 220 | 46.82% | 0.64390 | 1.07434 |
| Smoothed competition-frequency baseline | 220 | 44.09% | 0.64205 | 1.06134 |
| Nested K selection (K=3/6/10) | 220 | 47.27% | 0.63100 | 1.04943 |
| League-specific nested selection (production / frequency / K=3/6/10) | 220 | 46.82% | 0.62843 | 1.04660 |
| Fixed K=10 | 220 | 47.27% | 0.63100 | 1.04943 |

The shrinkage candidates improve the aggregate point estimates, especially Brier and Log Loss. Accuracy improves only slightly for the global nested K candidate and is unchanged for league-specific selection.

## Paired date-block bootstrap

For league-specific nested selection minus production baseline:

| Metric | Point difference | 95% interval |
|---|---:|---:|
| Accuracy | 0.00 percentage points | -4.32 to +4.17 pp |
| Brier | -0.01547 | -0.04420 to +0.01088 |
| Log Loss | -0.02774 | -0.07096 to +0.01170 |

All intervals cross zero. The observed improvements are promising but not statistically established by this sample. The global nested K candidate also has intervals crossing zero (accuracy +0.45 pp [-3.47,+3.97], Brier -0.01290 [-0.04077,+0.01503], Log Loss -0.02492 [-0.06737,+0.01715]).

## League-level differences

League-specific nested selection minus production baseline:

| League | Matches | Accuracy difference | Brier difference | Log Loss difference |
|---|---:|---:|---:|---:|
| J1 | 60 | -3.33 pp | +0.00219 | -0.00176 |
| J2 | 80 | 0.00 pp | -0.04821 | -0.08148 |
| J3 | 80 | +2.50 pp | +0.00401 | +0.00651 |

The probability-quality improvement is concentrated in J2. J1 and J3 have slightly worse Brier under the selected candidates. The smoothed frequency baseline is also much stronger on J2 than on J1/J3. This is evidence against applying one universal shrinkage setting to all leagues.

## Decision and next work

1. Do not change the production default or merge the experiment yet.
2. Keep the current candidate as an audit-only option.
3. Continue collecting future untouched rounds and rerun the same frozen evaluation protocol.
4. Investigate J2-specific calibration separately; do not transfer its gains to J1/J3.
5. Validate the 2025 mirror against an independent/official historical source before relying on it for a production decision.
6. Preserve all original prediction snapshots and append any future revisions; do not rewrite historical predictions.

## Reproducibility

- Workflow run: https://github.com/sigulun555-lab/gpt/actions/runs/41454586099
- Artifact: `toto-1x2-model-experiment` (contains the 2026-only report, 2025+2026 report, and round-1668 prediction snapshot).
- The `npm test` and experiment report-generation steps passed. The later live official GOAL3 HTML check failed due a remote socket termination; this is an external live-fetch issue and does not invalidate the completed model report generation.


## Official 2025 league-size cross-check (2026-11-02)

The 2025 historical CSV mirror contains 1,140 rows: J1 380, J2 380, and J3 380. The official J.League season-format documents independently confirm that all three 2025 divisions had 20 clubs and a 38-round home-and-away format, totaling 380 league matches per division:

- 2025 season overview: https://aboutj.jleague.jp/corporate/about_competitions/2025season/
- J1 format: https://aboutj.jleague.jp/corporate/pressrelease/article/15329
- J2 format: https://aboutj.jleague.jp/corporate/pressrelease/article/15333
- J3 format: https://aboutj.jleague.jp/corporate/pressrelease/article/15332

This validates the expected league-level match totals, not every row's date, fixture identity, or 90-minute score in the community mirror. Row-level reconciliation against the official J.League Data Site remains outstanding. The equal 2025 league sizes provide no basis to prioritize one league by match count.

The 2026 outer evaluation currently contains J1=60, J2=80, J3=80 matches. J1 remains in scope and requires more untouched chronological evaluation; its smaller current outer sample is not grounds to stop evaluating it.
