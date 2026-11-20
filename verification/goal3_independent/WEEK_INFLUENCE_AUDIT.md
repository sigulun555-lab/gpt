# GOAL3 leave-one-ISO-week-out influence audit

This audit supplements the existing date/team/match leave-one-cluster-out
analysis. It uses the frozen paired CSV and keeps all matches and both
team-side targets from an ISO week together when excluding that week.

It is a fixed-prediction influence diagnostic only:
- no model refit or k reselection after exclusion;
- no bootstrap or inferential confidence interval;
- no prospective forecast generation;
- no production model or prediction-log changes.

The fixed cohort is expected to contain 236 matches, 472 team-side targets,
17 target dates, and 8 ISO-week clusters. The output records the exact input
CSV SHA-256 and script SHA-256. It reports per-week target-weighted k=6 minus
k=10 score differences, the contribution of each week to the overall Brier
and LogLoss difference sums, and leave-one-week-out sign reversals.

A result in which one week accounts for more than the full net improvement
means other weeks collectively offset part of that week's gain. Such a result
is a warning about concentration, not a reason to drop the week.

Reproduce:
python verification/goal3_independent/week_influence_audit.py --csv verification/goal3_independent/goal3_outer_matches.csv --out artifacts/goal3_week_influence_audit.json

The eight-week cluster count is too small to interpret the jackknife as a
sampling distribution. The result cannot satisfy the prospective protocol
or justify production adoption.
