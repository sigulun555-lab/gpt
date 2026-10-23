# Data quality and walk-forward validation

## Why this check exists
The previous baseline calculated league scoring averages from a combined J1/J2/J3 history. This can contaminate expected-goal rates across competitions. The model now infers the target league from the maintained team registry and filters historical averages and team form to that league. An explicit `filterCompetition:false` mode remains available for an apples-to-apples mixed-league baseline comparison.

## Run
- `npm test`: parser, alias, league-isolation and probability tests.
- `npm run audit`: checks missing/invalid rows, duplicate fixtures, unmapped teams, league counts and compares chronological walk-forward metrics for mixed versus league-filtered models.

## Caveats
The current historical window is short (2026-08-07 through 2026-10-21) and the source is a community mirror. A 350-match count does not mean 350 matches for every target league; report metrics per league and extend the history with verified past seasons before trusting conclusions. Model selection should use chronological holdout, and calibration must be fitted only on earlier data.
