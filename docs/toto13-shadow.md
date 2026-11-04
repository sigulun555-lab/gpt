# toto13 T1 shadow runner

This folder is an isolated research path. It does not import or modify production `model.js`, `data/predictions.json`, market weights, or existing prediction logs.

## Variants

- **V0 control:** T1 Poisson with shrinkage (k=6), last 12 team matches.
- **V1 candidate:** identical model with only shrinkage changed to (k=10), last 12 team matches.
- **Baseline reference:** (k=0), last 12 team matches.

All three use the same history snapshot, fixture list, strict chronological cutoff, and 0..10 score grid. No Dixon–Coles correction, draw boost, or market blend is applied. This isolates the single shrinkage-constant change.

## Input fixture format

Pass a JSON document containing exactly 13 fixtures. Do not use a 3-match totoGOAL3 fixture file.

```json
{
  "round": 1234,
  "marketType": "toto",
  "deadline": "2026-11-07T11:00:00+09:00",
  "fixtures": [
    {
      "number": 1,
      "date": "2026-11-07",
      "competition": "J1",
      "home": "Home team",
      "away": "Away team"
    }
  ]
}
```

The example above shows the shape only; all 13 real fixtures must be supplied before execution.

## Run

```sh
node scripts/run_toto13_shadow.js path/to/round_fixtures.json shadow/toto13/predictions/round_XXXX_<unique>.json
node --test tests/toto13-shadow.test.js
```

The runner reads `data/history.json`, validates all history rows, sorts chronologically, and uses only matches strictly earlier than each target fixture date. Same-day matches are excluded. The output records SHA-256 hashes for the history and fixture inputs, all three probability vectors, xG values, prior-data counts, and creation time.

The output path must not already exist. Existing shadow logs are never overwritten. A run is not scored until official results and the pre-deadline timestamp are independently verified.

## Current readiness

- The GPT main history snapshot contains 429 completed rows as of the 2026-11-03 23:21:25Z refresh, but it is sourced from a community mirror and is not fully official-result-audited.
- The GPT repository's current live `data/fixtures.json` is a 3-match totoGOAL3 round 1670 file. It is intentionally rejected by this toto13 runner.
- Therefore the engine and tests can be prepared now, but no round prediction is generated until the correct 13-match toto fixtures are available.
- No scheduled or production workflow is changed by this branch.
