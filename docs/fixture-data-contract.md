# Fixture data contract

`data/fixtures.json` is the shared, prediction-free fixture feed for GPT and Claude. It contains only confirmed fixtures from the current official toto page; if round/deadline/fixture validation fails, `fixtures` is an empty array so consumers do not accidentally reuse stale targets.

## Schema

```json
{
  "schemaVersion": 1,
  "fixtures": [
    {
      "marketType": "totoGOAL3",
      "matchNumber": 1,
      "home": "水戸",
      "away": "Ｃ大阪",
      "round": 1666,
      "deadline": "2026-10-28T18:50:00+09:00"
    }
  ]
}
```

Each fixture has exactly these fields: `marketType`, `matchNumber`, `home`, `away`, `round`, and `deadline`. No probabilities, picks, model metadata, or prediction timestamps belong in this file.

- Standard toto: one record per match, normally match numbers 1–13.
- totoGOAL3: one record per actual match (3 records total). The six team rows on the official page are paired in source order into home/away for matches 1–3; they are not treated as six matches.
- The pipeline only publishes records after validating market type, expected match count, sequential match numbers, team names, round, and a future sale deadline.
- `data/predictions.json` remains the prediction output. The model's prediction logic is unchanged by this fixture-feed addition.
