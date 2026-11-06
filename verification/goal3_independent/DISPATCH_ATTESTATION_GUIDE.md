# GOAL3 dispatch fixture-attestation guide

The manual dispatch requires the exact snapshot that was reviewed on the official totoGOAL3 page. A boolean confirmation is not sufficient.

## Before dispatch

1. Open the pinned official URL:
   `https://sp.toto-dream.com/dcs/subos/screen/ss01/sssl014/PGSSSL01401InittotoSP.form?commodityId=02&gameAssortment=1`
2. Confirm that the selected market is totoGOAL3, then manually verify the current round, all three fixture numbers, dates, home/away teams, and sales deadline.
3. Immediately before dispatch, paste a JSON object in the `attested_fixture_snapshot` input. The `reviewedAt` value must be an ISO-8601 timestamp with timezone and must be no more than 10 minutes old when the runner validates it.

## Required JSON shape

```json
{
  "officialUrl": "https://sp.toto-dream.com/dcs/subos/ss01/sssl014/PGSSSL01401InittotoSP.form?commodityId=02&gameAssortment=1",
  "marketType": "totoGOAL3",
  "round": 1670,
  "deadline": "2026-11-07T12:50:00+09:00",
  "fixtures": [
    {"number": 1, "date": "2026-11-07", "home": "清水", "away": "東京Ｖ"},
    {"number": 2, "date": "2026-11-07", "home": "福岡", "away": "長崎"},
    {"number": 3, "date": "2026-11-07", "home": "柏", "away": "川崎Ｆ"}
  ],
  "reviewedAt": "2026-11-06T08:55:00Z"
}
```

The values above are an illustrative historical snapshot only. **Do not reuse them for a later round.** Copy the exact current round, deadline, fixture dates and team labels from the official page, and set `reviewedAt` to the actual review time.

## Fail-closed checks

After the official snapshot is fetched, but before the paired forecast is generated, the validator requires an exact match on the official URL, market type, round, deadline, fixture count, fixture order/numbers, dates, and home/away labels. It also checks that the source metadata and each prediction row point to the pinned official URL, the pipeline timestamps are recent, and the review timestamp is fresh.

Any mismatch, missing field, stale review, non-official URL, malformed date, unconfirmed target, or expired deadline stops the workflow before forecast generation. The attestation JSON, observed snapshot, canonical SHA-256 fingerprint, and validation time are retained in the immutable run artifact and referenced by the dispatch provenance sidecar.

Pull-request CI runs the validator self-test only. It does not invoke workflow_dispatch or create a prospective forecast.
