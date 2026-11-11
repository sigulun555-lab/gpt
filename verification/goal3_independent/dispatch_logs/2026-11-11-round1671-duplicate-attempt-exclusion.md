# Round 1671 duplicate-attempt disposition — 2026-11-11

This is a research-only audit record. Both forecast artifacts remain immutable.

## Attempt A — first artifact
- Workflow run: 42960073093 / run #355
- Commit: 31c1dfc5ad698a1412b626a1e5d6304d8a5fe498
- Event: push on `research/goal3-prospective-shadow-20261104-v2`
- Commit message: `[goal3-shadow] round 1671 strict v1.5 prospective run`
- Run created: 2026-11-11T04:34:02Z
- Artifact: 13828832120
- Artifact name: `goal3-shadow-round-1671-run-42960073093-attempt-1`
- ZIP SHA-256: `2d3c32526aaab8cbf5cf5611c5f48a96cf68d22d653107091d2c243df17f54b7`
- Artifact created: 2026-11-11T04:35:50Z
- Forecast as-of: 2026-11-11T04:35:49.039309Z
- Deadline: 2026-11-11T09:50:00Z
- k=6 maxAbsGradient: `9.652471248955408e-7` (within 1e-6)
- k=10 maxAbsGradient: `1.2605898014328432e-7` (within 1e-6)

## Attempt B — later diagnostic rerun
- Workflow run: 42981817161 / run #374
- Commit: 3130cd5420767d3a5a727e27f125653f1960d2ab
- Event: push on the same research branch
- Commit message: `[goal3-shadow] record isolated round 1671 prospective rerun trigger`
- Run created: 2026-11-11T07:45:25Z
- Artifact: 13837723447
- Artifact name: `goal3-shadow-round-1671-run-42981817161-attempt-1`
- ZIP SHA-256: `a3c4a5ebd60b383155ad66ee0905128bb0ac412cd39d16229cd4d52f5eaa0bbc`
- Artifact created: 2026-11-11T07:47:42Z
- Forecast as-of: 2026-11-11T07:47:41.454669Z
- Deadline: 2026-11-11T09:50:00Z
- k=6 maxAbsGradient: `1.0547355469014974e-6` (over 1e-6)
- k=10 maxAbsGradient: `1.260590565266284e-7` (within 1e-6)

## Input comparison
- Both snapshots contain 460 history rows and the same 3 official round-1671 fixtures.
- After removing only `updatedAt` and per-row `fetchedAt`, the 460 history rows are byte-equivalent as canonical JSON.
- The official fixture payloads are identical after removing generatedAt timestamp fields.
- The two history snapshot hashes differ because fetch timestamps differ; no substantive history-row or fixture change was found.

## Primary-cohort disposition
The second artifact was created after the first artifact's diagnostics had been observed, and a same-round canonical-attempt rule had not yet been documented. To avoid any retrospective attempt selection, **exclude round 1671 entirely from the primary prospective cohort**. Retain both artifacts as immutable diagnostics only. Do not choose Attempt A merely because its k=6 gradient passes, and do not choose Attempt B merely because it has a later snapshot.

A first-artifact canonical policy was added to `PROSPECTIVE_SHADOW_PROTOCOL.md` in commit `fc131d122c1828d3581934172205b4f89f5a7f63` for future rounds. That prospective rule does not retroactively qualify either round-1671 attempt.

No production model, main, market weight, existing prediction log, or existing shadow artifact was modified. No PR was merged.
