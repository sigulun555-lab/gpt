# toto13 Window-20 shadow arm — round 1674

Status: preregistered implementation draft; **no forecast has been generated**.

- Product: ordinary toto13 only (13 matches, 1/X/2).
- First eligible round: 1674; official network deadline 2026-11-25 18:50 JST.
- Earliest generation: 2026-11-24 22:00 JST.
- Primary candidate: `W20_k10_window20_cap`; paired control: `V1_k10_window12`.
- Only the history cap changes. Both arms use the existing isolated T1 predictor and k=10.
- 12–19 prior matches: use all available, `partial_window=true`; 20+: use latest 20; fewer than 12: no probabilities, fixture excluded from paired scoring.
- Pooled home/away baseline stays fixed. No league baseline, market blend, injuries, lineups, Dixon–Coles or draw boost.
- This arm is separate from V0 k=6/window12, the existing GOAL3 k=6/k=10 study, mini toto A/B, and the ordinary-toto13 V1/V2 submission logs.
- The workflow validates on push/PR. It only generates a forecast when a dedicated push commit includes `[window20-shadow]` after the earliest time and before the official deadline. The output is a new immutable Actions artifact under `toto13-window20-round-1674`; no prediction file is committed and no existing shadow is overwritten.
- The official page was revalidated at 2026-11-21 14:01 JST. The 2026-11-18 PR #129 snapshot was stale: 12 of 13 fixture pairs changed (match 7 alone stayed the same). The W20 fixture file now uses the corrected 2026-11-21 official card from PR #129 commit f5cc6e2570dbd053cdfc8ef274beef8a33cb48e9; the old snapshot remains preserved as data/schedule/round_1674_20261118_superseded.json.
- History is fetched from GPT main at execution time. The source is a community mirror; historical first-known availability is not proven by `fetchedAt`.
