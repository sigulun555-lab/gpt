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
- The official page was revalidated at 2026-11-21 14:01 JST and compared with the original 2026-11-18 snapshot: all 13 home/away fixture pairs are identical (0 changed, 13 unchanged). This later observation refreshes page/retrieval timestamps and the sales summary only; it is not a fixture correction. The fixture manifest records both the current PR #129 schedule blob and the original/preserved 11/18 snapshot SHAs. The old snapshot remains preserved as data/schedule/round_1674_20261118_superseded.json. Before generation, the page must still be re-opened within five minutes of the actual run. The isolated base runner is pinned to GPT PR #18 commit 8e30fc35a746ff329fae36c6684fba87abd68ae4.
- History is fetched from GPT main at execution time. The source is a community mirror; historical first-known availability is not proven by `fetchedAt`.
