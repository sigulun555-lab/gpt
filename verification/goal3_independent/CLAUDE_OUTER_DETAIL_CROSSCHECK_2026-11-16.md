# Claude outer-detail cross-check — 2026-11-16

## Sources and immutable references

- GPT PR #20 head at audit: `a24c01e09806a91b82d54e037b4441d4fa4757a3`
- GPT projection: `verification/goal3_independent/goal3_outer_matches.json`
  - blob: `5f8bc3b119a2c59d158b1bdddef7b590ee0bdc2e`
  - schema: `targetRows`, 6-decimal probability projection
- Claude detail branch: `claude/goal3-outer-detail-2026-11-15`
  - branch head: `95d9426d4230de8718ec4751ba6682ec3e3fe722`
- Claude raw detail: `analysis/goal3_outer_detail/goal3_outer_matches.json`
  - blob: `6f8ac699cd907d72a97575f4ff15efa842fa043b`
  - schema: `window/k/targets`, 10-decimal probabilities plus per-row metrics
- Claude script: `analysis/goal3_outer_detail/goal3_detail.py`
  - blob: `49314449fc708d4b21c78ed7d4e1ab162551cfe1`

## Independent row alignment

Rows were aligned by `date | home | away | side`, not by array position.

| Check | Result |
|---|---:|
| GPT targets | 472 |
| Claude targets | 472 |
| matched keys | 472 / 472 |
| missing / extra | 0 / 0 |
| actual-category mismatches | 0 |
| league mismatches | 0 |
| fold mismatches | 0 |
| probability cells compared | 3,776 |

The two artifacts use different schemas and precisions. After rounding Claude's probabilities to the GPT projection's documented 6 decimal places, all 472 k=6 rows and all 472 k=10 rows match. Before rounding, the maximum absolute probability difference is `4.997e-7` (mean `2.5093e-7`), consistent with the 6dp versus 10dp representation.

Recomputing per-row scores from the 6dp GPT projection versus Claude's 10dp values gives maximum absolute differences of `1.2141e-6` for Brier and `4.7361e-6` for LogLoss. These are precision effects, not row/key or actual-label mismatches.

## Portability and inclusion decision

- GPT PR #20 already contains a frozen 6dp projection and its provenance chain. Copying Claude's raw JSON into the same canonical path would replace a different schema and create ambiguity, so it is not copied over.
- Claude's raw file remains available on the separate Claude branch and is now cross-checked against the PR #20 projection.
- The Claude script currently hard-codes `/home/claude/gpt/verification/goal3_independent` and a `/tmp/claude-.../scratchpad` output path. It is not clean-checkout portable as-is and is not added to the portable CI.
- No model, market weight, production prediction, main branch, or existing shadow artifact was changed.

## Interpretation

The 472-row outer-detail data are aligned at 6dp. The artifacts are not byte-identical because their schemas and probability precision differ. The retrospective 17-date comparison remains separate from prospective shadow evaluation. k=10 remains the research baseline; k=6 remains a candidate with no established statistical superiority.
