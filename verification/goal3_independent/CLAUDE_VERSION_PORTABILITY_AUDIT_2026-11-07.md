# Claude GOAL3 version/provenance portability audit — 2026-11-07

Scope: GOAL3 research-only cross-check for PR #20. This document does not alter Claude's repository, production code, main, market weights, saved predictions, or existing shadow artifacts.

## Pinned source material

- Claude all-k share commit: 17da09beedee57c571038c8204cd51112fb1d959
- Claude version source commit: 7401bc2f640a22f7f0cac36fca541ca127cfb91b
- Claude README documents GPT dependency snapshot: 2adf65a20247dd30c5b86ae95c22bd75980cda89

The original Claude generation manifest does not contain SHA-256 values for data/history.json, data/team_aliases.json, or nested_validation.py. The README identifies a dependency commit, so the independent verifier now fetches that exact commit and records SHA-256 plus Git blob SHA-1 for history, aliases, 2025.csv, and nested_validation.py. It also compares the documented data inputs byte-for-byte with the current PR #20 inputs.

The verifier also imports the documented nested_validation.py from that commit and reruns its loader against the byte-identical history, aliases, and 2025.csv inputs. The old and current loaders return exactly the same 1,140 2025 rows and 429 2026 rows. The complete nested_validation.py files differ because the PR #20 version adds strict fail-closed input validation, but the code from def bin_goals onward (including fit/predict) is byte-identical. This establishes that the loader hardening did not change the historical training population or the model core.

Important limitation: this proves the documented commit's file contents and the GPT-side reproduction inputs; it does not prove which local files were read during the original Claude generation. The report therefore leaves execution_linkage_proven=false and does not call the missing original manifest complete.

## Portable version tests

Claude's original research/goal3_versions/test_versions.py remains unchanged and contains machine-specific /home/claude/ and /tmp/claude-0/ paths. Its test 3 injects rows through GPT's loader rather than exercising goal3_versions.load_rows(); its test 4 does not execute run_version("v2") because the call is guarded by if False.

GPT adds a separate portable harness at tests/test_goal3_claude_versions_portable.py. The workflow first downloads and verifies the pinned Claude module, then runs the harness using temporary fixture files. It covers:

- load_rows() using a temporary history path and the CLI's own file-loading route;
- strict date < target_date cutoff;
- exact v0/v1 equivalence (both lambda=10);
- v2 integration with a synthetic market fixture, and not_run when no market is supplied;
- preservation of separate raw-dispersion and conditional-Pearson fields in the audit report.

The synthetic v2 test is a code-path test only. It does not validate the 0.7 market weight, a real market snapshot, or any production blend.

## Loader and model reproduction

The pinned-source verifier independently calls Claude's load_rows(history_path), canonicalizes the returned team names only for population-key comparison, and checks all 429 eligible 2026 rows against the strict GPT nested-validation population. It then re-fits k=6 by each target date and compares the 472 team-side probabilities to Claude's all-k JSON and compact CSV. The report records raw SHA-256 and Git blob SHA-1 for the independent reproduction inputs.

The original Claude loader itself is not the strict fail-closed loader: its int(...) conversions and isinstance(..., int) filters should not be treated as a substitute for the GPT strict input validation. The fixed 429-row PR #20 input passes the GPT loader's checks; this does not upgrade Claude's original generation provenance.

## Raw VMR versus conditional Pearson residual

These are separate diagnostics and must not be merged:

- Claude diag2.py reports unconditional observed-goal variance divided by observed-goal mean (raw VMR).
- GPT's GOAL3 residual audit computes a model-conditional Pearson dispersion using lambda = -log(P(goal=0)) and (goal-lambda)^2/lambda.

The independent audit report retains separate keys for both quantities and explicitly labels raw VMR as not a conditional Poisson residual. The toto13 raw VMR work remains a separate project and is not used as GOAL3 Poisson-residual evidence.

## Reproduction

The PR workflow runs, in order:

1. python verification/goal3_independent/verify_claude_goal3_share.py --out verification/goal3_independent/results/claude_goal3_share_audit.json
2. python tests/test_goal3_claude_versions_portable.py -v

The second step uses only temporary test inputs and the pinned module produced by step 1. A successful workflow is required before describing these checks as passed.
