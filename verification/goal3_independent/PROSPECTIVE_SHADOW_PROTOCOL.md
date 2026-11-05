# GOAL3 Prospective Shadow: k=6 vs k=10

## Protocol version and purpose

Scoring-manifest protocol v1.2 (2026-11-05). Forecast artifacts retain their
existing embedded evaluation protocolVersion 1.1; v1.2 versions the scorer's
manifest, code-lock, and result-source validation contract.

Research-only prospective comparison of the shrinkage-MLE penalty coefficient. The single L2 penalty currently named `lambda` is called `k` in this experiment; it is not a second penalty.

This experiment does not alter production forecasts, model code, market weights, or published prediction logs. The runner is manually dispatched and stores one immutable workflow artifact per run.

## Frozen comparison

- Arm A: k=10, current research reference.
- Arm B: k=6, leading candidate from the 2026-only nested outer-fold sweep.
- Both arms use the same shrinkage-MLE implementation, same official fixtures, same 2026-only result-history snapshot, same team aliases, and same forecast cutoff.
- No k=8 sensitivity arm in the primary test.
- No market blend, Dixon-Coles correction, draw boost, or other simultaneous model change.
- Each fixture is trained only on matches with a date strictly earlier than its fixture date.
- The existing snapshot pipeline is run first and writes the input files in the workflow workspace. Its `generatedAt` and `lastFetchedAt` values are captured before its HTTP requests; they are retained as declared pipeline metadata and are **not** treated as actual fetch-completion time.
- The research runner starts only after that fetch step completes and the files are available. It records `snapshotAvailableBy` after reading the inputs; every eligible history row must have a `fetchedAt` timestamp no later than this conservative post-fetch bound. Missing timestamps fail closed. This bound is not claimed to be the exact HTTP completion timestamp.
- `forecastAsOf` is recorded after both k arms finish fitting. Both `snapshotAvailableBy` and `forecastAsOf` must be strictly before the official deadline; a snapshot or completed forecast at/after the deadline is rejected.
- The official GOAL3 round, three fixtures, six team targets, and deadline must be confirmed.
- The artifact records SHA-256 hashes for history, official target snapshot, aliases, runner, and model implementation.

## Outcome definitions (pre-registered)

The target for each team-side is its full-time goal count categorized as 0, 1, 2, or 3+.

For target i and category c in {0,1,2,3}, let p(i,c) be the forecast probability and y(i,c) the one-hot observed outcome.

- **Multiclass Brier (lower is better):** mean over team-side targets of `sum_c (p(i,c) - y(i,c))^2`. This is the four-category sum, not the category-average; its range is 0 to 2.
- **Log Loss (lower is better):** mean of `-ln(max(p(i, observed_category), 1e-12))`, using the natural logarithm and a fixed probability floor of 1e-12.
- **Paired difference:** for each target, calculate score(k=6) - score(k=10), then average over the same paired targets. A negative difference favors k=6; a positive difference favors k=10.
- The same target set and cluster draws must be used for both arms and both primary metrics. Do not treat the two team-sides of one match as independent bootstrap clusters.

## Evidence and bootstrap plan

- **Interim descriptive floor:** at least 10 completed rounds and 30 matches. Below this, report only the available counts and do not present an inferential adoption result.
- **Adoption-review floor:** at least 20 completed rounds and 60 matches, with all three fixtures and all six paired team-side forecasts/results present for every included round. Results from 10-19 rounds remain descriptive and cannot trigger adoption, even if a nominal interval happens to exclude zero.
- **Primary resampling unit:** the official GOAL3 round. Resample complete rounds with replacement; retain all three matches and all six team-side targets in each sampled round. Calculate paired k=6 minus k=10 differences inside each replicate.
- **Primary bootstrap:** 10,000 replicates, NumPy PCG64, seed 20261105, two-sided 95% percentile interval, NumPy linear quantile definition. Use the same sampled round indices for Brier and Log Loss.
- **Sensitivity analysis:** if fixtures in the included sample span multiple calendar dates, also report a date-cluster bootstrap that retains all matches and both team-sides on each sampled date. Keep this sensitivity result separate from the primary round-cluster result.
- Repeated teams across different rounds are not separately clustered. This limitation must be stated; the bootstrap is not proof of independence across rounds.

## League non-inferiority guard

Report J1/J2/J3 separately. Define a material deterioration margin in advance as:

- Brier: +0.01 absolute points (k=6 minus k=10).
- Log Loss: +0.02 natural-log units (k=6 minus k=10).

For adoption review, the one-sided 90% round-cluster bootstrap upper bound for each league and each primary metric must be below its corresponding margin. For this scorer, the operational minimum is 20 distinct complete rounds containing at least one fixture from that league; if a league has fewer, its non-inferiority guard fails and adoption is not eligible. These are conservative guardrails, not claims that the margins are universal measures of practical importance.

## Result and exclusion handling

- Score the official full-time result after 90 minutes plus stoppage time; exclude extra time and penalty shoot-outs.
- If any of the three fixtures in a round is postponed, abandoned, cancelled, or lacks a final official result at the scoring lock, exclude the **entire round** from the primary paired analysis. List every excluded round and reason; never silently drop only the missing team-side or match.
- Preserve the official result source URL and the observed publication/retrieval timestamp where available. If the history source is community-maintained rather than independently reconciled to official J.League results, label it as community-source data and state that limitation; do not describe internal consistency as official-source validation.
- Report fit failures and missing forecasts for both arms. No adoption if k=6 has more fit failures or missing targets than k=10; every included scored round must have complete paired forecasts.

## Adoption rule

Consider k=6 for adoption review only after the 20-round / 60-match floor is met. Adopt k=6 only if:

1. The upper bound of the primary 95% round-cluster bootstrap interval is below zero for **both** overall Brier and overall Log Loss.
2. The J1/J2/J3 non-inferiority guards above pass for both metrics.
3. k=6 has no increase in fit failures or missing forecasts relative to k=10, and every included round is fully paired.
4. The data provenance, exclusions, and deadline evidence are complete and auditable.

Otherwise retain k=10 for this research comparison. Retaining k=10 is not a claim that it is statistically superior.

## Scoring implementation and immutable result manifest

The research-only scorer is verification/goal3_independent/score_prospective_shadow.py.
It does not fetch results, generate forecasts, write to data/, or alter any
existing shadow artifact. It consumes a separate manifest and writes a new
output path exclusively; an existing output file is never overwritten.

A pre-result independent reviewer lock is committed at
`verification/goal3_independent/REVIEWER_APPROVED_CODE_HASHES_2026-11-06.json`.
It records the four raw-byte SHA-256 values, the exact source commits, and the
Git blob SHA-1 cross-checks. The lock was independently computed from the
contents at codeHashSourceCommit
`47bac0e2c716479355c12aed054fe2c2f713ee33`; the three generation files are
byte-identical at generationSourceCommit
`d5f48314159ffff2f8980738fa02d33ddf2a188b`. The scorer resolves the lock
file's exact last-touch commit SHA and committer timestamp from Git history
(`git log -1 --format=%H%x00%cI -- <lock-path>`) and fails closed unless the
manifest pins that exact `codeLockCommitSha`, `codeLockCommitAt`, and
`codeLockFileSha256`. It also verifies that `reviewedAt` is later than the
code-hash source commit and no later than the lock commit, that both source
commits are ancestors of the lock commit, and that the lock commit is in the
checked-out history. Locked source bytes/blob IDs must match the recorded
source commits. Every official result
fixture must have a timezone-aware `retrievedAt` strictly later than the lock
commit time; equal, earlier, or missing timestamps exclude the entire round.
The GOAL3 research CI checkout fetches full Git history so this check is
reproducible. If any of the four source files changes, this lock is stale and a
new independent review/lock is required.
This lock is a code-identity record only; it does not replace manual fixture
and official-result verification, and it does not authorize merge, dispatch,
or production adoption.

The manifest is JSON with schemaVersion: 1, protocolVersion: "1.2",
an approvedCodeHashes object, codeLockCommitSha, codeLockCommitAt,
codeLockFileSha256, and a rounds array. Before scoring, the reviewer must
independently pin the exact lock commit SHA, its Git committer timestamp, and
the lock file SHA-256, as well as the expected SHA-256 values for runnerSha256,
modelSha256, snapshotPipelineSha256, and scorerSha256 from the exact approved
research commit. scorerSha256 is the SHA-256 of score_prospective_shadow.py
itself; it prevents a modified scoring implementation from running under an
unchanged runner/model/pipeline lock. The scorer first compares the complete
manifest lock with the actual checked-out runner/model/pipeline/scorer files,
then rejects any forecast whose self-reported runner/model/pipeline hashes do
not match that same lock. The forecast artifact does not self-report the scorer
hash; it belongs only in the reviewer-controlled manifest lock. Merely copying
hashes from the forecast into the manifest is not an independent approval.
The output records both the approved lock and the checked-out hashes for audit. Each entry supplies the
round number, a relative forecastPath
and its exact forecastSha256, plus a relative resultsPath and its exact
resultsSha256. Paths that are absolute or escape the manifest directory are
rejected. Hash mismatches fail closed.

Each result JSON contains the matching round number,
resultBasis: "full_time_90_plus_stoppage", and exactly three fixture records.
Every fixture record must match the forecast number/date/home/away and provide:
competition (J1/J2/J3), status: "final", integer fullTimeHomeGoals /
fullTimeAwayGoals, sourceType: "official", an HTTPS sourceUrl whose hostname
is exactly www.jleague.jp (no userinfo, nonstandard port, or fragment), and a
timezone-aware retrievedAt strictly later than codeLockCommitAt. If publishedAt
is supplied, it must also be timezone-aware, strictly later than codeLockCommitAt,
and no later than retrievedAt. A later retrieval cannot make a result published
before the reviewer lock eligible. The exact host allowlist is intentionally narrow; a new official host requires
a reviewed code change and tests, not a manifest override. The official result
page and publication/retrieval evidence must still be manually checked for
each fixture. This includes confirming the competition assignment (J1/J2/J3),
fixture identity, final score, and full-time basis against that exact official
page; do not infer league assignment from team names. A domain allowlist alone
does not prove the page content is official. Extra time and shoot-outs are not
valid inputs.

The v1.2 manifest scorer verifies the reviewer lock commit/file/time against
Git history, the lock's source commits and raw bytes, every result retrievedAt,
and each supplied publishedAt against the lock commit time. If publishedAt is
unavailable, retrieval time alone is not proof of first public availability;
manual evidence must establish chronology or the round remains ineligible. It
also verifies the v1.1 forecast
timing bounds, input/code hashes, exact k=6/k=10 arms, all six successful fit
diagnostics, probability simplexes, fixture identity, and result provenance. If any one fixture in a round is
not final or any paired target/provenance field is invalid, the entire round
is excluded and the reason is recorded. It never scores only the available
matches or one team-side.

The primary analysis resamples complete official rounds with replacement,
retaining all three matches and six paired team-side targets. It uses 10,000
NumPy PCG64 replicates, seed 20261105, the same round draws for Brier and
LogLoss, and linear 2.5/97.5 percentiles. League non-inferiority resamples
complete rounds containing that league, retaining all that league's targets
from each sampled round, and reports one-sided 90% upper bounds. If included
fixtures span multiple calendar dates, a separate date-cluster sensitivity
bootstrap retains all matches and both team-sides on each sampled date.

The output records the manifest, forecast, results, and scorer SHA-256 values,
per-round exclusions and official result provenance, fit/missing-target counts,
overall paired scores, league bounds, and adoption gates. The scorer can only
return a research status or eligibility for human review; automatic production
adoption is hard-disabled. The v1.2 manifest code-hash lock is a separate
reviewer-controlled approval step; its exact Git commit and file hash must be
pinned in the manifest, and all official result retrievals must occur strictly
after that commit. The lock cannot be inferred from the forecast artifact. CLI runs are fixed to the pre-registered 10,000
replicates and seed 20261105; smaller settings are available only to unit tests.

Forecast artifacts created before protocol v1.1 that lack its explicit
snapshotAvailableBy, historyMaxDeclaredFetchedAt, and protocol-version
fields are not silently upgraded or included in the v1.1 primary cohort. Keep
them unchanged and list them as legacy/excluded unless a separately reviewed,
pre-scoring provenance procedure is documented.

## Operational safeguards

- Actual paired shadow forecast generation runs only on `workflow_dispatch`; pull-request actions, including `ready_for_review`, never launch it.
- Pull-request CI may run syntax/unit tests and read-only retrospective diagnostics, but it does not create or publish prospective shadow forecasts.
- Each run artifact name includes the official round and GitHub run ID. Existing output files are never overwritten.
- The runner does not write to `data/`, production files, or the default branch.
- Production `scripts/auto_pipeline.js` is not modified by this experiment; timestamp handling specific to the shadow is implemented in the research runner.
- A shadow artifact is evidence of forecast generation, not a claim of improved accuracy.
- The retrospective 236-match / 17-date cohort is exploratory only. Its single retrospective `fetchedAt` snapshot does not prove first-known result availability at each historical forecast date and cannot satisfy the prospective adoption rule.
