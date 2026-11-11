# GOAL3 Prospective Shadow: k=6 vs k=10

## Protocol version and purpose

Protocol v1.5 (strict-optimizer prospective amendment before the first v1.5 shadow run/result scoring; 2026-11-11). The statistical metrics, bootstrap settings, non-inferiority margins, and 20-round/60-match adoption floor are unchanged. v1.2 added byte-identical input retention and GitHub artifact deadline evidence; v1.3 added a branch-restricted push trigger; v1.4 added a gradient eligibility gate; v1.5 uses a fixed stricter L-BFGS-B preset for prospective fits after independent retrospective optimizer sensitivity showed that the default success flag can conceal non-stationary fits. The pinned nested_validation implementation remains unchanged for retrospective parity.

Research-only prospective comparison of the shrinkage-MLE penalty coefficient. **Terminology boundary:** within the pinned GPT implementation used by this PR, the single L2 penalty coefficient (source variable `lambda`) is exposed as `k` in this experiment; it is not a second penalty. This is an internal naming statement about the GPT implementation only. It does **not** establish equivalence with Claude's separate v0.5 (`λ=6`) implementation. Claude withdrew the earlier equivalence assertion because its generating code, objective function, regularization term, and coefficient application have not been verified. Until those are inspected and compared term by term, keep Claude v0.5 and GPT k-arms as separate implementations; do not treat their `6` values as the same parameter or pool their results.

This experiment does not alter production forecasts, production model code, market weights, or published prediction logs. The runner stores one immutable workflow artifact per run and can be launched by workflow_dispatch or the exact branch-restricted push marker described below.

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
- The artifact records SHA-256 hashes for history, official target snapshot, aliases, runner, pinned model implementation, independent optimizer-gradient helper, and snapshot pipeline.
- Each of the six fits must record optimizerStatus, optimizerName, optimizerOptions, maxAbsGradient (infinity norm of the independent analytic objective gradient, finite-difference tested), gradientL2, and gradientDimension. A success flag alone is insufficient.
- The v1.5 prospective optimizer is L-BFGS-B with exactly maxiter=10000, maxfun=200000, ftol=0.0, gtol=1e-8, maxls=100, maxcor=50. The pinned default nested_validation.fit_shrinkage remains unchanged and is used for retrospective parity; the prospective strict helper minimizes the identical penalized Poisson objective.
- The v1.5 primary cohort requires optimizerStatus=0, optimizerName/options exactly matching the preset, finite nonnegative gradient diagnostics, and maxAbsGradient <= 1e-6 for every fixture/arm fit. Any missing or over-tolerance fit excludes the entire round from the primary cohort; retain the immutable forecast as a diagnostic artifact.
- The same immutable workflow artifact must also contain byte-identical copies of the raw history, official predictions snapshot, and aliases files. The forecast JSON records each relative path, byte size, and SHA-256. Hashes without the corresponding raw input bytes are insufficient for a prospective adoption cohort.

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
- Preserve the official result source URL and the observed publication/retrieval timestamp where available. For J.League result evidence, accept HTTPS URLs only on `jleague.jp` / its subdomains or `j-league.or.jp` / its subdomains; reject credentials, nonstandard ports, and lookalike hosts. A `sourceType: official` label alone is not sufficient. If the history source is community-maintained rather than independently reconciled to official J.League results, label it as community-source data and state that limitation; do not describe internal consistency as official-source validation.
- Report fit failures and missing forecasts for both arms. No adoption if k=6 has more fit failures or missing targets than k=10; every included scored round must have complete paired forecasts.

## Adoption rule

Consider k=6 for adoption review only after the 20-round / 60-match floor is met. Adopt k=6 only if:

1. The upper bound of the primary 95% round-cluster bootstrap interval is below zero for **both** overall Brier and overall Log Loss.
2. The J1/J2/J3 non-inferiority guards above pass for both metrics.
3. k=6 has no increase in fit failures or missing forecasts relative to k=10, every included round is fully paired, and all six fits satisfy the v1.5 strict-optimizer gradient convergence gate.
4. The data provenance, exclusions, and deadline evidence are complete and auditable.

Otherwise retain k=10 for this research comparison. Retaining k=10 is not a claim that it is statistically superior.

## Scoring implementation and immutable result manifest

The research-only scorer is verification/goal3_independent/score_prospective_shadow.py.
It does not fetch results, generate forecasts, write to data/, or alter any
existing shadow artifact. It consumes a separate manifest and writes a new
output path exclusively; an existing output file is never overwritten.

The manifest is JSON with schemaVersion: 1, protocolVersion: "1.5", and a
rounds array. Each entry supplies the round number, a relative forecastPath
and its exact forecastSha256, a relative resultsPath and its exact
resultsSha256, plus an artifactEvidence object transcribed from GitHub's
workflow-run and artifact APIs. Paths that are absolute or escape the
manifest directory are rejected. Hash mismatches fail closed.

Each forecast artifact contains a sibling <forecast-stem>-inputs directory
with byte-identical history.json, official_predictions.json, and
team_aliases.json copies. The scorer re-hashes each copied file and requires
the hash to match both the forecast's input hash and the inputSnapshots
entry. Missing, changed, path-escaping, or size-mismatched inputs exclude the
entire round. It also requires optimizerDiagnostics and every
fitDiagnostics entry to match the v1.5 strict L-BFGS-B preset, with status 0,
finite gradientL2, and maxAbsGradient <= 1e-6. A fitSuccess flag without these
diagnostics is not eligible. The workflow uploads the whole artifacts/
directory so the forecast and its exact inputs remain in one immutable ZIP
artifact.

artifactEvidence must include workflowRunId, workflowRunAttempt,
workflowRunEvent, workflowRunConclusion, workflowRunHeadSha,
workflowRunHeadBranch, workflowRunHeadCommitMessage, workflowRunCreatedAt,
workflowRunUrl, artifactId, artifactName, artifactDigest, artifactCreatedAt,
artifactApiUrl, and forecastSha256. The reviewer must retrieve and verify
these fields against the GitHub Actions run, commit, and artifact API
responses; the scorer validates their internal consistency but does not make
network calls. The event must be either workflow_dispatch or a push on the
exact research branch research/goal3-prospective-shadow-20261104-v2 whose
head commit message contains [goal3-shadow]. For a push run, the branch and
full commit message are preserved in the forecast and repeated in the
manifest; the reviewer verifies them against the GitHub run/commit API. The
run conclusion must be success, the artifact name must include round, run ID,
and attempt, and the extracted forecast SHA must match. Both
workflowRunCreatedAt and artifactCreatedAt must be strictly before the
official deadline. The artifact must be created after the forecast record;
an artifact uploaded at/after the deadline is excluded even if forecastAsOf
was earlier. The artifact digest must match the GitHub API ZIP digest and the
artifact's extracted contents must contain the forecast and all three raw
input copies. The workflow's static/unit-test job must succeed before the
shadow job is allowed to run.

Each result JSON contains the matching round number,
resultBasis: "full_time_90_plus_stoppage", and exactly three fixture records.
Every fixture record must match the forecast number/date/home/away and provide:
competition (J1/J2/J3), status: "final", integer fullTimeHomeGoals /
fullTimeAwayGoals, sourceType: "official", an HTTPS sourceUrl, and a
timezone-aware retrievedAt; publishedAt is retained and checked when supplied.
The official source designation and URL must be independently checked by the
reviewer. The URL validator requires a non-root path on an allowed official
host; a bare official homepage is not a result locator. This is only a
syntactic host/path gate, not proof of page content. The reviewer must verify
that the page identifies the same round, fixture, full-time score and
90-minute-plus-stoppage basis, and record the publication/retrieval evidence.
Extra time and shoot-outs are not valid inputs.

The scorer verifies the v1.5 forecast timing bounds, workflow-run metadata,
byte-identical input snapshots, GitHub artifact evidence, input/code hashes,
exact k=6/k=10 arms, all six successful fit diagnostics, probability
simplexes, fixture identity, and result provenance. If any one fixture in a round is
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

The output records the manifest, forecast, results, exact input-snapshot
hashes, reviewer-transcribed GitHub artifact evidence, and scorer SHA-256
values, per-round exclusions and official result provenance,
fit/missing-target counts,
overall paired scores, league bounds, and adoption gates. The scorer can only
return a research status or eligibility for human review; automatic production
adoption is hard-disabled. CLI runs are fixed to the pre-registered 10,000
replicates and seed 20261105; smaller settings are available only to unit tests.

Forecast artifacts created before protocol v1.5, including any v1.1-v1.4
artifacts, are not silently upgraded or included in the v1.5 primary cohort.
Keep them unchanged and list them as legacy/excluded unless a separately
reviewed, pre-scoring provenance procedure is documented.

## Operational safeguards

- Actual paired shadow forecast generation runs only on `workflow_dispatch` or the guarded push marker on the exact research branch; pull-request actions, including `ready_for_review`, never launch it.
- Pull-request CI may run syntax/unit tests and read-only retrospective diagnostics, but it does not create or publish prospective shadow forecasts.
- Each run artifact name includes the official round, GitHub run ID, and run attempt. The artifact contains the forecast JSON plus byte-identical input snapshots. Existing output files and snapshot directories are never overwritten.
- A controlled push to the exact research branch runs the shadow job only when the commit message contains [goal3-shadow]; ordinary pushes to other branches or without the marker cannot create a shadow artifact.
- The runner does not write to `data/`, production files, or the default branch.
- Production `scripts/auto_pipeline.js` is not modified by this experiment; timestamp handling specific to the shadow is implemented in the research runner.
- A shadow artifact is evidence of forecast generation, not a claim of improved accuracy.
- The retrospective 236-match / 17-date cohort is exploratory only. Its single retrospective `fetchedAt` snapshot does not prove first-known result availability at each historical forecast date and cannot satisfy the prospective adoption rule.
