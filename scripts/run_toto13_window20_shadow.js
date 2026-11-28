#!/usr/bin/env node
'use strict';

// Separate toto13 Window-20 shadow arm. Reuses the exact isolated T1 predictor
// from scripts/run_toto13_shadow.js; only the team-history cap changes.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {
  normalizeTeam,
  validateHistory,
  validateFixtures,
  predictToto13,
} = require('./run_toto13_shadow.js');

const ROUND = 1674;
const PRODUCT = 'toto13';
const EARLIEST_GENERATION_AT = '2026-11-24T22:00:00+09:00';
const OFFICIAL_DEADLINE = '2026-11-25T18:50:00+09:00';
const MIN_PRIOR_MATCHES = 12;
const MAX_FIXTURE_AGE_MS = 5 * 60 * 1000;
const WINDOW20 = Object.freeze({ id: 'W20_k10_window20_cap', shrink: 10, window: 20 });
const CONTROL12 = Object.freeze({ id: 'V1_k10_window12', shrink: 10, window: 12 });

function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function validateWindow20OutputPath(outputPath, repoRoot = path.resolve(__dirname, '..')) {
  if (typeof outputPath !== 'string' || !outputPath.trim()) throw new Error('output path is required');
  const allowedDir = path.resolve(repoRoot, 'shadow', 'toto13', 'window20', 'predictions');
  const resolved = path.resolve(repoRoot, outputPath);
  const relative = path.relative(allowedDir, resolved);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative) || path.extname(resolved) !== '.json') {
    throw new Error('output path must be a new .json file inside shadow/toto13/window20/predictions/');
  }
  return resolved;
}

function priorTeamMatchCount(history, team, fixtureDate) {
  const canonical = normalizeTeam(team);
  return history.filter((m) =>
    m.date < fixtureDate &&
    (normalizeTeam(m.home) === canonical || normalizeTeam(m.away) === canonical)
  ).length;
}

function teamWindowStatus(count) {
  if (count < MIN_PRIOR_MATCHES) {
    return { eligible: false, status: 'missing_below_12_match_floor', priorCompletedMatches: count,
      minimumRequired: MIN_PRIOR_MATCHES, partial_window: null, usedMatches: 0 };
  }
  return {
    eligible: true,
    status: 'eligible',
    priorCompletedMatches: count,
    minimumRequired: MIN_PRIOR_MATCHES,
    partial_window: count < WINDOW20.window,
    usedMatches: Math.min(WINDOW20.window, count),
    controlWindowUsedMatches: Math.min(CONTROL12.window, count),
  };
}

function buildWindow20Report(historyInput, fixtureDocument, options = {}) {
  const history = validateHistory(historyInput);
  const fixtures = validateFixtures(fixtureDocument);
  const nowIso = options.nowIso || new Date().toISOString();
  const nowMs = Date.parse(nowIso);
  const earliestMs = Date.parse(EARLIEST_GENERATION_AT);
  const deadlineMs = Date.parse(OFFICIAL_DEADLINE);

  if (!Number.isFinite(nowMs)) throw new Error('nowIso must be a valid timestamp');
  if (fixtureDocument.round !== ROUND) throw new Error('Window-20 arm is preregistered for round 1674 only; no backfill');
  if (fixtureDocument.marketType !== 'toto') throw new Error('marketType must be ordinary toto13; mini toto and GOAL3 are excluded');
  if (fixtureDocument.deadline !== OFFICIAL_DEADLINE) throw new Error('official round-1674 network deadline does not match the preregistered deadline');
  if (nowMs < earliestMs) throw new Error('Window-20 forecast generation is not allowed before 2026-11-24 22:00 JST');
  if (nowMs >= deadlineMs) throw new Error('official round-1674 deadline has passed; no backfill is allowed');
  if (fixtureDocument.fixtures.length !== 13) throw new Error('exactly 13 ordinary-toto fixtures are required');
  if (!fixtureDocument.source || !String(fixtureDocument.source.url || '').includes('holdCntId=1674')) {
    throw new Error('official round-1674 source URL is required');
  }
  const pageDisplayAt = fixtureDocument.source.pageDisplayAt;
  const revalidatedAt = fixtureDocument.source.revalidatedAt;
  const retrievedAt = fixtureDocument.source.retrievedAt;
  const pageDisplayMs = Date.parse(pageDisplayAt);
  const revalidatedMs = Date.parse(revalidatedAt);
  const retrievedMs = Date.parse(retrievedAt);
  if (!Number.isFinite(pageDisplayMs) || !Number.isFinite(revalidatedMs) || !Number.isFinite(retrievedMs)) {
    throw new Error('official fixture source must include valid pageDisplayAt, revalidatedAt, and retrievedAt timestamps');
  }
  if (pageDisplayMs > revalidatedMs || revalidatedMs > retrievedMs || retrievedMs > nowMs) {
    throw new Error('official fixture timestamps must be ordered pageDisplayAt <= revalidatedAt <= retrievedAt <= generation');
  }
  if (nowMs - pageDisplayMs > MAX_FIXTURE_AGE_MS || nowMs - retrievedMs > MAX_FIXTURE_AGE_MS || retrievedMs - pageDisplayMs > MAX_FIXTURE_AGE_MS) {
    throw new Error('official fixture snapshot must be revalidated and retrieved within 5 minutes of generation');
  }

  const historyDocument = options.historyDocument || {};
  if (historyDocument.updatedAt && Date.parse(historyDocument.updatedAt) > nowMs) {
    throw new Error('history snapshot updatedAt is later than generation time');
  }
  const fetchedAtValues = [...new Set(history.map((m) => m.fetchedAt).filter(Boolean))].sort();
  const fetchedAtMissingCount = history.filter((m) => !m.fetchedAt).length;
  if (fetchedAtValues.some((v) => Date.parse(v) > nowMs)) {
    throw new Error('history contains fetchedAt timestamps later than generation time');
  }

  const predictions = fixtures.map((fixture) => {
    const home = normalizeTeam(fixture.home);
    const away = normalizeTeam(fixture.away);
    const homeCount = priorTeamMatchCount(history, home, fixture.date);
    const awayCount = priorTeamMatchCount(history, away, fixture.date);
    const homeStatus = teamWindowStatus(homeCount);
    const awayStatus = teamWindowStatus(awayCount);
    const pairedEligible = homeStatus.eligible && awayStatus.eligible;

    const base = {
      number: fixture.number,
      date: fixture.date,
      competition: fixture.competition || null,
      home,
      away,
      pairedEligible,
      teamStatus: { home: homeStatus, away: awayStatus },
      arms: {},
    };

    if (!pairedEligible) {
      base.arms[WINDOW20.id] = { status: 'not_emitted_below_12_match_floor', probabilities: null };
      base.arms[CONTROL12.id] = { status: 'not_emitted_below_12_match_floor', probabilities: null };
      return base;
    }

    const w20 = predictToto13(history, fixture, WINDOW20);
    const control = predictToto13(history, fixture, CONTROL12);
    base.arms[WINDOW20.id] = {
      status: 'predicted',
      probabilities: w20.probabilities,
      homeExpectedGoals: w20.homeExpectedGoals,
      awayExpectedGoals: w20.awayExpectedGoals,
      historyHomeMatches: w20.historyHomeMatches,
      historyAwayMatches: w20.historyAwayMatches,
      homePartialWindow: homeCount < 20,
      awayPartialWindow: awayCount < 20,
      homeWindowUsed: Math.min(20, homeCount),
      awayWindowUsed: Math.min(20, awayCount),
      priorMaxDate: w20.priorMaxDate,
    };
    base.arms[CONTROL12.id] = {
      status: 'predicted',
      probabilities: control.probabilities,
      homeExpectedGoals: control.homeExpectedGoals,
      awayExpectedGoals: control.awayExpectedGoals,
      historyHomeMatches: control.historyHomeMatches,
      historyAwayMatches: control.historyAwayMatches,
      homePartialWindow: false,
      awayPartialWindow: false,
      homeWindowUsed: 12,
      awayWindowUsed: 12,
      priorMaxDate: control.priorMaxDate,
    };
    return base;
  });

  const eligible = predictions.filter((p) => p.pairedEligible);
  const partialTeamSides = predictions.reduce((n, p) =>
    n + Number(p.teamStatus.home.eligible && p.teamStatus.home.partial_window) +
      Number(p.teamStatus.away.eligible && p.teamStatus.away.partial_window), 0);
  const missingTeamSides = predictions.reduce((n, p) =>
    n + Number(!p.teamStatus.home.eligible) + Number(!p.teamStatus.away.eligible), 0);

  return {
    schemaVersion: 1,
    artifactType: 'toto13_window20_shadow_only',
    product: PRODUCT,
    round: ROUND,
    generatedAt: nowIso,
    deadline: OFFICIAL_DEADLINE,
    earliestGenerationAt: EARLIEST_GENERATION_AT,
    primaryContrast: { candidate: WINDOW20.id, control: CONTROL12.id, onlyChangedParameter: 'teamHistoryWindowCap: 12 -> up to 20' },
    protocol: {
      shrinkageK: 10,
      minimumPriorMatchesPerTeamSide: MIN_PRIOR_MATCHES,
      window20Rule: '12-19: use all available and partial_window=true; 20+: use latest 20 and partial_window=false; below 12: mark missing and emit no probability',
      controlRule: 'k=10, latest 12 prior matches',
      historyCutoff: 'strictly earlier than each fixture date; same-day and future matches excluded',
      pooledHomeAwayBaseline: 'same existing isolated T1 pooled baseline for both arms',
      leagueSpecificBaseline: 'not used; separate experiment',
      marketBlend: 'none',
      injuryOrLineupAdjustment: 'none',
      dixonColes: 'none',
      drawBoost: 'none',
      scoreGrid: '0..10 inclusive',
      adoption: 'no adoption before 20 eligible prospective rounds with both paired LogLoss and Brier 95% CI upper bounds below zero',
    },
    source: {
      officialFixtureUrl: fixtureDocument.source.url,
      officialFixtureObservedAt: fixtureDocument.source.observedAt || null,
      officialFixtureRevalidatedAt: revalidatedAt,
      officialFixturePreviousSnapshot: fixtureDocument.source.previousSnapshot || null,
      historySource: historyDocument.source || null,
      historyUpdatedAt: historyDocument.updatedAt || null,
      historyFetchedAtValues: fetchedAtValues,
      historyFetchedAtMissingCount: fetchedAtMissingCount,
      teamAliasesUsed: false,
      teamNormalization: 'NFKC only, exactly as the isolated base runner; no alias-map substitutions',
    },
    input: {
      historyRows: history.length,
      historyDistinctDates: new Set(history.map((m) => m.date)).size,
      historySha256: options.historySha256 || null,
      fixtureSha256: options.fixtureSha256 || null,
      baseRunnerSha256: options.baseRunnerSha256 || null,
      runnerSha256: options.runnerSha256 || null,
      teamAliasesSha256AuditOnly: options.teamAliasesSha256 || null,
    },
    coverage: {
      totalFixtures: 13,
      pairedEligibleFixtures: eligible.length,
      ineligibleFixtures: predictions.length - eligible.length,
      totalTeamSides: 26,
      missingTeamSides,
      partialWindowTeamSides: partialTeamSides,
      fullWindowTeamSides: 26 - missingTeamSides - partialTeamSides,
    },
    predictions,
    execution: {
      repository: options.execution?.repository || null,
      workflowRunId: options.execution?.workflowRunId || null,
      workflowRunAttempt: options.execution?.workflowRunAttempt || null,
      commitSha: options.execution?.commitSha || null,
      headCommitSha: options.execution?.headCommitSha || null,
      ref: options.execution?.ref || null,
      eventName: options.execution?.eventName || null,
      triggerMarker: '[window20-shadow]',
      timestampEvidence: 'verify GitHub run and artifact upload completion metadata independently; generatedAt alone is insufficient',
    },
    warnings: [
      'Research-only ordinary-toto13 shadow. Not a production prediction.',
      'Do not mix this arm with mini toto A/B or totoGOAL3.',
      'History source may be a community mirror; historical first-known availability is not proven by fetchedAt.',
      'No market, injury, suspension, or lineup adjustment is applied.',
      ...predictions.filter((p) => !p.pairedEligible).map((p) => 'Fixture ' + p.number + ' excluded from paired scoring because at least one team has fewer than 12 prior matches.'),
    ],
  };
}

function main() {
  const fixturesPath = process.argv[2];
  const outputPath = process.argv[3];
  if (!fixturesPath || !outputPath || !process.argv.includes('--generate')) {
    throw new Error('usage: node scripts/run_toto13_window20_shadow.js <round-1674-fixtures.json> <new-output.json> --generate');
  }
  if (process.env.WINDOW20_TRIGGER_CONFIRMED !== '1' ||
      process.env.GITHUB_EVENT_NAME !== 'push' ||
      process.env.GITHUB_REF !== 'refs/heads/research/toto13-window20-round1674-20261121-run2' ||
      !String(process.env.GITHUB_EVENT_HEAD_COMMIT_MESSAGE || '').includes('[window20-shadow]')) {
    throw new Error('generation requires the dedicated round-1674 marker push on the preregistered research branch');
  }

  const repoRoot = path.resolve(__dirname, '..');
  const safeOutputPath = validateWindow20OutputPath(outputPath, repoRoot);
  if (fs.existsSync(safeOutputPath)) throw new Error('refusing to overwrite an existing Window-20 shadow artifact');

  const historyPath = process.env.WINDOW20_HISTORY_PATH || path.join(repoRoot, 'data', 'history.json');
  const historyRaw = fs.readFileSync(historyPath, 'utf8');
  const fixtureRaw = fs.readFileSync(fixturesPath, 'utf8');
  const historyDocument = JSON.parse(historyRaw);
  const fixtureDocument = JSON.parse(fixtureRaw);
  const baseRunnerRaw = fs.readFileSync(path.join(repoRoot, 'scripts', 'run_toto13_shadow.js'), 'utf8');
  const runnerRaw = fs.readFileSync(__filename, 'utf8');
  const aliasesPath = path.join(repoRoot, 'data', 'team_aliases.json');
  const aliasesRaw = fs.existsSync(aliasesPath) ? fs.readFileSync(aliasesPath, 'utf8') : null;
  const nowIso = new Date().toISOString();

  const report = buildWindow20Report(historyDocument.history, fixtureDocument, {
    nowIso,
    historyDocument,
    historySha256: sha256(historyRaw),
    fixtureSha256: sha256(fixtureRaw),
    baseRunnerSha256: sha256(baseRunnerRaw),
    runnerSha256: sha256(runnerRaw),
    teamAliasesSha256: aliasesRaw ? sha256(aliasesRaw) : null,
    execution: {
      repository: process.env.GITHUB_REPOSITORY,
      workflowRunId: process.env.GITHUB_RUN_ID,
      workflowRunAttempt: process.env.GITHUB_RUN_ATTEMPT,
      commitSha: process.env.GITHUB_SHA,
      headCommitSha: process.env.SHADOW_HEAD_SHA,
      ref: process.env.GITHUB_REF,
      eventName: process.env.GITHUB_EVENT_NAME,
    },
  });

  fs.mkdirSync(path.dirname(safeOutputPath), { recursive: true });
  fs.writeFileSync(safeOutputPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  process.stdout.write(JSON.stringify({
    saved: path.relative(repoRoot, safeOutputPath),
    round: report.round,
    product: report.product,
    fixtures: report.predictions.length,
    eligibleFixtures: report.coverage.pairedEligibleFixtures,
    missingTeamSides: report.coverage.missingTeamSides,
    historyRows: report.input.historyRows,
    historySha256: report.input.historySha256,
    fixtureSha256: report.input.fixtureSha256,
    runnerSha256: report.input.runnerSha256,
    arms: [WINDOW20.id, CONTROL12.id],
  }, null, 2) + '\n');
}

if (require.main === module) {
  try { main(); }
  catch (error) { process.stderr.write('ERROR: ' + error.message + '\n'); process.exitCode = 1; }
}

module.exports = {
  WINDOW20, CONTROL12, EARLIEST_GENERATION_AT, OFFICIAL_DEADLINE,
  validateWindow20OutputPath, priorTeamMatchCount, teamWindowStatus, buildWindow20Report,
};
