#!/usr/bin/env node
'use strict';

// Isolated toto13 T1 shadow runner. Does not import or modify production model.js.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const CONFIGS = Object.freeze([
  { id: 'V0_k6_window12', shrink: 6, window: 12 },
  { id: 'V1_k10_window12', shrink: 10, window: 12 },
  { id: 'baseline_k0_window12', shrink: 0, window: 12 },
]);

function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function validateOutputPath(outputPath, repoRoot = path.resolve(__dirname, '..')) {
  if (typeof outputPath !== 'string' || !outputPath.trim()) {
    throw new Error('output path is required');
  }
  const allowedDir = path.resolve(repoRoot, 'shadow', 'toto13', 'predictions');
  const resolved = path.resolve(repoRoot, outputPath);
  const relative = path.relative(allowedDir, resolved);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative) || path.extname(resolved) !== '.json') {
    throw new Error('output path must be a new .json file inside shadow/toto13/predictions/');
  }
  return resolved;
}

function normalizeTeam(value) {
  return String(value || '').normalize('NFKC').trim();
}

function poisson(lambda, goals) {
  let factorial = 1;
  for (let i = 2; i <= goals; i += 1) factorial *= i;
  return Math.exp(-lambda) * (lambda ** goals) / factorial;
}

function validateHistory(history) {
  if (!Array.isArray(history) || history.length === 0) {
    throw new Error('history must be a non-empty array');
  }
  const valid = history.filter((m) =>
    typeof m.date === 'string' &&
    Number.isFinite(Date.parse(m.date)) &&
    Number.isFinite(m.home_goals) &&
    Number.isFinite(m.away_goals) &&
    typeof m.home === 'string' &&
    typeof m.away === 'string'
  );
  if (valid.length !== history.length) {
    throw new Error(`history contains invalid rows: ${history.length - valid.length}`);
  }
  return [...valid].sort((a, b) =>
    a.date.localeCompare(b.date) ||
    String(a.competition || '').localeCompare(String(b.competition || '')) ||
    a.home.localeCompare(b.home) ||
    a.away.localeCompare(b.away)
  );
}

function predictToto13(history, fixture, config) {
  const targetDate = fixture.date;
  if (typeof targetDate !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(targetDate) || !Number.isFinite(Date.parse(targetDate))) {
    throw new Error('fixture.date must be a valid YYYY-MM-DD date string');
  }
  const home = normalizeTeam(fixture.home);
  const away = normalizeTeam(fixture.away);
  if (!home || !away || home === away) throw new Error('fixture home/away are invalid');

  // Strict chronological split: same-day matches are never used.
  const prior = history.filter((m) => m.date < targetDate);
  if (prior.length === 0) throw new Error(`no prior history before ${targetDate}`);

  const leagueHome = prior.reduce((sum, m) => sum + m.home_goals, 0) / prior.length;
  const leagueAway = prior.reduce((sum, m) => sum + m.away_goals, 0) / prior.length;
  const base = (leagueHome + leagueAway) / 2;
  if (!(base > 0)) throw new Error('historical baseline must be positive');

  function teamStats(team) {
    const matches = prior.filter((m) =>
      normalizeTeam(m.home) === team || normalizeTeam(m.away) === team
    ).slice(-config.window);
    const n = matches.length;
    if (n === 0) return { attack: 1, defense: 1, matches: 0, unknown: true };
    const goalsFor = matches.map((m) =>
      normalizeTeam(m.home) === team ? m.home_goals : m.away_goals
    );
    const goalsAgainst = matches.map((m) =>
      normalizeTeam(m.home) === team ? m.away_goals : m.home_goals
    );
    let attack = (goalsFor.reduce((a, b) => a + b, 0) / n) / base;
    let defense = (goalsAgainst.reduce((a, b) => a + b, 0) / n) / base;
    if (config.shrink > 0) {
      const trust = n / (n + config.shrink);
      attack = 1 + (attack - 1) * trust;
      defense = 1 + (defense - 1) * trust;
    }
    return { attack, defense, matches: n, unknown: false };
  }

  const h = teamStats(home);
  const a = teamStats(away);
  const homeXg = Math.max(0.15, Math.min(3.5, leagueHome * h.attack * a.defense));
  const awayXg = Math.max(0.15, Math.min(3.5, leagueAway * a.attack * h.defense));

  let homeWin = 0;
  let draw = 0;
  let awayWin = 0;
  for (let hg = 0; hg <= 10; hg += 1) {
    for (let ag = 0; ag <= 10; ag += 1) {
      const probability = poisson(homeXg, hg) * poisson(awayXg, ag);
      if (hg > ag) homeWin += probability;
      else if (hg === ag) draw += probability;
      else awayWin += probability;
    }
  }
  const total = homeWin + draw + awayWin;
  const probabilities = {
    '1': homeWin / total,
    '0': draw / total,
    '2': awayWin / total,
  };

  return {
    probabilities,
    homeExpectedGoals: homeXg,
    awayExpectedGoals: awayXg,
    historyHomeMatches: h.matches,
    historyAwayMatches: a.matches,
    priorMatchCount: prior.length,
    priorMaxDate: prior[prior.length - 1].date,
    unknownTeams: [h.unknown ? home : null, a.unknown ? away : null].filter(Boolean),
  };
}

function validateFixtures(document) {
  if (!document || !Number.isInteger(document.round) || document.round <= 0) {
    throw new Error('fixture document must include a positive integer round');
  }
  if (document.marketType && document.marketType !== 'toto') {
    throw new Error('marketType must be toto; GOAL3 fixtures are not accepted');
  }
  if (!Array.isArray(document.fixtures) || document.fixtures.length !== 13) {
    throw new Error('exactly 13 toto fixtures are required; GOAL3 3-fixture files are rejected');
  }
  const numbers = document.fixtures.map((f) => f.number).sort((a, b) => a - b);
  if (numbers.some((n, i) => n !== i + 1)) {
    throw new Error('fixture numbers must be unique integers 1..13');
  }
  for (const fixture of document.fixtures) {
    if (typeof fixture.date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(fixture.date) || !Number.isFinite(Date.parse(fixture.date)) || !fixture.home || !fixture.away) {
      throw new Error(`fixture ${fixture.number} must include YYYY-MM-DD date, home and away`);
    }
  }
  return document.fixtures;
}

function runRound(historyInput, fixtureDocument, createdAt = new Date().toISOString(), execution = {}) {
  const history = validateHistory(historyInput);
  const fixtures = validateFixtures(fixtureDocument);
  const predictions = fixtures.map((fixture) => {
    const variants = {};
    for (const config of CONFIGS) {
      variants[config.id] = predictToto13(history, fixture, config);
    }
    return {
      number: fixture.number,
      date: fixture.date,
      competition: fixture.competition || null,
      home: fixture.home,
      away: fixture.away,
      variants,
    };
  });
  return {
    schemaVersion: 1,
    artifactType: 'toto13_shadow_only',
    round: fixtureDocument.round,
    createdAt,
    deadline: fixtureDocument.deadline || null,
    modelFamily: 'T1 independent Poisson 1X2',
    configs: CONFIGS,
    protocol: {
      evaluation: 'prospective shadow; no production replacement',
      historyWindow: 12,
      goalGrid: '0..10 inclusive',
      trainingCutoff: 'strictly earlier than each fixture date; same-day matches excluded',
      poissonCorrection: 'none',
      drawBoost: 'none',
      marketBlend: 'none',
      variants: 'V0 k=6; V1 candidate k=10; baseline k=0',
    },
    input: {
      historyRows: history.length,
      historySha256: historyInput.sha256 || null,
      fixtureSha256: fixtureDocument.sha256 || null,
      historySources: [...new Set(history.map((m) => m.source || 'unspecified'))].sort(),
    },
    predictions,
    execution: {
      repository: execution.repository || null,
      workflowRunId: execution.workflowRunId || null,
      workflowRunAttempt: execution.workflowRunAttempt || null,
      commitSha: execution.commitSha || null,
      ref: execution.ref || null,
      eventName: execution.eventName || null,
      timestampEvidence: 'createdAt is runner-generated; corroborate against the linked GitHub Actions run and artifact creation metadata before scoring',
    },
    warnings: [
      'Shadow-only research output. Not a production prediction.',
      'History rows may include community-mirror results; source quality must be audited.',
      'Do not score until official results and the pre-deadline prediction timestamp are verified.',
      ...[...new Set(predictions.flatMap((p) => p.variants.V0_k6_window12.unknownTeams.concat(p.variants.V1_k10_window12.unknownTeams, p.variants.baseline_k0_window12.unknownTeams)))].map((team) => `No prior history for team: ${team}; neutral attack/defense fallback used.`),
    ],
  };
}

function main() {
  const fixturesPath = process.argv[2];
  const outputPath = process.argv[3];
  if (!fixturesPath || !outputPath) {
    throw new Error('usage: node scripts/run_toto13_shadow.js <fixtures.json> <new-output.json>');
  }
  const repoRoot = path.resolve(__dirname, '..');
  const safeOutputPath = validateOutputPath(outputPath, repoRoot);
  if (fs.existsSync(safeOutputPath)) throw new Error(`refusing to overwrite existing shadow log: ${safeOutputPath}`);

  const historyPath = path.join(__dirname, '..', 'data', 'history.json');
  const historyRaw = fs.readFileSync(historyPath, 'utf8');
  const fixtureRaw = fs.readFileSync(fixturesPath, 'utf8');
  const historyDoc = JSON.parse(historyRaw);
  const fixtureDoc = JSON.parse(fixtureRaw);
  const report = runRound(
    historyDoc.history,
    fixtureDoc,
    new Date().toISOString(),
    {
      repository: process.env.GITHUB_REPOSITORY,
      workflowRunId: process.env.GITHUB_RUN_ID,
      workflowRunAttempt: process.env.GITHUB_RUN_ATTEMPT,
      commitSha: process.env.GITHUB_SHA,
      ref: process.env.GITHUB_REF,
      eventName: process.env.GITHUB_EVENT_NAME,
    }
  );
  report.input.historySha256 = sha256(historyRaw);
  report.input.fixtureSha256 = sha256(fixtureRaw);

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  process.stdout.write(JSON.stringify({
    saved: path.relative(repoRoot, safeOutputPath),
    round: report.round,
    fixtures: report.predictions.length,
    historyRows: report.input.historyRows,
    historySha256: report.input.historySha256,
    variants: CONFIGS.map((c) => c.id),
  }, null, 2) + '\n');
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`ERROR: ${error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = { CONFIGS, normalizeTeam, validateHistory, validateFixtures, validateOutputPath, predictToto13, runRound };
