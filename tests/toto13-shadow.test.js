'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {
  CONFIGS, normalizeTeam, validateHistory, validateFixtures, predictToto13, runRound,
} = require('../scripts/run_toto13_shadow.js');

function historyRows() {
  return [
    { date: '2026-01-01', competition: 'J1', home: 'Alpha', away: 'Beta', home_goals: 2, away_goals: 0 },
    { date: '2026-01-02', competition: 'J1', home: 'Beta', away: 'Alpha', home_goals: 1, away_goals: 1 },
    { date: '2026-01-03', competition: 'J1', home: 'Alpha', away: 'Beta', home_goals: 3, away_goals: 1 },
    { date: '2026-01-04', competition: 'J1', home: 'Beta', away: 'Alpha', home_goals: 0, away_goals: 2 },
  ];
}

function fixtures() {
  return {
    round: 9999,
    marketType: 'toto',
    deadline: '2026-02-01T12:00:00+09:00',
    fixtures: Array.from({ length: 13 }, (_, i) => ({
      number: i + 1, date: '2026-02-01', competition: 'J1',
      home: i % 2 ? 'Alpha' : 'Beta', away: i % 2 ? 'Beta' : 'Alpha',
    })),
  };
}

test('normalizes full-width team labels with NFKC', () => {
  assert.equal(normalizeTeam('東京Ｖ'), '東京V');
});

test('history must be valid and is sorted chronologically', () => {
  const rows = historyRows().reverse();
  const sorted = validateHistory(rows);
  assert.equal(sorted[0].date, '2026-01-01');
  assert.equal(sorted.length, 4);
});

test('strict date cutoff excludes same-day and future results', () => {
  const rows = historyRows();
  rows.push({ date: '2026-02-01', competition: 'J1', home: 'Alpha', away: 'Beta', home_goals: 9, away_goals: 0 });
  rows.push({ date: '2026-02-02', competition: 'J1', home: 'Alpha', away: 'Beta', home_goals: 8, away_goals: 0 });
  const prediction = predictToto13(validateHistory(rows), fixtures().fixtures[0], CONFIGS[0]);
  assert.equal(prediction.priorMatchCount, 4);
  assert.equal(prediction.priorMaxDate, '2026-01-04');
});

test('k6, k10 and k0 probabilities are normalized and isolated', () => {
  const history = validateHistory(historyRows());
  const fixture = fixtures().fixtures[0];
  for (const config of CONFIGS) {
    const result = predictToto13(history, fixture, config);
    const sum = Object.values(result.probabilities).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-12);
    assert.ok(Object.values(result.probabilities).every((p) => Number.isFinite(p) && p >= 0 && p <= 1));
  }
});

test('GOAL3 three-fixture input is rejected', () => {
  assert.throws(() => validateFixtures({
    round: 1670, marketType: 'totoGOAL3',
    fixtures: [{ number: 1 }, { number: 2 }, { number: 3 }],
  }), /marketType/);
});

test('fixture dates must be date-only to preserve strict same-day exclusion', () => {\n  const invalid = fixtures();\n  invalid.fixtures[0].date = '2026-02-01T12:00:00+09:00';\n  assert.throws(() => validateFixtures(invalid), /YYYY-MM-DD/);\n});\n\ntest('incomplete or duplicate fixture numbers are rejected', () => {
  assert.throws(() => validateFixtures({ round: 1, fixtures: fixtures().fixtures.slice(0, 12) }), /13/);
  const invalid = fixtures();
  invalid.fixtures[12].number = 12;
  assert.throws(() => validateFixtures(invalid), /unique integers/);
});

test('round report is clearly shadow-only and records all 13 fixtures', () => {
  const report = runRound(historyRows(), fixtures(), '2026-01-05T00:00:00.000Z');
  assert.equal(report.artifactType, 'toto13_shadow_only');
  assert.equal(report.predictions.length, 13);
  assert.equal(report.predictions[0].variants.V0_k6_window12.priorMatchCount, 4);
  assert.equal(report.predictions[0].variants.V1_k10_window12.priorMatchCount, 4);
  assert.match(report.warnings[0], /Shadow-only/);
});
