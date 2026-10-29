'use strict';
const assert = require('node:assert/strict');
const { runAblation, predictAblation, variants } = require('../scripts/experiment_toto_ablation.js');
const history = require('../data/history.json').history;
const fixture = { date: '2026-10-31', competition: 'J1', home: '岡山', away: 'FC東京' };
for (const config of [
  {},
  { competitionOnly: true },
  { competitionOnly: true, splitLeagueBaselines: true },
  { competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true },
  { competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: 3 },
  { competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: 6 },
  { competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: 10 }
]) {
  const result = predictAblation(history, fixture, config);
  assert.deepEqual(Object.keys(result.probabilities).sort(), ['0', '1', '2']);
  assert.ok(Object.values(result.probabilities).every(p => Number.isFinite(p) && p >= 0 && p <= 1));
  assert.ok(Math.abs(Object.values(result.probabilities).reduce((a, b) => a + b, 0) - 1) < 1e-8);
}
const report = runAblation();
assert.ok(report.historyMatches > 100);
assert.ok(report.holdoutStartDate);
assert.equal(Object.keys(report.variants).length, variants.length);
for (const variant of variants) {
  const result = report.variants[variant.id];
  assert.ok(result.all.n > 100, variant.id + ' should score enough historical matches');
  assert.ok(result.chronologicalHoldout.n > 0, variant.id + ' should score holdout matches');
  assert.ok(Number.isFinite(result.chronologicalHoldout.brier));
  assert.ok(Number.isFinite(result.chronologicalHoldout.logloss));
}
console.log(JSON.stringify({
  historyMatches: report.historyMatches,
  holdoutStartDate: report.holdoutStartDate,
  variants: Object.fromEntries(Object.entries(report.variants).map(([key, value]) => [key, {
    all: value.all, chronologicalHoldout: value.chronologicalHoldout
  }]))
}, null, 2));
