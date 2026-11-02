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
const nested = report.nestedKSelection;
assert.equal(nested.folds.length, 3);
assert.ok(nested.aggregateOuterMetrics.productionBaseline.n > 0);
assert.ok(nested.aggregateOuterMetrics.nestedSelectedK.n > 0);
assert.equal(nested.aggregateOuterMetrics.nestedLeagueSelected.n, nested.aggregateOuterMetrics.productionBaseline.n);
for (const fold of nested.folds) {
  assert.ok([3, 6, 10].includes(fold.selectedK));
  assert.ok(fold.trainEndDateInclusive < fold.testStartDate, 'outer train must end before test starts');
  assert.ok(fold.innerValidationMatches > 0);
  assert.ok(fold.testMatches > 0);
  assert.equal(fold.outerMetrics.nestedSelectedK.n, fold.testMatches);
  assert.equal(fold.outerMetrics.nestedLeagueSelected.n, fold.testMatches);
  for (const selection of Object.values(fold.selectedByCompetition)) {
    assert.ok(['productionBaseline', 'shrinkageK3', 'shrinkageK6', 'shrinkageK10'].includes(selection.model));
    if (selection.model === 'productionBaseline') assert.equal(selection.k, null);
    else assert.ok([3, 6, 10].includes(selection.k));
  }
}
for (const comparison of Object.values(nested.pairedComparisons)) {
  assert.equal(comparison.targetMatches, nested.aggregateOuterMetrics.productionBaseline.n);
  assert.ok(comparison.dateBlocks >= 3, 'paired bootstrap should retain multiple date blocks');
  assert.equal(comparison.replicates, 2000);
  for (const metric of ['accuracy', 'brier', 'logloss']) {
    assert.ok(Number.isFinite(comparison.deltaCandidateMinusBaseline[metric]));
    assert.equal(comparison.confidenceInterval95[metric].length, 2);
    assert.ok(comparison.confidenceInterval95[metric][0] <= comparison.confidenceInterval95[metric][1]);
  }
}
assert.ok(Object.keys(nested.perCompetition.nestedSelectedK).length >= 2);
assert.ok(Object.keys(nested.perCompetition.nestedLeagueSelected).length >= 2);
for (const result of Object.values(nested.perCompetition.nestedLeagueSelected)) {
  assert.ok(result.targetMatches > 0);
  assert.ok(Number.isFinite(result.deltaCandidateMinusBaseline.brier));
}
for (const result of Object.values(nested.perCompetition.nestedSelectedK)) {
  assert.ok(result.targetMatches > 0);
  assert.ok(Number.isFinite(result.deltaCandidateMinusBaseline.brier));
}
console.log(JSON.stringify({
  historyMatches: report.historyMatches,
  holdoutStartDate: report.holdoutStartDate,
  nestedKSelection: nested,
  variants: Object.fromEntries(Object.entries(report.variants).map(([key, value]) => [key, {
    all: value.all, chronologicalHoldout: value.chronologicalHoldout
  }]))
}, null, 2));
