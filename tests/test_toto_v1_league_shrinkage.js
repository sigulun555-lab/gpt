'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { runExperiment } = require('../scripts/experiment_toto_v1_league_shrinkage.js');

test('league-adaptive shrinkage V1 is compared with fixed-k6 V0 on identical 2026 targets', () => {
  const report = runExperiment();
  assert.equal(report.folds.length, 3);
  assert.equal(report.pairedDateBlockBootstrap.replicates, 2000);
  assert.equal(report.metrics.v0.n, report.metrics.v1.n);
  assert.equal(report.pairedDateBlockBootstrap.targetMatches, report.metrics.v0.n);
  assert.ok(report.historySources.historical2025.rows >= 1000);
  assert.ok(report.trainingHistoryMatches >= 1000);
  assert.ok(report.targetMatches >= 100);
  for (const fold of report.folds) {
    assert.ok(fold.trainEndDateInclusive < fold.testStartDate);
    assert.ok(fold.innerValidationStartDate <= fold.trainEndDateInclusive);
    for (const league of ['J1', 'J2', 'J3']) {
      assert.ok([3, 6, 10].includes(fold.selectedKByLeague[league]));
      assert.ok(fold.innerScoresByLeague[league].validationMatches > 0);
    }
    assert.equal(fold.v0.n, fold.v1.n);
    assert.equal(fold.v0.n, fold.outerTestMatches);
  }
  console.log(JSON.stringify({
    v0: report.metrics.v0,
    v1: report.metrics.v1,
    delta: report.pairedDateBlockBootstrap.deltaV1MinusV0,
    confidenceInterval95: report.pairedDateBlockBootstrap.confidenceInterval95,
    historySources: report.historySources,
    trainingHistoryMatches: report.trainingHistoryMatches,
    targetMatches: report.targetMatches,
    folds: report.folds.map(f => ({
      fold: f.fold,
      selectedKByLeague: f.selectedKByLeague,
      v0: f.v0,
      v1: f.v1
    }))
  }, null, 2));
});
