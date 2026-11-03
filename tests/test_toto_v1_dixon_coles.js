'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { dixonColesProbabilities, runExperiment } = require('../scripts/experiment_toto_v1_dixon_coles.js');
const { loadCombinedHistory, predictShrinkage } = require('../scripts/experiment_toto_v1_recency.js');

test('Dixon-Coles rho=0 is identical to independent Poisson and all rho candidates normalize', () => {
  const loaded = loadCombinedHistory();
  const fixture = { date: '2026-10-31', competition: 'J1', home: '岡山', away: 'FC東京' };
  const base = predictShrinkage(loaded.history, fixture, { k: 6 });
  const identity = dixonColesProbabilities(base.homeExpectedGoals, base.awayExpectedGoals, 0);
  for (const key of ['1', '0', '2']) assert.ok(Math.abs(identity[key] - base.probabilities[key]) < 1e-10);
  for (const rho of [-0.08, -0.04, 0, 0.04, 0.08]) {
    const p = dixonColesProbabilities(base.homeExpectedGoals, base.awayExpectedGoals, rho);
    assert.ok(Math.abs(Object.values(p).reduce((a, b) => a + b, 0) - 1) < 1e-10);
    assert.ok(Object.values(p).every(value => Number.isFinite(value) && value >= 0 && value <= 1));
  }
  assert.throws(() => dixonColesProbabilities(3.5, 3.5, 0.2), /non-positive/);
});

test('nested Dixon-Coles V1 compares with shrinkage V0 on all 2026 target fixtures', () => {
  const report = runExperiment();
  assert.equal(report.folds.length, 3);
  assert.equal(report.pairedDateBlockBootstrap.replicates, 2000);
  assert.equal(report.metrics.v0.n, report.metrics.v1.n);
  assert.equal(report.metrics.v0.n, report.targetMatches);
  assert.equal(report.pairedDateBlockBootstrap.targetMatches, report.metrics.v0.n);
  assert.ok(report.historySources.historical2025.rows >= 1000);
  assert.ok(report.trainingHistoryMatches >= 1000);
  assert.ok(report.targetMatches >= 100);
  for (const fold of report.folds) {
    assert.ok(fold.trainEndDateInclusive < fold.testStartDate);
    assert.ok(fold.innerValidationStartDate <= fold.trainEndDateInclusive);
    assert.ok([-0.08, -0.04, 0, 0.04, 0.08].includes(fold.selectedRho));
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
      selectedRho: f.selectedRho,
      v0: f.v0,
      v1: f.v1
    }))
  }, null, 2));
});
