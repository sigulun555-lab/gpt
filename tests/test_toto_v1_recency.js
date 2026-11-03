'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { predictShrinkage, runExperiment } = require('../scripts/experiment_toto_v1_recency.js');
const history = require('../data/history.json').history;

test('V0 and V1 return normalized 1X2 probabilities and preserve the V0 definition', () => {
  const fixture = { date: '2026-10-31', competition: 'J1', home: '岡山', away: 'FC東京' };
  const v0 = predictShrinkage(history, fixture, { k: 6 });
  const v1 = predictShrinkage(history, fixture, { k: 6, halfLifeMatches: 6 });
  for (const result of [v0, v1]) {
    const values = Object.values(result.probabilities);
    assert.equal(values.length, 3);
    assert.ok(values.every(x => Number.isFinite(x) && x >= 0 && x <= 1));
    assert.ok(Math.abs(values.reduce((a, b) => a + b, 0) - 1) < 1e-10);
  }
  assert.equal(v0.version, 'V0-shrinkage-k6');
  assert.equal(v0.halfLifeMatches, null);
  assert.equal(v1.version, 'V1-recency-shrinkage-k6');
  assert.equal(v1.halfLifeMatches, 6);
  assert.ok(Number.isFinite(v1.homeExpectedGoals));
  assert.ok(Number.isFinite(v1.awayExpectedGoals));
});

test('nested walk-forward compares V1 against fixed V0 without same-day leakage', () => {
  const report = runExperiment();
  assert.equal(report.folds.length, 3);
  assert.equal(report.pairedDateBlockBootstrap.replicates, 2000);
  assert.ok(report.metrics.v0.n > 0);
  assert.equal(report.metrics.v0.n, report.metrics.v1.n);
  assert.equal(report.pairedDateBlockBootstrap.targetMatches, report.metrics.v0.n);
  assert.ok(report.pairedDateBlockBootstrap.dateBlocks >= 3);
  for (const fold of report.folds) {
    assert.ok(fold.trainEndDateInclusive < fold.testStartDate);
    assert.ok(fold.innerValidationStartDate <= fold.trainEndDateInclusive);
    assert.ok([3, 6, 9, 12].includes(fold.selectedHalfLifeMatches));
    assert.equal(fold.v0.n, fold.v1.n);
    assert.equal(fold.v0.n, fold.outerTestMatches);
  }
  console.log(JSON.stringify({
    v0: report.metrics.v0,
    v1: report.metrics.v1,
    delta: report.pairedDateBlockBootstrap.deltaV1MinusV0,
    confidenceInterval95: report.pairedDateBlockBootstrap.confidenceInterval95,
    folds: report.folds.map(f => ({
      fold: f.fold,
      selectedHalfLifeMatches: f.selectedHalfLifeMatches,
      v0: f.v0,
      v1: f.v1
    }))
  }, null, 2));
});
