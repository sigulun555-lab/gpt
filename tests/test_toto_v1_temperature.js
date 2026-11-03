'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { temperatureScale, runExperiment } = require('../scripts/experiment_toto_v1_temperature.js');

test('temperature scaling preserves normalized 1X2 probabilities and T=1 is identity', () => {
  const input = { '1': 0.52, '0': 0.27, '2': 0.21 };
  const identity = temperatureScale(input, 1);
  for (const key of ['1', '0', '2']) assert.ok(Math.abs(identity[key] - input[key]) < 1e-12);
  const scaled = temperatureScale(input, 1.2);
  assert.ok(Math.abs(Object.values(scaled).reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(Object.values(scaled).every(p => Number.isFinite(p) && p >= 0 && p <= 1));
  assert.throws(() => temperatureScale(input, 0), /positive/);
});

test('nested temperature V1 compares with fixed shrinkage V0 on identical fixtures', () => {
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
    assert.ok([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6].includes(fold.selectedTemperature));
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
      selectedTemperature: f.selectedTemperature,
      v0: f.v0,
      v1: f.v1
    }))
  }, null, 2));
});
