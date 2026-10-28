'use strict';
const assert = require('node:assert/strict');
const { runExperiment } = require('../scripts/experiment_toto_model.js');
const report = runExperiment();
assert.equal(report.round, 1668);
assert.equal(report.predictions.length, 13, 'round 1668 must have exactly 13 fixtures');
assert.equal(new Set(report.predictions.map(x => x.matchNumber)).size, 13, 'match numbers must be unique');
for (const p of report.predictions) {
  for (const version of ['baseline', 'candidate']) {
    const probs = p[version].probabilities;
    assert.deepEqual(Object.keys(probs).sort(), ['0', '1', '2']);
    assert.ok(Object.values(probs).every(x => Number.isFinite(x) && x >= 0 && x <= 1));
    assert.ok(Math.abs(Object.values(probs).reduce((a, b) => a + b, 0) - 1) < 1e-8);
  }
  assert.ok(p.marketVoteShare);
  assert.ok(p.candidateK10);
  assert.ok(Math.abs(Object.values(p.trialBlend70Market30K10).reduce((a, b) => a + b, 0) - 1) < 1e-8);
}
assert.equal(report.marketSnapshot.retrievedAt, '2026-10-28T23:50:00+09:00');
assert.ok(report.evaluation.baselineCurrentProduction.n > 100);
assert.equal(report.evaluation.baselineCurrentProduction.n, report.evaluation.candidateCompetitionShrinkageK3.n);
assert.equal(report.evaluation.baselineCurrentProduction.n, report.evaluation.candidateCompetitionShrinkageK6.n);
assert.equal(report.evaluation.baselineCurrentProduction.n, report.evaluation.candidateCompetitionShrinkageK10.n);
console.log(JSON.stringify({
  round: report.round,
  evaluation: report.evaluation,
  predictions: report.predictions.map(p => ({
    n: p.matchNumber, match: p.home + ' vs ' + p.away,
    baseline: p.baseline.probabilities,
    candidate: p.candidate.probabilities,
    candidateTopPick: p.topPick,
    expectedGoals: [p.candidate.homeExpectedGoals, p.candidate.awayExpectedGoals]
  }))
}, null, 2));
