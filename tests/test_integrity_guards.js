'use strict';

const assert = require('node:assert/strict');
const {
  sha256,
  createInputProvenance,
  validateHistory,
  validateProbabilityVector,
  validatePredictions,
  validateProvenance,
  assertBuildIntegrity
} = require('../scripts/integrity_guards.js');

const generatedAt = '2026-10-22T00:00:00.000Z';
const deadlineAt = '2026-10-28T18:50:00+09:00';
const sourceUrl = 'https://example.test/schedule.csv';
const totoUrl = 'https://example.test/toto.html';

function historyRow(overrides = {}) {
  return {
    date: '2025-01-01',
    competition: 'J1',
    home: 'Home',
    away: 'Away',
    home_goals: 2,
    away_goals: 1,
    source: 'community_jleague_csv',
    sourceUrl,
    fetchedAt: generatedAt,
    ...overrides
  };
}

function fixture(number, home, away) {
  return { number, date: '2026-10-28', home, away };
}

function prediction(number, home, away) {
  return {
    number,
    date: '2026-10-28',
    home,
    away,
    totoRound: 1666,
    totoTargetConfirmed: true,
    prediction: {
      probabilities: { '1': 0.4, '0': 0.3, '2': 0.3 },
      homeGoalProbabilities: [0.25, 0.3, 0.2, 0.25],
      awayGoalProbabilities: [0.2, 0.3, 0.25, 0.25],
      homeExpectedGoals: 1.2,
      awayExpectedGoals: 1.1
    }
  };
}

const goodHistory = [historyRow()];
assert.deepEqual(validateHistory(goodHistory, generatedAt), []);
assert.deepEqual(validateProbabilityVector([0.25, 0.25, 0.25, 0.25], 4, 'test'), []);
assert.ok(validateProbabilityVector([0.2, 0.2, 0.2, 0.2], 4, 'test').length > 0);
assert.ok(validateProbabilityVector([0.5, -0.1, 0.3, 0.3], 4, 'test').length > 0);
assert.ok(validateProbabilityVector([0.5, NaN, 0.5, 0], 4, 'test').length > 0);
assert.ok(validateProbabilityVector([0.5, 0.5], 4, 'test').length > 0);

assert.ok(validateHistory([historyRow(), historyRow()], generatedAt).some(x => x.includes('duplicates match key')));
assert.ok(validateHistory([historyRow({ competition: '' })], generatedAt).some(x => x.includes('.competition')));
assert.ok(validateHistory([historyRow({ home: '' })], generatedAt).some(x => x.includes('.home is missing')));
assert.ok(validateHistory([historyRow({ date: '2025-01-02' }), historyRow({ date: '2025-01-01', home: 'Home 2' })], generatedAt).some(x => x.includes('not in chronological order')));
assert.ok(validateHistory([historyRow({ date: '2026-02-30' })], generatedAt).some(x => x.includes('real YYYY-MM-DD')));
assert.ok(validateHistory([historyRow({ home_goals: 1.5 })], generatedAt).some(x => x.includes('non-negative integer')));
assert.ok(validateHistory([historyRow({ date: '2026-10-23' })], generatedAt).some(x => x.includes('after the generation date')));
assert.ok(validateHistory([historyRow({ fetchedAt: '2026-10-22T00:00:01.000Z' })], generatedAt).some(x => x.includes('later than generatedAt')));

const fixtures = [fixture(1, 'Home A', 'Away A'), fixture(2, 'Home B', 'Away B'), fixture(3, 'Home C', 'Away C')];
const predictions = [
  prediction(1, 'Home A', 'Away A'),
  prediction(2, 'Home B', 'Away B'),
  prediction(3, 'Home C', 'Away C')
];
assert.deepEqual(validatePredictions(predictions, fixtures, 'totoGOAL3', 1666), []);
const badPrediction = structuredClone(predictions);
badPrediction[0].prediction.homeGoalProbabilities = [0.5, 0.5, 0.5, 0.5];
assert.ok(validatePredictions(badPrediction, fixtures, 'totoGOAL3', 1666).some(x => x.includes('must sum to 1')));
const badOutcomePrediction = structuredClone(predictions);
badOutcomePrediction[0].prediction.probabilities = { '1': 0.5, '0': 0.5, '2': 0.5 };
assert.ok(validatePredictions(badOutcomePrediction, fixtures, 'totoGOAL3', 1666).some(x => x.includes('must sum to 1')));
assert.ok(validatePredictions(predictions.slice(0, 2), fixtures, 'totoGOAL3', 1666).some(x => x.includes('count must match')));

const provenance = createInputProvenance('schedule bytes', 'official page bytes', generatedAt, deadlineAt, sourceUrl, totoUrl);
assert.match(provenance.scheduleInput.sha256, /^[a-f0-9]{64}$/);
assert.match(provenance.officialTotoInput.sha256, /^[a-f0-9]{64}$/);
assert.notEqual(sha256('schedule bytes'), sha256('different schedule bytes'));
assert.deepEqual(validateProvenance(provenance, generatedAt, deadlineAt, true), []);
assert.ok(validateProvenance(provenance, generatedAt, '2026-10-21T00:00:00Z', true).some(x => x.includes('earlier than deadlineAt')));

assert.equal(assertBuildIntegrity({
  history: goodHistory,
  predictions,
  fixtures,
  marketType: 'totoGOAL3',
  round: 1666,
  generatedAt,
  deadlineAt,
  provenance,
  confirmed: true
}), true);

assert.throws(() => assertBuildIntegrity({
  history: goodHistory,
  predictions: badPrediction,
  fixtures,
  marketType: 'totoGOAL3',
  round: 1666,
  generatedAt,
  deadlineAt,
  provenance,
  confirmed: true
}), /Prediction integrity validation failed/);

console.log('PASS history quality, duplicate detection, probability simplex, confirmed fixture parity, timestamp/deadline, and input SHA-256 provenance guards');
