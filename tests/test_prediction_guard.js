'use strict';
const assert = require('node:assert/strict');
const { sha256, validateHistory, validatePredictionSet } = require('../scripts/prediction_guard.js');
const generatedAt = '2026-11-10T06:00:00.000Z', deadlineAt = '2026-11-11T09:50:00.000Z';
const history = [
 { date: '2026-11-08', competition: 'J1', home: 'Home', away: 'Away', home_goals: 2, away_goals: 1, fetchedAt: '2026-11-09T00:00:00.000Z' },
 { date: '2026-11-09', competition: 'J2', home: 'Other', away: 'Side', home_goals: 0, away_goals: 0, fetchedAt: '2026-11-09T00:00:00.000Z' }
];
assert.equal(sha256('abc').length, 64);
assert.equal(validateHistory(history, { generatedAt }).status, 'passed');
assert.throws(() => validateHistory([...history, history[0]], { generatedAt }), /duplicate fixture key/i);
assert.throws(() => validateHistory([{ ...history[0], home_goals: -1 }], { generatedAt }), /invalid goal counts/i);
assert.throws(() => validateHistory([{ ...history[0], date: '2026-11-12' }], { generatedAt }), /future-dated/i);
assert.throws(() => validateHistory([{ ...history[0], fetchedAt: '2026-11-10T07:00:00.000Z' }], { generatedAt }), /fetched after generatedAt/i);
const fixtures = [
 { number: 1, date: '2026-11-11', home: 'Home', away: 'Away' },
 { number: 2, date: '2026-11-11', home: 'Other', away: 'Side' },
 { number: 3, date: '2026-11-11', home: 'Third', away: 'Fourth' }
];
const makePrediction = number => ({ number, prediction: {
 probabilities: { '1': 0.5, '0': 0.25, '2': 0.25 },
 homeGoalProbabilities: [0.1, 0.2, 0.3, 0.4], awayGoalProbabilities: [0.4, 0.3, 0.2, 0.1],
 homeExpectedGoals: 1.5, awayExpectedGoals: 1.2
}});
const predictions = fixtures.map(f => makePrediction(f.number));
const goal3 = validatePredictionSet({ marketType: 'totoGOAL3', fixtures, predictions, generatedAt, deadlineAt });
assert.equal(goal3.status, 'passed'); assert.equal(goal3.distributionsChecked, 9); assert.ok(goal3.maxProbabilitySumError <= 1e-9);
assert.throws(() => validatePredictionSet({ marketType: 'totoGOAL3', fixtures, predictions: [{ ...predictions[0], prediction: { ...predictions[0].prediction, homeGoalProbabilities: [0.1, 0.2, 0.3, 0.3] }}, ...predictions.slice(1)], generatedAt, deadlineAt }), /does not sum to 1/i);
assert.throws(() => validatePredictionSet({ marketType: 'totoGOAL3', fixtures, predictions: [{ ...predictions[0], prediction: { ...predictions[0].prediction, probabilities: { '1': 0.5, '0': NaN, '2': 0.5 } }}, ...predictions.slice(1)], generatedAt, deadlineAt }), /finite probability/i);
assert.throws(() => validatePredictionSet({ marketType: 'totoGOAL3', fixtures, predictions, generatedAt: deadlineAt, deadlineAt }), /earlier than deadlineAt/i);
const duplicateFixtures = [fixtures[0], { ...fixtures[1], home: 'Home', away: 'Away' }, fixtures[2]];
assert.throws(() => validatePredictionSet({ marketType: 'totoGOAL3', fixtures: duplicateFixtures, predictions, generatedAt, deadlineAt }), /duplicate target fixture/i);
assert.throws(() => validatePredictionSet({ marketType: 'totoGOAL3', fixtures: fixtures.slice(0, 2), predictions: predictions.slice(0, 2), generatedAt, deadlineAt }), /exactly 3 fixtures/i);
const totoFixtures = Array.from({ length: 13 }, (_, i) => ({
 number: i + 1, date: '2026-11-11', home: 'TotoHome' + i, away: 'TotoAway' + i
}));
const totoPredictions = totoFixtures.map(f => makePrediction(f.number));
const totoAudit = validatePredictionSet({ marketType: 'toto', fixtures: totoFixtures, predictions: totoPredictions, generatedAt, deadlineAt });
assert.equal(totoAudit.status, 'passed');
assert.equal(totoAudit.fixtureCount, 13);
assert.throws(() => validatePredictionSet({
 marketType: 'toto', fixtures: totoFixtures.slice(0, 12), predictions: totoPredictions.slice(0, 12), generatedAt, deadlineAt
}), /exactly 13 fixtures/i);
assert.throws(() => validateHistory([{ ...history[0], home: '' }], { generatedAt }), /missing or identical teams/i);
assert.throws(() => validateHistory([{ ...history[0], away: 'Home' }], { generatedAt }), /missing or identical teams/i);
assert.throws(() => validateHistory([{ ...history[0], home_goals: 1.5 }], { generatedAt }), /invalid goal counts/i);
assert.throws(() => validateHistory([{ ...history[0], date: '2026-02-30' }], { generatedAt }), /invalid date/i);
assert.throws(() => validatePredictionSet({
 marketType: 'totoGOAL3', fixtures, predictions: [{ ...predictions[0], prediction: {
 ...predictions[0].prediction, probabilities: { '1': -0.1, '0': 0.5, '2': 0.6 }
}}, ...predictions.slice(1)], generatedAt, deadlineAt
}), /finite probability/i);
console.log('PASS runtime prediction guards: history quality, duplicate/future leakage, probability simplex, and deadline provenance');
