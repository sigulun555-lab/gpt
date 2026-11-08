'use strict';

const assert = require('node:assert/strict');
const { sha256, toLedger } = require('../scripts/prediction_ledger_v1.js');

const hashes = {
  history: sha256('history'),
  teamAliases: sha256('aliases'),
  fixtures: sha256('fixtures'),
  modelCode: sha256('model'),
  pipelineCode: sha256('pipeline'),
  adapterCode: sha256('adapter')
};

const goal3Payload = {
  schemaVersion: 1,
  generatedAt: '2026-11-08T05:58:08.594Z',
  round: 1671,
  deadline: '2026-11-11T18:50:00+09:00',
  totoSource: {
    marketType: 'totoGOAL3',
    lastFetchedAt: '2026-11-08T05:58:08.594Z',
    url: 'https://example.invalid/official'
  },
  predictions: [{
    number: 1,
    marketType: 'totoGOAL3',
    date: '2026-11-11',
    home: '横浜FM',
    away: '鹿島',
    sourceUrl: 'https://example.invalid/official',
    prediction: {
      homeGoalProbabilities: [0.1, 0.2, 0.3, 0.4],
      awayGoalProbabilities: [0.4, 0.3, 0.2, 0.1],
      homeExpectedGoals: 1.2,
      awayExpectedGoals: 0.9,
      canonicalHome: '横浜F・マリノス',
      canonicalAway: '鹿島アントラーズ',
      method: 'Poisson baseline'
    },
    availabilityNote: '未取得'
  }]
};

const ledger = toLedger(goal3Payload, {
  inputHashes: hashes,
  predictionPayloadSha256: sha256('source payload'),
  exportedAt: '2026-11-08T06:00:00.000Z'
});
assert.equal(ledger.schemaVersion, 1);
assert.equal(ledger.records.length, 1, 'canonical ledger must be one row per match');
const row = ledger.records[0];
assert.equal(row.product, 'totoGOAL3');
assert.equal(row.matchNumber, 1);
assert.equal(row.probabilities, null, 'GOAL3 must not be represented as nominal 1X2');
assert.deepEqual(row.goalDistributions.home, [0.1, 0.2, 0.3, 0.4]);
assert.deepEqual(row.goalDistributions.away, [0.4, 0.3, 0.2, 0.1]);
assert.equal(row.confirmedAt, null, 'generatedAt must not be silently copied into confirmedAt');
assert.equal(row.confirmedAtStatus, 'not_recorded_in_source');
assert.equal(row.marketSnapshotId, null, 'fixture source is not a market-share snapshot');
assert.equal(row.sourceRetrievedAt, '2026-11-08T05:58:08.594Z');
assert.equal(row.inputHashCompleteness, 'complete');
assert.match(row.inputsHash, /^[a-f0-9]{64}$/);
assert.match(row.predictionPayloadSha256, /^[a-f0-9]{64}$/);

const toto13Payload = {
  generatedAt: '2026-11-08T05:58:08.594Z',
  round: 1672,
  deadline: '2026-11-15T12:00:00+09:00',
  totoSource: { marketType: 'toto' },
  predictions: [{
    number: 1, marketType: 'toto', date: '2026-11-15',
    home: 'Home', away: 'Away',
    prediction: { probabilities: { '1': 0.5, '0': 0.25, '2': 0.25 } }
  }]
};
const toto13 = toLedger(toto13Payload, { exportedAt: '2026-11-08T06:00:00.000Z' }).records[0];
assert.equal(toto13.product, 'toto13');
assert.deepEqual(toto13.probabilities, { home: 0.5, draw: 0.25, away: 0.25 });
assert.equal(toto13.goalDistributions, null);
assert.equal(toto13.inputHashCompleteness, 'partial');
assert.equal(toto13.inputsHash, null);

assert.throws(() => toLedger({
  ...goal3Payload,
  predictions: [{ ...goal3Payload.predictions[0], prediction: {
    ...goal3Payload.predictions[0].prediction,
    homeGoalProbabilities: [0.1, 0.2, 0.3, 0.3]
  }}]
}, { inputHashes: hashes }), /must sum to 1/);

assert.throws(() => toLedger({
  ...toto13Payload,
  predictions: [toto13Payload.predictions[0], toto13Payload.predictions[0]]
}), /Duplicate prediction row/);

console.log('PASS prediction ledger v1: toto13 nominal probabilities, GOAL3 ordinal distributions, one-match-per-row, provenance hashes, and unknown-time preservation');
