'use strict';
const crypto = require('node:crypto');
const { canonical } = require('../model.js');
const SUM_TOLERANCE = 1e-9;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
function fail(message) { const error = new Error('PREDICTION_AUDIT_FAILED: ' + message); error.code = 'PREDICTION_AUDIT_FAILED'; throw error; }
function sha256(value) { return crypto.createHash('sha256').update(value).digest('hex'); }
function validDate(value) {
  if (typeof value !== 'string' || !DATE_RE.test(value)) return false;
  const parsed = new Date(value + 'T00:00:00.000Z');
  return Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}
function validateHistory(history, { generatedAt = new Date() } = {}) {
  if (!Array.isArray(history) || history.length === 0) fail('history must be a non-empty array');
  const generatedMs = new Date(generatedAt).getTime();
  if (!Number.isFinite(generatedMs)) fail('generatedAt is invalid');
  const tokyoDate = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Tokyo', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(generatedMs));
  const seen = new Set(); let latestFetchedAt = null;
  for (let i = 0; i < history.length; i++) {
    const row = history[i];
    if (!row || typeof row !== 'object') fail('history row ' + i + ' is not an object');
    if (!validDate(row.date)) fail('history row ' + i + ' has invalid date');
    if (row.date > tokyoDate) fail('history row ' + i + ' is future-dated relative to generatedAt');
    const home = String(row.home || '').trim(), away = String(row.away || '').trim();
    if (!home || !away || canonical(home) === canonical(away)) fail('history row ' + i + ' has missing or identical teams');
    if (!Number.isInteger(row.home_goals) || row.home_goals < 0 || !Number.isInteger(row.away_goals) || row.away_goals < 0) fail('history row ' + i + ' has invalid goal counts');
    const pair = [canonical(home), canonical(away)].sort();
    const key = [row.date, String(row.competition || '').trim(), ...pair].join('|');
    if (seen.has(key)) fail('duplicate fixture key detected: ' + key);
    seen.add(key);
    if (row.fetchedAt != null) {
      const fetchedMs = Date.parse(row.fetchedAt);
      if (!Number.isFinite(fetchedMs)) fail('history row ' + i + ' has invalid fetchedAt');
      if (fetchedMs > generatedMs) fail('history row ' + i + ' was fetched after generatedAt');
      if (latestFetchedAt === null || fetchedMs > latestFetchedAt) latestFetchedAt = fetchedMs;
    }
  }
  return { status: 'passed', rows: history.length, invalidRows: 0, duplicateFixtureKeys: 0,
    latestFetchedAt: latestFetchedAt === null ? null : new Date(latestFetchedAt).toISOString(), futureDatedRows: 0 };
}
function validateVector(values, label, expectedLength) {
  if (!Array.isArray(values) || values.length !== expectedLength) fail(label + ' must contain ' + expectedLength + ' values');
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) fail(label + '[' + i + '] is not a finite probability in [0,1]');
    sum += value;
  }
  const error = Math.abs(sum - 1);
  if (error > SUM_TOLERANCE) fail(label + ' does not sum to 1 (sum=' + sum + ')');
  return error;
}
function validatePredictionSet({ marketType, fixtures, predictions, generatedAt, deadlineAt }) {
  if (!['toto', 'totoGOAL3'].includes(marketType)) fail('unsupported marketType');
  if (!Array.isArray(fixtures) || !Array.isArray(predictions) || fixtures.length === 0 || fixtures.length !== predictions.length) fail('fixture/prediction count mismatch');
  if (marketType === 'totoGOAL3' && fixtures.length !== 3) fail('GOAL3 must contain exactly 3 fixtures');
  if (marketType === 'toto' && fixtures.length > 13) fail('toto fixture count exceeds 13');
  const generatedMs = Date.parse(generatedAt), deadlineMs = Date.parse(deadlineAt);
  if (!Number.isFinite(generatedMs) || !Number.isFinite(deadlineMs) || generatedMs >= deadlineMs) fail('generatedAt must be valid and earlier than deadlineAt');
  const seen = new Set(); let maxProbabilitySumError = 0;
  for (let i = 0; i < fixtures.length; i++) {
    const fixture = fixtures[i], prediction = predictions[i];
    if (!fixture || !prediction || fixture.number !== i + 1 || prediction.number !== fixture.number) fail('fixture numbering/order mismatch at index ' + i);
    if (!validDate(fixture.date) || !fixture.home || !fixture.away || canonical(fixture.home) === canonical(fixture.away)) fail('invalid fixture at index ' + i);
    const key = [fixture.date, canonical(fixture.home), canonical(fixture.away)].sort().join('|');
    if (seen.has(key)) fail('duplicate target fixture: ' + key);
    seen.add(key);
    const p = prediction.prediction;
    if (!p || typeof p !== 'object') fail('missing prediction object at fixture ' + fixture.number);
    maxProbabilitySumError = Math.max(maxProbabilitySumError, validateVector(
      [p.probabilities && p.probabilities['1'], p.probabilities && p.probabilities['0'], p.probabilities && p.probabilities['2']],
      '1X2 fixture ' + fixture.number, 3));
    if (marketType === 'totoGOAL3') {
      maxProbabilitySumError = Math.max(maxProbabilitySumError,
        validateVector(p.homeGoalProbabilities, 'GOAL3 home fixture ' + fixture.number, 4),
        validateVector(p.awayGoalProbabilities, 'GOAL3 away fixture ' + fixture.number, 4));
    }
    for (const side of ['homeExpectedGoals', 'awayExpectedGoals']) {
      if (typeof p[side] !== 'number' || !Number.isFinite(p[side]) || p[side] < 0) fail('invalid ' + side + ' at fixture ' + fixture.number);
    }
  }
  return { status: 'passed', fixtureCount: fixtures.length,
    distributionsChecked: marketType === 'totoGOAL3' ? fixtures.length * 3 : fixtures.length,
    maxProbabilitySumError, tolerance: SUM_TOLERANCE, generatedBeforeDeadline: true };
}
module.exports = { sha256, validateHistory, validatePredictionSet, SUM_TOLERANCE };
