'use strict';

const crypto = require('node:crypto');

const VALID_COMPETITIONS = new Set(['J1', 'J2', 'J3']);
const DEFAULT_TOLERANCE = 1e-9;

function isIsoDate(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(value + 'T00:00:00.000Z');
  return Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}

function isIsoDateTime(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/.test(value)) return false;
  return Number.isFinite(Date.parse(value));
}

function tokyoDate(value) {
  const date = new Date(value);
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Tokyo',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  }).formatToParts(date);
  const map = Object.fromEntries(parts.map(part => [part.type, part.value]));
  return map.year + '-' + map.month + '-' + map.day;
}

function sha256(text) {
  return crypto.createHash('sha256').update(String(text), 'utf8').digest('hex');
}

function createInputProvenance(scheduleText, totoHtml, generatedAt, deadlineAt, scheduleUrl, totoUrl) {
  return {
    schemaVersion: 1,
    generatedAt,
    deadlineAt: deadlineAt || null,
    scheduleInput: {
      url: scheduleUrl,
      fetchedAt: generatedAt,
      sha256: sha256(scheduleText)
    },
    officialTotoInput: {
      url: totoUrl,
      fetchedAt: generatedAt,
      sha256: sha256(totoHtml)
    }
  };
}

function validateHistory(history, generatedAt) {
  const errors = [];
  if (!Array.isArray(history) || history.length === 0) {
    return ['history must be a non-empty array'];
  }
  if (!isIsoDateTime(generatedAt)) errors.push('generatedAt must be an ISO-8601 timestamp with timezone');
  const generatedMs = Date.parse(generatedAt);
  const latestTokyoDate = isIsoDateTime(generatedAt) ? tokyoDate(generatedAt) : null;
  const seen = new Set();
  let previousDate = '';

  history.forEach((row, index) => {
    const prefix = 'history[' + index + ']';
    if (!row || typeof row !== 'object' || Array.isArray(row)) {
      errors.push(prefix + ' must be an object');
      return;
    }
    if (!isIsoDate(row.date)) errors.push(prefix + '.date must be a real YYYY-MM-DD date');
    if (!VALID_COMPETITIONS.has(row.competition)) errors.push(prefix + '.competition must be J1, J2, or J3');
    if (typeof row.home !== 'string' || !row.home.trim()) errors.push(prefix + '.home is missing');
    if (typeof row.away !== 'string' || !row.away.trim()) errors.push(prefix + '.away is missing');
    if (row.home && row.away && row.home.trim() === row.away.trim()) errors.push(prefix + ' has identical home and away teams');
    if (!Number.isInteger(row.home_goals) || row.home_goals < 0) errors.push(prefix + '.home_goals must be a non-negative integer');
    if (!Number.isInteger(row.away_goals) || row.away_goals < 0) errors.push(prefix + '.away_goals must be a non-negative integer');
    if (!isIsoDateTime(row.fetchedAt)) {
      errors.push(prefix + '.fetchedAt must be an ISO-8601 timestamp with timezone');
    } else if (isIsoDateTime(generatedAt) && Date.parse(row.fetchedAt) > generatedMs) {
      errors.push(prefix + '.fetchedAt is later than generatedAt');
    }
    if (row.source !== 'community_jleague_csv') errors.push(prefix + '.source must identify the community schedule source');
    if (typeof row.sourceUrl !== 'string' || !/^https:\/\//.test(row.sourceUrl)) errors.push(prefix + '.sourceUrl must be an https URL');

    if (isIsoDate(row.date)) {
      if (latestTokyoDate && row.date > latestTokyoDate) errors.push(prefix + '.date is after the generation date in Asia/Tokyo');
      if (previousDate && row.date < previousDate) errors.push(prefix + ' is not in chronological order');
      previousDate = row.date;
    }
    const key = [row.date, row.competition, row.home, row.away].join('|');
    if (seen.has(key)) errors.push(prefix + ' duplicates match key ' + key);
    seen.add(key);
  });

  return errors;
}

function validateProbabilityVector(vector, expectedLength, label, tolerance = DEFAULT_TOLERANCE) {
  const errors = [];
  if (!Array.isArray(vector) || vector.length !== expectedLength) {
    return [label + ' must contain exactly ' + expectedLength + ' probabilities'];
  }
  let total = 0;
  vector.forEach((value, index) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      errors.push(label + '[' + index + '] must be finite');
    } else if (value < 0 || value > 1) {
      errors.push(label + '[' + index + '] must be within [0, 1]');
    } else {
      total += value;
    }
  });
  if (errors.length === 0 && Math.abs(total - 1) > tolerance) {
    errors.push(label + ' must sum to 1 within tolerance ' + tolerance + ' (actual ' + total + ')');
  }
  return errors;
}

function validatePredictions(predictions, fixtures, marketType, round) {
  const errors = [];
  if (!Array.isArray(predictions) || !Array.isArray(fixtures) || predictions.length !== fixtures.length) {
    return ['prediction count must match the confirmed fixture count'];
  }
  if (marketType === 'totoGOAL3' && fixtures.length !== 3) errors.push('totoGOAL3 must contain exactly 3 fixtures');
  if (marketType === 'toto' && (fixtures.length < 1 || fixtures.length > 13)) errors.push('toto fixture count must be between 1 and 13');

  const seenNumbers = new Set();
  predictions.forEach((item, index) => {
    const fixture = fixtures[index] || {};
    const prefix = 'predictions[' + index + ']';
    if (!item || typeof item !== 'object') {
      errors.push(prefix + ' must be an object');
      return;
    }
    if (!Number.isInteger(item.number) || item.number < 1 || seenNumbers.has(item.number)) {
      errors.push(prefix + '.number must be a unique positive integer');
    }
    seenNumbers.add(item.number);
    if (item.number !== fixture.number || item.home !== fixture.home || item.away !== fixture.away || item.date !== fixture.date) {
      errors.push(prefix + ' does not match its confirmed fixture');
    }
    if (item.totoRound !== round || item.totoTargetConfirmed !== true) {
      errors.push(prefix + ' must carry the confirmed round and target flag');
    }
    const prediction = item.prediction || {};
    const p = prediction.probabilities;
    if (!p || typeof p !== 'object' || Array.isArray(p) || Object.keys(p).sort().join(',') !== '0,1,2') {
      errors.push(prefix + '.prediction.probabilities must have exactly keys 0, 1, and 2');
    } else {
      errors.push(...validateProbabilityVector([p['1'], p['0'], p['2']], 3, prefix + '.prediction.probabilities'));
    }
    errors.push(...validateProbabilityVector(prediction.homeGoalProbabilities, 4, prefix + '.prediction.homeGoalProbabilities'));
    errors.push(...validateProbabilityVector(prediction.awayGoalProbabilities, 4, prefix + '.prediction.awayGoalProbabilities'));
    for (const key of ['homeExpectedGoals', 'awayExpectedGoals']) {
      if (typeof prediction[key] !== 'number' || !Number.isFinite(prediction[key]) || prediction[key] <= 0) {
        errors.push(prefix + '.prediction.' + key + ' must be a finite positive number');
      }
    }
  });
  return errors;
}

function validateProvenance(provenance, generatedAt, deadlineAt, requireDeadline) {
  const errors = [];
  if (!provenance || typeof provenance !== 'object') return ['provenance is required'];
  if (provenance.schemaVersion !== 1) errors.push('provenance.schemaVersion must be 1');
  if (provenance.generatedAt !== generatedAt) errors.push('provenance.generatedAt must equal payload generatedAt');
  if (!isIsoDateTime(generatedAt)) errors.push('generatedAt must be an ISO-8601 timestamp with timezone');
  if (requireDeadline) {
    if (!isIsoDateTime(deadlineAt)) errors.push('deadlineAt must be an ISO-8601 timestamp with timezone');
    else if (Date.parse(deadlineAt) <= Date.parse(generatedAt)) errors.push('generatedAt must be earlier than deadlineAt');
    if (provenance.deadlineAt !== deadlineAt) errors.push('provenance.deadlineAt must equal the confirmed deadline');
  }
  for (const [key, label] of [['scheduleInput', 'scheduleInput'], ['officialTotoInput', 'officialTotoInput']]) {
    const input = provenance[key];
    if (!input || typeof input !== 'object') {
      errors.push('provenance.' + label + ' is required');
      continue;
    }
    if (typeof input.url !== 'string' || !/^https:\/\//.test(input.url)) errors.push('provenance.' + label + '.url must be an https URL');
    if (!isIsoDateTime(input.fetchedAt) || Date.parse(input.fetchedAt) > Date.parse(generatedAt)) {
      errors.push('provenance.' + label + '.fetchedAt must be valid and not later than generatedAt');
    }
    if (typeof input.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(input.sha256)) {
      errors.push('provenance.' + label + '.sha256 must be a lowercase 64-character SHA-256');
    }
  }
  return errors;
}

function assertBuildIntegrity({ history, predictions, fixtures, marketType, round, generatedAt, deadlineAt, provenance, confirmed }) {
  const errors = [
    ...validateHistory(history, generatedAt),
    ...validateProvenance(provenance, generatedAt, deadlineAt, confirmed)
  ];
  if (confirmed) errors.push(...validatePredictions(predictions, fixtures, marketType, round));
  if (errors.length) throw new Error('Prediction integrity validation failed: ' + errors.join('; '));
  return true;
}

module.exports = {
  isIsoDate,
  isIsoDateTime,
  sha256,
  createInputProvenance,
  validateHistory,
  validateProbabilityVector,
  validatePredictions,
  validateProvenance,
  assertBuildIntegrity
};
