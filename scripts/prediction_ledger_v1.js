'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const ROOT = path.resolve(__dirname, '..');
const sha256 = value => crypto.createHash('sha256').update(value).digest('hex');
const isFiniteNumber = value => typeof value === 'number' && Number.isFinite(value);
const iso = value => typeof value === 'string' && Number.isFinite(Date.parse(value));

function productName(marketType) {
  if (marketType === 'toto') return 'toto13';
  if (marketType === 'totoGOAL3') return 'totoGOAL3';
  if (marketType === 'miniToto-A' || marketType === 'miniToto-B') return marketType;
  throw new Error('Unsupported marketType: ' + String(marketType));
}

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

function assertVector(values, label) {
  if (!Array.isArray(values) || values.length !== 4 ||
      !values.every(value => isFiniteNumber(value) && value >= 0 && value <= 1)) {
    throw new Error(label + ' must contain four finite probabilities in [0, 1]');
  }
  if (Math.abs(sum(values) - 1) > 1e-8) {
    throw new Error(label + ' must sum to 1');
  }
}

function assertOneXTwo(probabilities) {
  if (!probabilities || !['1', '0', '2'].every(key =>
    isFiniteNumber(probabilities[key]) && probabilities[key] >= 0 && probabilities[key] <= 1)) {
    throw new Error('toto13 requires finite 1/X/2 probabilities in [0, 1]');
  }
  if (Math.abs(probabilities['1'] + probabilities['0'] + probabilities['2'] - 1) > 1e-8) {
    throw new Error('toto13 1/X/2 probabilities must sum to 1');
  }
}

function normalizeInputHashes(inputHashes = {}) {
  const keys = ['history', 'teamAliases', 'fixtures', 'modelCode', 'pipelineCode', 'adapterCode'];
  const normalized = {};
  for (const key of keys) {
    const value = inputHashes[key] ?? null;
    if (value !== null && !/^[a-f0-9]{64}$/.test(value)) {
      throw new Error('Invalid SHA-256 for input ' + key);
    }
    normalized[key] = value;
  }
  return normalized;
}

function toLedger(payload, options = {}) {
  if (!payload || !Array.isArray(payload.predictions) || !payload.predictions.length) {
    throw new Error('Source payload must contain at least one prediction');
  }
  if (!Number.isInteger(payload.round) || !iso(payload.generatedAt) || !iso(payload.deadline)) {
    throw new Error('Source payload is missing a valid round, generatedAt, or deadline');
  }

  const hashes = normalizeInputHashes(options.inputHashes);
  const knownHashes = Object.values(hashes).filter(Boolean);
  const inputsHash = knownHashes.length
    ? sha256(Buffer.from(JSON.stringify(hashes), 'utf8'))
    : null;
  const inputHashCompleteness = knownHashes.length === Object.keys(hashes).length ? 'complete' : 'partial';
  const predictionPayloadSha256 = options.predictionPayloadSha256 ?? null;
  if (predictionPayloadSha256 !== null && !/^[a-f0-9]{64}$/.test(predictionPayloadSha256)) {
    throw new Error('predictionPayloadSha256 must be a lowercase SHA-256 hex digest');
  }

  const seen = new Set();
  const records = payload.predictions.map(item => {
    const marketType = item.marketType || payload.totoSource?.marketType;
    const product = productName(marketType);
    const matchNumber = item.number ?? item.matchNumber;
    if (!Number.isInteger(matchNumber) || matchNumber < 1 || matchNumber > 13 ||
        !item.home || !item.away || item.home === item.away || !iso(item.date)) {
      throw new Error('Invalid fixture row at match number ' + String(matchNumber));
    }
    const uniqueKey = product + ':' + payload.round + ':' + matchNumber;
    if (seen.has(uniqueKey)) throw new Error('Duplicate prediction row: ' + uniqueKey);
    seen.add(uniqueKey);

    const prediction = item.prediction || {};
    const isGoal3 = product === 'totoGOAL3';
    if (isGoal3) {
      assertVector(prediction.homeGoalProbabilities, 'homeGoalProbabilities');
      assertVector(prediction.awayGoalProbabilities, 'awayGoalProbabilities');
    } else {
      assertOneXTwo(prediction.probabilities);
    }

    const expectedGoals = isFiniteNumber(prediction.homeExpectedGoals) &&
      isFiniteNumber(prediction.awayExpectedGoals)
      ? { home: prediction.homeExpectedGoals, away: prediction.awayExpectedGoals }
      : null;
    const predictionId = [
      'gpt', product, payload.round, 'm' + matchNumber,
      payload.generatedAt.replace(/[^0-9A-Za-z]/g, '')
    ].join('-');

    return {
      predictionId,
      round: payload.round,
      product,
      matchNumber,
      date: item.date,
      competition: item.competition ?? null,
      home: item.home,
      away: item.away,
      canonicalHome: prediction.canonicalHome ?? null,
      canonicalAway: prediction.canonicalAway ?? null,
      modelName: 'GPT',
      modelVersion: item.modelVersion ?? null,
      modelVersionStatus: item.modelVersion ? 'recorded' : 'not_recorded_in_source',
      modelMethod: prediction.method ?? null,
      probabilities: isGoal3 ? null : {
        home: prediction.probabilities['1'],
        draw: prediction.probabilities['0'],
        away: prediction.probabilities['2']
      },
      goalDistributions: isGoal3 ? {
        home: [...prediction.homeGoalProbabilities],
        away: [...prediction.awayGoalProbabilities]
      } : null,
      expectedGoals,
      generatedAt: payload.generatedAt,
      confirmedAt: item.confirmedAt ?? null,
      confirmedAtStatus: item.confirmedAt ? 'recorded' : 'not_recorded_in_source',
      deadlineAt: payload.deadline,
      fixtureSourceRetrievedAt: payload.totoSource?.lastFetchedAt ?? null,
      sourceUrl: item.sourceUrl ?? payload.totoSource?.url ?? null,
      marketSnapshotId: item.marketSnapshotId ?? null,
      marketSnapshotStatus: item.marketSnapshotId ? 'linked' : 'not_included_in_source_payload',
      inputHashes: hashes,
      inputsHash,
      inputHashCompleteness,
      predictionPayloadSha256,
      availabilityNote: item.availabilityNote ?? null
    };
  });

  return {
    schemaVersion: 1,
    exportedAt: options.exportedAt || new Date().toISOString(),
    sourceRepo: 'sigulun555-lab/gpt',
    sourcePath: 'data/predictions.json',
    records
  };
}

function loadHash(relativePath) {
  const absolutePath = path.join(ROOT, relativePath);
  return sha256(fs.readFileSync(absolutePath));
}

function appendLedgerRecords(outputPath, records) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const existing = new Map();
  if (fs.existsSync(outputPath)) {
    const lines = fs.readFileSync(outputPath, 'utf8').split(/\\r?\\n/).filter(Boolean);
    for (const line of lines) {
      const record = JSON.parse(line);
      if (!record.predictionId || existing.has(record.predictionId)) {
        throw new Error('Existing ledger contains a missing or duplicate predictionId');
      }
      existing.set(record.predictionId, record);
    }
  }

  const append = [];
  let skipped = 0;
  for (const record of records) {
    const prior = existing.get(record.predictionId);
    if (prior) {
      if (JSON.stringify(prior) !== JSON.stringify(record)) {
        throw new Error('Append-only conflict for predictionId ' + record.predictionId +
          '; create an explicit revision instead of overwriting an existing row');
      }
      skipped++;
      continue;
    }
    existing.set(record.predictionId, record);
    append.push(JSON.stringify(record));
  }
  if (append.length) fs.appendFileSync(outputPath, append.join('\\n') + '\\n', 'utf8');
  return { appended: append.length, skipped, total: existing.size };
}

function main() {
  const sourcePath = path.join(ROOT, 'data', 'predictions.json');
  const sourceBytes = fs.readFileSync(sourcePath);
  const payload = JSON.parse(sourceBytes.toString('utf8'));
  const inputHashes = {
    history: loadHash('data/history.json'),
    teamAliases: loadHash('data/team_aliases.json'),
    fixtures: loadHash('data/fixtures.json'),
    modelCode: loadHash('model.js'),
    pipelineCode: loadHash('scripts/auto_pipeline.js'),
    adapterCode: loadHash('scripts/prediction_ledger_v1.js')
  };
  const ledger = toLedger(payload, {
    inputHashes,
    predictionPayloadSha256: sha256(sourceBytes)
  });
  const outputPath = path.join(ROOT, 'data', 'exports', 'prediction_ledger_v1.json');
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(ledger, null, 2) + '\n');
  console.log(JSON.stringify({
    output: path.relative(ROOT, outputPath),
    round: payload.round,
    product: ledger.records[0].product,
    records: ledger.records.length,
    inputHashCompleteness: ledger.records[0].inputHashCompleteness,
    confirmedAtRecorded: ledger.records.every(record => record.confirmedAtStatus === 'recorded'),
    marketSnapshotsLinked: ledger.records.filter(record => record.marketSnapshotId).length
  }, null, 2));
}

if (require.main === module) main();

module.exports = { sha256, toLedger, appendLedgerRecords };
