'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { parseSchedule } = require('./auto_pipeline.js');
const { canonical } = require('../model.js');

const ROOT = path.resolve(__dirname, '..');
const OUTCOMES = ['1', '0', '2'];
const K = 6;
const HALF_LIVES = [3, 6, 9, 12, 18, 24];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const avg = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 1.25;
const outcome = m => m.home_goals > m.away_goals ? '1' : m.home_goals === m.away_goals ? '0' : '2';

function normalizeCompetition(value) {
  const raw = String(value || '').trim().toUpperCase();
  return /^[123]$/.test(raw) ? 'J' + raw : raw;
}

function loadCombinedHistory() {
  const current = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'history.json'), 'utf8')).history
    .filter(m => m.date && m.competition && Number.isFinite(m.home_goals) && Number.isFinite(m.away_goals));
  const targetStartDate = current.map(m => m.date).sort()[0];
  if (!targetStartDate) throw new Error('Current-season target history is empty');

  const historicalPath = path.join(ROOT, 'verification', 'goal3_independent', '2025.csv');
  const historicalParsed = parseSchedule(fs.readFileSync(historicalPath, 'utf8'));
  const historical = historicalParsed
    .filter(m => m.date && m.date.startsWith('2025-') &&
      Number.isInteger(m.home_goals) && Number.isInteger(m.away_goals) &&
      m.home_goals >= 0 && m.away_goals >= 0)
    .map(m => ({ ...m, competition: normalizeCompetition(m.competition), season: 2025 }));

  const currentNormalized = current.map(m => ({
    ...m, competition: normalizeCompetition(m.competition), season: Number(m.date.slice(0, 4))
  }));
  const combined = new Map();
  for (const row of [...historical, ...currentNormalized]) {
    if (!['J1', 'J2', 'J3'].includes(row.competition)) continue;
    const key = [row.date, row.competition, canonical(row.home), canonical(row.away)].join('|');
    combined.set(key, row);
  }
  const rows = [...combined.values()].sort((a, b) => a.date.localeCompare(b.date));
  const historicalRows = rows.filter(m => m.date.startsWith('2025-'));
  const targetRows = rows.filter(m => m.date >= targetStartDate);
  if (historicalRows.length < 1000) {
    throw new Error('Expected at least 1,000 2025 history rows; found ' + historicalRows.length);
  }
  if (targetRows.length < 100) {
    throw new Error('Expected at least 100 current-season target rows; found ' + targetRows.length);
  }
  return {
    history: rows,
    targetStartDate,
    historySources: {
      historical2025: { path: 'verification/goal3_independent/2025.csv', rows: historicalRows.length },
      current2026: { path: 'data/history.json', rows: currentNormalized.length }
    }
  };
}

function poisson(lambda, goals) {
  let p = Math.exp(-lambda);
  for (let i = 1; i <= goals; i++) p *= lambda / i;
  return p;
}

function probabilitiesFromLambdas(homeLambda, awayLambda) {
  let home = 0, draw = 0, away = 0;
  for (let h = 0; h <= 10; h++) {
    for (let a = 0; a <= 10; a++) {
      const p = poisson(homeLambda, h) * poisson(awayLambda, a);
      if (h > a) home += p;
      else if (h === a) draw += p;
      else away += p;
    }
  }
  const total = home + draw + away;
  return { '1': home / total, '0': draw / total, '2': away / total };
}

/**
 * V0: competition-only Poisson, separate home/away league rates, last 12
 * same-venue team matches, equal weights, fixed shrinkage k=6.
 * V1: same model plus exponential recency weights on those same 12 matches.
 */
function predictShrinkage(allHistory, fixture, options = {}) {
  const halfLife = options.halfLifeMatches ?? null;
  const k = Number.isFinite(options.k) ? options.k : K;
  if (halfLife !== null && !(halfLife > 0)) throw new RangeError('halfLifeMatches must be positive');
  const prior = allHistory.filter(m =>
    m.date < fixture.date &&
    normalizeCompetition(m.competition) === normalizeCompetition(fixture.competition) &&
    Number.isFinite(m.home_goals) &&
    Number.isFinite(m.away_goals)
  ).sort((a, b) => a.date.localeCompare(b.date));

  const leagueHome = clamp(avg(prior.map(m => m.home_goals)), 0.2, 3.5);
  const leagueAway = clamp(avg(prior.map(m => m.away_goals)), 0.2, 3.5);

  function teamRates(name, venue) {
    const target = canonical(name);
    const played = prior.filter(m =>
      venue === 'home' ? canonical(m.home) === target : canonical(m.away) === target
    ).slice(-12);
    const baseAttack = venue === 'home' ? leagueHome : leagueAway;
    const baseConcede = venue === 'home' ? leagueAway : leagueHome;
    let weightedGoalsFor = 0, weightedGoalsAgainst = 0, weightSum = 0;
    played.forEach((match, index) => {
      const age = played.length - 1 - index;
      const weight = halfLife === null ? 1 : Math.pow(0.5, age / halfLife);
      const goalsFor = venue === 'home' ? match.home_goals : match.away_goals;
      const goalsAgainst = venue === 'home' ? match.away_goals : match.home_goals;
      weightedGoalsFor += weight * goalsFor;
      weightedGoalsAgainst += weight * goalsAgainst;
      weightSum += weight;
    });
    const attackRate = (weightedGoalsFor + k * baseAttack) / (weightSum + k);
    const concedeRate = (weightedGoalsAgainst + k * baseConcede) / (weightSum + k);
    return {
      attack: attackRate / Math.max(0.2, baseAttack),
      defense: concedeRate / Math.max(0.2, baseConcede),
      matches: played.length,
      effectiveWeight: weightSum
    };
  }

  const home = teamRates(fixture.home, 'home');
  const away = teamRates(fixture.away, 'away');
  const homeLambda = clamp(leagueHome * home.attack * away.defense, 0.15, 3.5);
  const awayLambda = clamp(leagueAway * away.attack * home.defense, 0.15, 3.5);
  return {
    probabilities: probabilitiesFromLambdas(homeLambda, awayLambda),
    homeExpectedGoals: homeLambda,
    awayExpectedGoals: awayLambda,
    homeHistoryMatches: home.matches,
    awayHistoryMatches: away.matches,
    homeEffectiveWeight: home.effectiveWeight,
    awayEffectiveWeight: away.effectiveWeight,
    version: halfLife === null ? 'V0-shrinkage-k6' : 'V1-recency-shrinkage-k6',
    halfLifeMatches: halfLife
  };
}

function score(rows) {
  let correct = 0, brier = 0, logloss = 0;
  for (const row of rows) {
    const actual = outcome(row.match);
    const predicted = OUTCOMES.reduce((best, key) =>
      row.probabilities[key] > row.probabilities[best] ? key : best, OUTCOMES[0]);
    if (predicted === actual) correct++;
    for (const key of OUTCOMES) brier += (row.probabilities[key] - (key === actual ? 1 : 0)) ** 2;
    logloss -= Math.log(Math.max(1e-12, row.probabilities[actual]));
  }
  const n = rows.length;
  return { n, accuracy: n ? correct / n : null, brier: n ? brier / n : null, logloss: n ? logloss / n : null };
}

function pairedDateBlockBootstrap(candidateRows, baselineRows, replicates = 2000, seed = 1670) {
  const key = row => [row.match.date, normalizeCompetition(row.match.competition),
    canonical(row.match.home), canonical(row.match.away)].join('|');
  const baselineByKey = new Map(baselineRows.map(row => [key(row), row]));
  const groups = new Map();
  for (const candidate of candidateRows) {
    const baseline = baselineByKey.get(key(candidate));
    if (!baseline) throw new Error('Missing paired V0 row: ' + key(candidate));
    if (!groups.has(candidate.match.date)) groups.set(candidate.match.date, []);
    groups.get(candidate.match.date).push({ candidate, baseline });
  }
  const dates = [...groups.keys()].sort();
  const pairs = dates.flatMap(date => groups.get(date));
  const deltaFor = selected => {
    let accuracy = 0, brier = 0, logloss = 0;
    for (const pair of selected) {
      const actual = outcome(pair.candidate.match);
      const metrics = row => {
        const predicted = OUTCOMES.reduce((best, k) =>
          row.probabilities[k] > row.probabilities[best] ? k : best, OUTCOMES[0]);
        return {
          accuracy: predicted === actual ? 1 : 0,
          brier: OUTCOMES.reduce((sum, k) => sum + (row.probabilities[k] - (k === actual ? 1 : 0)) ** 2, 0),
          logloss: -Math.log(Math.max(1e-12, row.probabilities[actual]))
        };
      };
      const c = metrics(pair.candidate), b = metrics(pair.baseline);
      accuracy += c.accuracy - b.accuracy;
      brier += c.brier - b.brier;
      logloss += c.logloss - b.logloss;
    }
    const n = selected.length;
    return { accuracy: accuracy / n, brier: brier / n, logloss: logloss / n };
  };
  const point = deltaFor(pairs);
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const samples = { accuracy: [], brier: [], logloss: [] };
  for (let i = 0; i < replicates; i++) {
    const selected = [];
    for (let j = 0; j < dates.length; j++) {
      const date = dates[Math.floor(random() * dates.length)];
      selected.push(...groups.get(date));
    }
    const delta = deltaFor(selected);
    for (const metric of Object.keys(samples)) samples[metric].push(delta[metric]);
  }
  const quantile = (values, q) => {
    const sorted = values.slice().sort((a, b) => a - b);
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  };
  return {
    targetMatches: pairs.length,
    dateBlocks: dates.length,
    replicates,
    resamplingUnit: 'calendar date blocks',
    deltaV1MinusV0: point,
    confidenceInterval95: Object.fromEntries(Object.entries(samples).map(([metric, values]) =>
      [metric, [quantile(values, 0.025), quantile(values, 0.975)]]))
  };
}

function runExperiment(historyInput = null, options = {}) {
  const loaded = Array.isArray(historyInput) ? null : loadCombinedHistory();
  const allRows = (Array.isArray(historyInput) ? historyInput : loaded.history).filter(m =>
    m.date && m.competition && Number.isFinite(m.home_goals) && Number.isFinite(m.away_goals)
  ).slice().sort((a, b) => a.date.localeCompare(b.date));
  const targetStartDate = options.targetStartDate || loaded?.targetStartDate ||
    allRows.filter(m => m.date.startsWith('2026-')).map(m => m.date).sort()[0];
  if (!targetStartDate) throw new Error('No current-season target start date');
  const targets = allRows.filter(m => m.date >= targetStartDate);
  const dates = [...new Set(targets.map(m => m.date))].sort();
  if (dates.length < 10) throw new Error('Insufficient unique target dates for nested walk-forward');

  const boundaries = [
    Math.max(2, Math.floor(dates.length * 0.55)),
    Math.floor(dates.length * 0.70),
    Math.floor(dates.length * 0.85),
    dates.length
  ];
  boundaries[1] = Math.max(boundaries[1], boundaries[0] + 1);
  boundaries[2] = Math.max(boundaries[2], boundaries[1] + 1);
  boundaries[3] = dates.length;

  const aggregate = { v0: [], v1: [] };
  const folds = [];
  for (let foldIndex = 0; foldIndex < 3; foldIndex++) {
    const testStart = dates[boundaries[foldIndex]];
    const testEnd = boundaries[foldIndex + 1] < dates.length ? dates[boundaries[foldIndex + 1]] : null;
    const outerTrain = allRows.filter(m => m.date < testStart);
    const innerDates = [...new Set(outerTrain.map(m => m.date))].sort();
    const innerStart = innerDates[Math.max(1, Math.floor(innerDates.length * 0.70))];
    const innerValidation = outerTrain.filter(m => m.date >= innerStart);
    const candidateSettings = [
      { id: 'V0', halfLife: null, priority: 0 },
      ...HALF_LIVES.map((halfLife, index) => ({
        id: 'H' + halfLife,
        halfLife,
        priority: index + 1
      }))
    ];
    const innerScores = {};
    for (const candidate of candidateSettings) {
      const rows = innerValidation.map(match => ({
        match,
        probabilities: predictShrinkage(outerTrain, match, {
          k: K,
          halfLifeMatches: candidate.halfLife
        }).probabilities
      }));
      innerScores[candidate.id] = score(rows);
    }
    candidateSettings.sort((a, b) =>
      innerScores[a.id].brier - innerScores[b.id].brier ||
      innerScores[a.id].logloss - innerScores[b.id].logloss ||
      a.priority - b.priority
    );
    const selectedCandidate = candidateSettings[0];
    const selectedHalfLife = selectedCandidate.halfLife;

    const foldRows = { v0: [], v1: [] };
    const outerTest = targets.filter(m => m.date >= testStart && (!testEnd || m.date < testEnd));
    for (const match of outerTest) {
      const v0 = predictShrinkage(allRows, match, { k: K }).probabilities;
      const v1 = predictShrinkage(allRows, match, { k: K, halfLifeMatches: selectedHalfLife }).probabilities;
      foldRows.v0.push({ match, probabilities: v0 });
      foldRows.v1.push({ match, probabilities: v1 });
    }
    aggregate.v0.push(...foldRows.v0);
    aggregate.v1.push(...foldRows.v1);
    folds.push({
      fold: foldIndex + 1,
      trainEndDateInclusive: outerTrain[outerTrain.length - 1]?.date || null,
      testStartDate: testStart,
      testEndDateExclusive: testEnd,
      trainMatches: outerTrain.length,
      innerValidationStartDate: innerStart,
      innerValidationMatches: innerValidation.length,
      innerScores,
      selectedModel: selectedCandidate.id,\n      selectedHalfLifeMatches: selectedHalfLife,
      outerTestMatches: outerTest.length,
      v0: score(foldRows.v0),
      v1: score(foldRows.v1)
    });
  }

  return {
    generatedAt: new Date().toISOString(),
    protocol: '3 expanding-window outer folds over 2026 targets; training includes 2025 and strictly earlier 2026 matches; target date and later rows are excluded; inner chronological validation selects V1 half-life; paired metrics and date-block bootstrap',
    targetStartDate,
    historySources: options.historySources || loaded?.historySources || null,
    trainingHistoryMatches: allRows.filter(m => m.date < dates[0]).length,
    targetMatches: targets.length,
    v0Definition: 'competition-only Poisson; separate home/away league scoring baselines; last 12 same-venue team matches; equal weights; fixed shrinkage k=6',
    v1Definition: 'V0 plus exponential recency weights on the last 12 same-venue matches; half-life selected from 3/6/9/12 matches using inner validation Brier, LogLoss tie-break; shrinkage k=6 unchanged',
    candidates: ['V0', ...HALF_LIVES.map(value => 'H' + value)],
    metrics: { v0: score(aggregate.v0), v1: score(aggregate.v1) },
    pairedDateBlockBootstrap: pairedDateBlockBootstrap(aggregate.v1, aggregate.v0),
    folds
  };
}

module.exports = { loadCombinedHistory, predictShrinkage, score, pairedDateBlockBootstrap, runExperiment };

if (require.main === module) {
  console.log(JSON.stringify(runExperiment(), null, 2));
}
