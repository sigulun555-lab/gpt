'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { predict, canonical } = require('../model.js');

const ROOT = path.resolve(__dirname, '..');
const history = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'history.json'), 'utf8')).history;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const avg = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 1.25;
const KEYS = ['1', '0', '2'];
const outcome = m => m.home_goals > m.away_goals ? '1' : m.home_goals === m.away_goals ? '0' : '2';

function poisson(lam, k) {
  let p = Math.exp(-lam);
  for (let i = 1; i <= k; i++) p *= lam / i;
  return p;
}
function outcomeProbabilities(homeLambda, awayLambda) {
  let home = 0, draw = 0, away = 0;
  for (let i = 0; i <= 10; i++) for (let j = 0; j <= 10; j++) {
    const p = poisson(homeLambda, i) * poisson(awayLambda, j);
    if (i > j) home += p; else if (i === j) draw += p; else away += p;
  }
  const total = home + draw + away;
  return { '1': home / total, '0': draw / total, '2': away / total };
}

/*
 * Ablation factors are switched on cumulatively:
 * competitionOnly: use only the target competition's past matches.
 * splitLeagueBaselines: normalize home attack/defense against home/away league rates.
 * sideSpecificStats: estimate team rates from same-venue matches among each team's last 12.
 * shrinkageK: shrink those same-venue goal rates toward the corresponding league rate.
 */
function predictAblation(allHistory, fixture, config = {}) {
  const prior = allHistory.filter(m =>
    m.date < fixture.date &&
    (!config.competitionOnly || m.competition === fixture.competition) &&
    Number.isFinite(m.home_goals) && Number.isFinite(m.away_goals)
  );
  const leagueHome = clamp(avg(prior.map(m => m.home_goals)), 0.2, 3.5);
  const leagueAway = clamp(avg(prior.map(m => m.away_goals)), 0.2, 3.5);
  const sharedBase = Math.max(0.2, (leagueHome + leagueAway) / 2);
  const stats = (name, side) => {
    const target = canonical(name);
    const played = prior.filter(m => canonical(m.home) === target || canonical(m.away) === target).slice(-12);
    const selected = config.sideSpecificStats
      ? played.filter(m => side === 'home' ? canonical(m.home) === target : canonical(m.away) === target)
      : played;
    const gf = selected.map(m => canonical(m.home) === target ? m.home_goals : m.away_goals);
    const ga = selected.map(m => canonical(m.home) === target ? m.away_goals : m.home_goals);
    return { gf: avg(gf), ga: avg(ga), n: selected.length };
  };
  const rates = (name, side) => {
    const s = stats(name, side);
    const attackBase = config.splitLeagueBaselines ? (side === 'home' ? leagueHome : leagueAway) : sharedBase;
    const defenseBase = config.splitLeagueBaselines ? (side === 'home' ? leagueAway : leagueHome) : sharedBase;
    const k = Number.isFinite(config.shrinkageK) ? config.shrinkageK : 0;
    const denominator = s.n + k;
    const gf = denominator > 0 ? (s.gf * s.n + attackBase * k) / denominator : attackBase;
    const ga = denominator > 0 ? (s.ga * s.n + defenseBase * k) / denominator : defenseBase;
    return { attack: gf / Math.max(0.2, attackBase), defense: ga / Math.max(0.2, defenseBase), n: s.n };
  };
  const h = rates(fixture.home, 'home');
  const a = rates(fixture.away, 'away');
  const homeLambda = clamp(leagueHome * h.attack * a.defense, 0.15, 3.5);
  const awayLambda = clamp(leagueAway * a.attack * h.defense, 0.15, 3.5);
  return { probabilities: outcomeProbabilities(homeLambda, awayLambda), homeLambda, awayLambda,
    historyHomeMatches: h.n, historyAwayMatches: a.n };
}

const variants = [
  { id: 'productionBaseline', label: 'Current production baseline', production: true },
  { id: 'competitionOnly', label: 'Competition filter only', config: { competitionOnly: true } },
  { id: 'splitLeagueBaselines', label: '+ split home/away league baselines', config: { competitionOnly: true, splitLeagueBaselines: true } },
  { id: 'sideSpecificStats', label: '+ same-venue team stats', config: { competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true } },
  { id: 'shrinkageK3', label: '+ shrinkage k=3', config: { competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: 3 } },
  { id: 'shrinkageK6', label: '+ shrinkage k=6', config: { competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: 6 } },
  { id: 'shrinkageK10', label: '+ shrinkage k=10', config: { competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: 10 } },
];

function score(rows) {
  let brier = 0, logloss = 0, correct = 0;
  for (const row of rows) {
    const actual = outcome(row.match);
    for (const key of KEYS) brier += (row.probabilities[key] - (key === actual ? 1 : 0)) ** 2;
    logloss -= Math.log(Math.max(1e-12, row.probabilities[actual]));
    if (KEYS.reduce((best, key) => row.probabilities[key] > row.probabilities[best] ? key : best, KEYS[0]) === actual) correct++;
  }
  const n = rows.length;
  return { n, accuracy: n ? correct / n : null, brier: n ? brier / n : null, logloss: n ? logloss / n : null };
}



function pairedDateBlockBootstrap(candidateRows, baselineRows, replicates = 2000, seed = 1668) {
  const key = row => [
    row.match.date, row.match.competition, canonical(row.match.home), canonical(row.match.away)
  ].join('|');
  const baselineByKey = new Map(baselineRows.map(row => [key(row), row]));
  const groups = new Map();
  for (const candidate of candidateRows) {
    const baseline = baselineByKey.get(key(candidate));
    if (!baseline) throw new Error('paired bootstrap missing baseline row for ' + key(candidate));
    const date = candidate.match.date;
    if (!groups.has(date)) groups.set(date, []);
    groups.get(date).push({ candidate, baseline });
  }
  const dates = [...groups.keys()].sort();
  const pairs = dates.flatMap(date => groups.get(date));
  const differences = selected => {
    let accuracy = 0, brier = 0, logloss = 0;
    for (const pair of selected) {
      const actual = outcome(pair.candidate.match);
      const metrics = row => {
        const predicted = KEYS.reduce((best, k) => row.probabilities[k] > row.probabilities[best] ? k : best, KEYS[0]);
        return {
          accuracy: predicted === actual ? 1 : 0,
          brier: KEYS.reduce((sum, k) => sum + (row.probabilities[k] - (k === actual ? 1 : 0)) ** 2, 0),
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
  const estimate = differences(pairs);
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
    const delta = differences(selected);
    for (const metric of Object.keys(samples)) samples[metric].push(delta[metric]);
  }
  const quantile = (values, q) => {
    const sorted = values.slice().sort((a, b) => a - b);
    const position = (sorted.length - 1) * q;
    const lower = Math.floor(position), upper = Math.ceil(position);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
  };
  return {
    targetMatches: pairs.length,
    dateBlocks: dates.length,
    replicates,
    resamplingUnit: 'calendar date blocks; all matches on a date stay together',
    deltaCandidateMinusBaseline: estimate,
    confidenceInterval95: Object.fromEntries(Object.entries(samples).map(([metric, values]) => [
      metric, [quantile(values, 0.025), quantile(values, 0.975)]
    ]))
  };
}

function compareByCompetition(candidateRows, baselineRows) {
  const competitions = [...new Set(baselineRows.map(row => row.match.competition))].sort();
  return Object.fromEntries(competitions.map(competition => {
    const baseline = baselineRows.filter(row => row.match.competition === competition);
    const candidate = candidateRows.filter(row => row.match.competition === competition);
    const b = score(baseline), c = score(candidate);
    return [competition, {
      targetMatches: baseline.length,
      baseline: b,
      candidate: c,
      deltaCandidateMinusBaseline: {
        accuracy: c.accuracy - b.accuracy,
        brier: c.brier - b.brier,
        logloss: c.logloss - b.logloss
      }
    }];
  }));
}

function runNestedKSelection(historyInput, targetInput = historyInput) {
  const sorted = historyInput.slice().sort((a, b) => a.date.localeCompare(b.date));
  const targets = targetInput.slice().sort((a, b) => a.date.localeCompare(b.date));
  const dates = [...new Set(targets.map(m => m.date))].sort();
  const boundaries = [
    Math.max(2, Math.floor(dates.length * 0.55)),
    Math.floor(dates.length * 0.70),
    Math.floor(dates.length * 0.85),
    dates.length
  ];
  boundaries[1] = Math.max(boundaries[1], boundaries[0] + 1);
  boundaries[2] = Math.max(boundaries[2], boundaries[1] + 1);
  boundaries[3] = dates.length;
  const folds = [];
  const aggregate = { productionBaseline: [], nestedSelectedK: [], nestedLeagueSelected: [], fixedK3: [], fixedK6: [], fixedK10: [] };
  for (let foldIndex = 0; foldIndex < 3; foldIndex++) {
    const startDate = dates[boundaries[foldIndex]];
    const endDate = boundaries[foldIndex + 1] < dates.length ? dates[boundaries[foldIndex + 1]] : null;
    const outerTrain = sorted.filter(m => m.date < startDate);
    const outerTest = targets.filter(m => m.date >= startDate && (!endDate || m.date < endDate));
    const innerDates = [...new Set(outerTrain.map(m => m.date))].sort();
    const innerStartDate = innerDates[Math.max(1, Math.floor(innerDates.length * 0.7))];
    const innerValidation = outerTrain.filter(m => m.date >= innerStartDate);
    const innerScores = {};
    for (const k of [3, 6, 10]) {
      const rows = innerValidation.map(match => ({
        match,
        probabilities: predictAblation(outerTrain, match, {
          competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: k
        }).probabilities
      }));
      innerScores[k] = score(rows);
    }
    const selectedK = [3, 6, 10].sort((a, b) =>
      innerScores[a].brier - innerScores[b].brier ||
      innerScores[a].logloss - innerScores[b].logloss ||
      a - b
    )[0];
    const innerSelectionByCompetition = {};
    const selectedByCompetition = {};
    const competitions = [...new Set(outerTrain.map(m => m.competition))].sort();
    for (const competition of competitions) {
      const validation = innerValidation.filter(m => m.competition === competition);
      if (!validation.length) continue;
      const candidateRows = [{ id: 'productionBaseline', k: null, rows: [] },
        ...[3, 6, 10].map(k => ({ id: 'shrinkageK' + k, k, rows: [] }))];
      for (const match of validation) {
        const prior = outerTrain.filter(m => m.date < match.date);
        if (!prior.length) continue;
        candidateRows[0].rows.push({
          match,
          probabilities: predict(prior, { date: match.date, home: match.home, away: match.away }).probabilities
        });
        for (const candidate of candidateRows.slice(1)) {
          candidate.rows.push({
            match,
            probabilities: predictAblation(outerTrain, match, {
              competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: candidate.k
            }).probabilities
          });
        }
      }
      const scored = candidateRows.map(candidate => ({ ...candidate, metrics: score(candidate.rows) }));
      scored.sort((a, b) =>
        a.metrics.brier - b.metrics.brier ||
        a.metrics.logloss - b.metrics.logloss ||
        (a.k ?? 0) - (b.k ?? 0)
      );
      const selected = scored[0];
      selectedByCompetition[competition] = { model: selected.id, k: selected.k };
      innerSelectionByCompetition[competition] = {
        selectedModel: selected.id,
        selectedK: selected.k,
        innerMetrics: selected.metrics,
        candidates: Object.fromEntries(scored.map(candidate => [candidate.id, candidate.metrics]))
      };
    }
    const foldRows = { productionBaseline: [], nestedSelectedK: [], nestedLeagueSelected: [], fixedK3: [], fixedK6: [], fixedK10: [] };
    for (const match of outerTest) {
      const prior = sorted.filter(m => m.date < match.date);
      if (!prior.length) continue;
      const baseline = predict(prior, { date: match.date, home: match.home, away: match.away }).probabilities;
      foldRows.productionBaseline.push({ match, probabilities: baseline });
      for (const k of [3, 6, 10]) {
        const probabilities = predictAblation(sorted, match, {
          competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: k
        }).probabilities;
        foldRows[`fixedK${k}`].push({ match, probabilities });
      }
      foldRows.nestedSelectedK.push({
        match,
        probabilities: predictAblation(sorted, match, {
          competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: selectedK
        }).probabilities
      });
      const leagueSelection = selectedByCompetition[match.competition] || { model: 'productionBaseline', k: null };
      const leagueProbabilities = leagueSelection.model === 'productionBaseline'
        ? baseline
        : predictAblation(sorted, match, {
          competitionOnly: true, splitLeagueBaselines: true, sideSpecificStats: true, shrinkageK: leagueSelection.k
        }).probabilities;
      foldRows.nestedLeagueSelected.push({ match, probabilities: leagueProbabilities });
    }
    for (const key of Object.keys(aggregate)) aggregate[key].push(...foldRows[key]);
    folds.push({
      fold: foldIndex + 1,
      trainEndDateInclusive: outerTrain[outerTrain.length - 1].date,
      testStartDate: startDate,
      testEndDateExclusive: endDate,
      trainMatches: outerTrain.length,
      innerValidationStartDate: innerStartDate,
      innerValidationMatches: innerValidation.length,
      innerScores,
      selectedK,
      innerSelectionByCompetition,
      selectedByCompetition,
      testMatches: outerTest.length,
      outerMetrics: Object.fromEntries(Object.entries(foldRows).map(([key, rows]) => [key, score(rows)]))
    });
  }
  const aggregateOuterMetrics = Object.fromEntries(Object.entries(aggregate).map(([key, rows]) => [key, score(rows)]));
  const pairedComparisons = {
    nestedSelectedK: pairedDateBlockBootstrap(aggregate.nestedSelectedK, aggregate.productionBaseline),
    nestedLeagueSelected: pairedDateBlockBootstrap(aggregate.nestedLeagueSelected, aggregate.productionBaseline),
    fixedK10: pairedDateBlockBootstrap(aggregate.fixedK10, aggregate.productionBaseline)
  };
  const perCompetition = {
    nestedSelectedK: compareByCompetition(aggregate.nestedSelectedK, aggregate.productionBaseline),
    nestedLeagueSelected: compareByCompetition(aggregate.nestedLeagueSelected, aggregate.productionBaseline),
    fixedK10: compareByCompetition(aggregate.fixedK10, aggregate.productionBaseline)
  };
  return {
    method: 'three expanding-window outer folds; choose k using only prior outer-training data, inner chronological validation, Brier primary and LogLoss tie-break; same-date matches remain together',
    candidates: [3, 6, 10],
    aggregateOuterMetrics,
    pairedComparisons,
    perCompetition,
    folds
  };
}

function runAblation(historyInput = history, options = {}) {
  const sorted = historyInput.filter(m => m.date && m.competition && Number.isFinite(m.home_goals) && Number.isFinite(m.away_goals))
    .slice().sort((a, b) => a.date.localeCompare(b.date));
  const targetStartDate = options.targetStartDate || sorted[0]?.date;
  const targets = sorted.filter(m => m.date >= targetStartDate);
  const scored = new Map(variants.map(v => [v.id, []]));
  const dates = [...new Set(targets.map(m => m.date))].sort();
  // Keep every match on a date in the same split; choose the cutoff closest to 20% of target rows.
  const targetHoldoutCount = Math.max(1, Math.round(targets.length * 0.2));
  let holdoutStartDate = dates[1] || dates[0];
  let bestHoldoutDistance = Infinity;
  for (const date of dates.slice(1)) {
    const holdoutCount = targets.filter(m => m.date >= date).length;
    const distance = Math.abs(holdoutCount - targetHoldoutCount);
    if (distance < bestHoldoutDistance) {
      holdoutStartDate = date;
      bestHoldoutDistance = distance;
    }
  }
  for (const match of targets) {
    const prior = sorted.filter(m => m.date < match.date);
    if (!prior.length) continue;
    for (const variant of variants) {
      const probabilities = variant.production
        ? predict(prior, { date: match.date, home: match.home, away: match.away }).probabilities
        : predictAblation(sorted, { date: match.date, home: match.home, away: match.away, competition: match.competition }, variant.config).probabilities;
      scored.get(variant.id).push({ match, probabilities, isHoldout: match.date >= holdoutStartDate });
    }
  }
  const metrics = {};
  for (const variant of variants) {
    const rows = scored.get(variant.id);
    metrics[variant.id] = {
      label: variant.label,
      all: score(rows),
      chronologicalHoldout: score(rows.filter(r => r.isHoldout)),
      holdoutStartDate,
      caveat: 'Exploratory only. The final chronological holdout is chosen to approximate 20% of target rows while keeping same-date matches together; this is a single target period and does not establish generalization.'
    };
  }
  return {
    generatedAt: new Date().toISOString(),
    source: options.source || 'data/history.json; community J.League CSV mirror, not official results feed',
    historyMatches: sorted.length,
    targetMatches: targets.length,
    targetStartDate,
    holdoutStartDate,
    factorOrder: ['competitionOnly', 'splitLeagueBaselines', 'sideSpecificStats', 'shrinkageK'],
    variants: metrics,
    nestedKSelection: runNestedKSelection(sorted, targets)
  };
}

if (require.main === module) process.stdout.write(JSON.stringify(runAblation(), null, 2) + '\n');
module.exports = { predictAblation, runAblation, variants };
