'use strict';
const {
  loadCombinedHistory, predictShrinkage, score, pairedDateBlockBootstrap
} = require('./experiment_toto_v1_recency.js');

const BASELINE_K = 6;
const RHO_VALUES = [-0.08, -0.04, 0, 0.04, 0.08];
const KEYS = ['1', '0', '2'];

function poisson(lambda, goals) {
  let p = Math.exp(-lambda);
  for (let i = 1; i <= goals; i++) p *= lambda / i;
  return p;
}

function dixonColesProbabilities(homeLambda, awayLambda, rho) {
  let home = 0, draw = 0, away = 0;
  for (let h = 0; h <= 10; h++) {
    for (let a = 0; a <= 10; a++) {
      let tau = 1;
      if (h === 0 && a === 0) tau = 1 - homeLambda * awayLambda * rho;
      else if (h === 0 && a === 1) tau = 1 + homeLambda * rho;
      else if (h === 1 && a === 0) tau = 1 + awayLambda * rho;
      else if (h === 1 && a === 1) tau = 1 - rho;
      if (!(tau > 0)) throw new RangeError('Dixon-Coles correction produced a non-positive low-score factor');
      const p = poisson(homeLambda, h) * poisson(awayLambda, a) * tau;
      if (h > a) home += p;
      else if (h === a) draw += p;
      else away += p;
    }
  }
  const total = home + draw + away;
  return { '1': home / total, '0': draw / total, '2': away / total };
}

function runExperiment() {
  const loaded = loadCombinedHistory();
  const allRows = loaded.history;
  const targets = allRows.filter(m => m.date >= loaded.targetStartDate);
  const dates = [...new Set(targets.map(m => m.date))].sort();
  if (dates.length < 10) throw new Error('Insufficient unique target dates for nested walk-forward');

  const boundaries = [
    0,
    Math.max(1, Math.floor(dates.length / 3)),
    Math.max(2, Math.floor((dates.length * 2) / 3)),
    dates.length
  ];
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

    const innerBaseRows = innerValidation.map(match => {
      const base = predictShrinkage(outerTrain, match, { k: BASELINE_K });
      return { match, homeLambda: base.homeExpectedGoals, awayLambda: base.awayExpectedGoals };
    });
    const innerScores = {};
    for (const rho of RHO_VALUES) {
      innerScores[rho] = score(innerBaseRows.map(row => ({
        match: row.match,
        probabilities: dixonColesProbabilities(row.homeLambda, row.awayLambda, rho)
      })));
    }
    const selectedRho = RHO_VALUES.slice().sort((a, b) =>
      innerScores[a].brier - innerScores[b].brier ||
      innerScores[a].logloss - innerScores[b].logloss ||
      (a === 0 ? -1 : b === 0 ? 1 : Math.abs(a) - Math.abs(b))
    )[0];

    const foldRows = { v0: [], v1: [] };
    const outerTest = targets.filter(m => m.date >= testStart && (!testEnd || m.date < testEnd));
    for (const match of outerTest) {
      const base = predictShrinkage(allRows, match, { k: BASELINE_K });
      foldRows.v0.push({ match, probabilities: base.probabilities });
      foldRows.v1.push({
        match,
        probabilities: dixonColesProbabilities(
          base.homeExpectedGoals, base.awayExpectedGoals, selectedRho
        )
      });
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
      selectedRho,
      outerTestMatches: outerTest.length,
      v0: score(foldRows.v0),
      v1: score(foldRows.v1)
    });
  }

  return {
    protocol: '3 expanding-window outer folds covering all 2026 targets; 2025 plus strictly earlier 2026 matches for training; inner chronological validation selects Dixon-Coles rho; same-day and future data excluded',
    targetStartDate: loaded.targetStartDate,
    historySources: loaded.historySources,
    trainingHistoryMatches: allRows.filter(m => m.date < dates[0]).length,
    targetMatches: targets.length,
    v0Definition: 'competition-only Poisson with separate home/away league baselines, last 12 same-venue matches and fixed shrinkage k=6',
    v1Definition: 'V0 plus Dixon-Coles low-score dependence correction for 0-0, 0-1, 1-0, 1-1; global rho selected from -0.08/-0.04/0/0.04/0.08 using inner validation Brier primary, LogLoss tie-break; rho=0 is the no-change fallback',
    metrics: { v0: score(aggregate.v0), v1: score(aggregate.v1) },
    pairedDateBlockBootstrap: pairedDateBlockBootstrap(aggregate.v1, aggregate.v0),
    folds
  };
}

module.exports = { dixonColesProbabilities, runExperiment };

if (require.main === module) {
  console.log(JSON.stringify(runExperiment(), null, 2));
}
