'use strict';
const { loadCombinedHistory, predictShrinkage, score, pairedDateBlockBootstrap } =
  require('./experiment_toto_v1_recency.js');

const BASELINE_K = 6;
const TEMPERATURES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6];
const KEYS = ['1', '0', '2'];

function temperatureScale(probabilities, temperature) {
  if (!(temperature > 0)) throw new RangeError('temperature must be positive');
  const logits = KEYS.map(key => Math.log(Math.max(1e-12, probabilities[key])) / temperature);
  const maxLogit = Math.max(...logits);
  const exp = logits.map(value => Math.exp(value - maxLogit));
  const total = exp.reduce((a, b) => a + b, 0);
  return Object.fromEntries(KEYS.map((key, index) => [key, exp[index] / total]));
}

function runExperiment() {
  const loaded = loadCombinedHistory();
  const allRows = loaded.history;
  const targets = allRows.filter(m => m.date >= loaded.targetStartDate);
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

    const innerBaseRows = innerValidation.map(match => ({
      match,
      probabilities: predictShrinkage(outerTrain, match, { k: BASELINE_K }).probabilities
    }));
    const innerScores = {};
    for (const temperature of TEMPERATURES) {
      innerScores[temperature] = score(innerBaseRows.map(row => ({
        match: row.match,
        probabilities: temperatureScale(row.probabilities, temperature)
      })));
    }
    const selectedTemperature = TEMPERATURES.slice().sort((a, b) =>
      innerScores[a].brier - innerScores[b].brier ||
      innerScores[a].logloss - innerScores[b].logloss ||
      (a === 1 ? -1 : b === 1 ? 1 : Math.abs(a - 1) - Math.abs(b - 1))
    )[0];

    const foldRows = { v0: [], v1: [] };
    const outerTest = targets.filter(m => m.date >= testStart && (!testEnd || m.date < testEnd));
    for (const match of outerTest) {
      const base = predictShrinkage(allRows, match, { k: BASELINE_K }).probabilities;
      foldRows.v0.push({ match, probabilities: base });
      foldRows.v1.push({
        match,
        probabilities: temperatureScale(base, selectedTemperature)
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
      selectedTemperature,
      outerTestMatches: outerTest.length,
      v0: score(foldRows.v0),
      v1: score(foldRows.v1)
    });
  }

  return {
    protocol: '3 expanding-window outer folds over 2026 targets; 2025 plus strictly earlier 2026 matches for training; inner chronological validation selects one global temperature; same-day and future data excluded',
    targetStartDate: loaded.targetStartDate,
    historySources: loaded.historySources,
    trainingHistoryMatches: allRows.filter(m => m.date < dates[0]).length,
    targetMatches: targets.length,
    v0Definition: 'competition-only Poisson with separate home/away league baselines, last 12 same-venue matches and fixed shrinkage k=6',
    v1Definition: 'V0 plus multiclass temperature scaling of 1X2 probabilities; choose T from 0.7/0.8/0.9/1.0/1.1/1.2/1.4/1.6 using inner validation Brier primary and LogLoss tie-break; T=1 is the no-change fallback',
    metrics: { v0: score(aggregate.v0), v1: score(aggregate.v1) },
    pairedDateBlockBootstrap: pairedDateBlockBootstrap(aggregate.v1, aggregate.v0),
    folds
  };
}

module.exports = { temperatureScale, runExperiment };

if (require.main === module) {
  console.log(JSON.stringify(runExperiment(), null, 2));
}
