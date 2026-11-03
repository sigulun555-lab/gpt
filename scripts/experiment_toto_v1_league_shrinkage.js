'use strict';
const { loadCombinedHistory, predictShrinkage, score, pairedDateBlockBootstrap } =
  require('./experiment_toto_v1_recency.js');

const K_VALUES = [3, 6, 10];
const BASELINE_K = 6;
const LEAGUES = ['J1', 'J2', 'J3'];

function runExperiment() {
  const loaded = loadCombinedHistory();
  const allRows = loaded.history;
  const targetStartDate = loaded.targetStartDate;
  const targets = allRows.filter(m => m.date >= targetStartDate);
  const dates = [...new Set(targets.map(m => m.date))].sort();
  if (dates.length < 10) throw new Error('Insufficient unique target dates for nested walk-forward');

  // Partition all 2026 target dates into three chronological outer test folds.
  // The first fold trains on 2025 only; later folds expand with prior 2026 results.
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

    const innerScoresByLeague = {};
    const selectedKByLeague = {};
    for (const league of LEAGUES) {
      const validation = innerValidation.filter(m => m.competition === league);
      const scoresByK = {};
      for (const k of K_VALUES) {
        const rows = validation.map(match => ({
          match,
          probabilities: predictShrinkage(outerTrain, match, { k }).probabilities
        }));
        scoresByK[k] = score(rows);
      }
      const selectedK = K_VALUES.slice().sort((a, b) =>
        scoresByK[a].brier - scoresByK[b].brier ||
        scoresByK[a].logloss - scoresByK[b].logloss ||
        (a === BASELINE_K ? -1 : b === BASELINE_K ? 1 : a - b)
      )[0];
      selectedKByLeague[league] = selectedK;
      innerScoresByLeague[league] = {
        validationMatches: validation.length,
        selectedK,
        candidates: scoresByK
      };
    }

    const foldRows = { v0: [], v1: [] };
    const outerTest = targets.filter(m => m.date >= testStart && (!testEnd || m.date < testEnd));
    for (const match of outerTest) {
      const v0 = predictShrinkage(allRows, match, { k: BASELINE_K }).probabilities;
      const selectedK = selectedKByLeague[match.competition] ?? BASELINE_K;
      const v1 = predictShrinkage(allRows, match, { k: selectedK }).probabilities;
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
      innerScoresByLeague,
      selectedKByLeague,
      outerTestMatches: outerTest.length,
      v0: score(foldRows.v0),
      v1: score(foldRows.v1)
    });
  }

  return {
    protocol: '3 expanding-window outer folds over 2026 targets; 2025 plus strictly earlier 2026 matches for training; inner chronological validation selects one shrinkage k per league; same-day and future data excluded',
    targetStartDate,
    historySources: loaded.historySources,
    trainingHistoryMatches: allRows.filter(m => m.date < dates[0]).length,
    targetMatches: targets.length,
    v0Definition: 'competition-only Poisson; separate home/away league scoring baselines; last 12 same-venue team matches; equal weights; fixed shrinkage k=6 for all leagues',
    v1Definition: 'same as V0, but choose shrinkage k independently for J1/J2/J3 from k=3/6/10 using only each outer fold inner validation; Brier primary, LogLoss tie-break; k=6 wins exact ties',
    metrics: { v0: score(aggregate.v0), v1: score(aggregate.v1) },
    pairedDateBlockBootstrap: pairedDateBlockBootstrap(aggregate.v1, aggregate.v0),
    folds
  };
}

module.exports = { runExperiment };

if (require.main === module) {
  console.log(JSON.stringify(runExperiment(), null, 2));
}
