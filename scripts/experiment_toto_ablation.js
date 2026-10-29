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

function runAblation() {
  const sorted = history.filter(m => m.date && m.competition && Number.isFinite(m.home_goals) && Number.isFinite(m.away_goals))
    .slice().sort((a, b) => a.date.localeCompare(b.date));
  const scored = new Map(variants.map(v => [v.id, []]));
  const dates = [...new Set(sorted.map(m => m.date))].sort();
  const holdoutStartDate = dates[Math.max(1, Math.floor(dates.length * 0.8))];
  for (const match of sorted) {
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
      caveat: 'Exploratory only. The final chronological 20% was not used to fit model parameters, but this is a single small 2026 sample and does not establish generalization.'
    };
  }
  return {
    generatedAt: new Date().toISOString(),
    source: 'data/history.json; community J.League CSV mirror, not official results feed',
    historyMatches: sorted.length,
    holdoutStartDate,
    factorOrder: ['competitionOnly', 'splitLeagueBaselines', 'sideSpecificStats', 'shrinkageK'],
    variants: metrics
  };
}

if (require.main === module) process.stdout.write(JSON.stringify(runAblation(), null, 2) + '\n');
module.exports = { predictAblation, runAblation, variants };
