'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { predict, canonical } = require('../model.js');

const ROOT = path.resolve(__dirname, '..');
const history = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'history.json'), 'utf8')).history;
const fixturesFile = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'toto_round_1668.json'), 'utf8'));
const KEYS = ['1', '0', '2'];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const avg = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 1.25;
const outcome = m => m.home_goals > m.away_goals ? '1' : m.home_goals === m.away_goals ? '0' : '2';
function poisson(lam, k) { let p = Math.exp(-lam); for (let i = 1; i <= k; i++) p *= lam / i; return p; }
function outcomeProbabilities(homeLambda, awayLambda) {
  let h = 0, d = 0, a = 0;
  for (let i = 0; i <= 10; i++) for (let j = 0; j <= 10; j++) {
    const p = poisson(homeLambda, i) * poisson(awayLambda, j);
    if (i > j) h += p; else if (i === j) d += p; else a += p;
  }
  const z = h + d + a;
  return { '1': h / z, '0': d / z, '2': a / z };
}
function teamStats(prior, name, competition, side, limit = 12) {
  const target = canonical(name);
  const matches = prior.filter(m => m.competition === competition &&
    (canonical(m.home) === target || canonical(m.away) === target)).slice(-limit);
  let gf = [], ga = [];
  for (const m of matches) {
    const isHome = canonical(m.home) === target;
    gf.push(isHome ? m.home_goals : m.away_goals);
    ga.push(isHome ? m.away_goals : m.home_goals);
  }
  const split = matches.filter(m => (side === 'home' ? canonical(m.home) === target : canonical(m.away) === target));
  return {
    n: split.length,
    gf: split.map(m => side === 'home' ? m.home_goals : m.away_goals),
    ga: split.map(m => side === 'home' ? m.away_goals : m.home_goals),
  };
}
function predictCompetitionShrinkage(allHistory, fixture, k = 6) {
  const prior = allHistory.filter(m => m.date < fixture.date && m.competition === fixture.competition &&
    Number.isFinite(m.home_goals) && Number.isFinite(m.away_goals));
  const leagueHome = clamp(avg(prior.map(m => m.home_goals)), 0.2, 3.5);
  const leagueAway = clamp(avg(prior.map(m => m.away_goals)), 0.2, 3.5);
  const rates = (name, side, base) => {
    const s = teamStats(prior, name, fixture.competition, side);
    const gfa = avg(s.gf), gaa = avg(s.ga);
    const gfShrunk = (gfa * s.n + base * k) / (s.n + k);
    const gaBase = side === 'home' ? leagueAway : leagueHome;
    const gaShrunk = (gaa * s.n + gaBase * k) / (s.n + k);
    return { attack: gfShrunk / base, defense: gaShrunk / gaBase, n: s.n };
  };
  const h = rates(fixture.home, 'home', leagueHome);
  const a = rates(fixture.away, 'away', leagueAway);
  const homeLambda = clamp(leagueHome * h.attack * a.defense, 0.15, 3.5);
  const awayLambda = clamp(leagueAway * a.attack * h.defense, 0.15, 3.5);
  return { probabilities: outcomeProbabilities(homeLambda, awayLambda), homeLambda, awayLambda,
    historyHomeMatches: h.n, historyAwayMatches: a.n, method: 'competition-specific Poisson with k=6 shrinkage' };
}
function summarize(rows) {
  let brier = 0, logloss = 0, correct = 0;
  for (const r of rows) {
    const actual = outcome(r.match);
    for (const k of KEYS) brier += (r.p[k] - (k === actual ? 1 : 0)) ** 2;
    logloss -= Math.log(Math.max(1e-12, r.p[actual]));
    if (KEYS.reduce((best, k) => r.p[k] > r.p[best] ? k : best, KEYS[0]) === actual) correct++;
  }
  const n = rows.length;
  return { n, accuracy: n ? correct / n : null, brier: n ? brier / n : null, logloss: n ? logloss / n : null };
}
function runExperiment() {
  const sorted = history.filter(m => m.date && m.competition && Number.isFinite(m.home_goals) && Number.isFinite(m.away_goals))
    .slice().sort((a, b) => a.date.localeCompare(b.date));
  const baseRows = [], candidateRows = [];
  for (const m of sorted) {
    const prior = sorted.filter(x => x.date < m.date);
    if (!prior.length) continue;
    const base = predict(prior, { date: m.date, home: m.home, away: m.away }).probabilities;
    const candidate = predictCompetitionShrinkage(sorted, { date: m.date, home: m.home, away: m.away, competition: m.competition }).probabilities;
    baseRows.push({ match: m, p: base });
    candidateRows.push({ match: m, p: candidate });
  }
  const predictions = fixturesFile.fixtures.map(f => {
    const fixture = { ...f, date: f.date, competition: f.competition };
    const base = predict(sorted, { date: fixture.date, home: fixture.home, away: fixture.away });
    const candidate = predictCompetitionShrinkage(sorted, fixture);
    return {
      matchNumber: fixture.matchNumber, date: fixture.date, competition: fixture.competition,
      home: fixture.home, away: fixture.away,
      baseline: { probabilities: base.probabilities, homeExpectedGoals: base.homeExpectedGoals, awayExpectedGoals: base.awayExpectedGoals },
      candidate: { probabilities: candidate.probabilities, homeExpectedGoals: candidate.homeLambda, awayExpectedGoals: candidate.awayLambda },
      candidateHistory: { home: candidate.historyHomeMatches, away: candidate.historyAwayMatches },
      topPick: Object.keys(candidate.probabilities).reduce((best, k) => candidate.probabilities[k] > candidate.probabilities[best] ? k : best, '1')
    };
  });
  return {
    generatedAt: new Date().toISOString(),
    round: fixturesFile.round,
    deadline: fixturesFile.deadline,
    source: 'data/history.json (community J.League CSV mirror); not official toto market odds',
    historyRows: sorted.length,
    evaluation: {
      method: 'chronological walk-forward; same-day/future rows excluded; scores are mean per match; competition-specific candidate uses same competition only',
      baselineCurrentProduction: summarize(baseRows),
      candidateCompetitionShrinkageK6: summarize(candidateRows),
      caveat: 'This is a model-selection experiment on the currently stored 2026 sample, not proof of future improvement. The candidate is not connected to production.'
    },
    predictions
  };
}
if (require.main === module) process.stdout.write(JSON.stringify(runExperiment(), null, 2) + '\n');
module.exports = { runExperiment, predictCompetitionShrinkage, outcomeProbabilities };
