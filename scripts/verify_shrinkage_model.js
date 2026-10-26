#!/usr/bin/env node
/**
 * Audit-only verification of the lambdaPen=30 penalized Poisson model.
 * Does not import or modify model.js and does not write prediction logs.
 * Run from repository root: node scripts/verify_shrinkage_model.js
 */
const fs = require("node:fs");
const path = require("node:path");

const source = JSON.parse(fs.readFileSync(path.join(__dirname, "../data/history.json"), "utf8"));
const history = source.history
  .filter(g => ["J1", "J2", "J3"].includes(g.competition))
  .sort((a, b) => a.date.localeCompare(b.date) || a.competition.localeCompare(b.competition));
if (history.length !== 380) throw new Error(`Expected 380 J1/J2/J3 matches, got ${history.length}`);

const teams = [...new Set(history.flatMap(g => [g.home, g.away]))].sort();
const teamIndex = new Map(teams.map((team, i) => [team, i]));
const matches = history.map(g => ({
  home: teamIndex.get(g.home), away: teamIndex.get(g.away),
  hg: Number(g.home_goals), ag: Number(g.away_goals),
  date: g.date, competition: g.competition
}));
const nTeams = teams.length, dim = 2 * nTeams + 2;
const muIndex = 2 * nTeams, homeEffectIndex = muIndex + 1;

function objectiveGradient(p, training, lambdaPen) {
  const grad = Array(dim).fill(0);
  let objective = 0;
  const mu = p[muIndex], h = p[homeEffectIndex];
  for (const g of training) {
    const lh = Math.exp(Math.max(-20, Math.min(20, mu + h + p[g.home] - p[nTeams + g.away])));
    const la = Math.exp(Math.max(-20, Math.min(20, mu + p[g.away] - p[nTeams + g.home])));
    objective += lh - g.hg * Math.log(lh) + la - g.ag * Math.log(la);
    const eh = lh - g.hg, ea = la - g.ag;
    grad[g.home] += eh; grad[nTeams + g.away] -= eh;
    grad[g.away] += ea; grad[nTeams + g.home] -= ea;
    grad[muIndex] += eh + ea; grad[homeEffectIndex] += eh;
  }
  // Penalize attack/defence only. No penalty on mu or home effect.
  for (let i = 0; i < 2 * nTeams; i++) {
    objective += lambdaPen * p[i] * p[i];
    grad[i] += 2 * lambdaPen * p[i];
  }
  return { f: objective, g: grad };
}
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const norm = a => Math.sqrt(dot(a, a));

// L-BFGS with backtracking Armijo line search; unconstrained parameters (equivalent
// to L-BFGS-B with no active bounds). Deterministic initial values per specification.
function fit(training, lambdaPen) {
  let p = Array(dim).fill(0);
  p[muIndex] = Math.log(1.3);
  let { f, g } = objectiveGradient(p, training, lambdaPen);
  const S = [], Y = [], R = [];
  for (let iteration = 0; iteration < 250; iteration++) {
    if (norm(g) < 1e-5) break;
    let q = g.slice();
    const alpha = [];
    for (let k = S.length - 1; k >= 0; k--) {
      alpha[k] = R[k] * dot(S[k], q);
      for (let j = 0; j < dim; j++) q[j] -= alpha[k] * Y[k][j];
    }
    let gamma = 1;
    if (S.length) {
      const k = S.length - 1;
      gamma = dot(S[k], Y[k]) / Math.max(1e-20, dot(Y[k], Y[k]));
    }
    const r = q.map(v => v * gamma);
    for (let k = 0; k < S.length; k++) {
      const beta = R[k] * dot(Y[k], r);
      for (let j = 0; j < dim; j++) r[j] += S[k][j] * (alpha[k] - beta);
    }
    let direction = r.map(v => -v), gd = dot(g, direction);
    if (gd >= 0) {
      direction = g.map(v => -v); gd = -dot(g, g);
      S.length = 0; Y.length = 0; R.length = 0;
    }
    let step = 1, nextP, next, nextF, nextG;
    for (let ls = 0; ls < 35; ls++) {
      nextP = p.map((v, j) => v + step * direction[j]);
      next = objectiveGradient(nextP, training, lambdaPen);
      nextF = next.f; nextG = next.g;
      if (Number.isFinite(nextF) && nextF <= f + 1e-4 * step * gd) break;
      step *= 0.5;
    }
    if (step < 1e-10) break;
    const s = nextP.map((v, j) => v - p[j]);
    const y = nextG.map((v, j) => v - g[j]);
    const sy = dot(s, y);
    if (sy > 1e-12) {
      S.push(s); Y.push(y); R.push(1 / sy);
      if (S.length > 7) { S.shift(); Y.shift(); R.shift(); }
    }
    p = nextP; f = nextF; g = nextG;
  }
  return p;
}
function poissonProbability(lambda, k) {
  let value = Math.exp(-lambda);
  for (let i = 1; i <= k; i++) value *= lambda / i;
  return value;
}
function predict(p, g) {
  const lh = Math.exp(Math.max(-20, Math.min(20, p[muIndex] + p[homeEffectIndex] + p[g.home] - p[nTeams + g.away])));
  const la = Math.exp(Math.max(-20, Math.min(20, p[muIndex] + p[g.away] - p[nTeams + g.home])));
  const hp = Array.from({ length: 10 }, (_, k) => poissonProbability(lh, k));
  const ap = Array.from({ length: 10 }, (_, k) => poissonProbability(la, k));
  const probs = [0, 0, 0]; // home win, draw, away win
  let gridTotal = 0;
  for (let hg = 0; hg < 10; hg++) for (let ag = 0; ag < 10; ag++) {
    const cell = hp[hg] * ap[ag]; gridTotal += cell;
    if (hg > ag) probs[0] += cell;
    else if (hg === ag) probs[1] += cell;
    else probs[2] += cell;
  }
  return probs.map(v => v / gridTotal);
}
function actualClass(g) { return g.hg > g.ag ? 0 : g.hg === g.ag ? 1 : 2; }
function summarize(rows) {
  return {
    n: rows.length,
    accuracy: rows.filter(r => r.predicted === r.actual).length / rows.length,
    brier: rows.reduce((s, r) => s + r.probs.reduce((v, p, k) => v + (p - (k === r.actual ? 1 : 0)) ** 2, 0), 0) / rows.length,
    logLoss: rows.reduce((s, r) => s - Math.log(Math.max(1e-15, r.probs[r.actual])), 0) / rows.length,
    drawPredictions: rows.filter(r => r.predicted === 1).length
  };
}
function evaluateLambda(lambdaPen, start = 60) {
  const rows = [];
  for (let i = start; i < matches.length; i++) {
    // Refit before each prediction using every match strictly earlier in time.
    const fitted = fit(matches.slice(0, i), lambdaPen);
    const probs = predict(fitted, matches[i]);
    rows.push({ probs, actual: actualClass(matches[i]), predicted: probs.indexOf(Math.max(...probs)) });
  }
  return summarize(rows);
}
function evaluateHistoricalRateBaseline(start = 60) {
  const rows = [];
  for (let i = start; i < matches.length; i++) {
    const counts = [0, 0, 0];
    for (const g of matches.slice(0, i)) counts[actualClass(g)]++;
    const probs = counts.map(c => c / i);
    rows.push({ probs, actual: actualClass(matches[i]), predicted: probs.indexOf(Math.max(...probs)) });
  }
  return summarize(rows);
}

const output = {
  audit: "lambdaPen=30 only; independent verification; no production prediction or log writes",
  source: "data/history.json; competition in J1/J2/J3",
  matches: matches.length,
  teams: teams.length,
  ordering: "date ascending, then competition ascending",
  model: {
    formulaHome: "exp(mu + h + a_home - d_away)",
    formulaAway: "exp(mu + a_away - d_home)",
    objective: "-sum(y*ln(lambda)-lambda) + lambdaPen*(sum(a^2)+sum(d^2))",
    lambdaPen: 30,
    penaltyAppliesTo: ["attack", "defence"],
    unpenalized: ["homeEffect", "mu"],
    initialValues: { attack: 0, defence: 0, homeEffect: 0, mu: "ln(1.3)" },
    optimizer: "unconstrained L-BFGS with Armijo backtracking; no bounds are active, so this addresses the unconstrained case of L-BFGS-B"
  },
  metrics: "accuracy; multiclass Brier=sum of 3 squared class errors; log loss=-ln(actual class probability); draw predictions count when draw probability is maximal",
  evaluations: Object.fromEntries([60, 100, 150].map(start => [start, {
    evaluationMatches: matches.length - start,
    baseline: evaluateHistoricalRateBaseline(start),
    lambda30: evaluateLambda(30, start)
  }]))
};
console.log(JSON.stringify(output, null, 2));
