'use strict';
// Additive, trial-only addition. Does NOT change predict()/backtestGoal3()
// in model.js and is NOT wired into auto_pipeline.js or any production
// path. Proposed by Claude per the user's request to try the same
// GOAL3 v2 market-blend idea used on the Claude side (see
// sigulun555-lab/toto-lab PR #17), adapted to this repo's model.js output
// shape (homeGoalProbabilities / awayGoalProbabilities, 4-bucket arrays).

// MARKET_WEIGHT carried over from round 1665's 1X2 market blend
// (Claude V2 logic, MARKET_WEIGHT = 0.7 there too). NOT independently
// derived or validated for GOAL3 -- flag this in any report that uses it.
const MARKET_WEIGHT = 0.7;

// probs, market: 4-length arrays [p0, p1, p2, p3plus], already normalized.
function blendGoal3WithMarket(probs, market, weight = MARKET_WEIGHT) {
  if (probs.length !== 4 || market.length !== 4) {
    throw new Error('blendGoal3WithMarket expects 4-bucket [0,1,2,3+] arrays');
  }
  const raw = probs.map((p, i) => weight * market[i] + (1 - weight) * p);
  const total = raw.reduce((a, b) => a + b, 0);
  return raw.map(v => v / total);
}

function argmaxBucket(probs) {
  return probs.reduce((best, v, i) => (v > probs[best] ? i : best), 0);
}

if (typeof module !== 'undefined') {
  module.exports = { blendGoal3WithMarket, argmaxBucket, MARKET_WEIGHT };
}
