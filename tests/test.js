'use strict';
const assert=require('node:assert/strict');
const {predict,backtest,canonical,leagueOf}=require('../model.js');
const {parseSchedule,parseToto,build}=require('../scripts/auto_pipeline.js');
assert.equal(canonical('Ｃ大阪'),canonical('C大阪'));
assert.equal(canonical('Ｇ大阪'),canonical('G大阪'));
assert.equal(canonical('東京Ｖ'),canonical('東京V'));
assert.equal(leagueOf('広島'),'J1');
assert.equal(leagueOf('富山'),'J2');
const hist=[
 {date:'2026-01-01',competition:'J1',home:'C大阪',away:'柏',home_goals:2,away_goals:0},
 {date:'2026-01-01',competition:'J2',home:'富山',away:'八戸',home_goals:4,away_goals:0},
 {date:'2026-01-08',competition:'J1',home:'鹿島',away:'セレッソ大阪',home_goals:1,away_goals:1},
 {date:'2026-01-08',competition:'J2',home:'八戸',away:'富山',home_goals:1,away_goals:2}
];
const p=predict(hist,{date:'2026-02-01',home:'Ｃ大阪',away:'柏レイソル'});
assert.equal(p.targetLeague,'J1');
assert.equal(p.historyCompetitionMatches,2);
assert.equal(p.historyHomeMatches,2);
assert.equal(p.historyAwayMatches,1);
assert.ok(Math.abs(Object.values(p.probabilities).reduce((a,b)=>a+b,0)-1)<1e-9);
for(const x of [p.homeGoalProbabilities,p.awayGoalProbabilities])assert.ok(Math.abs(x.reduce((a,b)=>a+b,0)-1)<1e-9);
const mixed=predict(hist,{date:'2026-02-01',home:'Ｃ大阪',away:'柏レイソル'},{filterCompetition:false});
assert.equal(mixed.historyCompetitionMatches,4);
const noShrinkOption=predict(hist,{date:'2026-02-01',home:'Ｃ大阪',away:'柏レイソル'},{shrinkageMatches:0});
assert.deepEqual(noShrinkOption,p,'K=0 must preserve the default prediction exactly');
const shrunk=predict(hist,{date:'2026-02-01',home:'Ｃ大阪',away:'柏レイソル'},{shrinkageMatches:8});
assert.ok(Math.abs(Object.values(shrunk.probabilities).reduce((a,b)=>a+b,0)-1)<1e-9);
assert.ok(shrunk.homeGoalProbabilities.every(x=>Number.isFinite(x)&&x>=0&&x<=1));
for(const k of [2,4,8,12,16]){const candidate=predict(hist,{date:'2026-02-01',home:'Ｃ大阪',away:'柏レイソン'},{shrinkageMatches:k});assert.ok(Math.abs(Object.values(candidate.probabilities).reduce((a,b)=>a+b,0)-1)<1e-9,`K=${k} must preserve normalized 1X2 probabilities`);assert.ok(candidate.homeExpectedGoals>=0.15&&candidate.homeExpectedGoals<=3.5);assert.ok(candidate.awayExpectedGoals>=0.15&&candidate.awayExpectedGoals<=3.5);}
const parsed=parseSchedule('"2025","J1","R1","2025/01/01","2025-01-01T12:00:00+09:00","Home","2-0","Away"');
assert.equal(parsed.length,1);
const noTargets=parseToto('<html><body>no fixture listing</body></html>',new Date('2026-10-22T00:00:00Z'));
assert.equal(noTargets.fixtures.length,0);
const noPrediction=build('"2025","J1","R1","2025/01/01","2025-01-01T12:00:00+09:00","Home","2-0","Away"','<html>none</html>',new Date('2026-10-22T00:00:00Z'));
assert.equal(noPrediction.payload.predictionsGenerated,false);
const bt=backtest(hist);
assert.equal(bt.method,'chronological walk-forward; competition-filtered');
console.log('PASS data parsing, league isolation, team aliases, probability normalization, and walk-forward backtest');
