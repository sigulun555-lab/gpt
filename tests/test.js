'use strict';
const assert=require('node:assert/strict');
const {predict,backtest,canonical,leagueOf}=require('../model.js');
const {parseSchedule,build}=require('../scripts/auto_pipeline.js');
const aliases=require('../data/team_aliases.json').aliases;
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
assert.equal(p.historyCompetitionMatches,2,'must exclude J2 rows');
assert.equal(p.historyHomeMatches,2);
assert.equal(p.historyAwayMatches,2);
assert.ok(Math.abs(Object.values(p.probabilities).reduce((a,b)=>a+b,0)-1)<1e-9);
for(const x of [p.homeGoalProbabilities,p.awayGoalProbabilities])assert.ok(Math.abs(x.reduce((a,b)=>a+b,0)-1)<1e-9);
const mixed=predict(hist,{date:'2026-02-01',home:'Ｃ大阪',away:'柏レイソル'},{filterCompetition:false});
assert.equal(mixed.historyCompetitionMatches,4,'explicit mixed mode remains available for comparison');
const bt=backtest(hist);
assert.equal(bt.method,'chronological walk-forward; competition-filtered');
console.log('PASS league filtering, team aliases, shared score probabilities, and walk-forward backtest');
