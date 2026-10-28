'use strict';
const assert=require('node:assert/strict');
const {csv,date,score,parseSchedule,parseToto,build}=require('../scripts/auto_pipeline.js');
const {predict}=require('../model.js');
assert.deepEqual(csv('"2026","J1","round","2026/10/22","2026-10-22T19:00:00+09:00","Home","2-1","Away"')[0].slice(0,4),['2026','J1','round','2026/10/22']);
assert.equal(date('2026/1/9'),'2026-01-09');
assert.deepEqual(score('2-1'),{home_goals:2,away_goals:1});
const hist=parseSchedule('"2025","J1","R1","2025/01/01","2025-01-01T12:00:00+09:00","Home","2-0","Away"\n"2025","J1","R2","2025/01/08","2025-01-08T12:00:00+09:00","Away","1-1","Home"');
assert.equal(hist.length,2);assert.equal(hist[0].home,'Home');assert.equal(hist[0].home_goals,2);
const p=predict(hist,{date:'2025-02-01',home:'Home',away:'Away'}).probabilities;
assert.ok(Math.abs(p['1']+p['0']+p['2']-1)<1e-9);assert.ok(Object.values(p).every(x=>x>=0&&x<=1));
const noTargets=parseToto('<html><body>no fixture listing</body></html>',new Date('2026-10-22T00:00:00Z'));assert.equal(noTargets.fixtures.length,0);
const html='<div>〖開催回〗 第 999 回 販売期間 2026/10/01 00:00 ～ 2026/10/30 12:00</div>'+Array.from({length:13},(_,i)=>'<tr><td>'+(i+1)+'</td><td>10/24</td><td>J</td><td>Home'+i+' VS Away'+i+'</td></tr>').join('');
const parsed=parseToto(html,new Date('2026-10-22T00:00:00Z'));assert.equal(parsed.fixtures.length,13);assert.equal(parsed.round,999);
const notConfirmed=build('"2025","J1","R1","2025/01/01","2025-01-01T12:00:00+09:00","Home","2-0","Away"','<html>none</html>',new Date('2026-10-22T00:00:00Z'));assert.equal(notConfirmed.payload.predictionsGenerated,false);assert.equal(notConfirmed.payload.predictions.length,0);

const goal3Html='<div>〖開催回〗 第 1666 回 くじ totoGOAL3 マルチ 販売期間 2026/10/21 08:00 ～ 2026/10/28 18:50</div>'+
'<tr><td>10/28</td><td>水戸 <a>その他データ</a></td><td>35.86%</td><td>0.40点</td></tr>'+
'<tr><td>Ｃ大阪 <a>その他データ</a></td><td>16.87%</td><td>1.60点</td></tr>'+
'<tr><td>10/28</td><td>横浜FM <a>その他データ</a></td><td>24.27%</td><td>1.00点</td></tr>'+
'<tr><td>広島 <a>その他データ</a></td><td>16.81%</td><td>2.00点</td></tr>'+
'<tr><td>10/28</td><td>名古屋 <a>その他データ</a></td><td>20.16%</td><td>2.00点</td></tr>'+
'<tr><td>浦和 <a>その他データ</a></td><td>18.83%</td><td>1.80点</td></tr>';
const goal3Parsed=parseToto(goal3Html,new Date('2026-10-22T00:00:00Z'));
assert.equal(goal3Parsed.marketType,'totoGOAL3');
assert.equal(goal3Parsed.round,1666);
assert.equal(goal3Parsed.fixtures.length,3);
assert.deepEqual(goal3Parsed.fixtures.map(f=>[f.number,f.home,f.away]),[[1,'水戸','Ｃ大阪'],[2,'横浜FM','広島'],[3,'名古屋','浦和']]);
const goal3Built=build('"2025","J1","R1","2025/01/01","2025-01-01T12:00:00+09:00","Home","2-0","Away"',goal3Html,new Date('2026-10-22T00:00:00Z'));
assert.equal(goal3Built.payload.totoTargetsConfirmed,true);
assert.equal(goal3Built.payload.marketType,'totoGOAL3');
assert.equal(goal3Built.payload.predictionsGenerated,false);
assert.equal(goal3Built.payload.predictions.length,0,'GOAL3 must not be passed to the existing 1X2 predictor');
assert.equal(goal3Built.fixtureData.length,3);
assert.deepEqual(Object.keys(goal3Built.fixtureData[0]),['marketType','matchNumber','home','away','round','deadline']);
assert.equal(goal3Built.fixtureData[0].deadline,'2026-10-28T18:50:00+09:00');
console.log('PASS totoGOAL3 team-row pairing, fixture-only output, and preservation of the 13-fixture toto parser');

console.log('PASS parser, score, probability normalization, toto 13-fixture parsing, and safe no-target behavior');

const {canonical,goalProbabilities}=require('../model.js');
const aliases=require('../data/team_aliases.json').aliases;
assert.equal(canonical('Ｃ大阪'),canonical('C大阪'));
assert.equal(canonical('Ｇ大阪'),canonical('G大阪'));
assert.equal(canonical('東京Ｖ'),canonical('東京V'));
assert.equal(canonical('川崎Ｆ'),canonical('川崎F'));
assert.equal(canonical('京都'),canonical('京都サンガF.C.'));
const histAliases=[
 {date:'2025-01-01',home:'C大阪',away:'柏',home_goals:2,away_goals:0},
 {date:'2025-01-08',home:'鹿島',away:'セレッソ大阪',home_goals:1,away_goals:1}
];
const aliasPrediction=predict(histAliases,{date:'2025-02-01',home:'Ｃ大阪',away:'柏レイソル'});
assert.ok(aliasPrediction.historyHomeMatches>0,'full-width C大阪 should match history abbreviation');
assert.ok(aliasPrediction.historyAwayMatches>0,'full club name should match history abbreviation');
for(const p of [aliasPrediction.homeGoalProbabilities,aliasPrediction.awayGoalProbabilities]){
 assert.equal(p.length,4);assert.ok(p.every(x=>Number.isFinite(x)&&x>=0&&x<=1));assert.ok(Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-9);
}
assert.ok(Math.abs(Object.values(aliasPrediction.probabilities).reduce((a,b)=>a+b,0)-1)<1e-9);
assert.equal(goalProbabilities(1).length,4);
console.log('PASS team alias normalization and shared score/1X2 probability outputs');
