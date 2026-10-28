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
const parsed=parseToto(html,new Date('2026-10-22T00:00:00Z'));assert.equal(parsed.fixtures.length,13);assert.equal(parsed.round,999);assert.equal(parsed.marketType,'toto');
const officialShapeHtml='<div>〖開催回〗 第 1000 回 販売期間 2026/10/24 08:00 ～ 2026/10/31 13:50</div>'+Array.from({length:13},(_,i)=>'<tr><td>'+(i+1)+'</td><td>'+(i<5?'10/31':'11/01')+'<br>14:00</td><td>会場'+i+'</td><td>Home'+i+'</td><td>VS</td><td>Away'+i+'</td></tr>').join('');
const officialShape=parseToto(officialShapeHtml,new Date('2026-10-28T00:00:00+09:00'));
assert.equal(officialShape.fixtures.length,13,'official-shaped rows with date/time line breaks should parse all 13 fixtures');
assert.deepEqual([officialShape.fixtures[0].number,officialShape.fixtures[0].dateLabel,officialShape.fixtures[0].home,officialShape.fixtures[0].away],[1,'10/31','Home0','Away0']);
assert.equal(officialShape.fixtures[12].number,13);
const officialRound1668=[
 ['10/31','14:00','岡山','FC東京'],['10/31','14:00','水戸','広島'],['10/31','15:00','神戸','名古屋'],
 ['10/31','15:00','鹿島','Ｃ大阪'],['10/31','16:00','柏','浦和'],['10/31','14:00','京都','清水'],
 ['11/01','14:00','千葉','福岡'],['11/01','14:00','長崎','東京Ｖ'],['11/01','16:00','町田','Ｇ大阪'],
 ['11/01','13:00','横浜FM','川崎Ｆ'],['10/31','14:00','仙台','大宮'],['10/31','14:00','新潟','富山'],
 ['10/31','14:00','宮崎','大分']
];
const official1668Html='<div>〖開催回〗 第 1668 回 toto 販売期間 2026/10/24 08:00 ～ 2026/10/31 13:50</div><table>'+officialRound1668.map((m,i)=>'<tr><td>'+(i+1)+'</td><td>'+m[0]+'<br>'+m[1]+'</td><td>会場'+i+'</td><td>'+m[2]+'</td><td>VS</td><td>'+m[3]+'</td><td>データ</td></tr>').join('')+'</table>';
const official1668Parsed=parseToto(official1668Html,new Date('2026-10-28T00:00:00+09:00'));
assert.equal(official1668Parsed.round,1668);
assert.equal(official1668Parsed.deadline,'2026-10-31T13:50:00+09:00');
assert.equal(official1668Parsed.fixtures.length,13,'round 1668 official-listed fixture data should parse all 13 matches');
assert.deepEqual(official1668Parsed.fixtures.map(f=>[f.number,f.dateLabel,f.home,f.away]),officialRound1668.map((m,i)=>[i+1,m[0],m[2],m[3]]));


const goal3Rows=[
 ['10/28','水戸','その他データ','35.85%','0.40点'],
 ['','Ｃ大阪','その他データ','16.83%','1.60点'],
 ['10/28','横浜FM','その他データ','24.29%','1.00点'],
 ['','広島','その他データ','16.88%','2.00点'],
 ['10/28','名古屋','その他データ','20.03%','2.00点'],
 ['','浦和','その他データ','18.68%','1.80点']
];
const goal3Html='<div>〖開催回〗 第 1666 回 〖くじ〗 totoGOAL3 マルチ 販売期間 2026/10/21 08:00 ～ 2026/10/28 18:50</div>'+goal3Rows.map(cells=>'<tr>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('');
const goal3Parsed=parseToto(goal3Html,new Date('2026-10-22T00:00:00Z'));
assert.equal(goal3Parsed.marketType,'totoGOAL3');
assert.equal(goal3Parsed.round,1666);
assert.equal(goal3Parsed.deadline,'2026-10-28T18:50:00+09:00');
assert.equal(goal3Parsed.fixtures.length,3,'GOAL3 should normalize six team rows to three matches');
assert.deepEqual(goal3Parsed.fixtures.map(f=>[f.number,f.home,f.away]),[[1,'水戸','Ｃ大阪'],[2,'横浜FM','広島'],[3,'名古屋','浦和']]);
const goal3Built=build('"2025","J1","R1","2025/01/01","2025-01-01T12:00:00+09:00","Home","2-0","Away"',goal3Html,new Date('2026-10-22T00:00:00Z'));
assert.equal(goal3Built.payload.totoTargetsConfirmed,true);
assert.equal(goal3Built.fixtures.fixtures.length,3);
assert.deepEqual(Object.keys(goal3Built.fixtures.fixtures[0]).sort(),['away','deadline','home','marketType','matchNumber','round'].sort());
assert.equal(goal3Built.fixtures.fixtures[0].marketType,'totoGOAL3');
assert.equal(goal3Built.fixtures.fixtures[0].matchNumber,1);
assert.equal(goal3Built.fixtures.fixtures[0].round,1666);
assert.equal(goal3Built.fixtures.fixtures[0].deadline,'2026-10-28T18:50:00+09:00');

const notConfirmed=build('"2025","J1","R1","2025/01/01","2025-01-01T12:00:00+09:00","Home","2-0","Away"','<html>none</html>',new Date('2026-10-22T00:00:00Z'));assert.equal(notConfirmed.payload.predictionsGenerated,false);assert.equal(notConfirmed.payload.predictions.length,0);
console.log('PASS parser, score, probability normalization, toto 13-fixture parsing, totoGOAL3 six-team-row normalization, fixture-only data contract, and safe no-target behavior');

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
