'use strict';
const assert=require('node:assert/strict');
const {csv,date,score,parseSchedule,parseToto,build}=require('../scripts/auto_pipeline.js');
const {predict,backtestGoal3}=require('../model.js');
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

const goal3BacktestHistory=[
 {date:'2025-01-01',home:'Home',away:'Away',home_goals:0,away_goals:1},
 {date:'2025-01-08',home:'Away',away:'Home',home_goals:2,away_goals:1},
 {date:'2025-01-15',home:'Home',away:'Away',home_goals:3,away_goals:0},
 {date:'2025-01-22',home:'Away',away:'Home',home_goals:2,away_goals:2}
];
const goal3Bt=backtestGoal3(goal3BacktestHistory);
assert.equal(goal3Bt.sourceRows,4);
assert.equal(goal3Bt.testMatches,4,'backtest should include every scored match, including the first fallback prediction');
assert.equal(goal3Bt.teamGoalTargets,8);
assert.equal(goal3Bt.testStartDate,'2025-01-01');
assert.equal(goal3Bt.testEndDate,'2025-01-22');
assert.match(goal3Bt.method,/strictly earlier/);
for(const modelName of ['teamAdjustedPoisson','leagueMeanPoisson','empiricalLaplaceBaseline']){
 const metrics=goal3Bt.models[modelName];
 assert.equal(metrics.targetCount,8);
 assert.ok(Number.isFinite(metrics.accuracy)&&metrics.accuracy>=0&&metrics.accuracy<=1);
 assert.ok(Number.isFinite(metrics.brier)&&metrics.brier>=0);
 assert.ok(Number.isFinite(metrics.logloss)&&metrics.logloss>=0);
 assert.ok(Number.isFinite(metrics.exactBothSidesAccuracy)&&metrics.exactBothSidesAccuracy>=0&&metrics.exactBothSidesAccuracy<=1);
}
const leakTarget={date:'2025-01-15',home:'Home',away:'Away'};
const earlierOnly=goal3BacktestHistory.filter(m=>m.date<leakTarget.date);
const forecastEarlier=predict(earlierOnly,leakTarget);
const forecastWithLaterRows=predict([...goal3BacktestHistory,
 {date:'2025-01-15',home:'Home',away:'Away',home_goals:99,away_goals:99},
 {date:'2025-02-01',home:'Home',away:'Away',home_goals:99,away_goals:99}
],leakTarget);
assert.deepEqual(forecastWithLaterRows.homeGoalProbabilities,forecastEarlier.homeGoalProbabilities,'same-day and future scores must not leak into GOAL3 probabilities');
assert.deepEqual(forecastWithLaterRows.awayGoalProbabilities,forecastEarlier.awayGoalProbabilities,'same-day and future scores must not leak into GOAL3 probabilities');
console.log('PASS GOAL3 chronological backtest metrics, baselines, and strict no-same-day/future-leakage regression');

const historicalRows=require('../data/history.json').history;
const historicalGoal3Bt=backtestGoal3(historicalRows);
assert.equal(historicalGoal3Bt.sourceRows,historicalRows.filter(m=>Number.isFinite(m.home_goals)&&Number.isFinite(m.away_goals)&&m.date).length);
assert.equal(historicalGoal3Bt.teamGoalTargets,historicalGoal3Bt.testMatches*2);
assert.ok(historicalGoal3Bt.testMatches>100,'historical GOAL3 evaluation should cover a useful sample');
for(const metrics of Object.values(historicalGoal3Bt.models)){
 assert.equal(metrics.targetCount,historicalGoal3Bt.teamGoalTargets);
 assert.ok(Number.isFinite(metrics.accuracy)&&Number.isFinite(metrics.brier)&&Number.isFinite(metrics.logloss));
}
const savedGoal3Bt=require('../data/goal3_backtest.json');
assert.equal(savedGoal3Bt.sourceRows,historicalGoal3Bt.sourceRows,'saved GOAL3 backtest must use the same history snapshot');
assert.equal(savedGoal3Bt.testMatches,historicalGoal3Bt.testMatches,'saved GOAL3 backtest must score every match');
assert.equal(savedGoal3Bt.teamGoalTargets,historicalGoal3Bt.teamGoalTargets,'saved GOAL3 backtest target count must match recomputation');
for(const modelName of Object.keys(historicalGoal3Bt.models)){
 for(const metricName of ['targetCount','accuracy','brier','logloss','exactBothSidesAccuracy']){
  assert.ok(Math.abs(savedGoal3Bt.models[modelName][metricName]-historicalGoal3Bt.models[modelName][metricName])<1e-12,
   'saved GOAL3 backtest metric mismatch: '+modelName+'.'+metricName);
 }
}
console.log('GOAL3 HISTORICAL BACKTEST '+JSON.stringify({
 period:[historicalGoal3Bt.testStartDate,historicalGoal3Bt.testEndDate],
 sourceRows:historicalGoal3Bt.sourceRows,
 testMatches:historicalGoal3Bt.testMatches,
 teamGoalTargets:historicalGoal3Bt.teamGoalTargets,
 models:historicalGoal3Bt.models
}));


// Exercise the actual app.js rendering logic with a minimal DOM shim. This complements
// parser tests and verifies the market-specific fixture counts and probability labels.
const vm = require('node:vm');
class TestElement {
  constructor(tagName='div') { this.tagName=tagName; this.children=[]; this._text=''; this.className=''; this.listeners={}; }
  set textContent(value) { this._text=String(value ?? ''); this.children=[]; }
  get textContent() { return this._text + this.children.map(child => child.textContent || '').join(''); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=[]; this._text=''; this.append(...nodes); }
  addEventListener(name, callback) { this.listeners[name]=callback; }
}
function createTestDocument() {
  const elements = new Map();
  return {
    getElementById(id) { if(!elements.has(id)) elements.set(id,new TestElement(id)); return elements.get(id); },
    createElement(tag) { return new TestElement(tag); },
    createTextNode(text) { const node=new TestElement('#text'); node.textContent=text; return node; },
    elements
  };
}
async function renderAppWithPayload(marketType) {
  const document=createTestDocument();
  const isGoal3=marketType==='totoGOAL3';
  const predictions=Array.from({length:isGoal3?3:13},(_,i)=>({
    number:i+1, date:'2026-10-28', home:'Home'+(i+1), away:'Away'+(i+1),
    prediction:isGoal3
      ? {homeGoalProbabilities:[0.1,0.2,0.3,0.4],awayGoalProbabilities:[0.4,0.3,0.2,0.1]}
      : {probabilities:{'1':0.5,'0':0.25,'2':0.25}}
  }));
  const status={predictionsGenerated:true,totoTargetsConfirmed:true,lastAttemptAt:'2026-10-28T02:00:00Z',lastSuccessfulFetch:'2026-10-28T02:00:00Z',warnings:[]};
  const payload={round:isGoal3?1666:1668,deadline:'2026-10-28T18:50:00+09:00',generatedAt:'2026-10-28T02:00:00Z',totoSource:{marketType},predictions,warnings:[]};
  const fetch=async url=>({ok:true,json:async()=>url.includes('status.json')?status:payload});
  vm.runInNewContext(require('node:fs').readFileSync(require('node:path').join(__dirname,'..','app.js'),'utf8'),{document,fetch,console});
  await new Promise(resolve=>setTimeout(resolve,0));
  return document;
}
(async()=>{
  const goal3Doc=await renderAppWithPayload('totoGOAL3');
  assert.equal(goal3Doc.getElementById('badge').textContent,'予想更新済み');
  assert.equal(goal3Doc.getElementById('roundTitle').textContent,'第1666回 totoGOAL3 予想');
  const goal3Cards=goal3Doc.getElementById('fixtures');
  assert.equal(goal3Cards.children.length,3,'GOAL3 UI must render exactly three fixture cards');
  assert.match(goal3Cards.textContent,/ホーム得点/);
  assert.match(goal3Cards.textContent,/アウェイ得点/);
  for(const label of ['0点','1点','2点','3+点']) assert.ok(goal3Cards.textContent.includes(label),'GOAL3 UI must show '+label);
  assert.equal(goal3Cards.children[0].children[2].children.length,10,'each GOAL3 card must show two headings and four bins per side');

  const totoDoc=await renderAppWithPayload('toto');
  assert.equal(totoDoc.getElementById('badge').textContent,'予想更新済み');
  assert.equal(totoDoc.getElementById('roundTitle').textContent,'第1668回 toto 予想');
  const totoCards=totoDoc.getElementById('fixtures');
  assert.equal(totoCards.children.length,13,'standard toto UI must continue to render all thirteen fixtures');
  for(const label of ['ホーム勝ち','引き分け','アウェイ勝ち']) assert.ok(totoCards.textContent.includes(label),'standard toto UI must show '+label);
  assert.equal(totoCards.children[0].children[2].children.length,3,'each standard toto card must show three 1X2 probabilities');
  console.log('PASS app.js UI rendering: GOAL3 3 fixtures with goal bins and standard toto 13 fixtures with 1X2 labels');
})().catch(error=>{console.error(error);process.exitCode=1;});
