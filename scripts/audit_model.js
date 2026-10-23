'use strict';
const fs=require('node:fs'),path=require('node:path');
const {backtest,canonical,leagueOf,predict}=require('../model.js');
const root=path.resolve(__dirname,'..');
const history=JSON.parse(fs.readFileSync(path.join(root,'data/history.json'),'utf8')).history;
const aliases=JSON.parse(fs.readFileSync(path.join(root,'data/team_aliases.json'),'utf8')).aliases;
const keys=history.map(m=>[m.date,m.competition,canonical(m.home),canonical(m.away)].join('|'));
const dup=keys.filter((k,i)=>keys.indexOf(k)!==i);
const missing=history.filter(m=>!m.date||!m.competition||!m.home||!m.away||!Number.isInteger(m.home_goals)||!Number.isInteger(m.away_goals)||m.home_goals<0||m.away_goals<0);
const counts={},unmapped=new Set(),dateRows=history.map(m=>m.date).sort();
for(const m of history){counts[m.competition]=(counts[m.competition]||0)+1;if(!leagueOf(m.home))unmapped.add(m.home);if(!leagueOf(m.away))unmapped.add(m.away);}
const sorted=[...history].sort((a,b)=>a.date.localeCompare(b.date));
function metrics(rows,probabilities){
 let n=0,brier=0,logloss=0,correct=0;
 for(const m of rows){const p=probabilities(m);if(!p)continue;const actual=m.home_goals>m.away_goals?'1':m.home_goals===m.away_goals?'0':'2';
  for(const k of ['1','0','2'])brier+=(p[k]-(k===actual?1:0))**2;
  logloss-=Math.log(Math.max(1e-12,p[actual]));
  if(Object.keys(p).sort((a,b)=>p[b]-p[a])[0]===actual)correct++;n++;
 }
 return {n,accuracy:n?correct/n:null,brier:n?brier/n:null,logloss:n?logloss/n:null};
}
const eligible=sorted.filter(m=>sorted.some(x=>x.date<m.date&&x.competition===m.competition));
const modelLeague=metrics(eligible,m=>{const before=sorted.filter(x=>x.date<m.date);return predict(before,m).probabilities;});
const modelMixed=metrics(eligible,m=>{const before=sorted.filter(x=>x.date<m.date);return predict(before,m,{filterCompetition:false}).probabilities;});
const frequency=metrics(eligible,m=>{
 const before=sorted.filter(x=>x.date<m.date&&x.competition===m.competition);
 const counts={'1':1,'0':1,'2':1};
 for(const x of before)counts[x.home_goals>x.away_goals?'1':x.home_goals===x.away_goals?'0':'2']++;
 const total=counts['1']+counts['0']+counts['2'];
 return {'1':counts['1']/total,'0':counts['0']/total,'2':counts['2']/total};
});
const perLeague={};
for(const league of Object.keys(counts)){
 const rows=eligible.filter(m=>m.competition===league);
 perLeague[league]={historyRows:counts[league],eligibleBacktestRows:rows.length,leagueFilteredModel:metrics(rows,m=>predict(sorted.filter(x=>x.date<m.date),m).probabilities),smoothedFrequencyBaseline:metrics(rows,m=>{
  const before=sorted.filter(x=>x.date<m.date&&x.competition===m.competition),p={'1':1,'0':1,'2':1};
  for(const x of before)p[x.home_goals>x.away_goals?'1':x.home_goals===x.away_goals?'0':'2']++;
  const total=p['1']+p['0']+p['2'];return {'1':p['1']/total,'0':p['0']/total,'2':p['2']/total};
 })};
}
const report={generatedAt:new Date().toISOString(),historyRows:history.length,dateRange:[dateRows[0],dateRows.at(-1)],competitionCounts:counts,duplicateFixtures:dup.length,invalidRows:missing.length,unmappedTeams:[...unmapped].sort(),sameDayGroups:Object.values(history.reduce((a,m)=>(a[m.date]=(a[m.date]||0)+1,a),{})).filter(n=>n>1).length,backtest:{eligibleRows:eligible.length,leagueFilteredModel:modelLeague,mixedCompetitionModel:modelMixed,smoothedHistoricalFrequencyBaseline:frequency,perLeague},notes:['All evaluated predictions use strictly earlier dates; same-day fixtures are not included in training.','Brier uses the sum of the three squared class errors, averaged across matches; lower is better.','Log loss is averaged across matches; lower is better.','Historical source is a community mirror, not an official results feed.','History covers only a short 2026 window, so this backtest is diagnostic rather than a reliable estimate of future performance.']};
console.log(JSON.stringify(report,null,2));
if(dup.length||missing.length||unmapped.size)process.exitCode=1;
