'use strict';
const fs=require('node:fs'),path=require('node:path');
const {canonical,leagueOf,predict}=require('../model.js');
const {parseSchedule,HISTORICAL_URL}=require('./auto_pipeline.js');
const root=path.resolve(__dirname,'..');
const current=JSON.parse(fs.readFileSync(path.join(root,'data/history.json'),'utf8')).history;
const historicalUrl=process.env.JLEAGUE_HISTORICAL_CSV_URL||HISTORICAL_URL;
async function main(){
  const response=await fetch(historicalUrl,{headers:{'User-Agent':'JLeagueTotoPredictor-audit/0.1','Accept':'text/csv,*/*'},signal:AbortSignal.timeout(20000)});
  if(!response.ok)throw new Error('Historical CSV fetch failed: HTTP '+response.status);
  const csv=await response.text();
  if(csv.length<100||!/^year,category,term,date,/i.test(csv.replace(/^\uFEFF/,'')))throw new Error('Historical CSV schema did not match expected source');
  const prior=parseSchedule(csv).filter(m=>Number.isInteger(m.home_goals)&&Number.isInteger(m.away_goals)).map(m=>({...m,season:Number(m.date.slice(0,4)),source:'community_jleague_historical_csv',sourceUrl:historicalUrl}));
  const validCurrent=current.filter(m=>Number.isInteger(m.home_goals)&&Number.isInteger(m.away_goals));
  const combined=new Map();
  for(const m of [...prior,...validCurrent])combined.set([m.date,m.competition,canonical(m.home),canonical(m.away)].join('|'),m);
  const history=[...combined.values()].sort((a,b)=>a.date.localeCompare(b.date));
  const keys=history.map(m=>[m.date,m.competition,canonical(m.home),canonical(m.away)].join('|'));
  const dup=keys.filter((k,i)=>keys.indexOf(k)!==i);
  const missing=history.filter(m=>!m.date||!m.competition||!m.home||!m.away||!Number.isInteger(m.home_goals)||!Number.isInteger(m.away_goals)||m.home_goals<0||m.away_goals<0);
  const counts={},unmapped=new Set(),dates=history.map(m=>m.date).sort();
  for(const m of history){counts[m.competition]=(counts[m.competition]||0)+1;if(!leagueOf(m.home))unmapped.add(m.home);if(!leagueOf(m.away))unmapped.add(m.away);}
  const eligible=history.filter(m=>history.some(x=>x.date<m.date&&x.competition===m.competition));
  function metrics(rows,fn){let n=0,brier=0,logloss=0,correct=0;for(const m of rows){const p=fn(m);if(!p)continue;const actual=m.home_goals>m.away_goals?'1':m.home_goals===m.away_goals?'0':'2';for(const k of ['1','0','2'])brier+=(p[k]-(k===actual?1:0))**2;logloss-=Math.log(Math.max(1e-12,p[actual]));if(Object.keys(p).sort((a,b)=>p[b]-p[a])[0]===actual)correct++;n++;}return {n,accuracy:n?correct/n:null,brier:n?brier/n:null,logloss:n?logloss/n:null};}
  const leagueModel=metrics(eligible,m=>predict(history.filter(x=>x.date<m.date),m).probabilities);
  const mixedModel=metrics(eligible,m=>predict(history.filter(x=>x.date<m.date),m,{filterCompetition:false}).probabilities);
  const shrinkageSweep={};
  for(const k of [2,4,8,12,16])shrinkageSweep['K'+k]=metrics(eligible,m=>predict(history.filter(x=>x.date<m.date),m,{shrinkageMatches:k}).probabilities);
  const frequency=metrics(eligible,m=>{const p={'1':1,'0':1,'2':1};for(const x of history.filter(x=>x.date<m.date&&x.competition===m.competition))p[x.home_goals>x.away_goals?'1':x.home_goals===x.away_goals?'0':'2']++;const n=p['1']+p['0']+p['2'];return {'1':p['1']/n,'0':p['0']/n,'2':p['2']/n};});
  const perLeague={};
  for(const league of Object.keys(counts)){const rows=eligible.filter(m=>m.competition===league);const shrinkageByK={};for(const k of [2,4,8,12,16])shrinkageByK['K'+k]=metrics(rows,m=>predict(history.filter(x=>x.date<m.date),m,{shrinkageMatches:k}).probabilities);perLeague[league]={historyRows:counts[league],eligibleBacktestRows:rows.length,leagueFilteredModel:metrics(rows,m=>predict(history.filter(x=>x.date<m.date),m).probabilities),shrinkageByK,smoothedFrequencyBaseline:metrics(rows,m=>{const p={'1':1,'0':1,'2':1};for(const x of history.filter(x=>x.date<m.date&&x.competition===m.competition))p[x.home_goals>x.away_goals?'1':x.home_goals===x.away_goals?'0':'2']++;const n=p['1']+p['0']+p['2'];return {'1':p['1']/n,'0':p['0']/n,'2':p['2']/n};})};}
  const report={generatedAt:new Date().toISOString(),historySources:[{url:historicalUrl,rows:prior.length},{url:'data/history.json',rows:validCurrent.length}],historyRows:history.length,dateRange:[dates[0],dates.at(-1)],competitionCounts:counts,duplicateFixtures:dup.length,invalidRows:missing.length,unmappedTeams:[...unmapped].sort(),backtest:{eligibleRows:eligible.length,leagueFilteredModel:leagueModel,mixedCompetitionModel:mixedModel,teamFormShrinkageSweep:shrinkageSweep,smoothedHistoricalFrequencyBaseline:frequency,perLeague},notes:['Uses the 2025 historical CSV fetched at audit time plus the checked-in current-season history.','All predictions use strictly earlier dates; same-day fixtures are excluded from training.','Brier is the sum of three squared class errors averaged across matches; lower is better.','Log loss is averaged across matches; lower is better.','Team-form shrinkage K=2,4,8,12,16 is an audit-only comparison; these full-window results are exploratory and must not be used to claim holdout performance.','The historical source is a community mirror, not an official results feed; results are diagnostic, not proof of future accuracy.']};
  console.log(JSON.stringify(report,null,2));
  if(dup.length||missing.length||unmapped.size||prior.length<1000)process.exitCode=1;
}
main().catch(e=>{console.error('AUDIT_FAILED:',e.message);process.exitCode=1;});
