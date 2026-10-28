'use strict';
const fs=require('node:fs');
const path=require('node:path');
const aliasFile=path.join(__dirname,'data','team_aliases.json');
let TEAM_ALIASES={};
try { TEAM_ALIASES=JSON.parse(fs.readFileSync(aliasFile,'utf8')).aliases||{}; } catch (_) {}
const canonical=name=>{const key=String(name||'').normalize('NFKC').trim();return TEAM_ALIASES[key]||key;};
const factorial=n=>{let v=1;for(let i=2;i<=n;i++)v*=i;return v;};
function poisson(lam,k){return Math.exp(-lam)*Math.pow(lam,k)/factorial(k);}
function avg(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:1.25;}
function goalProbabilities(lam){const p0=poisson(lam,0),p1=poisson(lam,1),p2=poisson(lam,2);return [p0,p1,p2,Math.max(0,1-p0-p1-p2)];}
function predict(history,fixture){
 const homeName=canonical(fixture.home),awayName=canonical(fixture.away);
 const prev=history.filter(m=>m.date<fixture.date&&Number.isFinite(m.home_goals)&&Number.isFinite(m.away_goals));
 const leagueHome=avg(prev.map(m=>m.home_goals)),leagueAway=avg(prev.map(m=>m.away_goals));
 const teamStats=name=>{
  const target=canonical(name);
  const played=prev.filter(m=>canonical(m.home)===target||canonical(m.away)===target).slice(-12);
  if(!played.length)return {attack:1,defense:1,n:0};
  const gf=played.map(m=>canonical(m.home)===target?m.home_goals:m.away_goals);
  const ga=played.map(m=>canonical(m.home)===target?m.away_goals:m.home_goals);
  const base=(leagueHome+leagueAway)/2;
  return {attack:avg(gf)/Math.max(.2,base),defense:avg(ga)/Math.max(.2,base),n:played.length};
 };
 const h=teamStats(homeName),a=teamStats(awayName);
 const lh=Math.max(.15,Math.min(3.5,leagueHome*h.attack*a.defense));
 const la=Math.max(.15,Math.min(3.5,leagueAway*a.attack*h.defense));
 let home=0,draw=0,away=0;
 for(let i=0;i<=10;i++)for(let j=0;j<=10;j++){
  const p=poisson(lh,i)*poisson(la,j);
  if(i>j)home+=p;else if(i===j)draw+=p;else away+=p;
 }
 const total=home+draw+away;
 return {
  probabilities:{"1":home/total,"0":draw/total,"2":away/total},
  homeGoalProbabilities:goalProbabilities(lh),
  awayGoalProbabilities:goalProbabilities(la),
  homeExpectedGoals:lh,awayExpectedGoals:la,
  historyHomeMatches:h.n,historyAwayMatches:a.n,
  canonicalHome:homeName,canonicalAway:awayName,
  method:"Poisson baseline using earlier results only; normalized team aliases; shared score-rate basis"
 };
}
function backtest(history){
 const sorted=history.filter(m=>Number.isFinite(m.home_goals)&&Number.isFinite(m.away_goals)).sort((a,b)=>a.date.localeCompare(b.date));
 let n=0,brier=0,logloss=0,correct=0;
 for(let i=1;i<sorted.length;i++){const m=sorted[i],p=predict(sorted.slice(0,i),m).probabilities,actual=m.home_goals>m.away_goals?"1":m.home_goals===m.away_goals?"0":"2";
 for(const k of ["1","0","2"]){const y=k===actual?1:0;brier+=(p[k]-y)**2;}
 logloss-=Math.log(Math.max(1e-12,p[actual]));if(Object.keys(p).sort((a,b)=>p[b]-p[a])[0]===actual)correct++;n++;
 }
 return {n,brier:n?brier/n:null,logloss:n?logloss/n:null,accuracy:n?correct/n:null,method:"chronological walk-forward"};
}

function backtestGoal3(history){
 const sorted=history.filter(m=>Number.isFinite(m.home_goals)&&Number.isFinite(m.away_goals)&&m.date).sort((a,b)=>a.date.localeCompare(b.date));
 const adjusted=[],league=[],empirical=[];
 const bin=g=>Math.max(0,Math.min(3,Math.floor(g)));
 const empiricalProb=(prior,field)=>{
  const counts=[1,1,1,1];
  for(const m of prior)counts[bin(m[field])]++;
  const total=counts.reduce((a,b)=>a+b,0);
  return counts.map(v=>v/total);
 };
 for(let i=0;i<sorted.length;i++){
  const m=sorted[i],prior=sorted.slice(0,i).filter(x=>x.date<m.date);
  const p=predict(prior,m);
  const leagueHome=Math.max(.15,Math.min(3.5,avg(prior.map(x=>x.home_goals))));
  const leagueAway=Math.max(.15,Math.min(3.5,avg(prior.map(x=>x.away_goals))));
  const key=i;
  for(const side of ['home','away']){
   const actual=bin(m[side+'_goals']);
   const modelProb=side==='home'?p.homeGoalProbabilities:p.awayGoalProbabilities;
   const leagueProb=goalProbabilities(side==='home'?leagueHome:leagueAway);
   const empiricalP=empiricalProb(prior,side+'_goals');
   adjusted.push({probabilities:modelProb,actual,key});
   league.push({probabilities:leagueProb,actual,key});
   empirical.push({probabilities:empiricalP,actual,key});
  }
 }
 const summarize=records=>{
  let correct=0,brier=0,logloss=0;
  const byMatch=new Map();
  for(const r of records){
   const predicted=r.probabilities.reduce((best,value,index)=>value>r.probabilities[best]?index:best,0);
   const hit=predicted===r.actual;
   if(hit)correct++;
   brier+=r.probabilities.reduce((sum,value,index)=>sum+(value-(index===r.actual?1:0))**2,0);
   logloss-=Math.log(Math.max(1e-12,r.probabilities[r.actual]));
   if(!byMatch.has(r.key))byMatch.set(r.key,[]);
   byMatch.get(r.key).push(hit);
  }
  const targetCount=records.length,matchCount=byMatch.size;
  return {
   targetCount,
   accuracy:targetCount?correct/targetCount:null,
   brier:targetCount?brier/targetCount:null,
   logloss:targetCount?logloss/targetCount:null,
   exactBothSidesAccuracy:matchCount?[...byMatch.values()].filter(v=>v.length===2&&v.every(Boolean)).length/matchCount:null
  };
 };
 return {
  method:'chronological walk-forward; training uses matches with date strictly earlier than each target date',
  target:'home and away goals categorized as 0, 1, 2, 3+',
  sourceRows:sorted.length,
  testMatches:sorted.length,
  teamGoalTargets:adjusted.length,
  testStartDate:sorted[0]?.date||null,
  testEndDate:sorted.length?sorted[sorted.length-1].date:null,
  models:{
   teamAdjustedPoisson:summarize(adjusted),
   leagueMeanPoisson:summarize(league),
   empiricalLaplaceBaseline:summarize(empirical)
  }
 };
}
if(typeof module!=="undefined")module.exports={predict,backtest,backtestGoal3,canonical,goalProbabilities};
