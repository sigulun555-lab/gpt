'use strict';
const fs=require('node:fs');
const path=require('node:path');
const aliasFile=path.join(__dirname,'data','team_aliases.json');
const leagueFile=path.join(__dirname,'data','team_leagues.json');
let TEAM_ALIASES={},LEAGUE_BY_NAME={};
try{TEAM_ALIASES=JSON.parse(fs.readFileSync(aliasFile,'utf8')).aliases||{};}catch(_){}
try{LEAGUE_BY_NAME=JSON.parse(fs.readFileSync(leagueFile,'utf8')).leagueByName||{};}catch(_){}
const canonical=name=>{const key=String(name||'').normalize('NFKC').trim();return TEAM_ALIASES[key]||key;};
const leagueOf=name=>LEAGUE_BY_NAME[canonical(name)]||LEAGUE_BY_NAME[String(name||'').normalize('NFKC').trim()]||null;
const factorial=n=>{let v=1;for(let i=2;i<=n;i++)v*=i;return v;};
function poisson(lam,k){return Math.exp(-lam)*Math.pow(lam,k)/factorial(k);}
function avg(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:1.25;}
function goalProbabilities(lam){const p0=poisson(lam,0),p1=poisson(lam,1),p2=poisson(lam,2);return [p0,p1,p2,Math.max(0,1-p0-p1-p2)];}
function predict(history,fixture,options={}){
 const homeName=canonical(fixture.home),awayName=canonical(fixture.away);
 const targetLeague=fixture.competition||leagueOf(homeName)||leagueOf(awayName);
 const valid=history.filter(m=>m.date<fixture.date&&Number.isFinite(m.home_goals)&&Number.isFinite(m.away_goals));
 const prev=options.filterCompetition===false||!targetLeague?valid:valid.filter(m=>m.competition===targetLeague);
 const leagueHome=avg(prev.map(m=>m.home_goals)),leagueAway=avg(prev.map(m=>m.away_goals));
 const teamStats=name=>{
  const target=canonical(name),played=prev.filter(m=>canonical(m.home)===target||canonical(m.away)===target).slice(-12);
  if(!played.length)return {attack:1,defense:1,n:0};
  const gf=played.map(m=>canonical(m.home)===target?m.home_goals:m.away_goals);
  const ga=played.map(m=>canonical(m.home)===target?m.away_goals:m.home_goals);
  const base=(leagueHome+leagueAway)/2;
  const rawAttack=avg(gf)/Math.max(.2,base),rawDefense=avg(ga)/Math.max(.2,base);
  const k=Number.isFinite(options.shrinkageMatches)?Math.max(0,options.shrinkageMatches):0;
  const weight=k===0?1:played.length/(played.length+k);
  return {attack:1+(rawAttack-1)*weight,defense:1+(rawDefense-1)*weight,n:played.length};
 };
 const h=teamStats(homeName),a=teamStats(awayName);
 const lh=Math.max(.15,Math.min(3.5,leagueHome*h.attack*a.defense));
 const la=Math.max(.15,Math.min(3.5,leagueAway*a.attack*h.defense));
 let home=0,draw=0,away=0;
 for(let i=0;i<=10;i++)for(let j=0;j<=10;j++){const p=poisson(lh,i)*poisson(la,j);if(i>j)home+=p;else if(i===j)draw+=p;else away+=p;}
 const total=home+draw+away;
 return {probabilities:{'1':home/total,'0':draw/total,'2':away/total},homeGoalProbabilities:goalProbabilities(lh),awayGoalProbabilities:goalProbabilities(la),homeExpectedGoals:lh,awayExpectedGoals:la,historyHomeMatches:h.n,historyAwayMatches:a.n,canonicalHome:homeName,canonicalAway:awayName,targetLeague,historyCompetitionMatches:prev.length,method:targetLeague?'League-filtered Poisson baseline; earlier results only; shared score-rate basis':'Poisson baseline; league unavailable; fallback uses all competitions'};
}
function backtest(history,options={}){
 const sorted=history.filter(m=>Number.isFinite(m.home_goals)&&Number.isFinite(m.away_goals)).sort((a,b)=>a.date.localeCompare(b.date));
 let n=0,brier=0,logloss=0,correct=0,zeroHistory=0;
 for(const m of sorted){
  const before=sorted.filter(x=>x.date<m.date);
  if(!before.length)continue;
  const p=predict(before,m,options).probabilities,actual=m.home_goals>m.away_goals?'1':m.home_goals===m.away_goals?'0':'2';
  for(const k of ['1','0','2']){const y=k===actual?1:0;brier+=(p[k]-y)**2;}
  logloss-=Math.log(Math.max(1e-12,p[actual]));
  if(Object.keys(p).sort((a,b)=>p[b]-p[a])[0]===actual)correct++;
  if(predict(before,m,options).historyHomeMatches===0||predict(before,m,options).historyAwayMatches===0)zeroHistory++;
  n++;
 }
 return {n,brier:n?brier/n:null,logloss:n?logloss/n:null,accuracy:n?correct/n:null,zeroHistory,method:options.filterCompetition===false?'chronological walk-forward; all competitions mixed':'chronological walk-forward; competition-filtered'};
}
module.exports={predict,backtest,canonical,goalProbabilities,leagueOf};
