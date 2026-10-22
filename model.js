'use strict';
const factorial=n=>{let v=1;for(let i=2;i<=n;i++)v*=i;return v};
function poisson(lam,k){return Math.exp(-lam)*Math.pow(lam,k)/factorial(k)}
function avg(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:1.25}
function predict(history,fixture){
 const prev=history.filter(m=>m.date<fixture.date && Number.isFinite(m.home_goals)&&Number.isFinite(m.away_goals));
 const leagueHome=avg(prev.map(m=>m.home_goals)),leagueAway=avg(prev.map(m=>m.away_goals));
 const teamStats=name=>{
  const played=prev.filter(m=>m.home===name||m.away===name).slice(-12);
  if(!played.length)return {attack:1,defense:1,n:0};
  const gf=played.map(m=>m.home===name?m.home_goals:m.away_goals);
  const ga=played.map(m=>m.home===name?m.away_goals:m.home_goals);
  const base=(leagueHome+leagueAway)/2;
  return {attack:avg(gf)/Math.max(.2,base),defense:avg(ga)/Math.max(.2,base),n:played.length};
 };
 const h=teamStats(fixture.home),a=teamStats(fixture.away),base=(leagueHome+leagueAway)/2;
 const lh=Math.max(.15,Math.min(3.5,leagueHome*h.attack*a.defense));
 const la=Math.max(.15,Math.min(3.5,leagueAway*a.attack*h.defense));
 let home=0,draw=0,away=0;
 for(let i=0;i<=10;i++)for(let j=0;j<=10;j++){
  const p=poisson(lh,i)*poisson(la,j);
  if(i>j)home+=p;else if(i===j)draw+=p;else away+=p;
 }
 const total=home+draw+away;
 return {probabilities:{ "1":home/total,"0":draw/total,"2":away/total },homeExpectedGoals:lh,awayExpectedGoals:la,historyHomeMatches:h.n,historyAwayMatches:a.n,method:"Poisson baseline using earlier results only"};
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
if(typeof module!=="undefined")module.exports={predict,backtest};
