'use strict';
const fs=require('node:fs'),path=require('node:path');
const ROOT=path.resolve(__dirname,'..');
const history=JSON.parse(fs.readFileSync(path.join(ROOT,'data/history.json'),'utf8')).history;
const aliases=JSON.parse(fs.readFileSync(path.join(ROOT,'data/team_aliases.json'),'utf8')).aliases;
const canonical=n=>aliases[String(n||'').normalize('NFKC').trim()]||String(n||'').normalize('NFKC').trim();
const avg=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:1.25;
const factorial=n=>{let v=1;for(let i=2;i<=n;i++)v*=i;return v;};
const poisson=(l,k)=>Math.exp(-l)*Math.pow(l,k)/factorial(k);
function outcomeProb(lh,la){let q=[0,0,0];for(let i=0;i<=10;i++)for(let j=0;j<=10;j++){const p=poisson(lh,i)*poisson(la,j);if(i>j)q[0]+=p;else if(i===j)q[1]+=p;else q[2]+=p;}const s=q.reduce((a,b)=>a+b,0);return q.map(x=>x/s);}
function poissonModel(prev,m,competitionSpecific=false){
 const base=competitionSpecific?prev.filter(x=>x.competition===m.competition):prev;
 const lh0=avg(base.map(x=>x.home_goals)),la0=avg(base.map(x=>x.away_goals));
 const stats=name=>{const t=canonical(name),played=prev.filter(x=>canonical(x.home)===t||canonical(x.away)===t).slice(-12);if(!played.length)return {attack:1,defense:1};const gf=played.map(x=>canonical(x.home)===t?x.home_goals:x.away_goals),ga=played.map(x=>canonical(x.home)===t?x.away_goals:x.home_goals),b=(lh0+la0)/2;return {attack:avg(gf)/Math.max(.2,b),defense:avg(ga)/Math.max(.2,b)};};
 const h=stats(m.home),a=stats(m.away),lh=Math.max(.15,Math.min(3.5,lh0*h.attack*a.defense)),la=Math.max(.15,Math.min(3.5,la0*a.attack*h.defense));return outcomeProb(lh,la);
}
function metrics(name,predict,rows){
 let n=0,brier=0,logLoss=0,correct=0;
 for(const m of rows){const prev=rows.filter(x=>x.date<m.date);if(prev.length<10)continue;const p=predict(prev,m),actual=m.home_goals>m.away_goals?0:m.home_goals===m.away_goals?1:2;for(let k=0;k<3;k++)brier+=(p[k]-(k===actual?1:0))**2;logLoss-=Math.log(Math.max(1e-12,p[actual]));if(p.indexOf(Math.max(...p))===actual)correct++;n++;}
 return {name,n,accuracy:n?correct/n:null,brierSum3:n?brier/n:null,brierMean3:n?brier/(3*n):null,logLoss:n?logLoss/n:null};
}
function run(){
 const valid=history.filter(m=>m.date&&Number.isInteger(m.home_goals)&&Number.isInteger(m.away_goals)&&m.home_goals>=0&&m.away_goals>=0);
 const invalid=history.length-valid.length,seen=new Set(),duplicates=[];
 for(const m of valid){const k=[m.date,canonical(m.home),canonical(m.away)].join('|');if(seen.has(k))duplicates.push(k);seen.add(k);}
 const teams=[...new Set(valid.flatMap(m=>[m.home,m.away]))],missingAliases=teams.filter(n=>!aliases[String(n).normalize('NFKC').trim()]);
 const competitions={};for(const m of valid)competitions[m.competition||'MISSING']=(competitions[m.competition||'MISSING']||0)+1;
 const sorted=valid.slice().sort((a,b)=>a.date.localeCompare(b.date));
 const frequency=metrics('smoothed historical outcome frequencies',prev=>{const c=[1,1,1];for(const m of prev)c[m.home_goals>m.away_goals?0:m.home_goals===m.away_goals?1:2]++;const s=c.reduce((a,b)=>a+b,0);return c.map(x=>x/s);},sorted);
 const homeAwayMeans=metrics('competition-specific home/away goal means',(prev,m)=>{const q=prev.filter(x=>x.competition===m.competition);return outcomeProb(avg(q.map(x=>x.home_goals)),avg(q.map(x=>x.away_goals)));},sorted);
 const mixed=metrics('current mixed-competition Poisson',(prev,m)=>poissonModel(prev,m,false),sorted);
 const competition=metrics('competition-specific league-rate Poisson',(prev,m)=>poissonModel(prev,m,true),sorted);
 return {generatedAt:new Date().toISOString(),dataQuality:{historyRows:history.length,validRows:valid.length,invalidRows:invalid,duplicateFixtureKeys:duplicates.length,missingTeamAliases,dateRange:valid.length?[sorted[0].date,sorted[sorted.length-1].date]:null,competitionCounts:competitions,uniqueTeams:teams.length},evaluation:{method:'chronological walk-forward; only strictly earlier dates; first 10 prior fixtures excluded; same scored fixtures for every model',scoredFixtures:frequency.n,metrics:[frequency,homeAwayMeans,mixed,competition]},warnings:['Small historical sample; metrics do not guarantee future performance.','Team form currently pools each team’s last 12 matches; competition-specific league rates are experimental.','Historical outcome baseline uses Laplace smoothing with one pseudo-count per outcome.','No injuries, suspensions, odds, lineups or travel/rest variables are included.']};
}
if(require.main===module){const r=run();console.log(JSON.stringify(r,null,2));if(r.dataQuality.invalidRows||r.dataQuality.duplicateFixtureKeys||r.dataQuality.missingTeamAliases.length)process.exitCode=1;}
module.exports={run};
