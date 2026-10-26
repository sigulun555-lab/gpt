#!/usr/bin/env node
// Audit-only λ=10/30 walk-forward evaluation for 2025 CSV and 2026 history.json.
// No production imports, predictions, or log writes.
const fs = require("node:fs");
const path = require("node:path");
const csvPath = path.join(__dirname, "../data/2025_schedule_with_ids.csv");
const csv = fs.readFileSync(csvPath, "utf8").trim().split(/\r?\n/);
const headers = csv.shift().split(",");
const ix = Object.fromEntries(headers.map((h,i)=>[h,i]));
const rows = csv.map(line => {
  // CSV fields may contain commas inside quotes; parse safely.
  const fields = []; let s="", q=false;
  for (let i=0;i<line.length;i++){const c=line[i]; if(c==='"'){if(q&&line[i+1]==='"'){s+='"';i++;}else q=!q;}else if(c===','&&!q){fields.push(s);s="";}else s+=c;}
  fields.push(s); const o=Object.fromEntries(headers.map((h,i)=>[h,fields[i]]));
  const m=/^(\d+)-(\d+)$/.exec(o.score||"");
  if(!m) return null;
  const date=o.date.replaceAll("/","-");
  if(date<"2025-02-14"||date>"2025-12-06") return null;
  if(!["1","2","3"].includes(o.category)) return null;
  return {date,competition:({1:"J1",2:"J2",3:"J3"})[o.category],home:o.homeTeam,away:o.awayTeam,hg:+m[1],ag:+m[2]};
}).filter(Boolean).sort((a,b)=>a.date.localeCompare(b.date)||a.competition.localeCompare(b.competition));
const source = JSON.parse(fs.readFileSync(path.join(__dirname,"../data/history.json"),"utf8"));
const hist = source.history.filter(g=>["J1","J2","J3"].includes(g.competition)).sort((a,b)=>a.date.localeCompare(b.date)||a.competition.localeCompare(b.competition)).map(g=>({date:g.date,competition:g.competition,home:g.home,away:g.away,hg:+g.home_goals,ag:+g.away_goals}));
function run(data, labels) {
 const teams=[...new Set(data.flatMap(g=>[g.home,g.away]))].sort(), ti=new Map(teams.map((t,i)=>[t,i]));
 const matches=data.map(g=>({...g,home:ti.get(g.home),away:ti.get(g.away)}));
 const nt=teams.length, dim=2*nt+2, mi=2*nt, hi=mi+1;
 const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0), norm=a=>Math.sqrt(dot(a,a));
 function fg(p,tr,lam){
  const gr=Array(dim).fill(0);let f=0;const mu=p[mi],h=p[hi];
  for(const g of tr){const lh=Math.exp(Math.max(-20,Math.min(20,mu+h+p[g.home]-p[nt+g.away]))),la=Math.exp(Math.max(-20,Math.min(20,mu+p[g.away]-p[nt+g.home])));
   f+=lh-g.hg*Math.log(lh)+la-g.ag*Math.log(la);const eh=lh-g.hg,ea=la-g.ag;
   gr[g.home]+=eh;gr[nt+g.away]-=eh;gr[g.away]+=ea;gr[nt+g.home]-=ea;gr[mi]+=eh+ea;gr[hi]+=eh;
  }
  for(let i=0;i<2*nt;i++){f+=lam*p[i]*p[i];gr[i]+=2*lam*p[i];}
  return {f,g:gr};
 }
 function fit(tr,lam,tol=1e-5){
  let p=Array(dim).fill(0);p[mi]=Math.log(1.3);let {f,g}=fg(p,tr,lam),S=[],Y=[],R=[],converged=false,status="iteration_limit",iterations=0;
  for(let it=0;it<250;it++){iterations=it;if(norm(g)<tol){converged=true;status="gradient_tolerance";break;}let q=g.slice(),alpha=[];
   for(let k=S.length-1;k>=0;k--){alpha[k]=R[k]*dot(S[k],q);for(let j=0;j<dim;j++)q[j]-=alpha[k]*Y[k][j];}
   let gamma=1;if(S.length){let k=S.length-1;gamma=dot(S[k],Y[k])/Math.max(1e-20,dot(Y[k],Y[k]));}
   let r=q.map(v=>v*gamma);for(let k=0;k<S.length;k++){const beta=R[k]*dot(Y[k],r);for(let j=0;j<dim;j++)r[j]+=S[k][j]*(alpha[k]-beta);}
   let d=r.map(v=>-v),gd=dot(g,d);if(gd>=0){d=g.map(v=>-v);gd=-dot(g,g);S=[];Y=[];R=[];}
   let step=1,np,nf,ng,accepted=false;for(let ls=0;ls<35;ls++){np=p.map((v,j)=>v+step*d[j]);const z=fg(np,tr,lam);nf=z.f;ng=z.g;if(Number.isFinite(nf)&&nf<=f+1e-4*step*gd){accepted=true;break;}step*=.5;}
   if(!accepted||step<1e-10){status="line_search_failure";break;}const s=np.map((v,j)=>v-p[j]),y=ng.map((v,j)=>v-g[j]),sy=dot(s,y);
   if(sy>1e-12){S.push(s);Y.push(y);R.push(1/sy);if(S.length>7){S.shift();Y.shift();R.shift();}}
   p=np;f=nf;g=ng;
  }
  if(!converged&&status==="iteration_limit"&&norm(g)<tol){converged=true;status="gradient_tolerance";}return {p,converged,status,iterations,gradientNorm:norm(g)};
 }
 function pred(p,g){const lh=Math.exp(Math.max(-20,Math.min(20,p[mi]+p[hi]+p[g.home]-p[nt+g.away]))),la=Math.exp(Math.max(-20,Math.min(20,p[mi]+p[g.away]-p[nt+g.home])));
  const pois=l=>Array.from({length:10},(_,k)=>{let v=Math.exp(-l);for(let i=1;i<=k;i++)v*=l/i;return v;});
  const hp=pois(lh),ap=pois(la),pr=[0,0,0];let tot=0;
  for(let h=0;h<10;h++)for(let a=0;a<10;a++){const v=hp[h]*ap[a];tot+=v;pr[h>a?0:h===a?1:2]+=v;}return pr.map(v=>v/tot);
 }
 const cls=g=>g.hg>g.ag?0:g.hg===g.ag?1:2;
 const results={matches:matches.length,teams:teams.length,byStart:{},thresholdSensitivity:{}};
 function evaluate(start,lam,tol,excludeFailed){
  let correct=0,brier=0,ll=0,draw=0,failed=0,used=0,iterationsAtLimit=0,gradientFailures=0,lineSearchFailures=0,otherFailures=0,failedByReason={},failureDetails=[],maxConvergedGradientNorm=0,maxConvergedIterations=0;
  for(let i=start;i<matches.length;i++){
   const f=fit(matches.slice(0,i),lam,tol);
   if(!f.converged){failed++;failedByReason[f.status]=(failedByReason[f.status]||0)+1;
    if(f.status==="iteration_limit")iterationsAtLimit++;else if(f.status==="line_search_failure")lineSearchFailures++;else otherFailures++;
    failureDetails.push({trainMatches:i,evalDate:matches[i].date,evalHome:teams[matches[i].home],evalAway:teams[matches[i].away],reason:f.status,iterations:f.iterations,gradientNorm:f.gradientNorm});
    if(excludeFailed)continue;
   } else {maxConvergedGradientNorm=Math.max(maxConvergedGradientNorm,f.gradientNorm);maxConvergedIterations=Math.max(maxConvergedIterations,f.iterations);}
   const pr=pred(f.p,matches[i]),actual=cls(matches[i]),guess=pr.indexOf(Math.max(...pr));used++;if(guess===actual)correct++;if(guess===1)draw++;for(let k=0;k<3;k++)brier+=(pr[k]-(k===actual?1:0))**2;ll-=Math.log(Math.max(1e-15,pr[actual]));
  }
  return {evaluated:matches.length-start,used,excluded:excludeFailed?failed:0,accuracy:used?correct/used:null,brier:used?brier/used:null,logLoss:used?ll/used:null,drawPredictions:draw,convergenceFailures:failed,failedByReason,iterationLimit:iterationsAtLimit,lineSearchFailure:lineSearchFailures,otherFailure:otherFailures,tolerance:tol,maxConvergedGradientNorm,maxConvergedIterations,failureDetails};
 }
 for(const start of [60,100,150]){const out={};for(const lam of [10,30]){
  out["lambda"+lam]={primary:evaluate(start,lam,1e-5,true),thresholdRelaxed:evaluate(start,lam,1e-4,true)};
 }results.byStart[start]=out;}
 return results;
}
const output={generatedAt:new Date().toISOString(),spec:"walk-forward, starts 60/100/150, unconstrained L-BFGS Armijo, max iterations 250; failed fits excluded from metrics; compare gradient norm tolerance 1e-5 vs 1e-4; per-fit failure details included; score grid 0..9, normalize grid",2025:run(rows),2026:run(hist)};
console.log(JSON.stringify(output,null,2));
