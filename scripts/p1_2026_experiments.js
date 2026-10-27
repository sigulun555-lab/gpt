#!/usr/bin/env node
// Audit-only P1 experiments. Writes only result artifacts; never touches production files.
"use strict";
const fs=require("node:fs"), path=require("node:path");
const ROOT=path.join(__dirname,"..");
const aliases=JSON.parse(fs.readFileSync(path.join(ROOT,"data/team_aliases.json"),"utf8")).aliases;
const canonical=n=>aliases[n]||n;
function parseCsvLine(line){const out=[];let s="",q=false;for(let i=0;i<line.length;i++){const c=line[i];if(c==='"'){if(q&&line[i+1]==='"'){s+='"';i++;}else q=!q;}else if(c===","&&!q){out.push(s);s="";}else s+=c;}out.push(s);return out;}
function load2025(){const lines=fs.readFileSync(path.join(ROOT,"data/2025_schedule_with_ids.csv"),"utf8").trim().split(/\r?\n/),heads=parseCsvLine(lines.shift());return lines.map(line=>{const o=Object.fromEntries(heads.map((h,i)=>[h,parseCsvLine(line)[i]]));const m=/^(\d+)-(\d+)$/.exec(o.score||"");if(!m)return null;const date=(o.date||"").replaceAll("/","-");if(date<"2025-02-14"||date>"2025-12-06"||!["1","2","3"].includes(o.category))return null;return {date,competition:({"1":"J1","2":"J2","3":"J3"})[o.category],home:canonical(o.homeTeam),away:canonical(o.awayTeam),hg:+m[1],ag:+m[2]};}).filter(Boolean).sort((a,b)=>a.date.localeCompare(b.date)||a.competition.localeCompare(b.competition));}
function load2026(){const src=JSON.parse(fs.readFileSync(path.join(ROOT,"data/history.json"),"utf8"));return src.history.filter(g=>["J1","J2","J3"].includes(g.competition)).map(g=>({date:g.date,competition:g.competition,home:canonical(g.home),away:canonical(g.away),hg:+g.home_goals,ag:+g.away_goals})).sort((a,b)=>a.date.localeCompare(b.date)||a.competition.localeCompare(b.competition));}
const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;};
const maxabs=a=>a.reduce((m,v)=>Math.max(m,Math.abs(v)),0);
const norm=a=>Math.sqrt(dot(a,a));
const clampExp=x=>Math.exp(Math.max(-20,Math.min(20,x)));
function makeModel(rows,kind,teamUniverse){const teams=[...new Set(teamUniverse||rows.flatMap(g=>[g.home,g.away]))].sort(),ti=new Map(teams.map((t,i)=>[t,i]));return {teams,rows:rows.map(g=>({...g,home:ti.get(g.home),away:ti.get(g.away)})),n:teams.length,kind,dim:(kind==="home_away"?4:2)*teams.length+2};}
function objective(p,train,m,lambda,weights){const n=m.n,ha=m.kind==="home_away",mu=p[(ha?4:2)*n],adv=p[(ha?4:2)*n+1],g=Array(m.dim).fill(0);let f=0;
 for(let j=0;j<train.length;j++){const r=train[j],w=weights?weights[j]:1;let lh,la;
  if(ha){lh=clampExp(mu+adv+p[r.home]-p[3*n+r.away]);la=clampExp(mu+p[2*n+r.away]-p[n+r.home]);
   const eh=w*(lh-r.hg),ea=w*(la-r.ag);f+=w*(lh-r.hg*Math.log(lh)+la-r.ag*Math.log(la));g[r.home]+=eh;g[3*n+r.away]-=eh;g[2*n+r.away]+=ea;g[n+r.home]-=ea;g[4*n]+=eh+ea;g[4*n+1]+=eh;
  }else{lh=clampExp(mu+adv+p[r.home]-p[n+r.away]);la=clampExp(mu+p[r.away]-p[n+r.home]);const eh=w*(lh-r.hg),ea=w*(la-r.ag);f+=w*(lh-r.hg*Math.log(lh)+la-r.ag*Math.log(la));g[r.home]+=eh;g[n+r.away]-=eh;g[r.away]+=ea;g[n+r.home]-=ea;g[2*n]+=eh+ea;g[2*n+1]+=eh;}
 }
 const teamParams=(ha?4:2)*n;for(let i=0;i<teamParams;i++){f+=lambda*p[i]*p[i];g[i]+=2*lambda*p[i];}return {f,g};}
function fit(train,m,lambda,weights,initialP){const p=initialP&&initialP.length===m.dim?initialP.slice():Array(m.dim).fill(0),n=m.n;if(!(initialP&&initialP.length===m.dim))p[(m.kind==="home_away"?4:2)*n]=Math.log(1.3);let z=objective(p,train,m,lambda,weights),f=z.f,g=z.g,S=[],Y=[],R=[],status="iteration_limit",iterations=0;const maxIter=2000,tol=1e-5;
 for(let it=0;it<maxIter;it++){iterations=it;if(maxabs(g)<=tol){status="max_projected_gradient";break;}let q=g.slice(),alpha=[];
  for(let k=S.length-1;k>=0;k--){alpha[k]=R[k]*dot(S[k],q);for(let j=0;j<q.length;j++)q[j]-=alpha[k]*Y[k][j];}
  let gamma=1;if(S.length){const k=S.length-1;gamma=dot(S[k],Y[k])/Math.max(1e-20,dot(Y[k],Y[k]));}
  let r=q.map(v=>gamma*v);for(let k=0;k<S.length;k++){const beta=R[k]*dot(Y[k],r);for(let j=0;j<r.length;j++)r[j]+=S[k][j]*(alpha[k]-beta);}let d=r.map(v=>-v),gd=dot(g,d);
  if(gd>=0){d=g.map(v=>-v);gd=-dot(g,g);S=[];Y=[];R=[];}
  let step=1,np,nz,accepted=false;for(let ls=0;ls<60;ls++){np=p.map((v,j)=>v+step*d[j]);nz=objective(np,train,m,lambda,weights);if(Number.isFinite(nz.f)&&nz.f<=f+1e-4*step*gd){accepted=true;break;}step*=.5;}
  if(!accepted||step<1e-14){status="line_search_failure";break;}const s=np.map((v,j)=>v-p[j]),y=nz.g.map((v,j)=>v-g[j]),sy=dot(s,y);if(sy>1e-12){S.push(s);Y.push(y);R.push(1/sy);if(S.length>10){S.shift();Y.shift();R.shift();}}for(let j=0;j<p.length;j++)p[j]=np[j];f=nz.f;g=nz.g;
 }
 const converged=status==="max_projected_gradient"||maxabs(g)<=tol;if(converged)status="max_projected_gradient";else if(status==="iteration_limit")status="iteration_limit";return {p,objective:f,gradientMaxAbs:maxabs(g),gradientL2:norm(g),converged,status,iterations};}
function predict(p,m,r){const n=m.n,ha=m.kind==="home_away";let lh,la;if(ha){lh=clampExp(p[4*n]+p[4*n+1]+p[r.home]-p[3*n+r.away]);la=clampExp(p[4*n]+p[2*n+r.away]-p[n+r.home]);}else{lh=clampExp(p[2*n]+p[2*n+1]+p[r.home]-p[n+r.away]);la=clampExp(p[2*n]+p[r.away]-p[n+r.home]);}
 const pois=l=>{const a=[Math.exp(-l)];for(let k=1;k<10;k++)a.push(a[k-1]*l/k);return a;},hp=pois(lh),ap=pois(la),pr=[0,0,0];let total=0;for(let h=0;h<10;h++)for(let a=0;a<10;a++){const v=hp[h]*ap[a];total+=v;pr[h>a?0:h===a?1:2]+=v;}return pr.map(v=>v/total);}
function metric(pr,r){const y=r.hg>r.ag?0:r.hg===r.ag?1:2,guess=pr.indexOf(Math.max(...pr));return {correct:guess===y?1:0,brier:pr.reduce((s,v,i)=>s+(v-(i===y?1:0))**2,0),ll:-Math.log(Math.max(1e-15,pr[y])),actual:y,guess};}
function seeded(seed){let x=seed>>>0;return ()=>{x=(1664525*x+1013904223)>>>0;return x/4294967296;};}
function pairedCI(a,b,key,seed){const diffs=a.map((r,i)=>r[key]-b[i][key]),mean=diffs.reduce((s,v)=>s+v,0)/diffs.length,rng=seeded(seed),boots=[],n=diffs.length;for(let k=0;k<4000;k++){let s=0;for(let j=0;j<n;j++)s+=diffs[Math.floor(rng()*n)];boots.push(s/n);}boots.sort((x,y)=>x-y);return {difference:mean,ci95_low:boots[Math.floor(.025*boots.length)],ci95_high:boots[Math.floor(.975*boots.length)]};}
const rows25=load2025(),rows26=load2026();
const allStarts=[60,100,150],allLambdas=[10,30],allTrainModes=["2026_only","2025_plus_2026"];const starts=process.env.P1_START?[Number(process.env.P1_START)]:allStarts,lambdas=process.env.P1_LAMBDA?[Number(process.env.P1_LAMBDA)]:allLambdas,trainModes=process.env.P1_TRAIN_MODE?[process.env.P1_TRAIN_MODE]:allTrainModes,halflives=[60,120,240];if(starts.some(v=>!allStarts.includes(v))||lambdas.some(v=>!allLambdas.includes(v))||trainModes.some(v=>!allTrainModes.includes(v)))throw new Error("Invalid P1 matrix selector environment variable");
const allResults=[],matchResults=[],fitFailures=[];
const teamUniverse=[...new Set(rows25.concat(rows26).flatMap(g=>[g.home,g.away]))];
for(const trainMode of trainModes){for(const start of starts){for(const lambda of lambdas){
 const evalRows=rows26.slice(start);const candidates=[{name:"baseline",kind:"standard",halfLife:null},{name:"home_away_attack_defense",kind:"home_away",halfLife:null},...halflives.map(h=>({name:"recency_half_life_"+h,kind:"standard",halfLife:h}))];
 const predictions=new Map(candidates.map(c=>[c.name,[]]));const warmStarts=new Map(candidates.map(c=>[c.name,null]));const conditionStarted=Date.now();console.log(JSON.stringify({event:"condition_start",train_mode:trainMode,start,lambda,eval_matches:rows26.length-start,candidates:candidates.length}));
 for(let i=start;i<rows26.length;i++){const current=rows26[i],train26=rows26.slice(0,i),train=(trainMode==="2025_plus_2026"?rows25:[]).concat(train26);
  for(const c of candidates){const m=makeModel(train,c.kind,teamUniverse);const ti=new Map(m.teams.map((t,j)=>[t,j]));const mappedTrain=train.map(r=>({...r,home:ti.get(r.home),away:ti.get(r.away)}));const mappedEval={...current,home:ti.get(current.home),away:ti.get(current.away)};
   if(mappedEval.home===undefined||mappedEval.away===undefined)throw new Error("evaluation team missing from training set at "+current.date+" "+current.home+"-"+current.away);
   let weights=null;if(c.halfLife){const dayDiff=d=>Math.max(0,(Date.parse(current.date+"T00:00:00Z")-Date.parse(d+"T00:00:00Z"))/86400000);weights=train.map(r=>Math.pow(2,-dayDiff(r.date)/c.halfLife));const avg=weights.reduce((s,v)=>s+v,0)/weights.length;weights=weights.map(w=>w/avg);}
   const f=fit(mappedTrain,m,lambda,weights,warmStarts.get(c.name));warmStarts.set(c.name,f.p);const pr=predict(f.p,m,mappedEval),met=metric(pr,current);predictions.get(c.name).push({...met,date:current.date,competition:current.competition,home:current.home,away:current.away,p_home:pr[0],p_draw:pr[1],p_away:pr[2],objective:f.objective,gradient_max_abs:f.gradientMaxAbs,gradient_l2:f.gradientL2,converged:f.converged,termination_reason:f.status,iterations:f.iterations});
   if(!f.converged)fitFailures.push({train_mode:trainMode,start,lambda,model:c.name,date:current.date,competition:current.competition,home:current.home,away:current.away,reason:f.status,iterations:f.iterations,gradient_max_abs:f.gradientMaxAbs,gradient_l2:f.gradientL2});
  }
  if((i-start+1)%10===0||i===rows26.length-1)console.log(JSON.stringify({event:"progress",train_mode:trainMode,start,lambda,processed:i-start+1,total:rows26.length-start,elapsed_seconds:Math.round((Date.now()-conditionStarted)/1000),last_date:current.date}));
 }
 const base=predictions.get("baseline"),summarize=(name,arr)=>{const n=arr.length;return {train_mode:trainMode,start,lambda,model:name,half_life_days:candidates.find(c=>c.name===name)?.halfLife??null,evaluated:n,accuracy:arr.reduce((s,r)=>s+r.correct,0)/n,brier:arr.reduce((s,r)=>s+r.brier,0)/n,log_loss:arr.reduce((s,r)=>s+r.ll,0)/n,convergence_failures:arr.filter(r=>!r.converged).length,all_fits_converged:arr.every(r=>r.converged)};};
 for(const c of candidates){const arr=predictions.get(c.name),s=summarize(c.name,arr);if(c.name!=="baseline"){s.accuracy_diff=pairedCI(arr,base,"correct",11+start+lambda+c.name.length);s.brier_diff=pairedCI(base,arr,"brier",21+start+lambda+c.name.length);s.log_loss_diff=pairedCI(base,arr,"ll",31+start+lambda+c.name.length);s.ci_note="paired percentile bootstrap, 4000 resamples, fixed seed; positive accuracy_diff favors candidate, positive Brier/log_loss_diff means candidate is better";}else{s.accuracy_diff={difference:0,ci95_low:0,ci95_high:0};s.brier_diff={difference:0,ci95_low:0,ci95_high:0};s.log_loss_diff={difference:0,ci95_low:0,ci95_high:0};}allResults.push(s);
  for(let j=0;j<arr.length;j++){const r=arr[j];matchResults.push({train_mode:trainMode,start,lambda,model:c.name,half_life_days:c.halfLife??"",date:r.date,competition:r.competition,home:r.home,away:r.away,correct:r.correct,brier:r.brier,log_loss:r.ll,p_home:r.p_home,p_draw:r.p_draw,p_away:r.p_away,objective:r.objective,gradient_max_abs:r.gradient_max_abs,gradient_l2:r.gradient_l2,converged:r.converged,termination_reason:r.termination_reason,iterations:r.iterations});}
 }
}}}
function csvEscape(v){const x=String(v??"");return /[",\n\r]/.test(x)?'"'+x.replace(/"/g,'""')+'"':x;}
function writeCsv(file,rows){const keys=Object.keys(rows[0]||{});fs.writeFileSync(path.join(ROOT,file),[keys.join(","),...rows.map(r=>keys.map(k=>csvEscape(r[k]&&typeof r[k]==="object"?JSON.stringify(r[k]):r[k])).join(","))].join("\n")+"\n");}
writeCsv("p1-2026-metrics.csv",allResults);writeCsv("p1-2026-per-match.csv",matchResults);writeCsv("p1-2026-fit-failures.csv",fitFailures);
const out={generatedAt:new Date().toISOString(),spec:{evaluation:"2026 walk-forward; starts 60/100/150; lambda 10/30",train_modes:trainModes,models:["baseline","home_away_attack_defense","recency_half_life_60","recency_half_life_120","recency_half_life_240"],recency:"weight=2^(-age_days/half_life_days), normalized to mean 1 separately for each evaluation fit",convergence:"unbounded model; maximum absolute raw gradient is the projected-gradient infinity norm; threshold <=1e-5; failed fits retained in metrics and fit-failures CSV",interval:"paired percentile bootstrap 4000 resamples, fixed deterministic seeds",failed_fits_retained:true},rows_2025:rows25.length,rows_2026:rows26.length,conditions:allResults,fit_failure_count:fitFailures.length,fit_failures_by_reason:fitFailures.reduce((o,r)=>(o[r.reason]=(o[r.reason]||0)+1,o),{})};
fs.writeFileSync(path.join(ROOT,"p1-2026-results.json"),JSON.stringify(out,null,2)+"\n");
console.log(JSON.stringify({generatedAt:out.generatedAt,rows_2025:out.rows_2025,rows_2026:out.rows_2026,condition_count:allResults.length,fit_failure_count:out.fit_failure_count,fit_failures_by_reason:out.fit_failures_by_reason,metrics:allResults},null,2));
