#!/usr/bin/env node
"use strict";
const fs = require("node:fs"), path = require("node:path"), assert = require("node:assert/strict");
const root = process.env.P1_SHARD_ARTIFACT_DIR || "shard-artifacts";
const expected = [];
for (const mode of ["2026_only", "2025_plus_2026"]) for (const start of [60,100,150]) for (const lambda of [10,30]) expected.push("p1-part-"+mode+"-s"+start+"-l"+lambda);
const required = ["p1-2026-results.json","p1-2026-metrics.csv","p1-2026-per-match.csv","p1-2026-fit-failures.csv"];
function parseLine(line) { const out=[]; let s="", q=false; for(let i=0;i<line.length;i++){const c=line[i]; if(c==='"'){if(q&&line[i+1]==='"'){s+='"';i++;}else q=!q;} else if(c===","&&!q){out.push(s);s="";}else s+=c;}out.push(s);return out; }
function readCsv(file){const raw=fs.readFileSync(file,"utf8").replace(/^\uFEFF/,"").trimEnd();assert(raw.length>0,"empty CSV "+file);const lines=raw.split(/\r?\n/),header=parseLine(lines.shift());assert(header.length>1,"invalid CSV header "+file);const rows=lines.map((line,i)=>{const v=parseLine(line);assert.equal(v.length,header.length,"CSV column mismatch "+file+":"+(i+2));return Object.fromEntries(header.map((h,j)=>[h,v[j]]));});return {header,rows};}
function esc(v){const s=String(v??"");return /[",\n\r]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
function writeCsv(file,header,rows){fs.writeFileSync(file,[header.join(","),...rows.map(r=>header.map(h=>esc(r[h])).join(","))].join("\n")+"\n");}
function n(v,label){const x=Number(v);assert(Number.isFinite(x),"invalid "+label+": "+v);return x;}
function close(a,b,label){assert(Math.abs(a-b)<=1e-10,label+": "+a+" != "+b);}
function mk(r){return [r.train_mode,r.start,r.lambda,r.model].join("|");}
function fk(r){return [r.train_mode,r.start,r.lambda,r.model,r.date,r.competition,r.home,r.away].join("|");}
const dirs=fs.readdirSync(root,{withFileTypes:true}).filter(d=>d.isDirectory()).map(d=>d.name).sort();
assert.equal(dirs.length,12,"expected exactly 12 shard folders");assert.deepEqual(dirs,[...expected].sort(),"missing/unexpected shard folder");
let ref25,ref26,refSpec;const metrics=[],matches=[],failures=[],manifest=[];
for(const name of dirs){
 const dir=path.join(root,name);
 for(const file of required){const p=path.join(dir,file);assert(fs.existsSync(p),"missing "+name+"/"+file);assert(fs.statSync(p).size>20,"too small "+name+"/"+file);}
 const result=JSON.parse(fs.readFileSync(path.join(dir,required[0]),"utf8"));
 assert(result.spec&&Array.isArray(result.conditions),"invalid results JSON "+name);assert.equal(result.conditions.length,5,name+" expected 5 model rows");
 const m=name.match(/^p1-part-(2026_only|2025_plus_2026)-s(\d+)-l(\d+)$/);assert(m,"invalid shard name "+name);
 const mode=m[1],start=Number(m[2]),lambda=Number(m[3]);
 assert.deepEqual(result.spec.train_modes,[mode],name+" train mode mismatch");assert.equal(result.rows_2026,380,name+" rows_2026 mismatch");assert.equal(result.spec.failed_fits_retained,true,name+" failed fits not retained");
 if(ref25===undefined){ref25=result.rows_2025;ref26=result.rows_2026;refSpec=result.spec;}
 assert.equal(result.rows_2025,ref25,name+" rows_2025 inconsistent");assert.equal(result.rows_2026,ref26,name+" rows_2026 inconsistent");
 const mr=readCsv(path.join(dir,"p1-2026-metrics.csv")),pr=readCsv(path.join(dir,"p1-2026-per-match.csv")),fr=readCsv(path.join(dir,"p1-2026-fit-failures.csv"));
 assert.equal(mr.rows.length,5,name+" metrics count");const evalN=380-start;assert.equal(pr.rows.length,evalN*5,name+" per-match count");
 assert.equal(new Set(pr.rows.map(r=>[r.date,r.competition,r.home,r.away].join("|"))).size,evalN,name+" unique match count");
 assert.equal(fr.rows.length,result.fit_failure_count,name+" failure count mismatch");
 const reasons={};for(const r of fr.rows)reasons[r.reason]=(reasons[r.reason]||0)+1;assert.deepEqual(reasons,result.fit_failures_by_reason,name+" failure reason mismatch");
 const jm=new Map(result.conditions.map(r=>[mk(r),r])), byM=new Map();
 for(const r of pr.rows){const k=mk(r);if(!byM.has(k))byM.set(k,[]);byM.get(k).push(r);}
 const fb=new Map(fr.rows.map(r=>[fk(r),r])), bad=pr.rows.filter(r=>r.converged.toLowerCase()!=="true");
 assert.equal(bad.length,fr.rows.length,name+" non-converged rows not fully retained");
 for(const r of bad){const f=fb.get(fk(r));assert(f,name+" failed fit missing from failure CSV");assert.equal(f.reason,r.termination_reason,name+" failure reason mismatch");}
 for(const r of fr.rows)assert(bad.some(p=>fk(p)===fk(r)),name+" orphan failure row");
 for(const mrow of mr.rows){
  const jr=jm.get(mk(mrow)), rows=byM.get(mk(mrow));assert(jr,name+" JSON metric missing "+mk(mrow));assert(rows,name+" per-match metric missing "+mk(mrow));assert.equal(rows.length,evalN,name+" per-match rows per model");
  const acc=rows.reduce((s,r)=>s+n(r.correct,"correct"),0)/rows.length,brier=rows.reduce((s,r)=>s+n(r.brier,"brier"),0)/rows.length,ll=rows.reduce((s,r)=>s+n(r.log_loss,"log_loss"),0)/rows.length,badN=rows.filter(r=>r.converged.toLowerCase()!=="true").length;
  close(acc,n(mrow.accuracy,"accuracy"),name+" accuracy CSV");close(brier,n(mrow.brier,"brier"),name+" Brier CSV");close(ll,n(mrow.log_loss,"log_loss"),name+" log loss CSV");
  close(acc,n(jr.accuracy,"accuracy"),name+" accuracy JSON");close(brier,n(jr.brier,"brier"),name+" Brier JSON");close(ll,n(jr.log_loss,"log_loss"),name+" log loss JSON");
  assert.equal(badN,Number(mrow.convergence_failures),name+" convergence count CSV");assert.equal(badN,Number(jr.convergence_failures),name+" convergence count JSON");
  for(const field of ["accuracy_diff","brier_diff","log_loss_diff"]){const d=JSON.parse(mrow[field]);assert(Number.isFinite(d.difference)&&Number.isFinite(d.ci95_low)&&Number.isFinite(d.ci95_high),name+" invalid "+field);assert(d.ci95_low<=d.ci95_high,name+" inverted "+field);close(d.difference,n(jr[field].difference,field),name+" "+field+" difference");close(d.ci95_low,n(jr[field].ci95_low,field),name+" "+field+" low");close(d.ci95_high,n(jr[field].ci95_high,field),name+" "+field+" high");}
 }
 metrics.push(...mr.rows.map(r=>({...r,start:String(start),lambda:String(lambda)})));
 matches.push(...pr.rows.map(r=>({...r,start:String(start),lambda:String(lambda)})));
 failures.push(...fr.rows.map(r=>({...r,start:String(start),lambda:String(lambda)})));
 manifest.push({artifact:name,train_mode:mode,start,lambda,eval_matches:evalN,metric_rows:mr.rows.length,per_match_rows:pr.rows.length,fit_failure_rows:fr.rows.length,fit_failures_by_reason:reasons});
}
assert.equal(metrics.length,60,"expected 60 model-condition metric rows");assert.equal(matches.length,16600,"expected 16,600 per-match predictions");assert(failures.length>0,"failed fits must remain in output");
const mh=readCsv(path.join(root,dirs[0],"p1-2026-metrics.csv")).header,ph=readCsv(path.join(root,dirs[0],"p1-2026-per-match.csv")).header,fh=readCsv(path.join(root,dirs[0],"p1-2026-fit-failures.csv")).header;
const ord=(a,b)=>({2026_only:0,2025_plus_2026:1}[a.train_mode]-{2026_only:0,2025_plus_2026:1}[b.train_mode]||Number(a.start)-Number(b.start)||Number(a.lambda)-Number(b.lambda)||a.model.localeCompare(b.model);
metrics.sort(ord);matches.sort((a,b)=>ord(a,b)||a.date.localeCompare(b.date)||a.competition.localeCompare(b.competition)||a.home.localeCompare(b.home)||a.away.localeCompare(b.away));failures.sort((a,b)=>ord(a,b)||a.date.localeCompare(b.date)||a.competition.localeCompare(b.competition)||a.home.localeCompare(b.home)||a.away.localeCompare(b.away));
writeCsv("p1-2026-metrics.csv",mh,metrics);writeCsv("p1-2026-per-match.csv",ph,matches);writeCsv("p1-2026-fit-failures.csv",fh,failures);
const byReason={};for(const r of failures)byReason[r.reason]=(byReason[r.reason]||0)+1;
const conditions=metrics.map(r=>({train_mode:r.train_mode,start:Number(r.start),lambda:Number(r.lambda),model:r.model,half_life_days:r.half_life_days===""?null:Number(r.half_life_days),evaluated:Number(r.evaluated),accuracy:Number(r.accuracy),brier:Number(r.brier),log_loss:Number(r.log_loss),convergence_failures:Number(r.convergence_failures),all_fits_converged:r.all_fits_converged.toLowerCase()==="true",accuracy_diff:JSON.parse(r.accuracy_diff),brier_diff:JSON.parse(r.brier_diff),log_loss_diff:JSON.parse(r.log_loss_diff),...(r.model==="baseline"?{}:{ci_note:"paired percentile bootstrap, 4000 resamples, fixed seed; positive accuracy_diff favors candidate, positive Brier/log_loss_diff means candidate is better"})}));
const now=new Date().toISOString(),out={generatedAt:now,spec:{...refSpec,train_modes:["2026_only","2025_plus_2026"],starts:[60,100,150],lambdas:[10,30],matrix_shards:12,metric_rows:60,failed_fits_retained:true,aggregation_validation:"all 12 shards checked; per-match metrics independently recomputed; all non-converged fits matched to failure rows"},rows_2025:ref25,rows_2026:ref26,matrix_shard_count:12,condition_count:12,metric_row_count:conditions.length,per_match_prediction_count:matches.length,fit_failure_count:failures.length,fit_failures_by_reason:byReason,conditions,shard_manifest:manifest};
fs.writeFileSync("p1-2026-results.json",JSON.stringify(out,null,2)+"\n");
fs.writeFileSync("p1-2026-validation-manifest.json",JSON.stringify({generatedAt:now,sourceArtifactDirectories:dirs,checks:["12 expected shards","four required files present/non-empty","JSON/CSV parse","per-match metrics independently recomputed","failed fits retained and reason-matched","95% interval fields finite and ordered"],matrix_shard_count:12,metric_row_count:metrics.length,per_match_prediction_count:matches.length,fit_failure_count:failures.length,fit_failures_by_reason:byReason,shards:manifest},null,2)+"\n");
console.log(JSON.stringify({status:"success",matrix_shards:12,metric_rows:metrics.length,per_match_predictions:matches.length,fit_failures:failures.length,fit_failures_by_reason:byReason},null,2));
