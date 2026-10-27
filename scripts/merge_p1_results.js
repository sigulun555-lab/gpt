#!/usr/bin/env node
"use strict";
const fs=require("node:fs"),path=require("node:path");
const ROOT=process.cwd(), PARTS=path.join(ROOT,"p1-parts");
function findFiles(dir,name,out=[]){if(!fs.existsSync(dir))return out;for(const e of fs.readdirSync(dir,{withFileTypes:true})){const p=path.join(dir,e.name);if(e.isDirectory())findFiles(p,name,out);else if(e.name===name)out.push(p);}return out;}
const resultFiles=findFiles(PARTS,"p1-2026-results.json").sort();
if(resultFiles.length!==12)throw new Error("Expected 12 P1 shard results, found "+resultFiles.length);
const shards=resultFiles.map(file=>({file,data:JSON.parse(fs.readFileSync(file,"utf8"))}));
const conditions=shards.flatMap(s=>s.data.conditions||[]);
if(conditions.length!==60)throw new Error("Expected 60 condition summaries (12 shards x 5 models), found "+conditions.length);
const fitFailures=shards.reduce((n,s)=>n+(s.data.fit_failure_count||0),0);
const byReason={};for(const s of shards)for(const [k,v] of Object.entries(s.data.fit_failures_by_reason||{}))byReason[k]=(byReason[k]||0)+v;
const merged={generatedAt:new Date().toISOString(),spec:{evaluation:"2026 walk-forward; starts 60/100/150; lambda 10/30",train_modes:["2026_only","2025_plus_2026"],models:["baseline","home_away_attack_defense","recency_half_life_60","recency_half_life_120","recency_half_life_240"],recency:"weight=2^(-age_days/half_life_days), normalized to mean 1 separately for each evaluation fit",convergence:"unbounded model; maximum absolute raw gradient is the projected-gradient infinity norm; threshold <=1e-5; failed fits retained",interval:"paired percentile bootstrap 4000 resamples, fixed deterministic seeds",failed_fits_retained:true,shards:resultFiles.length},rows_2025:shards[0].data.rows_2025,rows_2026:shards[0].data.rows_2026,conditions,fit_failure_count:fitFailures,fit_failures_by_reason:byReason};
fs.writeFileSync(path.join(ROOT,"p1-2026-results.json"),JSON.stringify(merged,null,2)+"\n");
function mergeCsv(name,fallback){const files=findFiles(PARTS,name).sort();if(!files.length)throw new Error("Missing shard CSV "+name);let header=null,rows=[];for(const file of files){const lines=fs.readFileSync(file,"utf8").split(/\r?\n/).filter((line,i,a)=>line.length>0||i<a.length-1);if(!lines.length||!lines[0])continue;if(!header&&lines[0].includes("train_mode"))header=lines[0];const start=lines[0]===header?1:(lines[0].includes("train_mode")?1:0);for(const line of lines.slice(start))if(line.trim())rows.push(line);}header=header||fallback;fs.writeFileSync(path.join(ROOT,name),[header,...rows].join("\n")+"\n");return {files:files.length,rows:rows.length};}
const metrics=mergeCsv("p1-2026-metrics.csv","train_mode,start,lambda,model,half_life_days,evaluated,accuracy,brier,log_loss,convergence_failures,all_fits_converged,accuracy_diff,brier_diff,log_loss_diff,ci_note");
const perMatch=mergeCsv("p1-2026-per-match.csv","train_mode,start,lambda,model,half_life_days,date,competition,home,away,correct,brier,log_loss,p_home,p_draw,p_away,objective,gradient_max_abs,gradient_l2,converged,termination_reason,iterations");
const failures=mergeCsv("p1-2026-fit-failures.csv","train_mode,start,lambda,model,date,competition,home,away,reason,iterations,gradient_max_abs,gradient_l2");
console.log(JSON.stringify({event:"merge_complete",shards:resultFiles.length,conditions:conditions.length,fit_failure_count:fitFailures,metrics,perMatch,failures},null,2));
