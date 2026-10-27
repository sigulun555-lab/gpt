#!/usr/bin/env node
"use strict";
const fs=require("node:fs"),path=require("node:path");
const root=path.join(process.cwd(),"p1-parts");
const dirs=fs.existsSync(root)?fs.readdirSync(root,{withFileTypes:true}).filter(x=>x.isDirectory()).map(x=>path.join(root,x.name)):[];
const jsons=[];
for(const d of dirs){const p=path.join(d,"p1-2026-results.json");if(fs.existsSync(p))jsons.push(JSON.parse(fs.readFileSync(p,"utf8")));}
const expected=12;
if(jsons.length!==expected)throw new Error("Expected "+expected+" shard JSON artifacts, found "+jsons.length+"; do not treat partial results as complete.");
const rows=jsons.flatMap(x=>x.conditions||[]);
const perMatch=dirs.flatMap(d=>{const p=path.join(d,"p1-2026-per-match.csv");return fs.existsSync(p)?[fs.readFileSync(p,"utf8").trimEnd()]:[];});
const failures=dirs.flatMap(d=>{const p=path.join(d,"p1-2026-fit-failures.csv");return fs.existsSync(p)?[fs.readFileSync(p,"utf8").trimEnd()]:[];});
function mergeCsv(chunks){if(!chunks.length)return "";return [chunks[0].split("\n")[0],...chunks.flatMap(x=>x.split("\n").slice(1))].join("\n")+"\n";}
const metricsChunks=dirs.flatMap(d=>{const p=path.join(d,"p1-2026-metrics.csv");return fs.existsSync(p)?[fs.readFileSync(p,"utf8").trimEnd()]:[];});
const out={generatedAt:new Date().toISOString(),status:"complete",shards:jsons.length,rows_2025:jsons[0].rows_2025,rows_2026:jsons[0].rows_2026,conditions:rows,fit_failure_count:jsons.reduce((s,x)=>s+(x.fit_failure_count||0),0),fit_failures_by_reason:jsons.reduce((o,x)=>{for(const [k,v] of Object.entries(x.fit_failures_by_reason||{}))o[k]=(o[k]||0)+v;return o;},{}),spec:jsons[0].spec};
fs.writeFileSync("p1-2026-results.json",JSON.stringify(out,null,2)+"\n");
fs.writeFileSync("p1-2026-metrics.csv",mergeCsv(metricsChunks));
fs.writeFileSync("p1-2026-per-match.csv",mergeCsv(perMatch));
fs.writeFileSync("p1-2026-fit-failures.csv",mergeCsv(failures));
console.log(JSON.stringify({status:out.status,shards:out.shards,conditions:out.conditions.length,fit_failure_count:out.fit_failure_count},null,2));
