#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
chunk_dir=ROOT/"audit-chunks"
rows=[]
for p in sorted(chunk_dir.glob("p1-scipy-failure-audit-*.csv")):
    with p.open(encoding="utf-8-sig",newline="") as f:
        rows.extend(csv.DictReader(f))
with (ROOT/"p1-2026-results.json").open(encoding="utf-8") as f:
    experiment=json.load(f)
expected=int(experiment["fit_failure_count"])
keys=lambda r:(r.get("train_mode"),r.get("start"),r.get("lambda"),r.get("model"),r.get("date"),r.get("competition"),r.get("home"),r.get("away"))
if len(rows)!=expected:
    raise SystemExit(f"Expected {expected} audited fits, got {len(rows)}")
if len({keys(r) for r in rows})!=len(rows):
    raise SystemExit("Duplicate fit keys in SciPy audit output")
errors=[r for r in rows if r.get("audit_error")]
passes=[r for r in rows if r.get("pass_max_abs_dP_1e-4","").lower()=="true"]
valid=[r for r in rows if r.get("max_abs_dP") not in ("",None)]
fails=[r for r in valid if r.get("pass_max_abs_dP_1e-4","").lower()!="true"]
status_counts={}
for r in rows: status_counts[r.get("reason","unknown")]=status_counts.get(r.get("reason","unknown"),0)+1
values=[float(r["max_abs_dP"]) for r in valid]
worst=max(valid,key=lambda r:float(r["max_abs_dP"])) if valid else None
summary={
 "experiment_run_id":40518879108,
 "expected_failed_fits":expected,"audited_rows":len(rows),"valid_comparisons":len(valid),
 "audit_errors":len(errors),"pass_max_abs_dP_le_1e-4":len(passes),
 "fail_max_abs_dP_gt_1e-4":len(fails),"status_counts":status_counts,
 "max_abs_dP_observed":max(values) if values else None,
 "median_abs_dP":sorted(values)[len(values)//2] if values else None,
 "worst_fit":({k:worst.get(k) for k in ("train_mode","start","lambda","model","date","competition","home","away","reason","gradient_max_abs","scipy_gmax","max_abs_dP","scipy_message","audit_error")} if worst else None),
 "pass_threshold":1e-4,
 "scipy_success_is_not_equivalent_to_gnorm_pass":True,
 "notes":["Every fit remains in the merged output, including SciPy errors and dP threshold failures.",
          "SciPy success/message and SciPy gmax are reported separately because relative-reduction termination does not guarantee gtol=1e-10."]
}
fields=list(rows[0].keys()) if rows else []
with (ROOT/"p1-scipy-failure-audit.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(ROOT/"p1-scipy-failure-audit-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
if len(errors)>0 or len(valid)!=expected:
    raise SystemExit("Audit incomplete: errors or missing valid dP comparisons")
