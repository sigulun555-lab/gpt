#!/usr/bin/env python3
"""Join per-match JS and SciPy audit CSVs and report every threshold violation."""
import csv, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def locate(name):
    candidates=(ROOT/name, ROOT/"scripts"/name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked=", ".join(str(p) for p in candidates)
    raise SystemExit(f"Required audit CSV not found; checked: {checked}")

JS=locate("lambda-audit-js-per-match.csv")
SCIPY=locate("lambda-audit-scipy-per-match.csv")
OUT=ROOT/"lambda-audit-per-match-comparison.csv"
REPORT=ROOT/"lambda-audit-per-match-comparison.json"
PROB_TOL=1e-4
OBJECTIVE_REL_TOL=1e-6
KEY=("year","start","lambda","tolerance","date","competition","home","away")

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

js=[r for r in read(JS) if float(r["tolerance"]) == 1e-5]
sp=[r for r in read(SCIPY) if float(r["tolerance"]) == 1e-5]
def key(r): return tuple(format(float(r[k]), ".0e") if k == "tolerance" else r[k] for k in KEY)
jm={key(r):r for r in js}; sm={key(r):r for r in sp}
if len(jm)!=len(js) or len(sm)!=len(sp):
    raise SystemExit("Duplicate match keys found; refusing ambiguous join")
keys=sorted(set(jm)|set(sm))
joined=[]; violations=[]; missing_js=[]; missing_scipy=[]
for k in keys:
    a=jm.get(k); b=sm.get(k)
    if a is None: missing_js.append(dict(zip(KEY,k))); continue
    if b is None: missing_scipy.append(dict(zip(KEY,k))); continue
    probs=[abs(float(a[x])-float(b[x])) for x in ("p_home","p_draw","p_away")]
    pmax=max(probs)
    oa=float(a["objective"]); ob=float(b["objective"])
    orel=abs(oa-ob)/max(abs(oa),abs(ob),1e-12)
    row={name:value for name,value in zip(KEY,k)}
    row.update({f"js_{x}":a[x] for x in ("p_home","p_draw","p_away","objective","gradient_l2","gradient_max_abs","converged","termination_reason","iterations")})
    row.update({f"scipy_{x}":b[x] for x in ("p_home","p_draw","p_away","objective","gradient_l2","gradient_max_abs","converged","termination_reason","iterations","scipy_message")})
    row.update(prob_diff_home=probs[0],prob_diff_draw=probs[1],prob_diff_away=probs[2],prob_diff_max=pmax,objective_relative_diff=orel,
               probability_pass=pmax<=PROB_TOL,objective_pass=orel<=OBJECTIVE_REL_TOL,
               overall_pass=pmax<=PROB_TOL and orel<=OBJECTIVE_REL_TOL)
    joined.append(row)
    if not row["overall_pass"]:
        violations.append({**{name:row[name] for name in KEY},"prob_diff_home":probs[0],"prob_diff_draw":probs[1],"prob_diff_away":probs[2],"prob_diff_max":pmax,"objective_relative_diff":orel,"probability_pass":pmax<=PROB_TOL,"objective_pass":orel<=OBJECTIVE_REL_TOL,
                           "js_objective":oa,"scipy_objective":ob,"js_converged":a["converged"],"js_reason":a["termination_reason"],"scipy_converged":b["converged"],"scipy_reason":b["termination_reason"],"scipy_message":b["scipy_message"]})
fields=list(joined[0]) if joined else list(KEY)
with OUT.open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(joined)
report={"keys":["year","start","lambda","tolerance","date","competition","home","away"],"compared_tolerance":1e-5,"probability_max_abs_threshold":PROB_TOL,"objective_relative_threshold":OBJECTIVE_REL_TOL,
        "js_rows":len(js),"scipy_rows":len(sp),"matched_rows":len(joined),"missing_js_count":len(missing_js),"missing_scipy_count":len(missing_scipy),
        "matched_rows_expected_equal":len(js)==len(sp)==len(joined),"probability_max_abs_observed":max((r["prob_diff_max"] for r in joined),default=None),
        "objective_relative_max_observed":max((r["objective_relative_diff"] for r in joined),default=None),"threshold_violation_count":len(violations),
        "threshold_violations":violations,"missing_js_keys":missing_js,"missing_scipy_keys":missing_scipy,
        "gtol_note":"SciPy L-BFGS-B gtol is based on the projected gradient infinity norm. No bounds are supplied here, so projected gradient equals the raw gradient. gradient_l2 is a separate L2 diagnostic; gradient_max_abs is the raw gradient infinity norm, not necessarily the internal projected-gradient stopping test.",
        "status":"pass" if len(joined)==len(js)==len(sp) and not missing_js and not missing_scipy and not violations else "needs_review"}
REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ("threshold_violations","missing_js_keys","missing_scipy_keys")},ensure_ascii=False,indent=2))
