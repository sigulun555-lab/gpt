#!/usr/bin/env python3
"""Audit-only cold-start precision-stall diagnostic for one representative P1 fit."""
import csv, json, math, os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
TARGET_DATE = "2026-09-06"
TARGET_HOME = "ヴィッセル神戸"
TARGET_AWAY = "V・ファーレン長崎"
LAMBDA = 10.0
STALL_STEPS = 3
TOL_REPORT = 1e-5

aliases = json.loads((ROOT / "data/team_aliases.json").read_text(encoding="utf-8"))["aliases"]
canon = lambda name: aliases.get(name, name)

def parse_2025():
    rows=[]
    with (ROOT / "data/2025_schedule_with_ids.csv").open(encoding="utf-8-sig", newline="") as f:
        for o in csv.DictReader(f):
            score = (o.get("score") or "").strip()
            import re
            m=re.match(r"^(\d+)-(\d+)$", score)
            if not m: continue
            date=(o.get("date") or "").replace("/", "-")
            if date < "2025-02-14" or date > "2025-12-06" or o.get("category") not in ("1","2","3"): continue
            rows.append({"date":date,"competition":{"1":"J1","2":"J2","3":"J3"}[o["category"]],
                         "home":canon(o["homeTeam"]),"away":canon(o["awayTeam"]),
                         "hg":int(m.group(1)),"ag":int(m.group(2))})
    return sorted(rows,key=lambda r:(r["date"],r["competition"]))

def parse_2026():
    src=json.loads((ROOT / "data/history.json").read_text(encoding="utf-8"))
    rows=[{"date":g["date"],"competition":g["competition"],"home":canon(g["home"]),"away":canon(g["away"]),
           "hg":int(g["home_goals"]),"ag":int(g["away_goals"])}
          for g in src["history"] if g["competition"] in ("J1","J2","J3")]
    return sorted(rows,key=lambda r:(r["date"],r["competition"]))

def exp_clamped(x): return math.exp(max(-20.0,min(20.0,float(x))))

def make_arrays(train, teams):
    ix={t:i for i,t in enumerate(teams)}
    return [(ix[r["home"]],ix[r["away"]],r["hg"],r["ag"]) for r in train]

def objective_grad(p, data, n, lam):
    dim=len(p); grad=np.zeros(dim); f=0.0
    mu=p[2*n]; adv=p[2*n+1]
    for h,a,hg,ag in data:
        lh=exp_clamped(mu+adv+p[h]-p[n+a])
        la=exp_clamped(mu+p[a]-p[n+h])
        eh=lh-hg; ea=la-ag
        f += lh - hg*math.log(lh) + la - ag*math.log(la)
        grad[h]+=eh; grad[n+a]-=eh
        grad[a]+=ea; grad[n+h]-=ea
        grad[2*n]+=eh+ea; grad[2*n+1]+=eh
    f += lam*float(np.dot(p[:2*n],p[:2*n]))
    grad[:2*n] += 2*lam*p[:2*n]
    return f,grad

def fit_lbfgs_precision_stall(data,n,lam):
    p=np.zeros(2*n+2); p[2*n]=math.log(1.3)
    f,g=objective_grad(p,data,n,lam)
    S=[];Y=[];R=[];status="iteration_limit"; iterations=0; unchanged=0; trace=[]
    max_iter=4000
    for it in range(max_iter):
        iterations=it
        if float(np.max(np.abs(g))) <= TOL_REPORT:
            status="max_projected_gradient"; break
        q=g.copy(); alphas=[]
        for k in range(len(S)-1,-1,-1):
            alpha=R[k]*float(np.dot(S[k],q)); alphas.insert(0,alpha); q-=alpha*Y[k]
        gamma=1.0
        if S:
            k=len(S)-1; gamma=float(np.dot(S[k],Y[k]))/max(1e-20,float(np.dot(Y[k],Y[k])))
        rr=gamma*q
        for k in range(len(S)):
            beta=R[k]*float(np.dot(Y[k],rr)); rr += S[k]*(alphas[k]-beta)
        d=-rr; gd=float(np.dot(g,d))
        if gd>=0:
            d=-g.copy(); gd=-float(np.dot(g,g)); S=[];Y=[];R=[]
        step=1.0; accepted=False; np_=None; nf=None; ng=None
        for _ in range(60):
            cand=p+step*d
            cf,cg=objective_grad(cand,data,n,lam)
            if math.isfinite(cf) and cf <= f+1e-4*step*gd:
                accepted=True; np_=cand; nf=cf; ng=cg; break
            step*=0.5
        if not accepted or step<1e-14:
            status="line_search_failure"; break
        delta=abs(nf-f)
        ulp=math.ulp(f)
        unchanged = unchanged+1 if delta <= 4.0*ulp else 0
        s=np_-p; y=ng-g; sy=float(np.dot(s,y))
        if sy>1e-12:
            S.append(s);Y.append(y);R.append(1.0/sy)
            if len(S)>10: S.pop(0);Y.pop(0);R.pop(0)
        p=np_; f=nf; g=ng
        if it < 70 or unchanged:
            trace.append({"iteration":it+1,"objective":f,"delta_f":delta,"four_ulp":4*ulp,
                          "gmax":float(np.max(np.abs(g))),"step":step,"unchanged_count":unchanged})
        if unchanged>=STALL_STEPS:
            status="precision_stall"; iterations=it+1; break
    return {"x":p,"objective":f,"gradient":g,"status":status,"iterations":iterations,"trace":trace}

def probabilities(p,n,home,away):
    lh=exp_clamped(p[2*n]+p[2*n+1]+p[home]-p[n+away])
    la=exp_clamped(p[2*n]+p[away]-p[n+home])
    def pois(l):
        out=[math.exp(-l)]
        for k in range(1,10): out.append(out[-1]*l/k)
        return out
    hp,ap=pois(lh),pois(la); pr=[0.0,0.0,0.0]; total=0.0
    for h in range(10):
        for a in range(10):
            v=hp[h]*ap[a]; total+=v; pr[0 if h>a else 1 if h==a else 2]+=v
    return [v/total for v in pr]

def main():
    rows25=parse_2025(); rows26=parse_2026()
    target_home,target_away=canon(TARGET_HOME),canon(TARGET_AWAY)
    indices=[i for i,r in enumerate(rows26) if r["date"]==TARGET_DATE and r["home"]==target_home and r["away"]==target_away]
    if len(indices)!=1: raise RuntimeError(f"Expected one target row, got {len(indices)}: {indices}")
    idx=indices[0]; target=rows26[idx]
    train=rows25+rows26[:idx]
    teams=sorted({x for r in rows25+rows26 for x in (r["home"],r["away"])})
    data=make_arrays(train,teams); n=len(teams)
    js=fit_lbfgs_precision_stall(data,n,LAMBDA)
    def fun(x):
        f,g=objective_grad(x,data,n,LAMBDA); return f,g
    x0=np.zeros(2*n+2); x0[2*n]=math.log(1.3)
    scipy=minimize(fun,x0,jac=True,method="L-BFGS-B",
                   options={"gtol":1e-10,"ftol":0.0,"maxiter":10000,"maxfun":1000000,"maxls":100,"maxcor":10})
    ix={t:i for i,t in enumerate(teams)}
    p_js=probabilities(js["x"],n,ix[target_home],ix[target_away])
    p_sp=probabilities(scipy.x,n,ix[target_home],ix[target_away])
    max_dp=0.0; max_pair=None
    for h in range(n):
        for a in range(n):
            if h==a: continue
            a1=probabilities(js["x"],n,h,a); a2=probabilities(scipy.x,n,h,a)
            d=max(abs(x-y) for x,y in zip(a1,a2))
            if d>max_dp: max_dp=d; max_pair={"home":teams[h],"away":teams[a],"js":a1,"scipy":a2,"max_abs_diff":d}
    out={
      "target":{"date":target["date"],"competition":target["competition"],"home":target_home,"away":target_away,
                "training_rows_2025":len(rows25),"training_rows_2026_before_target":idx,"total_training_rows":len(train),
                "team_count":n,"lambda":LAMBDA,"start_mode":"cold_start"},
      "precision_stall_fit":{"status":js["status"],"iterations":js["iterations"],"objective":js["objective"],
                "gmax":float(np.max(np.abs(js["gradient"]))),"gradient_l2":float(np.linalg.norm(js["gradient"])),
                "threshold_1e-5_met":bool(np.max(np.abs(js["gradient"]))<=TOL_REPORT),"trace":js["trace"]},
      "scipy_reference":{"success":bool(scipy.success),"message":str(scipy.message),"nit":int(scipy.nit),
                "objective":float(scipy.fun),"gmax":float(np.max(np.abs(scipy.jac)))},
      "target_match_probabilities":{"outcomes":["home_win","draw","away_win"],"precision_stall":p_js,"scipy":p_sp,
                "max_abs_diff":max(abs(x-y) for x,y in zip(p_js,p_sp))},
      "all_ordered_distinct_team_pairs":{"pair_count":n*(n-1),"max_abs_probability_diff":max_dp,"worst_pair":max_pair,
                "pass_threshold_1e-4":bool(max_dp<=1e-4)},
      "note":"One representative fit only. This is a cold-start diagnostic, not a full-matrix rerun. Gradient threshold is reported, not used as the sole stopping criterion after precision stall."
    }
    (ROOT/"p1-precision-stall-diagnostic.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
