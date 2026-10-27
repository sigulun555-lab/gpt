#!/usr/bin/env python3
"""Independent audit-only walk-forward fit using SciPy L-BFGS-B.
Keeps the JS objective, regularization, starting point, score grid, and evaluation windows.
Never writes predictions or touches production files.
"""
import csv, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = (10,) if __import__("os").environ.get("AUDIT_SMOKE") == "1" else (10, 30)
STARTS = (60,) if __import__("os").environ.get("AUDIT_SMOKE") == "1" else (60, 100, 150)
TOL = 1e-5
EVAL_LIMIT = max(0, int(__import__("os").environ.get("AUDIT_MAX_EVAL_MATCHES", "0")))
MAX_ITERATIONS = max(1, int(__import__("os").environ.get("AUDIT_MAX_ITERATIONS", "2000")))
AUDIT_SMOKE = __import__("os").environ.get("AUDIT_SMOKE") == "1"

def load_2025():
    p = ROOT / "data/2025_schedule_with_ids.csv"
    rows=[]
    with p.open(encoding="utf-8-sig", newline="") as f:
        for o in csv.DictReader(f):
            score = (o.get("score") or "").strip()
            if "-" not in score: continue
            a,b=score.split("-",1)
            if not (a.isdigit() and b.isdigit()): continue
            date=(o.get("date") or "").replace("/","-")
            if date < "2025-02-14" or date > "2025-12-06": continue
            cat=str(o.get("category") or "")
            if cat not in ("1","2","3"): continue
            rows.append(dict(date=date,competition={"1":"J1","2":"J2","3":"J3"}[cat],
                             home=o["homeTeam"],away=o["awayTeam"],hg=int(a),ag=int(b)))
    return sorted(rows,key=lambda g:(g["date"],g["competition"]))

def load_2026():
    raw=json.loads((ROOT/"data/history.json").read_text(encoding="utf-8"))
    rows=[]
    for g in raw["history"]:
        if g.get("competition") not in ("J1","J2","J3"): continue
        rows.append(dict(date=g["date"],competition=g["competition"],home=g["home"],away=g["away"],
                         hg=int(g["home_goals"]),ag=int(g["away_goals"])))
    return sorted(rows,key=lambda g:(g["date"],g["competition"]))

def audit(data, year):
    teams=sorted({g["home"] for g in data}|{g["away"] for g in data})
    ti={t:i for i,t in enumerate(teams)}
    rows=[{**g,"home":ti[g["home"]],"away":ti[g["away"]]} for g in data]
    nt=len(teams); dim=2*nt+2; mi=2*nt; hi=mi+1
    actual=np.array([0 if g["hg"]>g["ag"] else 1 if g["hg"]==g["ag"] else 2 for g in rows],dtype=int)
    def fg(p, train, lam):
        mu=p[mi]; h=p[hi]
        ih=np.array([g["home"] for g in train]); ia=np.array([g["away"] for g in train])
        hg=np.array([g["hg"] for g in train],dtype=float); ag=np.array([g["ag"] for g in train],dtype=float)
        lh=np.exp(np.clip(mu+h+p[ih]-p[nt+ia],-20,20))
        la=np.exp(np.clip(mu+p[ia]-p[nt+ih],-20,20))
        f=float(np.sum(lh-hg*np.log(lh)+la-ag*np.log(la))+lam*np.dot(p[:2*nt],p[:2*nt]))
        gr=np.zeros(dim)
        eh=lh-hg; ea=la-ag
        np.add.at(gr,ih,eh); np.add.at(gr,nt+ia,-eh)
        np.add.at(gr,ia,ea); np.add.at(gr,nt+ih,-ea)
        gr[mi]=np.sum(eh+ea); gr[hi]=np.sum(eh)
        gr[:2*nt]+=2*lam*p[:2*nt]
        return f,gr
    def fit(train, lam):
        p0=np.zeros(dim); p0[mi]=math.log(1.3)
        result=minimize(lambda p:fg(p,train,lam),p0,jac=True,method="L-BFGS-B",
            options={"maxiter":MAX_ITERATIONS,"maxls":60,"gtol":TOL,"ftol":1e-12,"maxcor":10})
        gradnorm=float(np.linalg.norm(result.jac))
        # Treat SciPy's termination as converged only if success and gradient meets tolerance.
        converged=bool(result.success and gradnorm < TOL)
        status="converged" if converged else str(result.message)
        return result.x,converged,status,int(result.nit),gradnorm,float(result.fun)
    def predict(p,g):
        lh=math.exp(max(-20,min(20,p[mi]+p[hi]+p[g["home"]]-p[nt+g["away"]])))
        la=math.exp(max(-20,min(20,p[mi]+p[g["away"]]-p[nt+g["home"]])))
        def pois(l):
            arr=[math.exp(-l)]
            for k in range(1,10): arr.append(arr[-1]*l/k)
            return arr
        hp=pois(lh); ap=pois(la); pr=[0.,0.,0.]; total=0.
        for h in range(10):
            for a in range(10):
                v=hp[h]*ap[a]; total+=v; pr[0 if h>a else 1 if h==a else 2]+=v
        return np.array(pr)/total
    output={"matches":len(rows),"teams":nt,"byStart":{}}
    for start in STARTS:
        out={}
        for lam in LAMBDAS:
            correct=0; brier=0.; ll=0.; draws=0; failed=0; maxgrad=0.; maxiter=0; failure_reasons={}
            # Every evaluation match is retained, including fits SciPy marks unsuccessful.
            for i in range(start,min(len(rows), start + EVAL_LIMIT) if EVAL_LIMIT else len(rows)):
                p,ok,status,nit,gn,fun=fit(rows[:i],lam)
                if not ok:
                    failed+=1; failure_reasons[status]=failure_reasons.get(status,0)+1
                else:
                    maxgrad=max(maxgrad,gn); maxiter=max(maxiter,nit)
                pr=predict(p,rows[i]); y=actual[i]; guess=int(np.argmax(pr))
                correct+=int(guess==y); draws+=int(guess==1)
                brier+=float(np.sum((pr-np.eye(3)[y])**2)); ll-=math.log(max(1e-15,float(pr[y])))
            n=min(EVAL_LIMIT, len(rows)-start) if EVAL_LIMIT else len(rows)-start
            out[f"lambda{lam}"]={"evaluated":n,"used":n,"excluded":0,"accuracy":correct/n,
                "brier":brier/n,"logLoss":ll/n,"drawPredictions":draws,"convergenceFailures":failed,
                "failedByReason":failure_reasons,"tolerance":TOL,"maxConvergedGradientNorm":maxgrad,
                "maxConvergedIterations":maxiter,"optimizer":"scipy.optimize.minimize L-BFGS-B",
                "options":{"maxiter":MAX_ITERATIONS,"maxls":60,"gtol":TOL,"ftol":1e-12,"maxcor":10}}
        output["byStart"][str(start)]=out
    return output

result={"generatedAt":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "spec":"Independent SciPy L-BFGS-B implementation; same Poisson negative log-likelihood, L2 penalty, zero init except log(1.3) intercept, walk-forward splits, score grid 0..9. All evaluation matches retained even when optimizer reports failure; failure counts shown explicitly. eval cap=" + str(EVAL_LIMIT or "none") + ", maxiter=" + str(MAX_ITERATIONS) + ", smoke=" + str(AUDIT_SMOKE),
        "2025":audit(load_2025(),"2025"),"2026":audit(load_2026(),"2026")}
print(json.dumps(result,ensure_ascii=False,indent=2))
