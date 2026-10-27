#!/usr/bin/env python3
"""Chunked audit: compare every recorded non-converged P1 fit with a SciPy reference."""
import csv, json, math, os, re
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parent.parent
OFFSET=int(os.environ.get("P1_AUDIT_OFFSET","0"))
LIMIT=int(os.environ.get("P1_AUDIT_LIMIT","1000"))
THRESHOLD=1e-4
aliases=json.loads((ROOT/"data/team_aliases.json").read_text(encoding="utf-8"))["aliases"]
def canon(s): return aliases.get(s,s)

def rows2025():
    out=[]
    with (ROOT/"data/2025_schedule_with_ids.csv").open(encoding="utf-8-sig",newline="") as f:
        for o in csv.DictReader(f):
            m=re.match(r"^(\d+)-(\d+)$",(o.get("score") or "").strip())
            if not m: continue
            date=(o.get("date") or "").replace("/","-")
            if date<"2025-02-14" or date>"2025-12-06" or o.get("category") not in ("1","2","3"): continue
            out.append({"date":date,"competition":{"1":"J1","2":"J2","3":"J3"}[o["category"]],
                        "home":canon(o["homeTeam"]),"away":canon(o["awayTeam"]),
                        "hg":int(m.group(1)),"ag":int(m.group(2))})
    return sorted(out,key=lambda r:(r["date"],r["competition"]))

def rows2026():
    raw=json.loads((ROOT/"data/history.json").read_text(encoding="utf-8"))["history"]
    out=[{"date":g["date"],"competition":g["competition"],"home":canon(g["home"]),"away":canon(g["away"]),
          "hg":int(g["home_goals"]),"ag":int(g["away_goals"])}
         for g in raw if g["competition"] in ("J1","J2","J3")]
    return sorted(out,key=lambda r:(r["date"],r["competition"]))

def expc(x): return np.exp(np.clip(x,-20,20))
def map_rows(rows,idx): return [(idx[r["home"]],idx[r["away"]],r["hg"],r["ag"]) for r in rows]

def objgrad(p,data,n,lam,kind,weights=None):
    ha=kind=="home_away_attack_defense"
    base=4*n if ha else 2*n
    mu=p[base]; adv=p[base+1]
    g=np.zeros_like(p); f=0.0
    for j,(h,a,hg,ag) in enumerate(data):
        w=1.0 if weights is None else float(weights[j])
        if ha:
            lh=float(expc(mu+adv+p[h]-p[3*n+a]))
            la=float(expc(mu+p[2*n+a]-p[n+h]))
        else:
            lh=float(expc(mu+adv+p[h]-p[n+a]))
            la=float(expc(mu+p[a]-p[n+h]))
        eh=w*(lh-hg); ea=w*(la-ag)
        f+=w*(lh-hg*math.log(lh)+la-ag*math.log(la))
        if ha:
            g[h]+=eh; g[3*n+a]-=eh; g[2*n+a]+=ea; g[n+h]-=ea
            g[4*n]+=eh+ea; g[4*n+1]+=eh
        else:
            g[h]+=eh; g[n+a]-=eh; g[a]+=ea; g[n+h]-=ea
            g[2*n]+=eh+ea; g[2*n+1]+=eh
    f+=lam*float(np.dot(p[:base],p[:base]))
    g[:base]+=2*lam*p[:base]
    return float(f),g

def pois_matrix(lam):
    out=np.empty((len(lam),10),dtype=float)
    out[:,0]=np.exp(-lam)
    for k in range(1,10): out[:,k]=out[:,k-1]*lam/k
    return out

def pair_probs(p,teams,kind,home_ids,away_ids):
    n=len(teams); ha=kind=="home_away_attack_defense"; base=4*n if ha else 2*n
    if ha:
        lh=expc(p[base]+p[base+1]+p[home_ids]-p[3*n+away_ids])
        la=expc(p[base]+p[2*n+away_ids]-p[n+home_ids])
    else:
        lh=expc(p[base]+p[base+1]+p[home_ids]-p[n+away_ids])
        la=expc(p[base]+p[away_ids]-p[n+home_ids])
    hp,ap=pois_matrix(lh),pois_matrix(la)
    win=np.zeros(len(home_ids)); draw=np.zeros(len(home_ids)); loss=np.zeros(len(home_ids))
    for h in range(10):
        for a in range(10):
            v=hp[:,h]*ap[:,a]
            if h>a: win+=v
            elif h==a: draw+=v
            else: loss+=v
    total=win+draw+loss
    return np.column_stack((win/total,draw/total,loss/total))

def weights_for(train,current_date,half_life):
    if not half_life: return None
    current=datetime.strptime(current_date,"%Y-%m-%d")
    w=np.array([2**(-max(0,(current-datetime.strptime(r["date"],"%Y-%m-%d")).days)/half_life) for r in train],dtype=float)
    avg=float(w.mean())
    return w/avg if avg else w

def main():
    with (ROOT/"p1-2026-fit-failures.csv").open(encoding="utf-8-sig",newline="") as f:
        failures=list(csv.DictReader(f))
    batch=failures[OFFSET:OFFSET+LIMIT]
    r25=rows2025(); r26=rows2026()
    teams=sorted({t for r in r25+r26 for t in (r["home"],r["away"])})
    ix={t:i for i,t in enumerate(teams)}
    H=np.repeat(np.arange(len(teams)),len(teams)); A=np.tile(np.arange(len(teams)),len(teams)); mask=H!=A
    H,A=H[mask],A[mask]
    out=[]
    for seq,row in enumerate(batch,OFFSET+1):
        rec={k:row.get(k,"") for k in ("train_mode","start","lambda","model","date","competition","home","away","reason","iterations","objective","gradient_max_abs","gradient_l2","gradient_threshold_1e5_met","p_home","p_draw","p_away")}
        try:
            target_home,target_away=canon(row["home"]),canon(row["away"])
            candidates=[i for i,r in enumerate(r26) if r["date"]==row["date"] and r["competition"]==row["competition"] and r["home"]==target_home and r["away"]==target_away]
            if len(candidates)!=1: raise RuntimeError(f"target match count {len(candidates)}")
            idx26=candidates[0]; target=r26[idx26]
            train=(r25 if row["train_mode"]=="2025_plus_2026" else [])+r26[:idx26]
            data=map_rows(train,ix); n=len(teams); lam=float(row["lambda"]); kind=row["model"]
            params=json.loads(row["parameters"])
            if len(params)!=(4*n+2 if kind=="home_away_attack_defense" else 2*n+2):
                raise RuntimeError(f"parameter length mismatch {len(params)}")
            p0=np.asarray(params,dtype=float)
            half=None
            m=re.match(r"recency_half_life_(\d+)$",kind)
            if m: half=int(m.group(1))
            wt=weights_for(train,target["date"],half)
            fun=lambda p: objgrad(p,data,n,lam,kind,wt)
            scipy=minimize(fun,p0,jac=True,method="L-BFGS-B",
                           options={"gtol":1e-10,"ftol":0.0,"maxiter":10000,"maxfun":1000000,"maxls":100,"maxcor":10})
            psp=np.asarray(scipy.x,dtype=float)
            jsprob=pair_probs(p0,teams,kind,np.array([ix[target_home]]),np.array([ix[target_away]]))[0]
            spprob=pair_probs(psp,teams,kind,np.array([ix[target_home]]),np.array([ix[target_away]]))[0]
            probs_js=pair_probs(p0,teams,kind,H,A); probs_sp=pair_probs(psp,teams,kind,H,A)
            diffs=np.abs(probs_js-probs_sp); flat=np.unravel_index(int(np.argmax(diffs)),diffs.shape)
            worst_home=teams[int(H[flat[0]])]; worst_away=teams[int(A[flat[0]])]
            sf,sg=fun(psp)
            rec.update({
                "scipy_success":bool(scipy.success),"scipy_message":str(scipy.message),
                "scipy_iterations":int(scipy.nit),"scipy_objective":float(scipy.fun),
                "objective_abs_diff":abs(float(row["objective"])-float(scipy.fun)),
                "scipy_gmax":float(np.max(np.abs(sg))),
                "max_abs_dP":float(np.max(diffs)),
                "pass_max_abs_dP_1e-4":bool(np.max(diffs)<=THRESHOLD),
                "js_target_p_home":float(jsprob[0]),"js_target_p_draw":float(jsprob[1]),"js_target_p_away":float(jsprob[2]),
                "scipy_target_p_home":float(spprob[0]),"scipy_target_p_draw":float(spprob[1]),"scipy_target_p_away":float(spprob[2]),
                "worst_pair_home":worst_home,"worst_pair_away":worst_away,
                "worst_pair_js_home":float(probs_js[flat[0],0]),"worst_pair_js_draw":float(probs_js[flat[0],1]),"worst_pair_js_away":float(probs_js[flat[0],2]),
                "worst_pair_scipy_home":float(probs_sp[flat[0],0]),"worst_pair_scipy_draw":float(probs_sp[flat[0],1]),"worst_pair_scipy_away":float(probs_sp[flat[0],2]),
                "audit_error":""
            })
        except Exception as e:
            rec.update({"scipy_success":False,"scipy_message":"","scipy_iterations":"","scipy_objective":"",
                        "objective_abs_diff":"","scipy_gmax":"","max_abs_dP":"","pass_max_abs_dP_1e-4":False,
                        "js_target_p_home":"","js_target_p_draw":"","js_target_p_away":"",
                        "scipy_target_p_home":"","scipy_target_p_draw":"","scipy_target_p_away":"",
                        "worst_pair_home":"","worst_pair_away":"","worst_pair_js_home":"","worst_pair_js_draw":"",
                        "worst_pair_js_away":"","worst_pair_scipy_home":"","worst_pair_scipy_draw":"",
                        "worst_pair_scipy_away":"","audit_error":repr(e)})
        out.append(rec)
        if seq%25==0 or seq==OFFSET+len(batch):
            print(json.dumps({"event":"progress","processed":seq,"chunk_offset":OFFSET,"chunk_limit":LIMIT,"total_failure_rows":len(failures)},ensure_ascii=False))
    fields=list(out[0].keys()) if out else ["train_mode","start","lambda","model","date","competition","home","away","reason","iterations","objective","gradient_max_abs","gradient_l2","gradient_threshold_1e5_met","p_home","p_draw","p_away","scipy_success","scipy_message","scipy_iterations","scipy_objective","objective_abs_diff","scipy_gmax","max_abs_dP","pass_max_abs_dP_1e-4","js_target_p_home","js_target_p_draw","js_target_p_away","scipy_target_p_home","scipy_target_p_draw","scipy_target_p_away","worst_pair_home","worst_pair_away","worst_pair_js_home","worst_pair_js_draw","worst_pair_js_away","worst_pair_scipy_home","worst_pair_scipy_draw","worst_pair_scipy_away","audit_error"]
    name=f"p1-scipy-failure-audit-{OFFSET:05d}-{OFFSET+LIMIT-1:05d}.csv"
    with (ROOT/name).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
    passed=sum(r.get("pass_max_abs_dP_1e-4") is True for r in out)
    errors=sum(bool(r.get("audit_error")) for r in out)
    summary={"offset":OFFSET,"limit":LIMIT,"failure_rows_total":len(failures),"processed":len(out),
             "pass_count":passed,"not_pass_count":len(out)-passed,"error_count":errors,
             "max_abs_dP":max((float(r["max_abs_dP"]) for r in out if r.get("max_abs_dP") not in ("",None)),default=None),
             "threshold":THRESHOLD,"output":name}
    (ROOT/f"p1-scipy-failure-audit-{OFFSET:05d}-{OFFSET+LIMIT-1:05d}.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
