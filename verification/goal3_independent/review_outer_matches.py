#!/usr/bin/env python3
"""Independent audit of the frozen 472-row GOAL3 k=6/k=10 outer predictions."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEAGUES = ("J1", "J2", "J3")
CATS = (0, 1, 2, 3)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = ["date","comp","fold","home","away","hg","ag","side","actual"] + [
        f"k{k}_p{i}" for k in (6,10) for i in CATS
    ]
    if not rows or any(k not in rows[0] for k in required):
        raise ValueError("CSV missing required columns")
    for r in rows:
        for k in ("fold","hg","ag","actual"):
            r[k] = int(r[k])
        for k in (6,10):
            for i in CATS:
                r[f"k{k}_p{i}"] = float(r[f"k{k}_p{i}"])
    return rows

def probs(rows, k):
    return np.asarray([[r[f"k{k}_p{i}"] for i in CATS] for r in rows], dtype=float)

def score(rows):
    y = np.asarray([r["actual"] for r in rows], dtype=int)
    onehot = np.eye(4)[y]
    out, arr = {}, {}
    for k in (6,10):
        p = probs(rows,k)
        if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
            raise ValueError(f"Invalid k={k} probabilities")
        if np.max(np.abs(p.sum(axis=1)-1)) > 2e-6:
            raise ValueError(f"k={k} probability sums out of tolerance")
        b = np.sum((p-onehot)**2,axis=1)
        ll = -np.log(np.clip(p[np.arange(len(y)),y],1e-15,1))
        ac = (np.argmax(p,axis=1)==y).astype(float)
        arr[k] = {"brier":b,"logloss":ll,"accuracy":ac}
        out[str(k)] = {"n":len(y),"brier":float(b.mean()),"logloss":float(ll.mean()),"accuracy":float(ac.mean())}
    out["difference_k6_minus_k10"] = {
        m:float(arr[6][m].mean()-arr[10][m].mean()) for m in ("brier","logloss","accuracy")
    }
    return out,arr

def cluster_indices(rows,keyfn):
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[keyfn(r)].append(i)
    return list(groups.values())

def bootstrap(rows,arr,kind,keyfn,reps,seed):
    clusters=cluster_indices(rows,keyfn)
    rng=np.random.default_rng(seed)
    db=arr[6]["brier"]-arr[10]["brier"]
    dl=arr[6]["logloss"]-arr[10]["logloss"]
    sim=np.empty((reps,2),dtype=float)
    for j in range(reps):
        draw=rng.integers(0,len(clusters),size=len(clusters))
        ix=np.concatenate([clusters[int(i)] for i in draw])
        sim[j,0]=db[ix].mean()
        sim[j,1]=dl[ix].mean()
    return {
        "cluster":kind,"clusterCount":len(clusters),"replicates":reps,"seed":seed,
        "rng":"numpy.default_rng (PCG64)","interval":"95% percentile; linear quantile",
        "brier":{"mean":float(sim[:,0].mean()),"ci95":[float(x) for x in np.quantile(sim[:,0],[.025,.975],method="linear")]},
        "logloss":{"mean":float(sim[:,1].mean()),"ci95":[float(x) for x in np.quantile(sim[:,1],[.025,.975],method="linear")]}
    }

def calibration(rows,k):
    y=np.asarray([r["actual"] for r in rows],dtype=int)
    p=probs(rows,k)
    means=[]; bins=[]
    for c in CATS:
        obs=float(np.mean(y==c)); pred=float(p[:,c].mean())
        means.append({"category":c,"predictedMean":pred,"observedRate":obs,"gapPP":100*(pred-obs)})
        ordered=np.argsort(p[:,c],kind="stable")
        group=[]
        for ix in np.array_split(ordered,5):
            o=float(np.mean(y[ix]==c)); q=float(p[ix,c].mean())
            group.append({"n":len(ix),"predictedMean":q,"observedRate":o,"gapPP":100*(q-o)})
        bins.append({"category":c,"quintiles":group})
    return {"meanByCategory":means,"fiveQuantileBins":bins}

def residuals(rows):
    def calc(group):
        if not group: return {"n":0}
        goals=np.asarray([r["hg"] if r["side"]=="home" else r["ag"] for r in group],dtype=float)
        p0=np.asarray([r["k6_p0"] for r in group],dtype=float)
        lam=np.maximum(-np.log(np.clip(p0,1e-15,1)),1e-12)
        z=(goals-lam)/np.sqrt(lam)
        return {
            "n":len(group),"meanGoals":float(goals.mean()),"meanLambda":float(lam.mean()),
            "varianceToMean":float(goals.var()/goals.mean()) if goals.mean() else None,
            "observedZero":float(np.mean(goals==0)),"predictedZero":float(p0.mean()),
            "pearsonResidualMean":float(z.mean()),"pearsonResidualRMS":float(np.sqrt(np.mean(z*z))),
            "absResidualAbove2":float(np.mean(np.abs(z)>2))
        }
    return {
        "all":calc(rows),
        "home":calc([r for r in rows if r["side"]=="home"]),
        "away":calc([r for r in rows if r["side"]=="away"]),
        "league":{l:calc([r for r in rows if r["comp"]==l]) for l in LEAGUES}
    }

def source_audit(root):
    import unicodedata
    aliases=json.loads((root/"data/team_aliases.json").read_text(encoding="utf-8"))["aliases"]
    canon=lambda s:aliases.get(unicodedata.normalize("NFKC",str(s)).strip(),
                               unicodedata.normalize("NFKC",str(s)).strip())
    alias_counts=Counter(aliases.values())
    h=json.loads((root/"data/history.json").read_text(encoding="utf-8"))["history"]
    keys=set(); td=set(); dup=0; repeat=0; names=set(); apps=Counter(); unknown=set()
    source_names={"2026":defaultdict(set),"2025":defaultdict(set)}
    for r in h:
        if not isinstance(r.get("date"),str) or not r["date"].startswith("2026-"):
            raise ValueError(f"Non-2026 history date: {r.get('date')!r}")
        if r.get("competition") not in LEAGUES: raise ValueError("Unexpected history league")
        if not isinstance(r.get("home_goals"),int) or not isinstance(r.get("away_goals"),int) or min(r["home_goals"],r["away_goals"])<0:
            raise ValueError("Invalid history score")
        if not r.get("fetchedAt"): raise ValueError("Missing fetchedAt")
        key=(r["date"],r["competition"],r["home"],r["away"])
        if key in keys: dup+=1
        keys.add(key)
        for t in (r["home"],r["away"]):
            n=unicodedata.normalize("NFKC",str(t)).strip()
            if n not in aliases: unknown.add(n)
            c=canon(t); names.add(c); apps[c]+=1; source_names["2026"][c].add(n)
            if (r["date"],c) in td: repeat+=1
            td.add((r["date"],c))
    c25=[]
    with (root/"verification/goal3_independent/2025.csv").open(encoding="utf-8-sig",newline="") as f:
        c25=list(csv.DictReader(f))
    c25keys=set(); c25dup=0; c25unknown=set(); t25=set(); cats=Counter()
    bad_score=bad_date=bad_ids=0
    for r in c25:
        k=(r.get("date"),r.get("category"),r.get("homeTeam"),r.get("awayTeam"))
        if k in c25keys: c25dup+=1
        c25keys.add(k); cats[r.get("category")] += 1
        if r.get("category") not in ("1","2","3"): raise ValueError("Unexpected 2025 category")
        if not re.fullmatch(r"\d+-\d+",r.get("score","")): bad_score+=1
        try: datetime.strptime(r.get("date",""),"%Y/%m/%d")
        except ValueError: bad_date+=1
        try:
            home_id=int(r.get("homeTeamId","")); away_id=int(r.get("awayTeamId",""))
            if home_id==away_id: bad_ids+=1
        except (TypeError,ValueError): bad_ids+=1
        for t in (r.get("homeTeam"),r.get("awayTeam")):
            n=unicodedata.normalize("NFKC",str(t)).strip()
            if n not in aliases: c25unknown.add(n)
            c=canon(t); t25.add(c); source_names["2025"][c].add(n)
    if bad_score or bad_date or bad_ids: raise ValueError("2025 CSV contains invalid score/date/team IDs")
    t26={canon(r[t]) for r in h for t in ("home","away")}
    dates=sorted(r["date"] for r in h)
    collisions={
        season:{c:sorted(ns) for c,ns in groups.items() if len(ns)>1}
        for season,groups in source_names.items()
    }
    return {
        "history2026":{
            "rows":len(h),"leagueRows":dict(Counter(r["competition"] for r in h)),
            "dateRange":[dates[0],dates[-1]],"distinctDates":len({r["date"] for r in h}),
            "uniqueMatchKeys":len(keys),"duplicateMatchKeys":dup,
            "sameTeamSameDateConflicts":repeat,"canonicalTeams":len(names),
            "perTeamAppearancesMinMax":[min(apps.values()),max(apps.values())],
            "unmappedNames":sorted(unknown),"missingFetchedAt":sum(not r.get("fetchedAt") for r in h),
            "sourceNameCollisionsWithinSeason":collisions["2026"],
            "gitBlobSha":"f5cc8c044758e45b1e679e4ce796e68b3f27a854",
            "source":"community_jleague_csv"
        },
        "aliases":{
            "keys":len(aliases),"canonicalValues":len(alias_counts),
            "aliasKeysPerCanonicalDistribution":dict(Counter(alias_counts.values())),
            "canonicalValuesMissingSelfKey":sorted(v for v in alias_counts if v not in aliases),
            "oneToManyMappings":0,
            "gitBlobSha":"5de22584524ff4c07ddac1d27f49afbbbc5f88ab"
        },
        "history2025CommunityCsv":{
            "rows":len(c25),"categoryRows":dict(cats),"uniqueMatchKeys":len(c25keys),
            "duplicateMatchKeys":c25dup,"scoreParseFailures":bad_score,
            "dateParseFailures":bad_date,"teamIdFailures":bad_ids,
            "unmappedNames":sorted(c25unknown),
            "sourceNameCollisionsWithinSeason":collisions["2025"],
            "teamsOnly2025":sorted(t25-t26),"teamsOnly2026":sorted(t26-t25),
            "gitBlobSha":"847d324c5a6168d02efe328f8173418a951af720",
            "caveat":"nested_validation.py does not carry 2025 category into training rows; 2025_plus_2026 pools leagues."
        }
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default=str(ROOT/"verification/goal3_independent/goal3_outer_matches.csv"))
    ap.add_argument("--out",default="goal3_outer_review.json")
    ap.add_argument("--replicates",type=int,default=10000)
    ap.add_argument("--seed",type=int,default=1)
    args=ap.parse_args()
    path=Path(args.csv); rows=read_rows(path)
    dates=sorted({r["date"] for r in rows})
    matches=cluster_indices(rows,lambda r:(r["date"],r["comp"],r["home"],r["away"]))
    if len(rows)!=472 or len(dates)!=17 or len(matches)!=236:
        raise ValueError("Frozen cohort mismatch: expected 472 targets, 17 dates, 236 matches")
    if {r["fold"] for r in rows}!={1,2,3}: raise ValueError("Expected folds 1,2,3")
    if len({(r["date"],r["comp"],r["home"],r["away"],r["side"]) for r in rows})!=472:
        raise ValueError("Duplicate team-side key")
    for group in matches:
        g=[rows[i] for i in group]
        if len(g)!=2 or {r["side"] for r in g}!={"home","away"}:
            raise ValueError("Match does not contain exactly one home and away row")
        if len({(r["hg"],r["ag"],r["fold"]) for r in g})!=1:
            raise ValueError("Paired match metadata mismatch")
        for r in g:
            expected=min(3,r["hg"] if r["side"]=="home" else r["ag"])
            if r["actual"]!=expected: raise ValueError("Actual category does not match score")
    overall,arr=score(rows)
    league={l:score([r for r in rows if r["comp"]==l])[0] for l in LEAGUES}
    fold={str(f):score([r for r in rows if r["fold"]==f])[0] for f in (1,2,3)}
    report={
        "experiment":"retrospective fixed-k GOAL3 outer diagnostic; not prospective adoption evidence",
        "source":{
            "predictionRows":"gpt PR #20 comments #6626337643 and #6626339347",
            "probabilityDecimals":6,"csvSha256":sha256(path)
        },
        "cohort":{
            "teamSideRows":len(rows),"matches":len(matches),"dates":len(dates),
            "dateRange":[dates[0],dates[-1]],"leagueRows":dict(Counter(r["comp"] for r in rows)),
            "foldRows":dict(Counter(str(r["fold"]) for r in rows)),
            "maxProbabilitySumAbsError":float(max(np.max(np.abs(probs(rows,k).sum(axis=1)-1)) for k in (6,10)))
        },
        "scores":{"overall":overall,"byLeague":league,"byFold":fold},
        "bootstrap":{
            "dateCluster":bootstrap(rows,arr,"date",lambda r:r["date"],args.replicates,args.seed),
            "matchCluster":bootstrap(rows,arr,"match",lambda r:(r["date"],r["comp"],r["home"],r["away"]),args.replicates,args.seed)
        },
        "calibration":{
            "k6":calibration(rows,6),"k10":calibration(rows,10),
            "byLeague":{str(k):{l:calibration([r for r in rows if r["comp"]==l],k)["meanByCategory"] for l in LEAGUES} for k in (6,10)},
            "bySide":{str(k):{side:calibration([r for r in rows if r["side"]==side],k)["meanByCategory"] for side in ("home","away")} for k in (6,10)}
        },
        "poissonResidualsK6":residuals(rows),
        "sourceDataAudit":source_audit(ROOT),
        "interpretation":[
            "Negative k6-minus-k10 differences favor k=6; both primary metric confidence-interval upper bounds must be below zero for adoption review.",
            "This retrospective 17-date diagnostic is not prospective round-level evidence.",
            "Date-cluster bootstrap is primary for this retrospective diagnostic; match-cluster bootstrap is sensitivity analysis.",
            "The CSV probabilities are rounded to six decimals; small differences from full-precision upstream values are expected.",
            "The 2026 history and 2025 season CSV are community-source data, not a full official-source reconciliation."
        ]
    }
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),"cohort":report["cohort"],"overall":overall,
                      "bootstrap":report["bootstrap"],"sourceDataAudit":report["sourceDataAudit"]},
                     ensure_ascii=False,indent=2))

if __name__=="__main__": main()
