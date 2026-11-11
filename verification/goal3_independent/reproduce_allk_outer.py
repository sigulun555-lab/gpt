#!/usr/bin/env python3
"""Rebuild the fixed-2026-only GOAL3 outer k sweep from the shared reference fit code."""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, io, json, math, os, platform, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import scipy
from threadpoolctl import threadpool_info
import nested_validation as nv

ROOT=Path(__file__).resolve().parents[2]
K_VALUES=list(nv.LAMBDAS)
LOGLOSS_FLOOR=1e-12

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_blob_sha(path):
    raw=Path(path).read_bytes()
    header=f"blob {len(raw)}\0".encode("utf-8")
    return hashlib.sha1(header+raw).hexdigest()

def canonical_threadpool_info(infos):
    """Normalize and sort BLAS/OpenMP runtime records independent of discovery order."""
    normalized = []
    for info in infos:
        if not isinstance(info, dict):
            raise ValueError("threadpool runtime entries must be objects")
        normalized.append({str(key): info[key] for key in sorted(info)})
    sort_fields = (
        "user_api", "internal_api", "prefix", "version", "filepath",
        "threading_layer", "architecture", "num_threads",
    )
    normalized.sort(key=lambda item: tuple(str(item.get(key, "")) for key in sort_fields))
    return normalized


def execution_environment():
    """Capture numerical-library and thread-pool details in canonical order."""
    try:
        runtime_records = canonical_threadpool_info(threadpool_info())
        runtime_config = json.dumps(
            runtime_records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except Exception as exc:  # Environment introspection must not block the audit.
        runtime_config = f"unavailable: {type(exc).__name__}: {exc}"
    thread_names = (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
    return {
        "pythonVersion": sys.version,
        "numpyVersion": np.__version__,
        "scipyVersion": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpuCount": os.cpu_count(),
        "threadEnvironment": {name: os.environ.get(name) for name in thread_names},
        "numpyRuntimeConfig": runtime_config,
        "numpyRuntimeConfigSource": "threadpoolctl.threadpool_info; sorted by stable library identifiers",
        "numpyBuildConfigRepr": repr(np.__config__.CONFIG),
    }

def build_team_alias_provenance(raw_history_rows, aliases_doc, target_dates, aliases_path):
    """Describe how the current alias map changes historical team identities."""
    alias_map = aliases_doc.get("aliases", {})
    if not isinstance(alias_map, dict):
        raise ValueError("team alias map must be an object")
    eligible = [
        row for row in raw_history_rows
        if isinstance(row, dict) and row.get("competition") in ("J1", "J2", "J3")
    ]
    alias_counts = Counter()
    team_name_occurrences = 0
    for row in eligible:
        for side in ("home", "away"):
            name = row.get(side)
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"missing historical {side} team name")
            team_name_occurrences += 1
            if alias_map.get(name, name) != name:
                alias_counts[name] += 1

    updated_at = aliases_doc.get("updatedAt")
    prior_by_target = []
    for target_date in target_dates:
        prior = [row for row in eligible if isinstance(row.get("date"), str) and row["date"] < target_date]
        prior_alias_occurrences = sum(
            1
            for row in prior
            for side in ("home", "away")
            if alias_map.get(row[side], row[side]) != row[side]
        )
        prior_by_target.append({
            "targetDate": target_date,
            "priorHistoryMatches": len(prior),
            "priorTeamNameOccurrences": 2 * len(prior),
            "priorAliasMappedOccurrences": prior_alias_occurrences,
        })

    dates_before_update = [
        target_date for target_date in target_dates
        if isinstance(updated_at, str) and target_date < updated_at
    ]
    return {
        "source": aliases_doc.get("source"),
        "updatedAt": updated_at,
        "fileSha256": sha(aliases_path),
        "gitBlobSha": git_blob_sha(aliases_path),
        "historyTeamNameOccurrences": team_name_occurrences,
        "aliasMappedHistoryTeamNameOccurrences": sum(alias_counts.values()),
        "uniqueAliasKeysUsed": len(alias_counts),
        "targetDatesBeforeAliasUpdatedAt": dates_before_update,
        "priorHistoryByTargetDate": prior_by_target,
        "interpretation": (
            "This records the current retrospective normalization map and its material use. "
            "It does not prove the mappings were unknowable earlier, but no contemporaneous "
            "as-of alias-map snapshot or forecast execution record is established here."
        ),
    }

def load_frozen(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in ("fold","hg","ag","actual"):
            r[k]=int(r[k])
        for k in (6,10):
            for i in range(4):
                r[f"k{k}_p{i}"]=float(r[f"k{k}_p{i}"])
    groups=defaultdict(list)
    for r in rows:
        groups[(r["date"],r["comp"],r["home"],r["away"])].append(r)
    matches=[]
    for key,group in sorted(groups.items()):
        if len(group)!=2 or {x["side"] for x in group}!={"home","away"}:
            raise ValueError(f"Invalid frozen home/away pair: {key}")
        if len({(x["fold"],x["hg"],x["ag"]) for x in group})!=1:
            raise ValueError(f"Pair metadata mismatch: {key}")
        row=group[0]
        matches.append({
            "date":row["date"],"comp":row["comp"],"fold":row["fold"],
            "home":row["home"],"away":row["away"],"hg":row["hg"],"ag":row["ag"],
            "sourceRows":{x["side"]:x for x in group}
        })
    if len(rows)!=472 or len(matches)!=236:
        raise ValueError("Expected frozen cohort of 472 team-sides / 236 matches")
    fold_by_date=defaultdict(set)
    for m in matches: fold_by_date[m["date"]].add(m["fold"])
    if any(len(v)!=1 for v in fold_by_date.values()) or len(fold_by_date)!=17:
        raise ValueError("Expected 17 dates with one consistent outer fold per date")
    return rows,matches,dict((d,next(iter(v))) for d,v in fold_by_date.items())

def metrics(targets):
    y=np.asarray([x["actual"] for x in targets],dtype=int)
    p=np.asarray([x["p"] for x in targets],dtype=float)
    if len(y)==0: raise ValueError("Empty metric cohort")
    if not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1):
        raise ValueError("Invalid reconstructed probabilities")
    if np.max(np.abs(p.sum(axis=1)-1))>1e-9:
        raise ValueError("Reconstructed probability vectors do not sum to one")
    one=np.eye(4)[y]
    return {
        "n":int(len(y)),
        "brier":float(np.mean(np.sum((p-one)**2,axis=1))),
        "logloss":float(np.mean(-np.log(np.clip(p[np.arange(len(y)),y],LOGLOSS_FLOOR,1)))),
        "accuracy":float(np.mean(np.argmax(p,axis=1)==y))
    }

def moving_block_bootstrap(targets,k_values,reps,seed,rng_kind):
    by_date=defaultdict(list)
    for i,r in enumerate(targets[k_values[0]]): by_date[r["date"]].append(i)
    dates=sorted(by_date); n=len(dates); block_len=2; n_blocks=math.ceil(n/block_len)
    if rng_kind=="PCG64":
        rng=np.random.default_rng(seed)
        draw=lambda: rng.integers(0,n,size=n_blocks)
    elif rng_kind=="MT19937":
        rng=np.random.RandomState(seed)
        draw=lambda: rng.randint(0,n,size=n_blocks)
    else:
        raise ValueError(rng_kind)
    d_b=np.asarray([r["brier"] for r in targets[k_values[0]]])-np.asarray([r["brier"] for r in targets[k_values[1]]])
    d_l=np.asarray([r["logloss"] for r in targets[k_values[0]]])-np.asarray([r["logloss"] for r in targets[k_values[1]]])
    sims=np.empty((reps,2),dtype=float)
    for j in range(reps):
        starts=draw()
        sampled=[]
        for start in starts:
            sampled.extend([dates[(int(start)+offset)%n] for offset in range(block_len)])
        sampled=sampled[:n]
        ix=np.concatenate([by_date[d] for d in sampled])
        sims[j,0]=d_b[ix].mean()
        sims[j,1]=d_l[ix].mean()
    return {
        "rng":rng_kind,"seed":seed,"replicates":reps,"blockLengthDates":block_len,
        "circular":True,"sampledDatesPerReplicate":n,"blocksPerReplicate":n_blocks,
        "targetCountWeighted":True,"interval":"95% percentile, linear quantile",
        "brier":{"mean":float(sims[:,0].mean()),"ci95":[float(x) for x in np.quantile(sims[:,0],[.025,.975],method="linear")]},
        "logloss":{"mean":float(sims[:,1].mean()),"ci95":[float(x) for x in np.quantile(sims[:,1],[.025,.975],method="linear")]}
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frozen-csv",default=str(ROOT/"verification/goal3_independent/goal3_outer_matches.csv"))
    ap.add_argument("--history2026",default=str(ROOT/"data/history.json"))
    ap.add_argument("--aliases",default=str(ROOT/"data/team_aliases.json"))
    ap.add_argument("--csv2025",default=str(ROOT/"verification/goal3_independent/2025.csv"))
    ap.add_argument("--out",default="goal3_outer_allk_reproduction.json")
    ap.add_argument("--bootstrap-replicates",type=int,default=50000)
    ap.add_argument("--bootstrap-seed",type=int,default=20261104)
    args=ap.parse_args()

    frozen_rows,matches,fold_by_date=load_frozen(args.frozen_csv)
    raw_history_doc=json.loads(Path(args.history2026).read_text(encoding="utf-8"))
    raw_history_rows=raw_history_doc.get("history",[])
    fetched_at_values=sorted({str(r.get("fetchedAt")) for r in raw_history_rows if r.get("fetchedAt")})
    missing_fetched_at=sum(not r.get("fetchedAt") for r in raw_history_rows)
    aliases_doc=json.loads(Path(args.aliases).read_text(encoding="utf-8"))
    team_alias_audit=build_team_alias_provenance(
        raw_history_rows, aliases_doc, sorted(fold_by_date), Path(args.aliases)
    )
    h2025,h2026=nv.load_data(args.history2026,args.aliases,args.csv2025)
    source_dates=sorted({m["date"] for m in h2026})
    target_dates=sorted(fold_by_date)
    if len(source_dates)!=37 or len(h2026)!=429:
        raise ValueError(f"Unexpected 2026 history snapshot: rows={len(h2026)}, dates={len(source_dates)}")
    if any(d not in source_dates for d in target_dates):
        raise ValueError("Frozen target date absent from 2026 history")

    pred_by_k={}; fit_diagnostics=[]
    for k in K_VALUES:
        by_match={}
        for date in target_dates:
            prior=nv.select_training(h2025,h2026,date,"2026_only")
            if any(x["date"]>=date for x in prior):
                raise AssertionError("Temporal leakage in reconstructed training data")
            params,idx,info=nv.fit_shrinkage(prior,k)
            fit_diagnostics.append({
                "k":float(k),"date":date,"fold":fold_by_date[date],
                "trainingMatches":len(prior),"trainingDates":len({x["date"] for x in prior}),
                "optimizer":info
            })
            for m in [x for x in matches if x["date"]==date]:
                ph,pa=nv.predict_shrinkage(params,idx,m["home"],m["away"])
                by_match[(m["date"],m["comp"],m["home"],m["away"])]={"home":ph,"away":pa}
        pred_by_k[float(k)]=by_match

    targets_by_k={}; all_target_rows=[]
    for k in K_VALUES:
        rows=[]
        for m in matches:
            pred=pred_by_k[float(k)][(m["date"],m["comp"],m["home"],m["away"])]
            for side in ("home","away"):
                goals=m["hg"] if side=="home" else m["ag"]
                rows.append({
                    "date":m["date"],"comp":m["comp"],"fold":m["fold"],
                    "home":m["home"],"away":m["away"],"side":side,
                    "actual":nv.bin_goals(goals),"p":pred[side]
                })
        targets_by_k[float(k)]=rows

    k6_rows=targets_by_k[6.0]; k10_rows=targets_by_k[10.0]
    frozen_lookup={(r["date"],r["comp"],r["home"],r["away"],r["side"]):r for r in frozen_rows}
    max_error={6:0.0,10:0.0}; checked={6:0,10:0}
    for k in (6,10):
        for r in targets_by_k[float(k)]:
            src=frozen_lookup[(r["date"],r["comp"],r["home"],r["away"],r["side"])]
            for c in range(4):
                max_error[k]=max(max_error[k],abs(r["p"][c]-src[f"k{k}_p{c}"]))
                checked[k]+=1
        if max_error[k]>2e-6:
            raise ValueError(f"Rebuilt k={k} probabilities differ from frozen CSV: max abs error={max_error[k]}")

    metric_by_k={}
    for k in K_VALUES:
        key=str(float(k))
        rows=targets_by_k[float(k)]
        by_side={s:metrics([r for r in rows if r["side"]==s]) for s in ("home","away")}
        by_league_side={
            l:{s:metrics([r for r in rows if r["comp"]==l and r["side"]==s]) for s in ("home","away")}
            for l in ("J1","J2","J3")
        }
        by_fold_side={
            str(f):{s:metrics([r for r in rows if r["fold"]==f and r["side"]==s]) for s in ("home","away")}
            for f in (1,2,3)
        }
        # Fail closed if a paired cohort unexpectedly loses either side.
        if any(v["n"]!=236 for v in by_side.values()):
            raise ValueError(f"Unexpected home/away target counts for k={k}: {by_side}")
        metric_by_k[key]={
            "overall":metrics(rows),
            "byLeague":{l:metrics([r for r in rows if r["comp"]==l]) for l in ("J1","J2","J3")},
            "byFold":{str(f):metrics([r for r in rows if r["fold"]==f]) for f in (1,2,3)},
            "bySide":by_side,
            "byLeagueSide":by_league_side,
            "byFoldSide":by_fold_side
        }

    # For bootstrap, align every k to the same target-row order.
    aligned={}
    for k in K_VALUES:
        lookup={(r["date"],r["comp"],r["home"],r["away"],r["side"]):r for r in targets_by_k[float(k)]}
        aligned[float(k)]=[lookup[(r["date"],r["comp"],r["home"],r["away"],r["side"])] for r in k6_rows]
    bootstrap_targets={}
    for k in K_VALUES:
        bootstrap_targets[float(k)]=[{
            "date":r["date"],
            "brier":float(np.sum((np.asarray(r["p"])-np.eye(4)[r["actual"]])**2)),
            "logloss":float(-np.log(max(LOGLOSS_FLOOR,r["p"][r["actual"]])))
        } for r in aligned[float(k)]]
    # Report paired moving-block intervals for every candidate versus the
    # fixed k=10 reference. These are exploratory intervals on the same
    # retrospective cohort, not independent model-selection evidence.
    bootstrap_comparisons={}
    for k in K_VALUES:
        k=float(k)
        if k==10.0:
            continue
        bootstrap_comparisons[f"k{k:g}_minus_k10"]={
            "PCG64":moving_block_bootstrap(bootstrap_targets,(k,10.0),args.bootstrap_replicates,args.bootstrap_seed,"PCG64"),
            "MT19937":moving_block_bootstrap(bootstrap_targets,(k,10.0),args.bootstrap_replicates,args.bootstrap_seed,"MT19937")
        }
    bootstrap={"referenceK":10.0,"comparisonsVsK10":bootstrap_comparisons}

    per_match=[]
    for m in matches:
        entry={k:m[k] for k in ("date","comp","fold","home","away","hg","ag")}
        entry["byK"]={}
        for k in K_VALUES:
            p=pred_by_k[float(k)][(m["date"],m["comp"],m["home"],m["away"])]
            entry["byK"][str(float(k))]={"home":p["home"],"away":p["away"]}
        per_match.append(entry)

    report={
        "experiment":"GPT reconstruction of fixed-2026-only all-k outer diagnostics using nested_validation.py fit/predict functions",
        "executionEnvironment":execution_environment(),
        "provenance":{
            "frozenCsvSha256":sha(args.frozen_csv),
            "reproduceAllkOuterPySha256":sha(Path(__file__)),
            "reproduceAllkOuterPyGitBlobSha":git_blob_sha(Path(__file__)),
            "history2026Sha256":sha(args.history2026),
            "aliasesSha256":sha(args.aliases),
            "history2025CsvSha256":sha(args.csv2025),
            "nestedValidationPySha256":sha(Path(__file__).with_name("nested_validation.py")),
            "nestedValidationPyGitBlobSha":git_blob_sha(Path(__file__).with_name("nested_validation.py")),
            "pr19NestedValidationPyGitBlobSha":"5f73df9e6fdb43d5cf333c04036f87ae16c20d1f",
            "modelCodeDifference":"PR #20 retains PR #19's penalized Poisson objective, gradient, L-BFGS-B settings, and prediction transform; it adds a 2026-only date guard and fail-closed history input validation for score types/ranges and team identity."
        },
        "cohort":{
            "teamSideTargets":len(k6_rows),"matches":len(matches),"dates":len(target_dates),
            "dateRange":[target_dates[0],target_dates[-1]],
            "foldDates":dict(Counter(str(fold_by_date[d]) for d in target_dates)),
            "leagueTargets":dict(Counter(r["comp"] for r in k6_rows)),
            "history2026Rows":len(h2026),"history2026DistinctDates":len(source_dates),
            "historySnapshotUpdatedAt":raw_history_doc.get("updatedAt"),
            "historyFetchedAtUniqueCount":len(fetched_at_values),
            "historyFetchedAtValues":fetched_at_values,
            "historyFetchedAtMissingCount":int(missing_fetched_at)
        },
        "historicalAvailabilityCaveat":"Strict earlier-match-date cutoffs exclude later-dated results and same-day matches, but this retrospective snapshot does not preserve each match's original first-known timestamp. If all rows share the snapshot fetchedAt, exact historical ingestion-time availability cannot be audited.",
        "teamAliasAsOfAudit":team_alias_audit,
        "kCandidates":[float(k) for k in K_VALUES],
        "metricsByK":metric_by_k,
        "reconstructionCheck":{
            "maxProbabilityAbsErrorVsFrozenCsv":{str(k):v for k,v in max_error.items()},
            "probabilityCellsChecked":{str(k):v for k,v in checked.items()},
            "tolerance":2e-6
        },
        "fitDiagnostics":{
            "fitCount":len(fit_diagnostics),"failureCount":sum(not x["optimizer"]["success"] for x in fit_diagnostics),
            "minTrainingMatches":min(x["trainingMatches"] for x in fit_diagnostics),
            "maxTrainingMatches":max(x["trainingMatches"] for x in fit_diagnostics),
            "maxOptimizerIterations":max(x["optimizer"]["nit"] for x in fit_diagnostics),
            "perDateAndK":fit_diagnostics
        },
        "movingBlockBootstrap":bootstrap,
        "allKPredictions":per_match,
        "interpretation":[
            "The all-k reconstruction uses the same fit_shrinkage and predict_shrinkage functions as the PR #19 reference implementation; it independently rebuilds per-date training cutoffs, per-target predictions, aggregation, and metrics.",
            "k=6 and k=10 are required to reproduce the frozen six-decimal CSV within 2e-6; a larger mismatch aborts the run.",
            "This remains a retrospective 17-date diagnostic on the same outer cohort used in the original k sweep, not prospective evidence and not a basis for selecting k.",
            "The source history snapshot has 429 rows over 37 distinct 2026 dates. The PR #19 report's statement of 34 dates is inconsistent with the recorded source SHA and should be corrected."
        ]
    }
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "output":str(out),"cohort":report["cohort"],"reconstructionCheck":report["reconstructionCheck"],
        "metricsByK":{k:v["overall"] for k,v in metric_by_k.items()},
        "fitDiagnostics":{k:v for k,v in report["fitDiagnostics"].items() if k!="perDateAndK"},
        "movingBlockBootstrap":bootstrap
    },ensure_ascii=False,indent=2))

if __name__=="__main__": main()
