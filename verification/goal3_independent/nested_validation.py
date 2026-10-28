#!/usr/bin/env python3
"""Nested expanding-window validation for totoGOAL3.

Hyperparameters are selected only on inner, earlier date blocks. Each outer
block is evaluated once with its selected (lambda, training-window) pair.
Predictions for a date use strictly earlier dates; same-day fixtures never
enter that date's training set. This script is read-only except for its output.
"""
import argparse
import csv
import json
import math
import os
import unicodedata
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(__file__)
LAMBDAS = [1.0, 3.0, 10.0, 30.0, 100.0]
WINDOWS = ["2026_only", "2025_plus_2026"]
OUTER_START_FRACTIONS = [0.55, 0.70, 0.85]
INNER_START_FRACTION = 0.40

def canonical(name, aliases):
    name = unicodedata.normalize("NFKC", str(name)).strip()
    return aliases.get(name, name)

def load_data(history_path, aliases_path, csv2025_path):
    with open(aliases_path, encoding="utf-8") as f:
        aliases = json.load(f)["aliases"]
    with open(history_path, encoding="utf-8") as f:
        raw = json.load(f)["history"]
    h2026 = []
    for m in raw:
        if m.get("competition") not in ("J1", "J2", "J3"):
            continue
        if not isinstance(m.get("home_goals"), int) or not isinstance(m.get("away_goals"), int):
            continue
        h2026.append({
            "date": m["date"], "home": canonical(m["home"], aliases),
            "away": canonical(m["away"], aliases),
            "hg": int(m["home_goals"]), "ag": int(m["away_goals"]), "year": 2026,
        })
    h2026.sort(key=lambda m: (m["date"], m["home"], m["away"]))
    h2025 = []
    with open(csv2025_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            score = (row.get("score") or "").strip()
            if "-" not in score:
                continue
            try:
                hg, ag = (int(x) for x in score.split("-", 1))
            except ValueError:
                continue
            date = (row.get("date") or "").replace("/", "-").strip()
            if not date:
                continue
            h2025.append({
                "date": date, "home": canonical(row["homeTeam"], aliases),
                "away": canonical(row["awayTeam"], aliases),
                "hg": hg, "ag": ag, "year": 2025,
            })
    h2025.sort(key=lambda m: (m["date"], m["home"], m["away"]))
    if not h2026 or not h2025:
        raise ValueError("Both valid 2025 and 2026 history are required.")
    return h2025, h2026

def bin_goals(g):
    return min(3, max(0, int(g)))

def poisson_probs(lam):
    lam = min(3.5, max(0.15, float(lam)))
    p0 = math.exp(-lam)
    p1 = p0 * lam
    p2 = p1 * lam / 2.0
    return [p0, p1, p2, max(0.0, 1.0 - p0 - p1 - p2)]

def avg(values, default=1.25):
    return sum(values) / len(values) if values else default

def select_training(h2025, h2026, target_date, window):
    # Strict date cutoff: no same-day fixtures, regardless of kickoff time.
    prior26 = [m for m in h2026 if m["date"] < target_date]
    if window == "2026_only":
        prior = prior26
    elif window == "2025_plus_2026":
        prior = h2025 + prior26
    else:
        raise ValueError("Unknown training window: " + str(window))
    if any(m["date"] >= target_date for m in prior):
        raise AssertionError("Temporal leakage in training selection")
    return prior

def fit_shrinkage(prior, lam_pen):
    teams = sorted({m["home"] for m in prior} | {m["away"] for m in prior})
    if not teams:
        return None, {}, {"success": True, "nit": 0, "message": "empty prior fallback"}
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    homes = np.array([idx[m["home"]] for m in prior], dtype=int)
    aways = np.array([idx[m["away"]] for m in prior], dtype=int)
    hg = np.array([m["hg"] for m in prior], dtype=float)
    ag = np.array([m["ag"] for m in prior], dtype=float)
    x0 = np.zeros(2 + 2*n)
    x0[0] = math.log(1.3)

    def objective(x):
        mu, home_adv = x[0], x[1]
        attack = x[2:2+n]
        defense = x[2+n:2+2*n]
        eta_h_raw = mu + home_adv + attack[homes] - defense[aways]
        eta_a_raw = mu + attack[aways] - defense[homes]
        eta_h = np.clip(eta_h_raw, -20, 20)
        eta_a = np.clip(eta_a_raw, -20, 20)
        lh, la = np.exp(eta_h), np.exp(eta_a)
        value = np.sum(lh - hg * eta_h) + np.sum(la - ag * eta_a)
        value += lam_pen * (np.sum(attack**2) + np.sum(defense**2))
        # Match the reference implementation's gradient convention. The
        # clip is a numerical guard; diagnostics record every optimizer fit.
        eh, ea = lh - hg, la - ag
        da, dd = np.zeros(n), np.zeros(n)
        np.add.at(da, homes, eh)
        np.add.at(dd, aways, -eh)
        np.add.at(da, aways, ea)
        np.add.at(dd, homes, -ea)
        da += 2 * lam_pen * attack
        dd += 2 * lam_pen * defense
        grad = np.concatenate(([eh.sum() + ea.sum(), eh.sum()], da, dd))
        return float(value), grad

    res = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
    info = {"success": bool(res.success), "status": int(res.status),
            "nit": int(res.nit), "message": str(res.message)}
    if not res.success:
        raise RuntimeError("L-BFGS-B failed: " + str(info))
    return res.x, idx, info

def predict_shrinkage(params, idx, home, away):
    if params is None:
        return poisson_probs(1.25), poisson_probs(1.25)
    n = len(idx)
    attack, defense = params[2:2+n], params[2+n:2+2*n]
    ah = attack[idx[home]] if home in idx else 0.0
    dh = defense[idx[home]] if home in idx else 0.0
    aa = attack[idx[away]] if away in idx else 0.0
    da = defense[idx[away]] if away in idx else 0.0
    lh = math.exp(float(np.clip(params[0] + params[1] + ah - da, -20, 20)))
    la = math.exp(float(np.clip(params[0] + aa - dh, -20, 20)))
    return poisson_probs(lh), poisson_probs(la)

def predict_baselines(prior, home, away):
    league_h = avg([m["hg"] for m in prior])
    league_a = avg([m["ag"] for m in prior])
    lm_h, lm_a = poisson_probs(league_h), poisson_probs(league_a)
    counts_h, counts_a = [1,1,1,1], [1,1,1,1]
    for m in prior:
        counts_h[bin_goals(m["hg"])] += 1
        counts_a[bin_goals(m["ag"])] += 1
    total_h, total_a = sum(counts_h), sum(counts_a)
    emp_h = [x / total_h for x in counts_h]
    emp_a = [x / total_a for x in counts_a]
    played_home = [m for m in prior if m["home"] == home or m["away"] == home][-12:]
    played_away = [m for m in prior if m["home"] == away or m["away"] == away][-12:]
    def team_factors(team, played):
        gf = [m["hg"] if m["home"] == team else m["ag"] for m in played]
        ga = [m["ag"] if m["home"] == team else m["hg"] for m in played]
        base = max(0.2, (league_h + league_a) / 2)
        return (avg(gf) / base, avg(ga) / base) if played else (1.0, 1.0)
    h_atk, h_def = team_factors(home, played_home)
    a_atk, a_def = team_factors(away, played_away)
    simple_h = poisson_probs(min(3.5, max(0.15, league_h * h_atk * a_def)))
    simple_a = poisson_probs(min(3.5, max(0.15, league_a * a_atk * h_def)))
    return {"leagueMean": (lm_h, lm_a), "empiricalLaplace": (emp_h, emp_a),
            "simpleTeamAdjusted": (simple_h, simple_a)}

def score(records):
    # records are (probability-vector, observed-goals-category)
    if not records:
        raise ValueError("Cannot score an empty validation block")
    hits, brier, logloss = 0, 0.0, 0.0
    for probs, actual in records:
        pred = max(range(4), key=lambda k: probs[k])
        hits += int(pred == actual)
        brier += sum((probs[k] - (1.0 if k == actual else 0.0))**2 for k in range(4))
        logloss -= math.log(max(1e-12, probs[actual]))
    n = len(records)
    return {"targetCount": n, "accuracy": hits/n, "brier": brier/n, "logloss": logloss/n}

def date_blocks(dates, start, end, parts):
    if end <= start:
        return []
    width = end - start
    blocks = []
    for i in range(parts):
        lo = start + (width * i) // parts
        hi = start + (width * (i + 1)) // parts
        if hi > lo:
            blocks.append(dates[lo:hi])
    return [b for b in blocks if b]

def evaluate_dates(dates_to_score, h2025, h2026, window, lam_pen, collect_diagnostics=False):
    records = []
    fit_count = 0
    for date in dates_to_score:
        prior = select_training(h2025, h2026, date, window)
        params, idx, info = fit_shrinkage(prior, lam_pen)
        fit_count += 1
        targets = [m for m in h2026 if m["date"] == date]
        for m in targets:
            ph, pa = predict_shrinkage(params, idx, m["home"], m["away"])
            records.extend([(ph, bin_goals(m["hg"])), (pa, bin_goals(m["ag"]))])
    result = score(records)
    if collect_diagnostics:
        result["fitCount"] = fit_count
        result["fitFailureCount"] = 0
    return result

def choose_configuration(h2025, h2026, all_dates, outer_start_idx):
    inner_start = int(len(all_dates) * INNER_START_FRACTION)
    blocks = date_blocks(all_dates, inner_start, outer_start_idx, 3)
    if len(blocks) < 2:
        raise ValueError("Insufficient pre-outer dates for at least two inner folds")
    candidates = []
    for window in WINDOWS:
        for lam_pen in LAMBDAS:
            fold_scores = []
            for block in blocks:
                fold_scores.append(evaluate_dates(block, h2025, h2026, window, lam_pen))
            combined = {
                "targetCount": sum(s["targetCount"] for s in fold_scores),
                "brier": sum(s["brier"] * s["targetCount"] for s in fold_scores) /
                         sum(s["targetCount"] for s in fold_scores),
                "logloss": sum(s["logloss"] * s["targetCount"] for s in fold_scores) /
                           sum(s["targetCount"] for s in fold_scores),
                "accuracy": sum(s["accuracy"] * s["targetCount"] for s in fold_scores) /
                            sum(s["targetCount"] for s in fold_scores),
            }
            candidates.append({"window": window, "lambda": lam_pen,
                               "inner": combined, "innerFolds": fold_scores})
    # Pre-declared selection rule: lowest pooled inner Brier, then Log Loss,
    # then highest Accuracy, then prefer simpler/lower lambda as deterministic tie-break.
    candidates.sort(key=lambda c: (c["inner"]["brier"], c["inner"]["logloss"],
                                   -c["inner"]["accuracy"], c["lambda"], c["window"]))
    return candidates[0], candidates

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history2026", default=os.path.join(ROOT, "data", "history.json"))
    ap.add_argument("--aliases", default=os.path.join(ROOT, "data", "team_aliases.json"))
    ap.add_argument("--csv2025", default=os.path.join(HERE, "2025.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "nested_results.json"))
    args = ap.parse_args()
    h2025, h2026 = load_data(args.history2026, args.aliases, args.csv2025)
    dates = sorted({m["date"] for m in h2026})
    if len(dates) < 20:
        raise ValueError("Need at least 20 distinct 2026 dates for nested time-series validation")
    outer_starts = sorted(set(int(len(dates) * f) for f in OUTER_START_FRACTIONS))
    outer = []
    for fold_no, start_idx in enumerate(outer_starts, 1):
        if start_idx <= int(len(dates) * INNER_START_FRACTION) or start_idx >= len(dates):
            continue
        test_dates = dates[start_idx:(outer_starts[fold_no] if fold_no < len(outer_starts) else len(dates))]
        if not test_dates:
            continue
        selected, candidates = choose_configuration(h2025, h2026, dates, start_idx)
        window, lam_pen = selected["window"], selected["lambda"]
        selected_metrics = evaluate_dates(test_dates, h2025, h2026, window, lam_pen, True)
        baseline_records = defaultdict(list)
        for date in test_dates:
            prior = select_training(h2025, h2026, date, window)
            if any(m["date"] >= date for m in prior):
                raise AssertionError("Leakage guard failed")
            for m in [x for x in h2026 if x["date"] == date]:
                predictions = predict_baselines(prior, m["home"], m["away"])
                for model, (ph, pa) in predictions.items():
                    baseline_records[model].extend([(ph, bin_goals(m["hg"])), (pa, bin_goals(m["ag"]))])
        baselines = {name: score(records) for name, records in baseline_records.items()}
        outer.append({
            "fold": fold_no, "trainCutoffExclusive": test_dates[0],
            "testStart": test_dates[0], "testEnd": test_dates[-1],
            "testDates": len(test_dates),
            "selected": {"window": window, "lambda": lam_pen,
                         "innerSelectionMetrics": selected["inner"]},
            "innerCandidates": candidates,
            "outerMetrics": {"shrinkageMLE": selected_metrics, **baselines},
            "leakageCheck": "passed: each target date trained on strictly earlier dates",
        })
    if len(outer) < 2:
        raise ValueError("Nested validation produced fewer than two outer folds")
    result = {
        "method": "nested expanding-window time-series validation",
        "selectionRule": "lowest pooled inner Brier; tie-break Log Loss, Accuracy, lower lambda",
        "lambdaCandidates": LAMBDAS,
        "trainingWindowCandidates": WINDOWS,
        "outerStartFractions": OUTER_START_FRACTIONS,
        "history2025Rows": len(h2025), "history2026Rows": len(h2026),
        "history2026DistinctDates": len(dates),
        "history2025DateRange": [h2025[0]["date"], h2025[-1]["date"]],
        "history2026DateRange": [h2026[0]["date"], h2026[-1]["date"]],
        "outerFolds": outer,
        "warning": "Outer-fold results are rolling-origin estimates, not a guarantee of future performance. GOAL3 lambda=10 was inherited from 1X2 and was not independently tuned for GOAL3 before this analysis.",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "output": args.out,
        "outerFolds": [{
            "fold": x["fold"], "testStart": x["testStart"], "testEnd": x["testEnd"],
            "selected": x["selected"],
            "metrics": x["outerMetrics"],
        } for x in outer],
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
