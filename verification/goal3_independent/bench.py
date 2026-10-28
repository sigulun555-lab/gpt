#!/usr/bin/env python3
"""Independent re-implementation + extension of GPT's totoGOAL3 model comparison.

Reproduces (as a sanity check) GPT's model.js backtestGoal3() results for
leagueMeanPoisson and empiricalLaplaceBaseline on 2026-only data, then adds
our OWN shrinkage-MLE Poisson attack/defense model (lambda=10, the one
actually used for round 1666's saved predictions) to the same walk-forward
comparison, for both 2026-only and 2025+2026 training windows.

Note: GPT's "teamAdjustedPoisson" in their report is NOT our shrinkage-MLE
model. It's a much simpler last-12-games moving-average ratio model with no
regularization (see model.js teamStats()). We reproduce that too, separately,
to keep the comparison honest about which "team-adjusted" model is which.
"""
import json, csv, math, os, argparse
from datetime import datetime
import numpy as np
from scipy.optimize import minimize

# This script lives at verification/goal3_independent/bench.py inside the repo.
# By default it reads data/history.json and data/team_aliases.json from the
# repo root (two directories up). The 2025 season CSV is NOT part of this
# repo; it comes from the public k-addtimes/jleague_open_data repo:
#   git clone https://github.com/k-addtimes/jleague_open_data
#   -> processed/2025_schedule_with_ids.csv
# Pass its path with --csv2025, or place a copy at verification/goal3_independent/2025.csv.
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--history2026", default=os.path.join(REPO_ROOT, "data", "history.json"))
parser.add_argument("--aliases", default=os.path.join(REPO_ROOT, "data", "team_aliases.json"))
parser.add_argument("--csv2025", default=os.path.join(os.path.dirname(__file__), "2025.csv"))
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
args = parser.parse_args()

def nfkc(s):
    import unicodedata
    return unicodedata.normalize("NFKC", s).strip()

aliases = json.load(open(args.aliases))["aliases"]
def canonical(name):
    k = nfkc(name)
    return aliases.get(k, k)

# ---- load 2026 (bundled in this repo) ----
h2026_raw = json.load(open(args.history2026))["history"]
h2026 = []
for m in h2026_raw:
    if m.get("competition") not in ("J1", "J2", "J3"):
        continue
    if not isinstance(m.get("home_goals"), int) or not isinstance(m.get("away_goals"), int):
        continue
    h2026.append({"date": m["date"], "home": canonical(m["home"]), "away": canonical(m["away"]),
                  "home_goals": m["home_goals"], "away_goals": m["away_goals"]})
h2026.sort(key=lambda m: m["date"])

# ---- load 2025 (external: k-addtimes/jleague_open_data) ----
h2025 = []
with open(args.csv2025, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        score = row["score"]
        if not score or "-" not in score:
            continue
        try:
            hg, ag = score.split("-")
            hg, ag = int(hg), int(ag)
        except ValueError:
            continue
        date = row["date"].replace("/", "-")
        h2025.append({"date": date, "home": canonical(row["homeTeam"]), "away": canonical(row["awayTeam"]),
                      "home_goals": hg, "away_goals": ag})
h2025.sort(key=lambda m: m["date"])

print(f"2025 rows: {len(h2025)} ({h2025[0]['date']} .. {h2025[-1]['date']})")
print(f"2026 rows: {len(h2026)} ({h2026[0]['date']} .. {h2026[-1]['date']})")

def bin_goals(g):
    return max(0, min(3, int(math.floor(g))))

def poisson_pmf(lam, k):
    return math.exp(-lam) * lam**k / math.factorial(k)

def goal_probs(lam):
    p0, p1, p2 = poisson_pmf(lam, 0), poisson_pmf(lam, 1), poisson_pmf(lam, 2)
    return [p0, p1, p2, max(0.0, 1 - p0 - p1 - p2)]

def avg(xs, default=1.25):
    return sum(xs) / len(xs) if xs else default

def clip(x, lo, hi):
    return max(lo, min(hi, x))

# ---------- league-mean Poisson ----------
def league_mean_probs(prior):
    lh = clip(avg([m["home_goals"] for m in prior]), 0.15, 3.5)
    la = clip(avg([m["away_goals"] for m in prior]), 0.15, 3.5)
    return goal_probs(lh), goal_probs(la)

# ---------- empirical Laplace baseline ----------
def empirical_probs(prior, field):
    counts = [1, 1, 1, 1]
    for m in prior:
        counts[bin_goals(m[field])] += 1
    total = sum(counts)
    return [c / total for c in counts]

# ---------- GPT's simple ratio "team-adjusted Poisson" (reproduction) ----------
def team_stats_simple(name, prior, league_home, league_away):
    played = [m for m in prior if m["home"] == name or m["away"] == name][-12:]
    if not played:
        return 1.0, 1.0
    gf = [m["home_goals"] if m["home"] == name else m["away_goals"] for m in played]
    ga = [m["away_goals"] if m["home"] == name else m["home_goals"] for m in played]
    base = max(0.2, (league_home + league_away) / 2)
    return avg(gf) / base, avg(ga) / base

def simple_team_adjusted_probs(prior, home, away):
    league_home = avg([m["home_goals"] for m in prior])
    league_away = avg([m["away_goals"] for m in prior])
    h_atk, h_def = team_stats_simple(home, prior, league_home, league_away)
    a_atk, a_def = team_stats_simple(away, prior, league_home, league_away)
    lh = clip(league_home * h_atk * a_def, 0.15, 3.5)
    la = clip(league_away * a_atk * h_def, 0.15, 3.5)
    return goal_probs(lh), goal_probs(la)

# ---------- OUR shrinkage-MLE Poisson (lambda=10), used for round 1666 ----------
def fit_shrinkage(prior, lam_pen=10.0):
    teams = sorted(set(m["home"] for m in prior) | set(m["away"] for m in prior))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    if n == 0:
        return None, idx, {"success": True, "status": 0, "message": "empty prior fallback", "nit": 0, "fun": None}
    # params: [mu, h, a_0..a_{n-1}, d_0..d_{n-1}]
    x0 = np.zeros(2 + 2 * n)
    x0[0] = math.log(1.3)
    homes = np.array([idx[m["home"]] for m in prior])
    aways = np.array([idx[m["away"]] for m in prior])
    hg = np.array([m["home_goals"] for m in prior], dtype=float)
    ag = np.array([m["away_goals"] for m in prior], dtype=float)

    def negloglik_grad(x):
        mu, h = x[0], x[1]
        a = x[2:2 + n]
        d = x[2 + n:2 + 2 * n]
        # Match round-1666 goal3.py's clipped linear predictors.
        eta_h = np.clip(mu + h + a[homes] - d[aways], -20, 20)
        eta_a = np.clip(mu + a[aways] - d[homes], -20, 20)
        lam_h = np.exp(eta_h)
        lam_a = np.exp(eta_a)
        nll = np.sum(lam_h - hg * np.log(lam_h)) + np.sum(lam_a - ag * np.log(lam_a))
        pen = lam_pen * (np.sum(a**2) + np.sum(d**2))
        f = nll + pen
        # gradients
        dmu = np.sum(lam_h - hg) + np.sum(lam_a - ag)
        dh = np.sum(lam_h - hg)
        da = np.zeros(n); dd = np.zeros(n)
        gh = lam_h - hg  # d nll / d(a[home_i]-d[away_i] term)... build via scatter-add
        ga_ = lam_a - ag
        np.add.at(da, homes, gh)
        np.add.at(dd, aways, -gh)
        np.add.at(da, aways, ga_)
        np.add.at(dd, homes, -ga_)
        da += 2 * lam_pen * a
        dd += 2 * lam_pen * d
        grad = np.concatenate(([dmu, dh], da, dd))
        return f, grad

    res = minimize(negloglik_grad, x0, jac=True, method="L-BFGS-B",
                    options={"maxiter": 300})
    fit_info = {"success": bool(res.success), "status": int(res.status),
                "message": str(res.message), "nit": int(res.nit),
                "fun": float(res.fun)}
    return res.x, idx, fit_info

def shrinkage_predict(params, idx, home, away):
    if params is None:
        lh = la = 1.25
    else:
        n = len(idx)
        mu, h = params[0], params[1]
        a = params[2:2 + n]; d = params[2 + n:2 + 2 * n]
        ai = idx.get(home); di_h = idx.get(home)
        a_home = a[idx[home]] if home in idx else 0.0
        d_home = d[idx[home]] if home in idx else 0.0
        a_away = a[idx[away]] if away in idx else 0.0
        d_away = d[idx[away]] if away in idx else 0.0
        lh = math.exp(mu + h + a_home - d_away)
        la = math.exp(mu + a_away - d_home)
    lh = clip(lh, 0.15, 3.5)
    la = clip(la, 0.15, 3.5)
    return goal_probs(lh), goal_probs(la)

def summarize(records):
    correct = 0; brier = 0.0; logloss = 0.0
    by_match = {}
    for probs, actual, key in records:
        pred = max(range(4), key=lambda i: probs[i])
        hit = pred == actual
        if hit:
            correct += 1
        brier += sum((probs[i] - (1.0 if i == actual else 0.0)) ** 2 for i in range(4))
        logloss -= math.log(max(1e-12, probs[actual]))
        by_match.setdefault(key, []).append(hit)
    N = len(records)
    exact = sum(1 for v in by_match.values() if len(v) == 2 and all(v)) / len(by_match) if by_match else None
    return {"targetCount": N, "accuracy": correct / N, "brier": brier / N, "logloss": logloss / N,
            "exactBothSidesAccuracy": exact}

def run(mode):
    """mode: '2026only' or '2025plus2026'"""
    sorted_2026 = h2026  # already sorted
    records = {"leagueMean": [], "empirical": [], "simpleTeamAdj": [], "shrinkageMLE": []}
    fit_diagnostics = {"fitCount": 0, "successCount": 0, "failureCount": 0,
                       "failures": []}
    for i in range(0, len(sorted_2026)):
        m = sorted_2026[i]
        prior_2026 = [x for x in sorted_2026[:i] if x["date"] < m["date"]]
        if mode == "2026only":
            prior = prior_2026
        else:
            prior = h2025 + prior_2026
        # Do NOT skip empty-prior steps: model.js's backtestGoal3 evaluates
        # every match from sorted[0] onward regardless of prior size, and
        # every helper here already has a defined fallback for prior=[]
        # (avg() defaults to 1.25, empirical_probs falls back to a uniform
        # Laplace prior, fit_shrinkage returns params=None -> neutral 1.25).
        # Skipping empty-prior steps silently shrinks the evaluation set
        # (the 2026-08-07 season opener has no earlier 2026-only data; the
        # defined neutral/uniform fallbacks are therefore evaluated) and
        # was the exact source of the ~0.001-scale mismatch against GPT's
        # reported numbers.
        lm_h, lm_a = league_mean_probs(prior)
        emp_h = empirical_probs(prior, "home_goals")
        emp_a = empirical_probs(prior, "away_goals")
        simple_h, simple_a = simple_team_adjusted_probs(prior, m["home"], m["away"])
        params, idx, fit_info = fit_shrinkage(prior, lam_pen=10.0)
        fit_diagnostics["fitCount"] += 1
        if fit_info["success"]:
            fit_diagnostics["successCount"] += 1
        else:
            fit_diagnostics["failureCount"] += 1
            fit_diagnostics["failures"].append({
                "date": m["date"], "home": m["home"], "away": m["away"],
                **fit_info
            })
        shrink_h, shrink_a = shrinkage_predict(params, idx, m["home"], m["away"])

        actual_h = bin_goals(m["home_goals"])
        actual_a = bin_goals(m["away_goals"])
        key = i
        records["leagueMean"].append((lm_h, actual_h, key))
        records["leagueMean"].append((lm_a, actual_a, key))
        records["empirical"].append((emp_h, actual_h, key))
        records["empirical"].append((emp_a, actual_a, key))
        records["simpleTeamAdj"].append((simple_h, actual_h, key))
        records["simpleTeamAdj"].append((simple_a, actual_a, key))
        records["shrinkageMLE"].append((shrink_h, actual_h, key))
        records["shrinkageMLE"].append((shrink_a, actual_a, key))
    result = {name: summarize(recs) for name, recs in records.items()}
    result["_metadata"] = {
        "mode": mode,
        "targetMatches": len(sorted_2026),
        "targetDateStart": sorted_2026[0]["date"] if sorted_2026 else None,
        "targetDateEnd": sorted_2026[-1]["date"] if len(sorted_2026) > 1 else None,
        "history2026Rows": len(h2026),
        "history2025Rows": len(h2025),
        "shrinkageMLEFitDiagnostics": fit_diagnostics,
    }
    return result

if __name__ == "__main__":
    import sys
    results = {}
    for mode in ["2026only", "2025plus2026"]:
        print(f"=== mode: {mode} ===", file=sys.stderr)
        results[mode] = run(mode)
        for name, s in results[mode].items():
            if name == "_metadata":
                continue
            print(f"  {name:16s} acc={s['accuracy']:.4f} brier={s['brier']:.4f} logloss={s['logloss']:.4f}", file=sys.stderr)
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=2)
    print("done")
