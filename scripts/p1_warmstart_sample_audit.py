#!/usr/bin/env python3
"""Targeted warm-start follow-up for 20 recorded P1 failed fits. Audit branch only."""
import csv, json, math, random, re, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import p1_scipy_failure_audit as audit

THRESHOLD = 1e-4
SEED = 20261028

def select_rows(rows):
    rng = random.Random(SEED)
    groups = {}
    for row in rows:
        key = (row["train_mode"], row["start"], row["lambda"])
        groups.setdefault(key, []).append(row)
    chosen = []
    chosen_keys = set()
    for key in sorted(groups):
        row = rng.choice(groups[key])
        chosen.append(row)
        chosen_keys.add((row["train_mode"], row["start"], row["lambda"], row["model"], row["date"], row["competition"], row["home"], row["away"]))
    remaining = [r for r in rows if (r["train_mode"], r["start"], r["lambda"], r["model"], r["date"], r["competition"], r["home"], r["away"]) not in chosen_keys]
    # Add model coverage before filling the remaining slots randomly.
    for model in ["home_away_attack_defense", "recency_half_life_60", "recency_half_life_120", "recency_half_life_240", "baseline"]:
        if len(chosen) >= 20:
            break
        candidates = [r for r in remaining if r["model"] == model]
        if candidates:
            row = rng.choice(candidates)
            chosen.append(row)
            remaining.remove(row)
    if len(chosen) < 20:
        chosen.extend(rng.sample(remaining, 20-len(chosen)))
    return chosen[:20]

def main():
    failure_path = ROOT / "p1-2026-fit-failures.csv"
    with failure_path.open(encoding="utf-8-sig", newline="") as f:
        failures = list(csv.DictReader(f))
    selected = select_rows(failures)
    r25, r26 = audit.rows2025(), audit.rows2026()
    teams = sorted({t for r in r25+r26 for t in (r["home"], r["away"])})
    ix = {t:i for i,t in enumerate(teams)}
    H = np.repeat(np.arange(len(teams)), len(teams))
    A = np.tile(np.arange(len(teams)), len(teams))
    mask = H != A
    H, A = H[mask], A[mask]
    out = []
    for row in selected:
        rec = {k: row.get(k, "") for k in ("train_mode","start","lambda","model","date","competition","home","away","reason","iterations","gradient_max_abs","gradient_threshold_1e5_met")}
        target_home, target_away = audit.canon(row["home"]), audit.canon(row["away"])
        matches = [i for i,r in enumerate(r26) if r["date"] == row["date"] and r["competition"] == row["competition"] and r["home"] == target_home and r["away"] == target_away]
        if len(matches) != 1:
            raise RuntimeError(f"target match count {len(matches)} for {row['date']} {row['home']}-{row['away']}")
        target = r26[matches[0]]
        train = (r25 if row["train_mode"] == "2025_plus_2026" else []) + r26[:matches[0]]
        data = audit.map_rows(train, ix)
        lam = float(row["lambda"])
        kind = row["model"]
        p_js = np.asarray(json.loads(row["parameters"]), dtype=float)
        n = len(teams)
        base = 4*n if kind == "home_away_attack_defense" else 2*n
        half_match = re.match(r"recency_half_life_(\d+)$", kind)
        half = int(half_match.group(1)) if half_match else None
        weights = audit.weights_for(train, target["date"], half)
        fun = lambda p: audit.objgrad(p, data, n, lam, kind, weights)
        cold0 = np.zeros_like(p_js)
        cold0[base] = math.log(1.3)
        warm = minimize(fun, p_js, jac=True, method="L-BFGS-B",
                        options={"gtol":1e-10,"ftol":0.0,"maxiter":10000,"maxfun":1000000,"maxls":100,"maxcor":10})
        cold = minimize(fun, cold0, jac=True, method="L-BFGS-B",
                        options={"gtol":1e-10,"ftol":0.0,"maxiter":10000,"maxfun":1000000,"maxls":100,"maxcor":10})
        js_probs = audit.pair_probs(p_js, teams, kind, H, A)
        warm_probs = audit.pair_probs(np.asarray(warm.x), teams, kind, H, A)
        cold_probs = audit.pair_probs(np.asarray(cold.x), teams, kind, H, A)
        js_target = audit.pair_probs(p_js, teams, kind, np.array([ix[target_home]]), np.array([ix[target_away]]))[0]
        warm_target = audit.pair_probs(np.asarray(warm.x), teams, kind, np.array([ix[target_home]]), np.array([ix[target_away]]))[0]
        cold_target = audit.pair_probs(np.asarray(cold.x), teams, kind, np.array([ix[target_home]]), np.array([ix[target_away]]))[0]
        f_w, g_w = fun(np.asarray(warm.x))
        f_c, g_c = fun(np.asarray(cold.x))
        rec.update({
            "js_objective": float(row["objective"]),
            "warm_success": bool(warm.success), "warm_message": str(warm.message), "warm_iterations": int(warm.nit),
            "warm_objective": float(warm.fun), "warm_objective_abs_diff_vs_js": abs(float(row["objective"])-float(warm.fun)),
            "warm_gmax": float(np.max(np.abs(g_w))), "warm_max_abs_dP_vs_js": float(np.max(np.abs(js_probs-warm_probs))),
            "warm_target_argmax_match": int(np.argmax(js_target)==np.argmax(warm_target)),
            "cold_success": bool(cold.success), "cold_message": str(cold.message), "cold_iterations": int(cold.nit),
            "cold_objective": float(cold.fun), "cold_objective_abs_diff_vs_js": abs(float(row["objective"])-float(cold.fun)),
            "cold_gmax": float(np.max(np.abs(g_c))), "cold_max_abs_dP_vs_js": float(np.max(np.abs(js_probs-cold_probs))),
            "cold_target_argmax_match": int(np.argmax(js_target)==np.argmax(cold_target)),
            "warm_cold_max_abs_dP": float(np.max(np.abs(warm_probs-cold_probs))),
            "warm_pass_dP_1e-4": bool(np.max(np.abs(js_probs-warm_probs)) <= THRESHOLD),
            "cold_pass_dP_1e-4": bool(np.max(np.abs(js_probs-cold_probs)) <= THRESHOLD),
            "audit_error": ""
        })
        out.append(rec)
    fields = list(out[0].keys())
    with (ROOT/"p1-warmstart-sample-20.csv").open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    def stats(prefix):
        return {
            "success_count": sum(bool(r[prefix+"_success"]) for r in out),
            "max_abs_dP_vs_js": max(r[prefix+"_max_abs_dP_vs_js"] for r in out),
            "median_abs_dP_vs_js": float(np.median([r[prefix+"_max_abs_dP_vs_js"] for r in out])),
            "objective_abs_diff_max": max(r[prefix+"_objective_abs_diff_vs_js"] for r in out),
            "target_argmax_mismatches": sum(not bool(r[prefix+"_target_argmax_match"]) for r in out),
            "pass_dP_threshold_count": sum(bool(r[prefix+"_pass_dP_1e-4"]) for r in out),
            "message_counts": {m:sum(r[prefix+"_message"]==m for r in out) for m in sorted({r[prefix+"_message"] for r in out})}
        }
    summary = {
        "sample_size": len(out), "seed": SEED, "selection": "deterministic stratified sample: one row per train_mode/start/lambda cell, then model coverage and seeded fill",
        "interpretation": "warm-start means SciPy L-BFGS-B initialized from the recorded JS fit parameters; cold-start is independent zero parameters with intercept log(1.3). This tests sensitivity to optimizer initialization; it does not replay the entire JS sequential warm-start chain.",
        "warm_start": stats("warm"), "cold_start": stats("cold"),
        "warm_vs_cold_max_abs_dP": max(r["warm_cold_max_abs_dP"] for r in out),
        "all_target_argmax_matches_warm": all(bool(r["warm_target_argmax_match"]) for r in out),
        "all_target_argmax_matches_cold": all(bool(r["cold_target_argmax_match"]) for r in out),
        "audit_errors": sum(bool(r["audit_error"]) for r in out),
        "threshold": THRESHOLD
    }
    (ROOT/"p1-warmstart-sample-20-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
