#!/usr/bin/env python3
"""Independent numerical audit of the frozen GOAL3 k=6/k=10 outer cohort.

This implementation deliberately does not import nested_validation.py or
reproduce_allk_outer.py. It rebuilds the penalized Poisson fit using explicit
home/away design matrices and a separately written analytic gradient.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

EXPECTED_SHA256 = {
    "history2026": "efca00ea491fcfea6a1267af403cb5ba80ae540fca97a076def25815fc318201",
    "aliases": "92e6377068df8dd63935e0c003adfd7b355e7adac23f7aace4f947be4f5f027e",
    "frozenCsv": "5cc8960c10f3554d44ccc0d5388d0ceeb4c8a39dec4cc8b28bd713f0e3755678",
}
SOURCE_GIT_BLOB_SHA = {
    "history2026": "f5cc8c044758e45b1e679e4ce796e68b3f27a854",
    "aliases": "5de22584524ff4c07ddac1d27f49afbbbc5f88ab",
    "frozenCsv": "56892251679b8774a9e1b7ccc4b5beb42e660e36",
}
K_VALUES = (6.0, 10.0)
LEAGUES = ("J1", "J2", "J3")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(name: str, aliases: dict[str, str]) -> str:
    value = unicodedata.normalize("NFKC", str(name)).strip()
    return aliases.get(value, value)


def load_history(path: Path, aliases: dict[str, str]):
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    fetched_at = set()
    missing_fetched_at = 0
    for raw in doc.get("history", []):
        date = raw.get("date")
        comp = raw.get("competition")
        hg, ag = raw.get("home_goals"), raw.get("away_goals")
        if not isinstance(date, str) or not date.startswith("2026-"):
            raise ValueError(f"non-2026 history row: {date!r}")
        if comp not in LEAGUES:
            continue
        if not isinstance(hg, int) or not isinstance(ag, int) or hg < 0 or ag < 0:
            continue
        fetched = raw.get("fetchedAt")
        if fetched:
            fetched_at.add(str(fetched))
        else:
            missing_fetched_at += 1
        rows.append({
            "date": date,
            "comp": comp,
            "home": canonical(raw["home"], aliases),
            "away": canonical(raw["away"], aliases),
            "hg": int(hg),
            "ag": int(ag),
        })
    rows.sort(key=lambda x: (x["date"], x["home"], x["away"]))
    keys = [(r["date"], r["comp"], r["home"], r["away"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate canonical history match key")
    return rows, {
        "rows": len(rows),
        "distinctDates": len({r["date"] for r in rows}),
        "distinctTeams": len({r[s] for r in rows for s in ("home", "away")}),
        "fetchedAtUniqueCount": len(fetched_at),
        "fetchedAtValues": sorted(fetched_at),
        "missingFetchedAt": missing_fetched_at,
        "updatedAt": doc.get("updatedAt"),
    }


def load_frozen(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {
        "date", "comp", "fold", "home", "away", "hg", "ag",
        "side", "actual",
        *{f"k{k}_p{i}" for k in (6, 10) for i in range(4)},
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("frozen CSV is missing required columns")
    for r in rows:
        for key in ("fold", "hg", "ag", "actual"):
            r[key] = int(r[key])
        if r["comp"] not in LEAGUES or r["side"] not in ("home", "away"):
            raise ValueError("unexpected league or side in frozen CSV")
        if r["actual"] != min(3, r["hg"] if r["side"] == "home" else r["ag"]):
            raise ValueError("frozen actual category does not match raw score")
        for k in (6, 10):
            for i in range(4):
                r[f"k{k}_p{i}"] = float(r[f"k{k}_p{i}"])
    keyfn = lambda r: (r["date"], r["comp"], r["home"], r["away"], r["side"])
    keys = [keyfn(r) for r in rows]
    if len(rows) != 472 or len(set(keys)) != 472:
        raise ValueError("expected 472 unique team-side targets")
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["date"], r["comp"], r["home"], r["away"])].append(r)
    matches = []
    for key, pair in sorted(grouped.items()):
        if len(pair) != 2 or {x["side"] for x in pair} != {"home", "away"}:
            raise ValueError(f"invalid home/away pair: {key}")
        if len({(x["fold"], x["hg"], x["ag"]) for x in pair}) != 1:
            raise ValueError(f"pair metadata mismatch: {key}")
        x = pair[0]
        matches.append({
            "date": x["date"], "comp": x["comp"], "fold": x["fold"],
            "home": x["home"], "away": x["away"], "hg": x["hg"], "ag": x["ag"],
        })
    dates = sorted({r["date"] for r in rows})
    if len(matches) != 236 or len(dates) != 17:
        raise ValueError("expected 236 matches over 17 target dates")
    fold_by_date = defaultdict(set)
    for r in rows:
        fold_by_date[r["date"]].add(r["fold"])
    if any(len(v) != 1 for v in fold_by_date.values()):
        raise ValueError("target date crosses outer-fold boundary")
    return rows, matches, dates, {d: next(iter(v)) for d, v in fold_by_date.items()}


def build_design(prior):
    teams = sorted({m["home"] for m in prior} | {m["away"] for m in prior})
    if not teams:
        raise ValueError("empty training history")
    idx = {team: i for i, team in enumerate(teams)}
    n, p = len(teams), 2 + 2 * len(teams)
    m = len(prior)
    x_home = np.zeros((m, p), dtype=float)
    x_away = np.zeros((m, p), dtype=float)
    hg = np.asarray([r["hg"] for r in prior], dtype=float)
    ag = np.asarray([r["ag"] for r in prior], dtype=float)
    for row, match in enumerate(prior):
        hi, ai = idx[match["home"]], idx[match["away"]]
        x_home[row, 0] = 1.0
        x_home[row, 1] = 1.0
        x_home[row, 2 + hi] = 1.0
        x_home[row, 2 + n + ai] = -1.0
        x_away[row, 0] = 1.0
        x_away[row, 2 + ai] = 1.0
        x_away[row, 2 + n + hi] = -1.0
    return teams, idx, x_home, x_away, hg, ag


def independent_objective_gradient(theta, x_home, x_away, hg, ag, k, team_count):
    """Dense design-matrix implementation, independent of shared fit helpers."""
    eta_h_raw = x_home @ theta
    eta_a_raw = x_away @ theta
    eta_h = np.clip(eta_h_raw, -20.0, 20.0)
    eta_a = np.clip(eta_a_raw, -20.0, 20.0)
    rate_h = np.exp(eta_h)
    rate_a = np.exp(eta_a)
    attack = theta[2:2 + team_count]
    defense = theta[2 + team_count:2 + 2 * team_count]
    value = (
        np.sum(rate_h - hg * eta_h)
        + np.sum(rate_a - ag * eta_a)
        + k * (attack @ attack + defense @ defense)
    )
    active_h = ((eta_h_raw > -20.0) & (eta_h_raw < 20.0)).astype(float)
    active_a = ((eta_a_raw > -20.0) & (eta_a_raw < 20.0)).astype(float)
    residual_h = (rate_h - hg) * active_h
    residual_a = (rate_a - ag) * active_a
    grad = x_home.T @ residual_h + x_away.T @ residual_a
    grad[2:2 + team_count] += 2.0 * k * attack
    grad[2 + team_count:2 + 2 * team_count] += 2.0 * k * defense
    return float(value), grad


def fit_independent(prior, k):
    teams, idx, xh, xa, hg, ag = build_design(prior)
    n = len(teams)
    theta0 = np.zeros(2 + 2 * n, dtype=float)
    mean_goals = float((hg.sum() + ag.sum()) / (2.0 * len(prior)))
    theta0[0] = math.log(1.3)

    def fun_and_jac(theta):
        return independent_objective_gradient(theta, xh, xa, hg, ag, k, n)

    result = minimize(
        fun_and_jac, theta0, jac=True, method="L-BFGS-B",
        options={"maxiter": 300},
    )
    info = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "gradientInfinityNorm": float(np.max(np.abs(result.jac))) if result.jac is not None else None,
        "trainingMatches": len(prior),
        "trainingDates": len({r["date"] for r in prior}),
        "teamCount": n,
    }
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"independent optimizer failed for k={k}: {info}")
    return result.x, idx, info


def poisson_categories(rate):
    lam = float(rate)
    p0 = math.exp(-lam)
    p1 = p0 * lam
    p2 = p1 * lam / 2.0
    p3 = max(0.0, 1.0 - p0 - p1 - p2)
    probs = np.asarray([p0, p1, p2, p3], dtype=float)
    if not np.all(np.isfinite(probs)) or np.any(probs < 0) or abs(float(probs.sum()) - 1.0) > 1e-10:
        raise ValueError(f"invalid Poisson category probabilities for lambda={lam}")
    return probs


def predict_independent(theta, idx, home, away):
    n = len(idx)
    attack = theta[2:2 + n]
    defense = theta[2 + n:2 + 2 * n]
    h_attack = attack[idx[home]] if home in idx else 0.0
    h_defense = defense[idx[home]] if home in idx else 0.0
    a_attack = attack[idx[away]] if away in idx else 0.0
    a_defense = defense[idx[away]] if away in idx else 0.0
    eta_h = float(np.clip(theta[0] + theta[1] + h_attack - a_defense, -20.0, 20.0))
    eta_a = float(np.clip(theta[0] + a_attack - h_defense, -20.0, 20.0))
    return poisson_categories(math.exp(eta_h)), poisson_categories(math.exp(eta_a))


def score(probabilities, actual):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(actual, dtype=int)
    if p.shape != (len(y), 4) or not np.all(np.isfinite(p)):
        raise ValueError("invalid prediction matrix")
    if np.any(p < 0) or np.any(p > 1) or np.max(np.abs(p.sum(axis=1) - 1.0)) > 1e-9:
        raise ValueError("probability vector outside simplex")
    one_hot = np.eye(4)[y]
    return {
        "n": int(len(y)),
        "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "logloss": float(np.mean(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)))),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == y)),
    }


def calibration(rows, predictions, k, subset):
    selected = [i for i, r in enumerate(rows) if subset(r)]
    if not selected:
        return {"n": 0, "gapsPP": None}
    p = np.asarray([predictions[i] for i in selected])
    y = np.asarray([rows[i]["actual"] for i in selected], dtype=int)
    observed = np.asarray([(y == c).mean() for c in range(4)])
    predicted = p.mean(axis=0)
    return {
        "n": len(selected),
        "gapsPP": [float(100.0 * (predicted[c] - observed[c])) for c in range(4)],
        "predictedMean": predicted.tolist(),
        "observedRate": observed.tolist(),
    }


def poisson_residuals(rows, predictions, subset):
    selected = [i for i, r in enumerate(rows) if subset(r)]
    if not selected:
        return {"n": 0}
    goals, lambdas = [], []
    for i in selected:
        r = rows[i]
        goals.append(r["hg"] if r["side"] == "home" else r["ag"])
        p0 = float(predictions[i][0])
        lambdas.append(max(1e-12, -math.log(max(1e-15, min(1.0, p0)))))
    goals = np.asarray(goals, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    z = (goals - lambdas) / np.sqrt(lambdas)
    return {
        "n": len(selected),
        "meanGoals": float(goals.mean()),
        "meanLambda": float(lambdas.mean()),
        "pearsonResidualRMS": float(np.sqrt(np.mean(z * z))),
    }


def moving_block_bootstrap(rows, p6, p10, reps, seed):
    dates = sorted({r["date"] for r in rows})
    by_date = defaultdict(list)
    for i, r in enumerate(rows):
        by_date[r["date"]].append(i)
    d_brier = np.sum((p6 - np.eye(4)[[r["actual"] for r in rows]]) ** 2, axis=1) - np.sum(
        (p10 - np.eye(4)[[r["actual"] for r in rows]]) ** 2, axis=1
    )
    d_logloss = -np.log(np.clip(p6[np.arange(len(rows)), [r["actual"] for r in rows]], 1e-15, 1.0)) + np.log(
        np.clip(p10[np.arange(len(rows)), [r["actual"] for r in rows]], 1e-15, 1.0)
    )
    d_accuracy = (
        np.argmax(p6, axis=1) == np.asarray([r["actual"] for r in rows])
    ).astype(float) - (
        np.argmax(p10, axis=1) == np.asarray([r["actual"] for r in rows])
    ).astype(float)
    n_dates, block_len = len(dates), 2
    block_count = math.ceil(n_dates / block_len)
    rng = np.random.Generator(np.random.PCG64(seed))
    sims = np.empty((reps, 3), dtype=float)
    for b in range(reps):
        starts = rng.integers(0, n_dates, size=block_count)
        sampled_dates = []
        for start in starts:
            sampled_dates.extend(dates[(int(start) + offset) % n_dates] for offset in range(block_len))
        sampled_dates = sampled_dates[:n_dates]
        indices = np.concatenate([np.asarray(by_date[d], dtype=int) for d in sampled_dates])
        sims[b, 0] = d_brier[indices].mean()
        sims[b, 1] = d_logloss[indices].mean()
        sims[b, 2] = d_accuracy[indices].mean()
    point = {
        "brier": float(d_brier.mean()),
        "logloss": float(d_logloss.mean()),
        "accuracy": float(d_accuracy.mean()),
    }
    out = {
        "method": "circular moving-block bootstrap; dates sampled in length-2 blocks; target-count-weighted",
        "rng": "NumPy PCG64",
        "seed": seed,
        "replicates": reps,
        "dateClusters": n_dates,
        "blockLengthDates": block_len,
        "blocksPerReplicate": block_count,
        "sampledDatesPerReplicate": n_dates,
        "percentile": "95%; numpy quantile method=linear",
        "pointDifferenceK6MinusK10": point,
    }
    for j, name in enumerate(("brier", "logloss", "accuracy")):
        out[name] = {
            "bootstrapMean": float(sims[:, j].mean()),
            "ci95": [float(x) for x in np.quantile(sims[:, j], [0.025, 0.975], method="linear")],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--aliases", required=True)
    ap.add_argument("--frozen-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap-replicates", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--max-probability-error", type=float, default=2e-6)
    args = ap.parse_args()

    history_path, aliases_path, frozen_path = Path(args.history), Path(args.aliases), Path(args.frozen_csv)
    actual_sha = {
        "history2026": file_sha256(history_path),
        "aliases": file_sha256(aliases_path),
        "frozenCsv": file_sha256(frozen_path),
    }
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(f"frozen input SHA-256 mismatch: {actual_sha}")

    aliases_doc = json.loads(aliases_path.read_text(encoding="utf-8"))
    aliases = aliases_doc.get("aliases", {})
    history, history_audit = load_history(history_path, aliases)
    rows, matches, dates, fold_by_date = load_frozen(frozen_path)

    source_by_key = {
        (m["date"], m["comp"], m["home"], m["away"]): m for m in history
    }
    for m in matches:
        key = (m["date"], m["comp"], m["home"], m["away"])
        src = source_by_key.get(key)
        if src is None or (src["hg"], src["ag"]) != (m["hg"], m["ag"]):
            raise ValueError(f"frozen cohort/source history mismatch: {key}")

    predictions = {6: [None] * len(rows), 10: [None] * len(rows)}
    fit_diagnostics = []
    row_indices_by_date = defaultdict(list)
    for i, r in enumerate(rows):
        row_indices_by_date[r["date"]].append(i)
    for date in dates:
        prior = [m for m in history if m["date"] < date]
        if not prior:
            raise ValueError(f"empty prior for target date {date}")
        date_matches = [m for m in matches if m["date"] == date]
        for k in K_VALUES:
            theta, idx, info = fit_independent(prior, k)
            fit_diagnostics.append({
                "date": date, "fold": fold_by_date[date], "k": int(k), **info,
            })
            for m in date_matches:
                ph, pa = predict_independent(theta, idx, m["home"], m["away"])
                for i in row_indices_by_date[date]:
                    r = rows[i]
                    if (r["comp"], r["home"], r["away"]) != (m["comp"], m["home"], m["away"]):
                        continue
                    predictions[int(k)][i] = ph if r["side"] == "home" else pa

    if any(x is None for k in (6, 10) for x in predictions[k]):
        raise ValueError("one or more frozen target rows were not independently predicted")

    max_error, mean_abs_error = {}, {}
    for k in (6, 10):
        independent = np.asarray(predictions[k], dtype=float)
        frozen = np.asarray([[r[f"k{k}_p{c}"] for c in range(4)] for r in rows], dtype=float)
        abs_error = np.abs(independent - frozen)
        max_error[str(k)] = float(abs_error.max())
        mean_abs_error[str(k)] = float(abs_error.mean())

    p6, p10 = np.asarray(predictions[6]), np.asarray(predictions[10])
    actual = np.asarray([r["actual"] for r in rows], dtype=int)
    metrics = {"k6": score(p6, actual), "k10": score(p10, actual)}
    metrics["differenceK6MinusK10"] = {
        key: float(metrics["k6"][key] - metrics["k10"][key])
        for key in ("brier", "logloss", "accuracy")
    }
    bootstrap = moving_block_bootstrap(rows, p6, p10, args.bootstrap_replicates, args.seed)

    calibration_report = {}
    for k, p in ((6, p6), (10, p10)):
        calibration_report[str(k)] = {
            "all": calibration(rows, p, k, lambda r: True),
            "byLeague": {
                league: calibration(rows, p, k, lambda r, league=league: r["comp"] == league)
                for league in LEAGUES
            },
        }
    residual_report = {
        "k6": {
            "all": poisson_residuals(rows, p6, lambda r: True),
            "byLeague": {
                league: poisson_residuals(rows, p6, lambda r, league=league: r["comp"] == league)
                for league in LEAGUES
            },
        },
        "note": "Descriptive Pearson residual RMS only; not a formal goodness-of-fit test.",
    }

    passed = all(v <= args.max_probability_error for v in max_error.values())
    report = {
        "experiment": "Independent GOAL3 shrinkage-MLE optimizer audit for frozen k=6/k=10 outer cohort",
        "independentImplementation": {
            "doesNotImport": ["nested_validation.py", "reproduce_allk_outer.py"],
            "optimizer": "SciPy L-BFGS-B",
            "gradient": "separate dense home/away design-matrix gradient",
            "initialization": "log(1.3); zero home advantage and team effects, matching reference starting point",
            "objective": "Poisson negative log likelihood + k * (sum attack^2 + sum defense^2)",
            "logRateClip": [-20.0, 20.0],
            "maxIterations": 300,
            "stoppingControls": "SciPy L-BFGS-B defaults except maxiter=300",
        },
        "provenance": {
            "sha256": actual_sha,
            "gitBlobSha": SOURCE_GIT_BLOB_SHA,
            "2025CsvUsed": False,
            "cohortSource": "frozen goal3_outer_matches.csv; source match keys and raw scores cross-checked against 2026 history",
            "historicalAvailabilityCaveat": "History fetchedAt is a retrospective snapshot field; this audit does not prove when each result first became available.",
        },
        "cohort": {
            "matches": len(matches),
            "teamSideTargets": len(rows),
            "targetDates": len(dates),
            "dateRange": [dates[0], dates[-1]],
            "foldDates": dict(Counter(str(fold_by_date[d]) for d in dates)),
            "leagueTargets": dict(Counter(r["comp"] for r in rows)),
            "history2026": history_audit,
        },
        "fitDiagnostics": {
            "fitCount": len(fit_diagnostics),
            "failureCount": sum(not x["success"] for x in fit_diagnostics),
            "minTrainingMatches": min(x["trainingMatches"] for x in fit_diagnostics),
            "maxTrainingMatches": max(x["trainingMatches"] for x in fit_diagnostics),
            "perDateAndK": fit_diagnostics,
        },
        "reconstructionCheck": {
            "maxProbabilityAbsErrorVsFrozenCsv": max_error,
            "meanAbsoluteProbabilityErrorVsFrozenCsv": mean_abs_error,
            "tolerance": args.max_probability_error,
            "passed": passed,
        },
        "metrics": metrics,
        "bootstrap": bootstrap,
        "fourCategoryCalibration": calibration_report,
        "poissonResidualDiagnostics": residual_report,
        "reproductionCommand": (
            "python verification/goal3_independent/independent_optimizer_audit.py "
            "--history data/history.json --aliases data/team_aliases.json "
            "--frozen-csv verification/goal3_independent/goal3_outer_matches.csv "
            "--out artifacts/goal3_independent_optimizer_audit.json "
            f"--bootstrap-replicates {args.bootstrap_replicates} --seed {args.seed}"
        ),
        "decision": "Research-only. A successful reproduction is not evidence of prospective superiority or production adoption.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": passed,
        "maxProbabilityAbsError": max_error,
        "metrics": metrics,
        "bootstrap95": {k: bootstrap[k]["ci95"] for k in ("brier", "logloss", "accuracy")},
        "fitCount": len(fit_diagnostics),
        "failureCount": report["fitDiagnostics"]["failureCount"],
        "output": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
