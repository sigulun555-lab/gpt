#!/usr/bin/env python3
"""Date-cluster bootstrap audit of GOAL3 Poisson Pearson residual dispersion.

Input: immutable per-match all-k reconstruction JSON.
This is a retrospective descriptive diagnostic, not a formal goodness-of-fit test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCOPES = ("ALL", "J1", "J2", "J3")
K_VALUES = (6, 10)


def flatten_targets(report: dict) -> list[dict]:
    matches = report.get("allKPredictions")
    if not isinstance(matches, list) or not matches:
        raise ValueError("input has no allKPredictions")
    rows = []
    for match in matches:
        for side, goal_key in (("home", "hg"), ("away", "ag")):
            goals = match.get(goal_key)
            if not isinstance(goals, int) or goals < 0:
                raise ValueError("invalid observed goals")
            probs_by_k = {}
            for k in K_VALUES:
                probs = np.asarray(match["byK"][f"{k}.0"][side], dtype=float)
                if probs.shape != (4,) or not np.all(np.isfinite(probs)):
                    raise ValueError("invalid four-category probability vector")
                if np.any(probs < 0) or np.any(probs > 1) or abs(float(probs.sum()) - 1.0) > 1e-9:
                    raise ValueError("probability vector is outside the simplex")
                probs_by_k[k] = probs
            rows.append({
                "date": str(match["date"]),
                "league": str(match["comp"]),
                "goals": goals,
                "p": probs_by_k,
            })
    if len(rows) != 2 * len(matches):
        raise AssertionError("target expansion count mismatch")
    return rows


def audit(report: dict, replicates: int = 50_000, seed: int = 20261105) -> dict:
    if replicates < 1_000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    rows = flatten_targets(report)
    dates = sorted({row["date"] for row in rows})
    if len(rows) != 472 or len(report["allKPredictions"]) != 236 or len(dates) != 17:
        raise ValueError("expected 236 matches / 472 team-sides / 17 dates")

    # Each date contributes (n, sum Pearson residual, sum squared residual).
    sufficient = {}
    for scope in SCOPES:
        for k in K_VALUES:
            by_date = np.zeros((len(dates), 3), dtype=float)
            for di, date in enumerate(dates):
                selected = [r for r in rows if r["date"] == date and (scope == "ALL" or r["league"] == scope)]
                if not selected:
                    continue
                y = np.asarray([r["goals"] for r in selected], dtype=float)
                p0 = np.asarray([r["p"][k][0] for r in selected], dtype=float)
                lam = -np.log(np.clip(p0, 1e-15, 1.0))
                residual = (y - lam) / np.sqrt(lam)
                by_date[di] = (len(selected), residual.sum(), np.square(residual).sum())
            sufficient[(scope, k)] = by_date

    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, len(dates), size=(replicates, len(dates)))
    multiplicities = np.zeros((replicates, len(dates)), dtype=float)
    for j in range(len(dates)):
        multiplicities[np.arange(replicates), draws[:, j]] += 1.0

    observed, boot, counts, residual_means = {}, {}, {}, {}
    for key, date_stats in sufficient.items():
        total = date_stats.sum(axis=0)
        n = int(total[0])
        if n == 0:
            raise ValueError(f"empty cohort for {key}")
        observed[key] = float(total[2] / total[0])
        residual_means[key] = float(total[1] / total[0])
        counts[key] = n
        sampled = multiplicities @ date_stats
        boot[key] = sampled[:, 2] / sampled[:, 0]

    keys = [(scope, k) for scope in SCOPES for k in K_VALUES]
    ses = {key: float(np.std(boot[key], ddof=1)) for key in keys}
    studentized = np.column_stack([(boot[key] - observed[key]) / ses[key] for key in keys])
    critical = float(np.quantile(np.max(np.abs(studentized), axis=1), 0.95, method="linear"))

    results = {}
    for scope, k in keys:
        samples = boot[(scope, k)]
        point = observed[(scope, k)]
        se = ses[(scope, k)]
        results[f"{scope}_k{k}"] = {
            "n": counts[(scope, k)],
            "pearsonResidualMean": residual_means[(scope, k)],
            "dispersionIndexMeanSquaredPearsonResidual": point,
            "bootstrapSE": se,
            "nominalPercentileCI95": [
                float(np.quantile(samples, 0.025, method="linear")),
                float(np.quantile(samples, 0.975, method="linear")),
            ],
            "jointMaxTBand95": [point - critical * se, point + critical * se],
            "poissonReference": 1.0,
        }

    paired = {}
    for scope in SCOPES:
        delta = boot[(scope, 6)] - boot[(scope, 10)]
        paired[scope] = {
            "k6MinusK10DispersionIndex": observed[(scope, 6)] - observed[(scope, 10)],
            "percentileCI95": [
                float(np.quantile(delta, 0.025, method="linear")),
                float(np.quantile(delta, 0.975, method="linear")),
            ],
        }

    return {
        "experiment": "GOAL3 date-cluster bootstrap of Poisson Pearson residual dispersion",
        "cohort": {"matches": 236, "teamSideTargets": 472, "dates": 17},
        "method": {
            "lambda": "-log(P(goal=0)) from the four-category forecast",
            "pearsonResidual": "(observedGoals - lambda) / sqrt(lambda)",
            "dispersionIndex": "mean(PearsonResidual ** 2)",
            "resamplingUnit": "target date; all matches and home/away team-sides on a sampled date retained together",
            "rng": "NumPy PCG64",
            "seed": seed,
            "replicates": replicates,
            "quantile": "linear percentile",
            "jointFamily": "8 intervals: ALL/J1/J2/J3 x k=6/k=10",
            "jointBand": "studentized max-absolute-t bootstrap band",
            "jointMaxTCriticalValue": critical,
        },
        "results": results,
        "pairedK6MinusK10": paired,
        "limitations": [
            "Only 17 date clusters are available; bootstrap tails are uncertain.",
            "Repeated teams across different dates are not separately clustered.",
            "Reference value 1 is theoretical under a correctly specified conditional Poisson mean; parameter estimation and misspecification affect it.",
            "Descriptive residual diagnostic only; not a formal goodness-of-fit or model-selection test.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = audit(report, args.replicates, args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cohort": result["cohort"], "maxTCritical": result["method"]["jointMaxTCriticalValue"]}))


if __name__ == "__main__":
    main()
