#!/usr/bin/env python3
"""Date-cluster bootstrap audit of GOAL3 Poisson Pearson residual dispersion.

Input: immutable per-match all-k reconstruction JSON.
This is a retrospective descriptive diagnostic, not a formal goodness-of-fit test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from datetime import date as Date

import numpy as np

SCOPES = ("ALL", "J1", "J2", "J3")
K_VALUES = (6, 10)


def flatten_targets(report: dict) -> list[dict]:
    matches = report.get("allKPredictions")
    if not isinstance(matches, list) or not matches:
        raise ValueError("input has no allKPredictions")
    rows = []
    seen_match_keys = set()
    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            raise ValueError(f"match {index} is not an object")
        date = match.get("date")
        league = match.get("comp")
        home, away = match.get("home"), match.get("away")
        if not isinstance(date, str):
            raise ValueError(f"invalid target date at match {index}: {date!r}")
        try:
            parsed_date = Date.fromisoformat(date)
        except ValueError as exc:
            raise ValueError(f"invalid target date at match {index}: {date!r}") from exc
        if parsed_date.isoformat() != date or parsed_date.year != 2026:
            raise ValueError(f"invalid target date at match {index}: {date!r}")
        if league not in ("J1", "J2", "J3"):
            raise ValueError(f"invalid competition at match {index}: {league!r}")
        if not isinstance(home, str) or not home.strip() or not isinstance(away, str) or not away.strip():
            raise ValueError(f"missing home/away team at match {index}")
        home, away = home.strip(), away.strip()
        if home == away:
            raise ValueError(f"same home and away team at match {index}")
        match_key = (date, league, home, away)
        if match_key in seen_match_keys:
            raise ValueError(f"duplicate match key: {match_key!r}")
        seen_match_keys.add(match_key)

        by_k = match.get("byK")
        if not isinstance(by_k, dict):
            raise ValueError(f"missing byK object at match {index}")
        for side, goal_key in (("home", "hg"), ("away", "ag")):
            goals = match.get(goal_key)
            if type(goals) is not int or goals < 0:
                raise ValueError(f"invalid observed goals at match {index}, {side}: {goals!r}")
            probs_by_k = {}
            for k in K_VALUES:
                key = f"{k}.0"
                if key not in by_k or not isinstance(by_k[key], dict) or side not in by_k[key]:
                    raise ValueError(f"missing k={key} {side} probabilities at match {index}")
                probs = np.asarray(by_k[key][side], dtype=float)
                if probs.shape != (4,) or not np.all(np.isfinite(probs)):
                    raise ValueError(f"invalid four-category probability vector at match {index}, k={key}, side={side}")
                if np.any(probs < 0) or np.any(probs > 1) or abs(float(probs.sum()) - 1.0) > 1e-9:
                    raise ValueError(f"probability vector is outside the simplex at match {index}, k={key}, side={side}")
                if not 0.0 < float(probs[0]) < 1.0:
                    raise ValueError(f"goal-zero probability must be strictly between 0 and 1 at match {index}, k={key}, side={side}")
                probs_by_k[k] = probs
            rows.append({
                "date": date,
                "league": league,
                "matchKey": match_key,
                "side": side,
                "goals": goals,
                "p": probs_by_k,
            })
    if len(rows) != 2 * len(matches):
        raise AssertionError("target expansion count mismatch")
    return rows

def audit(
    report: dict,
    replicates: int = 50_000,
    seed: int = 20261105,
    input_sha256: str | None = None,
    code_sha256: str | None = None,
) -> dict:
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
                lam = -np.log(p0)
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
        "provenance": {
            "inputSha256": input_sha256,
            "codeSha256": code_sha256,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
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
    input_path = Path(args.input)
    input_bytes = input_path.read_bytes()
    report = json.loads(input_bytes)
    code_bytes = Path(__file__).read_bytes()
    result = audit(
        report,
        args.replicates,
        args.seed,
        input_sha256=hashlib.sha256(input_bytes).hexdigest(),
        code_sha256=hashlib.sha256(code_bytes).hexdigest(),
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cohort": result["cohort"], "maxTCritical": result["method"]["jointMaxTCriticalValue"]}))


if __name__ == "__main__":
    main()
