#!/usr/bin/env python3
"""Two-way date/team pigeonhole bootstrap sensitivity for GOAL3 all-k output.

Read-only analysis of the frozen all-k reconstruction artifact. It does not
fit a model, fetch live data, create forecasts, or touch production files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

K_VALUES = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0,
            15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)
REFERENCE_K = 10.0
PRIMARY_K = (6.0, 10.0)
SCOPES = ("ALL", "J1", "J2", "J3")
EXPECTED_COHORT = {
    "matches": 236,
    "teamSideTargets": 472,
    "dates": 17,
    "teams": 60,
    "leagueTargets": {"J1": 120, "J2": 178, "J3": 174},
    "dateRange": ["2026-09-19", "2026-11-03"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def product_cluster_weights(date_counts: np.ndarray, team_counts: np.ndarray,
                            date_index: np.ndarray, team_index: np.ndarray) -> np.ndarray:
    """Pigeonhole weight per target = date multiplicity * team multiplicity."""
    return date_counts[:, date_index] * team_counts[:, team_index]


def prepare_arrays(document: dict, enforce_expected_cohort: bool = True) -> dict:
    matches = document.get("allKPredictions")
    if not isinstance(matches, list) or not matches:
        raise ValueError("input must contain non-empty allKPredictions")

    records = []
    match_keys = set()
    target_keys = set()
    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            raise ValueError(f"match {index} must be an object")
        date = match.get("date")
        league = match.get("comp")
        home = match.get("home")
        away = match.get("away")
        hg = match.get("hg")
        ag = match.get("ag")
        if not isinstance(date, str) or league not in ("J1", "J2", "J3"):
            raise ValueError(f"invalid date or league in match {index}")
        if not isinstance(home, str) or not home or not isinstance(away, str) or not away or home == away:
            raise ValueError(f"invalid home/away teams in match {index}")
        if type(hg) is not int or type(ag) is not int or hg < 0 or ag < 0:
            raise ValueError(f"invalid integer score in match {index}")
        match_key = (date, league, home, away)
        if match_key in match_keys:
            raise ValueError(f"duplicate match key: {match_key}")
        match_keys.add(match_key)
        by_k = match.get("byK")
        if not isinstance(by_k, dict):
            raise ValueError(f"missing byK object in match {index}")
        for side, team, goals in (("home", home, hg), ("away", away, ag)):
            target_key = (*match_key, side)
            if target_key in target_keys:
                raise ValueError(f"duplicate target key: {target_key}")
            target_keys.add(target_key)
            probabilities = []
            for k in K_VALUES:
                key = f"{k:.1f}"
                if key not in by_k or side not in by_k[key]:
                    raise ValueError(f"missing k={key} {side} probabilities in match {index}")
                p = np.asarray(by_k[key][side], dtype=float)
                if p.shape != (4,) or not np.all(np.isfinite(p)):
                    raise ValueError(f"invalid k={key} {side} probability vector in match {index}")
                if np.any(p < 0.0) or np.any(p > 1.0) or abs(float(p.sum()) - 1.0) > 1e-9:
                    raise ValueError(f"probability vector outside simplex for k={key} {side}")
                probabilities.append(p)
            records.append({
                "date": date,
                "league": league,
                "team": team,
                "goals": goals,
                "actual": min(goals, 3),
                "probabilities": np.stack(probabilities),
            })

    dates, date_index = np.unique([r["date"] for r in records], return_inverse=True)
    teams, team_index = np.unique([r["team"] for r in records], return_inverse=True)
    leagues = np.asarray([r["league"] for r in records])
    actual = np.asarray([r["actual"] for r in records], dtype=np.int64)
    goals = np.asarray([r["goals"] for r in records], dtype=np.int64)
    probabilities = np.stack([r["probabilities"] for r in records])

    if len(records) != 2 * len(matches):
        raise ValueError("each match must contribute exactly two team-side targets")
    if enforce_expected_cohort:
        observed = {
            "matches": len(matches),
            "teamSideTargets": len(records),
            "dates": len(dates),
            "teams": len(teams),
            "leagueTargets": {s: int(np.sum(leagues == s)) for s in ("J1", "J2", "J3")},
            "dateRange": [str(dates[0]), str(dates[-1])],
        }
        if observed != EXPECTED_COHORT:
            raise ValueError(f"unexpected frozen cohort: {observed!r}")

    one_hot = np.eye(4, dtype=float)[actual]
    brier = np.sum((probabilities - one_hot[:, None, :]) ** 2, axis=2)
    selected_probability = probabilities[
        np.arange(len(records))[:, None],
        np.arange(len(K_VALUES))[None, :],
        actual[:, None],
    ]
    logloss = -np.log(np.clip(selected_probability, 1e-12, 1.0))
    accuracy = (np.argmax(probabilities, axis=2) == actual[:, None]).astype(float)

    return {
        "matches": matches,
        "dates": dates,
        "teams": teams,
        "dateIndex": date_index,
        "teamIndex": team_index,
        "leagues": leagues,
        "actual": actual,
        "goals": goals,
        "probabilities": probabilities,
        "brier": brier,
        "logloss": logloss,
        "accuracy": accuracy,
    }


def max_t_intervals(bootstrap: np.ndarray, point: np.ndarray, labels: list[dict]) -> dict:
    se = bootstrap.std(axis=0, ddof=1)
    if np.any(~np.isfinite(se)) or np.any(se <= 0.0):
        raise ValueError("invalid bootstrap standard error in max-t family")
    max_abs_t = np.max(np.abs((bootstrap - point[None, :]) / se[None, :]), axis=1)
    critical = float(np.quantile(max_abs_t, 0.95, method="linear"))
    intervals = []
    for i, label in enumerate(labels):
        intervals.append({
            **label,
            "point": float(point[i]),
            "bootstrapSE": float(se[i]),
            "nominalPercentile95CI": [
                float(np.quantile(bootstrap[:, i], 0.025, method="linear")),
                float(np.quantile(bootstrap[:, i], 0.975, method="linear")),
            ],
            "simultaneous95Band": [
                float(point[i] - critical * se[i]),
                float(point[i] + critical * se[i]),
            ],
        })
    return {"comparisonCount": len(labels), "studentizedMaxAbsTCriticalValue": critical, "intervals": intervals}


def run_audit(data: dict, replicates: int = 50_000, seed: int = 20261109) -> dict:
    if replicates < 1_000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    dates = data["dates"]
    teams = data["teams"]
    date_index = data["dateIndex"]
    team_index = data["teamIndex"]
    n = len(date_index)
    d_count = len(dates)
    t_count = len(teams)
    brier = data["brier"]
    logloss = data["logloss"]
    accuracy = data["accuracy"]
    probabilities = data["probabilities"]
    actual = data["actual"]
    goals = data["goals"]
    leagues = data["leagues"]

    point_brier = brier.mean(axis=0)
    point_logloss = logloss.mean(axis=0)
    point_accuracy = accuracy.mean(axis=0)

    scope_indices = {
        scope: np.arange(n) if scope == "ALL" else np.flatnonzero(leagues == scope)
        for scope in SCOPES
    }
    calibration_labels = []
    calibration_point = []
    calibration_data = []
    for scope in SCOPES:
        idx = scope_indices[scope]
        for k in PRIMARY_K:
            ki = K_VALUES.index(k)
            predicted = probabilities[idx, ki, :].mean(axis=0)
            observed = np.eye(4)[actual[idx]].mean(axis=0)
            gaps = (predicted - observed) * 100.0
            for category in range(4):
                calibration_labels.append({"scope": scope, "k": int(k), "category": category, "n": int(len(idx))})
                calibration_point.append(float(gaps[category]))
                calibration_data.append((idx, ki, category))
    calibration_point = np.asarray(calibration_point, dtype=float)

    residual_labels = []
    residual_point = []
    residual_data = []
    for scope in SCOPES:
        idx = scope_indices[scope]
        for k in PRIMARY_K:
            ki = K_VALUES.index(k)
            lam = -np.log(np.clip(probabilities[idx, ki, 0], 1e-300, 1.0))
            residual_squared = (goals[idx] - lam) ** 2 / lam
            residual_labels.append({"scope": scope, "k": int(k), "n": int(len(idx))})
            residual_point.append(float(residual_squared.mean()))
            residual_data.append((idx, ki))
    residual_point = np.asarray(residual_point, dtype=float)

    rng = np.random.Generator(np.random.PCG64(seed))
    boot_brier = np.empty((replicates, len(K_VALUES)), dtype=float)
    boot_logloss = np.empty_like(boot_brier)
    boot_accuracy = np.empty_like(boot_brier)
    boot_calibration = np.empty((replicates, len(calibration_labels)), dtype=float)
    boot_residual = np.empty((replicates, len(residual_labels)), dtype=float)
    empty_scope_repairs = 0

    batch_size = 500
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        date_counts = rng.multinomial(d_count, np.full(d_count, 1.0 / d_count), size=size)
        team_counts = rng.multinomial(t_count, np.full(t_count, 1.0 / t_count), size=size)
        weights = product_cluster_weights(date_counts, team_counts, date_index, team_index)
        # A replicate with no sampled member in a league has undefined league metrics.
        # Redraw the entire replicate; do not silently drop it or impute a value.
        bad = np.zeros(size, dtype=bool)
        for idx in scope_indices.values():
            bad |= weights[:, idx].sum(axis=1) == 0
        while bad.any():
            count = int(bad.sum())
            empty_scope_repairs += count
            date_counts[bad] = rng.multinomial(d_count, np.full(d_count, 1.0 / d_count), size=count)
            team_counts[bad] = rng.multinomial(t_count, np.full(t_count, 1.0 / t_count), size=count)
            weights[bad] = product_cluster_weights(date_counts[bad], team_counts[bad], date_index, team_index)
            bad = np.zeros(size, dtype=bool)
            for idx in scope_indices.values():
                bad |= weights[:, idx].sum(axis=1) == 0
        denominator = weights.sum(axis=1)
        boot_brier[start:start + size] = (weights @ brier) / denominator[:, None]
        boot_logloss[start:start + size] = (weights @ logloss) / denominator[:, None]
        boot_accuracy[start:start + size] = (weights @ accuracy) / denominator[:, None]

        for column, (idx, ki, category) in enumerate(calibration_data):
            scoped_weights = weights[:, idx]
            scoped_denominator = scoped_weights.sum(axis=1)
            predicted = scoped_weights @ probabilities[idx, ki, category]
            observed = scoped_weights @ (actual[idx] == category).astype(float)
            boot_calibration[start:start + size, column] = (
                (predicted - observed) / scoped_denominator * 100.0
            )
        for column, (idx, ki) in enumerate(residual_data):
            scoped_weights = weights[:, idx]
            scoped_denominator = scoped_weights.sum(axis=1)
            lam = -np.log(np.clip(probabilities[idx, ki, 0], 1e-300, 1.0))
            residual_squared = (goals[idx] - lam) ** 2 / lam
            boot_residual[start:start + size, column] = (
                (scoped_weights @ residual_squared) / scoped_denominator
            )

    best_frequency = {}
    for metric_name, bootstrap, mode in (
        ("brier", boot_brier, "min"),
        ("logloss", boot_logloss, "min"),
        ("accuracy", boot_accuracy, "max"),
    ):
        winners = np.argmin(bootstrap, axis=1) if mode == "min" else np.argmax(bootstrap, axis=1)
        frequencies = np.bincount(winners, minlength=len(K_VALUES)) / replicates
        best_frequency[metric_name] = {f"{k:g}": float(frequencies[i]) for i, k in enumerate(K_VALUES)}

    reference_index = K_VALUES.index(REFERENCE_K)
    contrast_columns = []
    contrast_meta = []
    for metric_name, bootstrap, point in (
        ("brier", boot_brier, point_brier),
        ("logloss", boot_logloss, point_logloss),
    ):
        for index, k in enumerate(K_VALUES):
            if index == reference_index:
                continue
            delta = bootstrap[:, index] - bootstrap[:, reference_index]
            observed = float(point[index] - point[reference_index])
            se = float(delta.std(ddof=1))
            if not np.isfinite(se) or se <= 0.0:
                raise ValueError(f"invalid SE for {metric_name}, k={k}")
            contrast_columns.append((delta - observed) / se)
            contrast_meta.append((metric_name, k, observed, se, delta))
    selection_critical = float(np.quantile(
        np.max(np.abs(np.column_stack(contrast_columns)), axis=1), 0.95, method="linear"
    ))
    selection_intervals = {}
    for metric_name, k, observed, se, delta in contrast_meta:
        selection_intervals.setdefault(f"{k:g}", {})[metric_name] = {
            "differenceKMinusK10": observed,
            "bootstrapSE": se,
            "nominalPercentile95CI": [
                float(np.quantile(delta, 0.025, method="linear")),
                float(np.quantile(delta, 0.975, method="linear")),
            ],
            "simultaneous95Band": [observed - selection_critical * se, observed + selection_critical * se],
        }

    calibration_family = max_t_intervals(boot_calibration, calibration_point, calibration_labels)
    calibration_family["intervalsContainingZero"] = sum(
        item["simultaneous95Band"][0] <= 0.0 <= item["simultaneous95Band"][1]
        for item in calibration_family["intervals"]
    )

    residual_family = max_t_intervals(boot_residual, residual_point, residual_labels)
    residual_family["intervalsContainingOne"] = sum(
        item["simultaneous95Band"][0] <= 1.0 <= item["simultaneous95Band"][1]
        for item in residual_family["intervals"]
    )

    paired_residual_labels = []
    paired_residual_bootstrap = []
    paired_residual_point = []
    for scope in SCOPES:
        i6 = next(i for i, item in enumerate(residual_labels) if item["scope"] == scope and item["k"] == 6)
        i10 = next(i for i, item in enumerate(residual_labels) if item["scope"] == scope and item["k"] == 10)
        paired_residual_bootstrap.append(boot_residual[:, i6] - boot_residual[:, i10])
        paired_residual_point.append(residual_point[i6] - residual_point[i10])
        paired_residual_labels.append({"scope": scope, "contrast": "k6_minus_k10"})
    paired_residual_family = max_t_intervals(
        np.column_stack(paired_residual_bootstrap),
        np.asarray(paired_residual_point, dtype=float),
        paired_residual_labels,
    )

    return {
        "experiment": "GOAL3 two-way date/team pigeonhole bootstrap sensitivity",
        "scope": "research-only retrospective fixed 2026-only outer cohort; sensitivity analysis only",
        "cohort": {
            "matches": len(data["matches"]),
            "teamSideTargets": n,
            "dates": d_count,
            "teams": t_count,
            "dateRange": [str(dates[0]), str(dates[-1])],
            "leagueTargets": {scope: int(len(scope_indices[scope])) for scope in ("J1", "J2", "J3")},
        },
        "method": {
            "resamplingUnit": "two-way pigeonhole bootstrap over target date and team identity",
            "dateClusters": d_count,
            "teamClusters": t_count,
            "dateResampling": "multinomial sample of D dates with replacement",
            "teamResampling": "independent multinomial sample of T teams with replacement",
            "observationWeight": "date multiplicity * team multiplicity for each team-side target",
            "weighting": "weighted target means within each scope",
            "rng": "NumPy PCG64",
            "seed": seed,
            "replicates": replicates,
            "quantile": "linear percentile",
            "emptyScopeReplicatesRedrawn": empty_scope_repairs,
            "selectionFamily": "15 non-reference k contrasts x 2 primary metrics (Brier, LogLoss), each against k=10",
            "calibrationFamily": "ALL/J1/J2/J3 x k=6/k=10 x categories 0/1/2/3+ (32 intervals)",
            "residualFamily": "ALL/J1/J2/J3 x k=6/k=10 (8 intervals)",
        },
        "pointMetricsByK": {
            f"{k:g}": {
                "brier": float(point_brier[i]),
                "logloss": float(point_logloss[i]),
                "accuracy": float(point_accuracy[i]),
            }
            for i, k in enumerate(K_VALUES)
        },
        "selectionStability": {
            "bestCandidateFrequency": best_frequency,
            "selectionAdjusted": {
                "referenceK": REFERENCE_K,
                "studentizedMaxAbsTCriticalValue": selection_critical,
                "intervals": selection_intervals,
                "k6MinusK10": selection_intervals["6"],
            },
        },
        "fourCategoryCalibration": calibration_family,
        "poissonResidualDispersion": {
            "definition": "lambda=-log(P(goal=0)); mean((observedGoals-lambda)^2/lambda)",
            "jointEightIntervalFamily": residual_family,
            "pairedK6MinusK10": paired_residual_family,
        },
        "limitations": [
            "This is a sensitivity analysis, not the predeclared date-only primary bootstrap.",
            "The two-way pigeonhole bootstrap assumes independent resampling of date and team cluster labels and uses product weights; it does not model every possible dependence structure.",
            "Only 17 target dates and 60 team identities are available; tail calibration remains uncertain.",
            "The input is retrospective and does not prove each match result's first-known availability time.",
            "Conditional Pearson residual dispersion is descriptive and is not the toto13 raw score VMR.",
            "No model fit, forecast generation, production change, or market-weight change is performed.",
        ],
        "decision": "Research-only sensitivity. Do not infer k=6 superiority or production adoption from this result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    data = prepare_arrays(document, enforce_expected_cohort=True)
    result = run_audit(data, replicates=args.replicates, seed=args.seed)
    result["provenance"] = {
        "inputFile": args.input.name,
        "inputSha256": sha256_file(args.input),
        "scriptSha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "reproductionCommand": (
            "python verification/goal3_independent/two_way_cluster_bootstrap_audit.py "
            "--input artifacts/goal3_outer_allk_reproduction.json "
            "--out artifacts/goal3_two_way_cluster_bootstrap_audit.json "
            f"--replicates {args.replicates} --seed {args.seed}"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "cohort": result["cohort"],
        "method": result["method"],
        "k6MinusK10": result["selectionStability"]["selectionAdjusted"]["k6MinusK10"],
        "calibrationIntervalsContainingZero": result["fourCategoryCalibration"]["intervalsContainingZero"],
        "residualIntervalsContainingOne": result["poissonResidualDispersion"]["jointEightIntervalFamily"]["intervalsContainingOne"],
        "output": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
