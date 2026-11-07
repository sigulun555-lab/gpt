#!/usr/bin/env python3
"""Pair-preserving date/team pigeonhole bootstrap sensitivity for GOAL3.

Each match is the base observation and contributes two linked team-side targets.
Independent date and team cluster multiplicities are sampled; the product of the
date count and both participating teams' counts is applied equally to both
sides of the match. This preserves the within-match home/away pairing that a
row-level date*team weighting can break. Research-only; no fitting or forecasts.
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_preserving_match_weights(
    date_counts: np.ndarray,
    team_counts: np.ndarray,
    date_index: np.ndarray,
    home_team_index: np.ndarray,
    away_team_index: np.ndarray,
) -> np.ndarray:
    """Return one weight per match; both team-side rows share this weight."""
    return (
        date_counts[:, date_index]
        * team_counts[:, home_team_index]
        * team_counts[:, away_team_index]
    )


def prepare_arrays(document: dict, enforce_expected_cohort: bool = True) -> dict:
    matches = document.get("allKPredictions") if isinstance(document, dict) else None
    if not isinstance(matches, list) or not matches:
        raise ValueError("input must contain non-empty allKPredictions")

    dates_raw, leagues_raw, homes_raw, aways_raw = [], [], [], []
    home_goals, away_goals = [], []
    probability_rows = []
    match_keys = set()

    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            raise ValueError(f"match {index} must be an object")
        date = match.get("date")
        league = match.get("comp")
        home, away = match.get("home"), match.get("away")
        hg, ag = match.get("hg"), match.get("ag")
        if not isinstance(date, str) or not date:
            raise ValueError(f"invalid date in match {index}")
        if league not in ("J1", "J2", "J3"):
            raise ValueError(f"invalid league in match {index}")
        if not isinstance(home, str) or not home or not isinstance(away, str) or not away or home == away:
            raise ValueError(f"invalid teams in match {index}")
        if type(hg) is not int or type(ag) is not int or hg < 0 or ag < 0:
            raise ValueError(f"invalid integer score in match {index}")
        key = (date, league, home, away)
        if key in match_keys:
            raise ValueError(f"duplicate match key: {key}")
        match_keys.add(key)

        by_k = match.get("byK")
        if not isinstance(by_k, dict):
            raise ValueError(f"missing byK object in match {index}")
        match_probabilities = []
        for k in K_VALUES:
            k_key = f"{k:.1f}"
            if k_key not in by_k or not isinstance(by_k[k_key], dict):
                raise ValueError(f"missing k={k_key} in match {index}")
            pair = []
            for side in ("home", "away"):
                if side not in by_k[k_key]:
                    raise ValueError(f"missing k={k_key} {side} in match {index}")
                p = np.asarray(by_k[k_key][side], dtype=float)
                if p.shape != (4,) or not np.all(np.isfinite(p)):
                    raise ValueError(f"invalid k={k_key} {side} probability vector")
                if np.any(p < 0.0) or np.any(p > 1.0) or abs(float(p.sum()) - 1.0) > 1e-9:
                    raise ValueError(f"probability vector outside simplex for k={k_key} {side}")
                pair.append(p)
            match_probabilities.append(pair)

        dates_raw.append(date)
        leagues_raw.append(league)
        homes_raw.append(home)
        aways_raw.append(away)
        home_goals.append(hg)
        away_goals.append(ag)
        probability_rows.append(match_probabilities)

    dates, date_index = np.unique(np.asarray(dates_raw), return_inverse=True)
    teams = np.unique(np.asarray(homes_raw + aways_raw))
    team_map = {name: i for i, name in enumerate(teams)}
    home_index = np.asarray([team_map[x] for x in homes_raw], dtype=np.int64)
    away_index = np.asarray([team_map[x] for x in aways_raw], dtype=np.int64)

    # Flatten in match order: home target then away target. Both share match weight.
    probabilities = np.asarray(probability_rows, dtype=float).reshape(len(matches), 2, len(K_VALUES), 4)
    probabilities = probabilities.reshape(2 * len(matches), len(K_VALUES), 4)
    actual = np.minimum(np.asarray([v for pair in zip(home_goals, away_goals) for v in pair]), 3)
    goals = np.asarray([v for pair in zip(home_goals, away_goals) for v in pair], dtype=np.int64)
    leagues = np.repeat(np.asarray(leagues_raw), 2)

    if enforce_expected_cohort:
        observed = {
            "matches": len(matches),
            "teamSideTargets": 2 * len(matches),
            "dates": len(dates),
            "teams": len(teams),
            "leagueTargets": {s: int(np.sum(leagues == s)) for s in ("J1", "J2", "J3")},
            "dateRange": [str(dates[0]), str(dates[-1])],
        }
        if observed != EXPECTED_COHORT:
            raise ValueError(f"unexpected frozen cohort: {observed!r}")

    one_hot = np.eye(4, dtype=float)[actual]
    brier = np.sum((probabilities - one_hot[:, None, :]) ** 2, axis=2)
    selected = probabilities[
        np.arange(2 * len(matches))[:, None],
        np.arange(len(K_VALUES))[None, :],
        actual[:, None],
    ]
    logloss = -np.log(np.clip(selected, 1e-12, 1.0))
    accuracy = (np.argmax(probabilities, axis=2) == actual[:, None]).astype(float)
    lam = -np.log(np.clip(probabilities[:, :, 0], 1e-300, 1.0))
    pearson = (goals[:, None] - lam) ** 2 / lam

    return {
        "matches": len(matches),
        "dates": dates,
        "teams": teams,
        "dateIndex": np.asarray(date_index, dtype=np.int64),
        "homeTeamIndex": home_index,
        "awayTeamIndex": away_index,
        "leagues": leagues,
        "actual": actual,
        "goals": goals,
        "probabilities": probabilities,
        "brier": brier,
        "logloss": logloss,
        "accuracy": accuracy,
        "pearson": pearson,
    }


def max_t_family(bootstrap: np.ndarray, point: np.ndarray, labels: list[dict]) -> dict:
    se = bootstrap.std(axis=0, ddof=1)
    if np.any(~np.isfinite(se)) or np.any(se <= 0.0):
        raise ValueError("invalid standard error in max-t family")
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
    return {
        "comparisonCount": len(labels),
        "studentizedMaxAbsTCriticalValue": critical,
        "intervals": intervals,
    }


def run_audit(data: dict, replicates: int = 50_000, seed: int = 20261110) -> dict:
    if replicates < 1_000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    match_count = data["matches"]
    target_count = 2 * match_count
    dates, teams = data["dates"], data["teams"]
    date_index = data["dateIndex"]
    home_index, away_index = data["homeTeamIndex"], data["awayTeamIndex"]
    leagues, actual = data["leagues"], data["actual"]
    probabilities = data["probabilities"]
    brier, logloss = data["brier"], data["logloss"]
    accuracy, pearson = data["accuracy"], data["pearson"]
    date_count, team_count = len(dates), len(teams)

    scope_indices = {
        scope: np.arange(target_count) if scope == "ALL" else np.flatnonzero(leagues == scope)
        for scope in SCOPES
    }
    point_brier = brier.mean(axis=0)
    point_logloss = logloss.mean(axis=0)
    point_accuracy = accuracy.mean(axis=0)

    calibration_labels, calibration_point, calibration_specs = [], [], []
    for scope in SCOPES:
        idx = scope_indices[scope]
        for k in PRIMARY_K:
            ki = K_VALUES.index(k)
            for category in range(4):
                pred = float(probabilities[idx, ki, category].mean())
                obs = float(np.mean(actual[idx] == category))
                calibration_labels.append({
                    "scope": scope, "k": int(k), "category": category, "n": int(len(idx))
                })
                calibration_point.append((pred - obs) * 100.0)
                calibration_specs.append((idx, ki, category))
    calibration_point = np.asarray(calibration_point, dtype=float)

    residual_labels, residual_point, residual_specs = [], [], []
    for scope in SCOPES:
        idx = scope_indices[scope]
        for k in PRIMARY_K:
            ki = K_VALUES.index(k)
            residual_labels.append({"scope": scope, "k": int(k), "n": int(len(idx))})
            residual_point.append(float(pearson[idx, ki].mean()))
            residual_specs.append((idx, ki))
    residual_point = np.asarray(residual_point, dtype=float)

    rng = np.random.Generator(np.random.PCG64(seed))
    boot_brier = np.empty((replicates, len(K_VALUES)), dtype=float)
    boot_logloss = np.empty_like(boot_brier)
    boot_accuracy = np.empty_like(boot_brier)
    boot_calibration = np.empty((replicates, len(calibration_labels)), dtype=float)
    boot_residual = np.empty((replicates, len(residual_labels)), dtype=float)
    redraws = 0
    batch_size = 500

    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        date_counts = rng.multinomial(
            date_count, np.full(date_count, 1.0 / date_count), size=size
        )
        team_counts = rng.multinomial(
            team_count, np.full(team_count, 1.0 / team_count), size=size
        )
        match_weights = pair_preserving_match_weights(
            date_counts, team_counts, date_index, home_index, away_index
        )
        weights = np.repeat(match_weights, 2, axis=1)

        # Redraw only replicates with zero total mass in any required league scope.
        bad = np.zeros(size, dtype=bool)
        for idx in scope_indices.values():
            bad |= weights[:, idx].sum(axis=1) == 0
        while bad.any():
            count = int(bad.sum())
            redraws += count
            date_counts[bad] = rng.multinomial(
                date_count, np.full(date_count, 1.0 / date_count), size=count
            )
            team_counts[bad] = rng.multinomial(
                team_count, np.full(team_count, 1.0 / team_count), size=count
            )
            match_weights[bad] = pair_preserving_match_weights(
                date_counts[bad], team_counts[bad], date_index, home_index, away_index
            )
            weights[bad] = np.repeat(match_weights[bad], 2, axis=1)
            bad = np.zeros(size, dtype=bool)
            for idx in scope_indices.values():
                bad |= weights[:, idx].sum(axis=1) == 0

        denominator = weights.sum(axis=1)
        boot_brier[start:start + size] = (weights @ brier) / denominator[:, None]
        boot_logloss[start:start + size] = (weights @ logloss) / denominator[:, None]
        boot_accuracy[start:start + size] = (weights @ accuracy) / denominator[:, None]

        for column, (idx, ki, category) in enumerate(calibration_specs):
            scoped = weights[:, idx]
            scoped_denominator = scoped.sum(axis=1)
            pred = scoped @ probabilities[idx, ki, category]
            observed = scoped @ (actual[idx] == category).astype(float)
            boot_calibration[start:start + size, column] = (
                (pred - observed) / scoped_denominator * 100.0
            )
        for column, (idx, ki) in enumerate(residual_specs):
            scoped = weights[:, idx]
            scoped_denominator = scoped.sum(axis=1)
            boot_residual[start:start + size, column] = (
                (scoped @ pearson[idx, ki]) / scoped_denominator
            )

    best_frequency = {}
    for metric, values, choose in (
        ("brier", boot_brier, np.argmin),
        ("logloss", boot_logloss, np.argmin),
        ("accuracy", boot_accuracy, np.argmax),
    ):
        winners = choose(values, axis=1)
        counts = np.bincount(winners, minlength=len(K_VALUES))
        best_frequency[metric] = {
            f"{k:g}": float(counts[i] / replicates) for i, k in enumerate(K_VALUES)
        }

    # Selection-adjusted family: 15 non-reference candidates x 2 primary metrics.
    reference_index = K_VALUES.index(REFERENCE_K)
    contrasts, contrast_meta = [], []
    for metric, values, point in (
        ("brier", boot_brier, point_brier),
        ("logloss", boot_logloss, point_logloss),
    ):
        for i, k in enumerate(K_VALUES):
            if i == reference_index:
                continue
            delta = values[:, i] - values[:, reference_index]
            observed = float(point[i] - point[reference_index])
            se = float(delta.std(ddof=1))
            if not np.isfinite(se) or se <= 0.0:
                raise ValueError(f"invalid SE for {metric}, k={k}")
            contrasts.append((delta - observed) / se)
            contrast_meta.append((metric, k, observed, se, delta))
    selection_critical = float(np.quantile(
        np.max(np.abs(np.column_stack(contrasts)), axis=1), 0.95, method="linear"
    ))
    selection_intervals = {}
    for metric, k, observed, se, delta in contrast_meta:
        selection_intervals.setdefault(f"{k:g}", {})[metric] = {
            "differenceKMinusK10": observed,
            "bootstrapSE": se,
            "nominalPercentile95CI": [
                float(np.quantile(delta, 0.025, method="linear")),
                float(np.quantile(delta, 0.975, method="linear")),
            ],
            "simultaneous95Band": [
                observed - selection_critical * se,
                observed + selection_critical * se,
            ],
        }

    calibration_family = max_t_family(boot_calibration, calibration_point, calibration_labels)
    residual_family = max_t_family(boot_residual, residual_point, residual_labels)
    return {
        "experiment": "GOAL3 pair-preserving date/team pigeonhole bootstrap sensitivity",
        "scope": "research-only retrospective fixed 2026-only outer cohort; sensitivity only",
        "cohort": {
            "matches": match_count,
            "teamSideTargets": target_count,
            "dates": date_count,
            "teams": team_count,
            "dateRange": [str(dates[0]), str(dates[-1])],
            "leagueTargets": {s: int(len(scope_indices[s])) for s in SCOPES if s != "ALL"},
        },
        "method": {
            "resamplingUnit": "match pairs; two linked team-side targets share one match weight",
            "dateClusters": date_count,
            "teamClusters": team_count,
            "dateResampling": "independent multinomial sample of D dates with replacement",
            "teamResampling": "independent multinomial sample of T team identities with replacement",
            "matchWeight": "date multiplicity * home-team multiplicity * away-team multiplicity",
            "pairing": "same match weight applied to both home and away team-side targets",
            "weighting": "weighted target means within each scope",
            "rng": "NumPy PCG64",
            "seed": seed,
            "replicates": replicates,
            "quantile": "linear percentile",
            "emptyScopeReplicatesRedrawn": redraws,
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
            },
        },
        "fourCategoryCalibration": calibration_family,
        "poissonResidualDispersion": {
            "definition": "lambda=-log(P(goal=0)); mean((observedGoals-lambda)^2/lambda)",
            **residual_family,
            "intervalsContainingOne": sum(
                x["simultaneous95Band"][0] <= 1.0 <= x["simultaneous95Band"][1]
                for x in residual_family["intervals"]
            ),
        },
        "limitations": [
            "This is a sensitivity analysis, not the predeclared date-only primary bootstrap.",
            "Date and team cluster labels are sampled independently; product weights are a pigeonhole approximation, not a full generative model.",
            "Pairing is preserved within each match, but only 17 dates and 60 teams are available.",
            "Only the fixed retrospective cohort is analyzed; historical first-known availability is unproven.",
            "Conditional Pearson residual dispersion is descriptive and is not toto13 raw score VMR.",
            "No model fit, forecast generation, production change, or market-weight change is performed.",
        ],
        "decision": "Research-only sensitivity. Do not infer k=6 superiority or production adoption from this result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()

    document = json.loads(args.input.read_text(encoding="utf-8"))
    arrays = prepare_arrays(document)
    result = run_audit(arrays, args.replicates, args.seed)
    result["provenance"] = {
        "inputFile": args.input.name,
        "inputSha256": sha256_file(args.input),
        "scriptSha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "reproductionCommand": (
            "python verification/goal3_independent/pair_preserving_date_team_bootstrap_audit.py "
            f"--input {args.input} --out {args.out} --replicates {args.replicates} --seed {args.seed}"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "cohort": result["cohort"],
        "k6MinusK10": result["selectionStability"]["selectionAdjusted"]["intervals"]["6"],
        "bestK6": {
            metric: result["selectionStability"]["bestCandidateFrequency"][metric]["6"]
            for metric in ("brier", "logloss", "accuracy")
        },
        "calibrationIntervalsContainingZero": sum(
            x["simultaneous95Band"][0] <= 0.0 <= x["simultaneous95Band"][1]
            for x in result["fourCategoryCalibration"]["intervals"]
        ),
        "residualIntervalsContainingOne": result["poissonResidualDispersion"]["intervalsContainingOne"],
        "redraws": result["method"]["emptyScopeReplicatesRedrawn"],
        "inputSha256": result["provenance"]["inputSha256"],
        "scriptSha256": result["provenance"]["scriptSha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
