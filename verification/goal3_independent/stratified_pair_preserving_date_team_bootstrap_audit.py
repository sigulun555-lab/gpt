#!/usr/bin/env python3
"""Stratified-team, match-pair-preserving date/team bootstrap sensitivity.

Date clusters are sampled globally to preserve common calendar shocks. Team
clusters are sampled independently within J1, J2, and J3, preserving the
observed 20-team composition of each league. Both team-side targets from a
match receive the same product weight. Research-only; no fitting or forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

import pair_preserving_date_team_bootstrap_audit as pair_audit

K_VALUES = pair_audit.K_VALUES
REFERENCE_K = pair_audit.REFERENCE_K
PRIMARY_K = pair_audit.PRIMARY_K
SCOPES = pair_audit.SCOPES
LEAGUES = ("J1", "J2", "J3")
EXPECTED_SOURCE_PROVENANCE = {
    "frozenCsvSha256": "5cc8960c10f3554d44ccc0d5388d0ceeb4c8a39dec4cc8b28bd713f0e3755678",
    "history2026Sha256": "efca00ea491fcfea6a1267af403cb5ba80ae540fca97a076def25815fc318201",
    "aliasesSha256": "92e6377068df8dd63935e0c003adfd7b355e7adac23f7aace4f947be4f5f027e",
    "history2025CsvSha256": "6175674bc28d4311b219eed5ea2aa5e478fe0f17b8e6b8653f37718057da4288",
    "nestedValidationPySha256": "ffca9d2b4b27564532c8964464af9ec4f4cb68b0b871266fb007afad236e670c",
    "nestedValidationPyGitBlobSha": "2b364ae31673b1f7471958084c416ca1ad8901f8",
    "pr19NestedValidationPyGitBlobSha": "5f73df9e6fdb43d5cf333c04036f87ae16c20d1f",
}
EXPECTED_TEAM_COUNTS = {"J1": 20, "J2": 20, "J3": 20}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_provenance(document: dict) -> None:
    provenance = document.get("provenance") if isinstance(document, dict) else None
    if not isinstance(provenance, dict):
        raise ValueError("all-k input is missing source provenance")
    mismatches = {
        key: {"expected": expected, "observed": provenance.get(key)}
        for key, expected in EXPECTED_SOURCE_PROVENANCE.items()
        if provenance.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"all-k source provenance mismatch: {mismatches}")


def build_team_strata(data: dict, expected_team_counts: dict[str, int] | None = None) -> dict:
    """Map each match's home/away teams to a league-specific team stratum."""
    expected = expected_team_counts or EXPECTED_TEAM_COUNTS
    match_leagues = np.asarray(data["leagues"])[::2]
    home_global = np.asarray(data["homeTeamIndex"], dtype=np.int64)
    away_global = np.asarray(data["awayTeamIndex"], dtype=np.int64)
    if len(match_leagues) != data["matches"]:
        raise ValueError("match league vector does not match match count")

    home_local = np.full(data["matches"], -1, dtype=np.int64)
    away_local = np.full(data["matches"], -1, dtype=np.int64)
    match_league_index = np.full(data["matches"], -1, dtype=np.int64)
    team_ids_by_league = []
    assigned_global_ids: list[int] = []

    for league_index, league in enumerate(LEAGUES):
        match_indices = np.flatnonzero(match_leagues == league)
        if len(match_indices) == 0:
            raise ValueError(f"empty league stratum: {league}")
        team_ids = np.unique(np.concatenate([
            home_global[match_indices], away_global[match_indices]
        ]))
        if len(team_ids) != int(expected[league]):
            raise ValueError(
                f"unexpected team count for {league}: {len(team_ids)} != {expected[league]}"
            )
        overlap = set(assigned_global_ids).intersection(int(x) for x in team_ids)
        if overlap:
            names = [str(data["teams"][i]) for i in sorted(overlap)]
            raise ValueError(f"team identity crosses league strata: {names}")
        assigned_global_ids.extend(int(x) for x in team_ids)
        local_map = {int(team_id): local for local, team_id in enumerate(team_ids)}
        home_local[match_indices] = [local_map[int(x)] for x in home_global[match_indices]]
        away_local[match_indices] = [local_map[int(x)] for x in away_global[match_indices]]
        match_league_index[match_indices] = league_index
        team_ids_by_league.append(team_ids)

    if np.any(home_local < 0) or np.any(away_local < 0) or np.any(match_league_index < 0):
        raise ValueError("one or more matches were not assigned to a league stratum")
    if set(assigned_global_ids) != set(range(len(data["teams"]))):
        raise ValueError("league strata do not partition the complete team set")

    return {
        "teamIdsByLeague": team_ids_by_league,
        "homeLocal": home_local,
        "awayLocal": away_local,
        "matchLeagueIndex": match_league_index,
    }


def stratified_pair_preserving_match_weights(
    date_counts: np.ndarray,
    team_counts_by_league: list[np.ndarray],
    date_index: np.ndarray,
    home_local: np.ndarray,
    away_local: np.ndarray,
    match_league_index: np.ndarray,
) -> np.ndarray:
    """One match weight; home and away targets later receive this same weight."""
    match_count = len(date_index)
    batch_size = date_counts.shape[0]
    weights = np.empty((batch_size, match_count), dtype=float)
    for league_index in range(len(LEAGUES)):
        match_indices = np.flatnonzero(match_league_index == league_index)
        counts = team_counts_by_league[league_index]
        weights[:, match_indices] = (
            date_counts[:, date_index[match_indices]]
            * counts[:, home_local[match_indices]]
            * counts[:, away_local[match_indices]]
        )
    return weights


def _summary(values: np.ndarray) -> dict:
    q = np.quantile(values, [0.05, 0.5, 0.95], method="linear")
    return {
        "p05": float(q[0]),
        "median": float(q[1]),
        "p95": float(q[2]),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def paired_residual_max_t_family(bootstrap: np.ndarray, point: np.ndarray, labels: list[dict]) -> dict:
    """Simultaneous paired k6-k10 residual-dispersion intervals.

    Zero-variance contrasts are retained as exact point intervals rather than
    causing the entire audit to fail (e.g. identical synthetic k arms).
    """
    bootstrap = np.asarray(bootstrap, dtype=float)
    point = np.asarray(point, dtype=float)
    if bootstrap.ndim != 2 or bootstrap.shape[1] != len(labels):
        raise ValueError("paired residual bootstrap shape does not match labels")
    if point.shape != (len(labels),) or not np.isfinite(bootstrap).all() or not np.isfinite(point).all():
        raise ValueError("paired residual inputs must be finite and aligned")
    se = bootstrap.std(axis=0, ddof=1)
    if np.any(~np.isfinite(se)) or np.any(se < 0.0):
        raise ValueError("invalid standard error in paired residual family")
    standardized = np.zeros_like(bootstrap)
    nonzero = se > 0.0
    standardized[:, nonzero] = (
        bootstrap[:, nonzero] - point[None, nonzero]
    ) / se[None, nonzero]
    critical = float(np.quantile(
        np.max(np.abs(standardized), axis=1), 0.95, method="linear"
    ))
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
        "intervalsContainingZero": sum(
            x["simultaneous95Band"][0] <= 0.0 <= x["simultaneous95Band"][1]
            for x in intervals
        ),
    }


def run_audit(data: dict, replicates: int = 50_000, seed: int = 20261111) -> dict:
    if replicates < 1_000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    strata = build_team_strata(data)
    match_count = data["matches"]
    target_count = 2 * match_count
    dates = data["dates"]
    date_index = data["dateIndex"]
    leagues = data["leagues"]
    actual = data["actual"]
    probabilities = data["probabilities"]
    brier = data["brier"]
    logloss = data["logloss"]
    accuracy = data["accuracy"]
    pearson = data["pearson"]
    date_count = len(dates)
    team_ids_by_league = strata["teamIdsByLeague"]
    home_local = strata["homeLocal"]
    away_local = strata["awayLocal"]
    match_league_index = strata["matchLeagueIndex"]
    team_counts_per_league = [len(x) for x in team_ids_by_league]

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
                observed = float(np.mean(actual[idx] == category))
                calibration_labels.append({
                    "scope": scope, "k": int(k), "category": category, "n": int(len(idx))
                })
                calibration_point.append((pred - observed) * 100.0)
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
    match_ess = np.empty(replicates, dtype=float)
    date_ess = np.empty(replicates, dtype=float)
    team_ess = [np.empty(replicates, dtype=float) for _ in LEAGUES]
    positive_match_count = np.empty(replicates, dtype=np.int64)
    redraws = 0
    batch_size = 500

    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        date_counts = rng.multinomial(
            date_count, np.full(date_count, 1.0 / date_count), size=size
        )
        team_counts_by_league = [
            rng.multinomial(n, np.full(n, 1.0 / n), size=size)
            for n in team_counts_per_league
        ]
        match_weights = stratified_pair_preserving_match_weights(
            date_counts, team_counts_by_league, date_index,
            home_local, away_local, match_league_index
        )
        weights = np.repeat(match_weights, 2, axis=1)

        bad = np.zeros(size, dtype=bool)
        for idx in scope_indices.values():
            bad |= weights[:, idx].sum(axis=1) == 0
        while bad.any():
            count = int(bad.sum())
            redraws += count
            date_counts[bad] = rng.multinomial(
                date_count, np.full(date_count, 1.0 / date_count), size=count
            )
            for league_index, n in enumerate(team_counts_per_league):
                team_counts_by_league[league_index][bad] = rng.multinomial(
                    n, np.full(n, 1.0 / n), size=count
                )
            match_weights[bad] = stratified_pair_preserving_match_weights(
                date_counts[bad],
                [x[bad] for x in team_counts_by_league],
                date_index, home_local, away_local, match_league_index
            )
            weights[bad] = np.repeat(match_weights[bad], 2, axis=1)
            bad = np.zeros(size, dtype=bool)
            for idx in scope_indices.values():
                bad |= weights[:, idx].sum(axis=1) == 0

        denominator = weights.sum(axis=1)
        sl = slice(start, start + size)
        boot_brier[sl] = (weights @ brier) / denominator[:, None]
        boot_logloss[sl] = (weights @ logloss) / denominator[:, None]
        boot_accuracy[sl] = (weights @ accuracy) / denominator[:, None]

        for column, (idx, ki, category) in enumerate(calibration_specs):
            scoped = weights[:, idx]
            scoped_denominator = scoped.sum(axis=1)
            predicted = scoped @ probabilities[idx, ki, category]
            observed = scoped @ (actual[idx] == category).astype(float)
            boot_calibration[sl, column] = (
                (predicted - observed) / scoped_denominator * 100.0
            )
        for column, (idx, ki) in enumerate(residual_specs):
            scoped = weights[:, idx]
            scoped_denominator = scoped.sum(axis=1)
            boot_residual[sl, column] = (
                (scoped @ pearson[idx, ki]) / scoped_denominator
            )

        match_ess[sl] = (match_weights.sum(axis=1) ** 2) / np.sum(match_weights ** 2, axis=1)
        date_ess[sl] = (date_counts.sum(axis=1) ** 2) / np.sum(date_counts ** 2, axis=1)
        positive_match_count[sl] = np.sum(match_weights > 0, axis=1)
        for league_index, counts in enumerate(team_counts_by_league):
            team_ess[league_index][sl] = (
                counts.sum(axis=1) ** 2
            ) / np.sum(counts ** 2, axis=1)

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

    calibration_family = pair_audit.max_t_family(
        boot_calibration, calibration_point, calibration_labels
    )
    residual_family = pair_audit.max_t_family(
        boot_residual, residual_point, residual_labels
    )
    residual_column = {
        (label["scope"], int(label["k"])): index
        for index, label in enumerate(residual_labels)
    }
    paired_residual_labels = [
        {
            "scope": scope,
            "contrast": "k6_minus_k10",
            "n": int(len(scope_indices[scope])),
        }
        for scope in SCOPES
    ]
    paired_residual_point = np.asarray([
        residual_point[residual_column[(scope, 6)]]
        - residual_point[residual_column[(scope, 10)]]
        for scope in SCOPES
    ], dtype=float)
    paired_residual_boot = np.column_stack([
        boot_residual[:, residual_column[(scope, 6)]]
        - boot_residual[:, residual_column[(scope, 10)]]
        for scope in SCOPES
    ])
    paired_residual_family = paired_residual_max_t_family(
        paired_residual_boot, paired_residual_point, paired_residual_labels
    )

    return {
        "experiment": "GOAL3 stratified-team match-pair-preserving date/team bootstrap sensitivity",
        "scope": "research-only retrospective fixed 2026-only outer cohort; sensitivity only",
        "cohort": {
            "matches": match_count,
            "teamSideTargets": target_count,
            "dates": date_count,
            "dateRange": [str(dates[0]), str(dates[-1])],
            "teamsByLeague": {
                league: int(team_counts_per_league[i]) for i, league in enumerate(LEAGUES)
            },
            "leagueTargets": {
                league: int(len(scope_indices[league])) for league in LEAGUES
            },
        },
        "method": {
            "resamplingUnit": "match pairs; two linked team-side targets share one match weight",
            "dateResampling": "global multinomial sample of D target dates with replacement",
            "teamResampling": "independent multinomial sample of n_l team identities within each league l, with replacement",
            "matchWeight": "date multiplicity * home-team multiplicity within league * away-team multiplicity within league",
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
            "pairedK6MinusK10": paired_residual_family,
        },
        "effectiveSampleSize": {
            "match": _summary(match_ess),
            "positiveMatchCount": _summary(positive_match_count.astype(float)),
            "dateCluster": _summary(date_ess),
            "teamClusterByLeague": {
                league: _summary(team_ess[i]) for i, league in enumerate(LEAGUES)
            },
            "note": "ESS describes bootstrap weight concentration, not additional independent observations.",
        },
        "limitations": [
            "This is a sensitivity analysis, not the predeclared date-only primary bootstrap.",
            "Date clusters are shared globally; team clusters are resampled within league strata to preserve the observed 20/20/20 team composition.",
            "Pairing is preserved within each match; only 17 date clusters and 20 teams per league are available.",
            "Product weights are a pigeonhole approximation, not a full generative model.",
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
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    input_sha = sha256_file(args.input)
    document = json.loads(args.input.read_text(encoding="utf-8"))
    validate_source_provenance(document)
    data = pair_audit.prepare_arrays(document)
    result = run_audit(data, args.replicates, args.seed)
    result["provenance"] = {
        "inputFile": args.input.name,
        "inputSha256": input_sha,
        "scriptSha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "reproductionCommand": (
            "python verification/goal3_independent/stratified_pair_preserving_date_team_bootstrap_audit.py "
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
        "ess": result["effectiveSampleSize"],
        "redraws": result["method"]["emptyScopeReplicatesRedrawn"],
        "inputSha256": result["provenance"]["inputSha256"],
        "scriptSha256": result["provenance"]["scriptSha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
