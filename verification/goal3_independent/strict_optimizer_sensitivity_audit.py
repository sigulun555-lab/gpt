#!/usr/bin/env python3
"""Full 16-k strict-optimizer sensitivity audit on the frozen 17-date cohort.

Research-only. The pinned default nested_validation implementation and its
existing all-k artifact are not modified. This audit refits the identical
penalized-Poisson objective with the fixed strict L-BFGS-B preset and reruns
the selection-adjusted paired date/team bootstrap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from nested_validation import bin_goals, canonical, load_data, predict_shrinkage
from optimizer_gradient import STRICT_LBFGSB_OPTIONS, fit_shrinkage_strict
import stratified_pair_preserving_date_team_bootstrap_audit as stratified_audit

ROOT = Path(__file__).resolve().parents[2]
K_VALUES = [float(x) for x in stratified_audit.K_VALUES]
EXPECTED_COHORT = {"matches": 236, "teamSideTargets": 472, "dates": 17}
PROBABILITY_SUM_TOL = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    header = ("blob " + str(len(raw))).encode("ascii") + b"\0"
    import hashlib as _hashlib
    return _hashlib.sha1(header + raw).hexdigest()


def execution_environment_fingerprint(allk: dict) -> dict:
    """Return a canonical fingerprint for the runtime used to build the all-k input.

    The strict sensitivity fit runs immediately after all-k reconstruction in the
    same CI job. Embedding this exact runtime record makes CPU/BLAS dispatch
    differences auditable from the strict artifact itself.
    """
    environment = allk.get("executionEnvironment")
    if not isinstance(environment, dict) or not environment:
        raise ValueError("all-k input is missing executionEnvironment CPU/BLAS provenance")
    canonical = json.dumps(
        environment, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "details": environment,
    }


def load_allk(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    stratified_audit.validate_source_provenance(document)
    cohort = document.get("cohort", {})
    for key, expected in EXPECTED_COHORT.items():
        if cohort.get(key) != expected:
            raise ValueError(f"unexpected frozen cohort {key}: {cohort.get(key)!r} != {expected}")
    if document.get("kCandidates") != K_VALUES:
        raise ValueError("all-k candidate grid differs from the pinned 16-value grid")
    rows = document.get("allKPredictions")
    if not isinstance(rows, list) or len(rows) != EXPECTED_COHORT["matches"]:
        raise ValueError("all-k prediction rows do not match the frozen 236-match cohort")
    return document


def reconstruct_strict_predictions(
    allk: dict, h2026: list[dict], aliases: dict, replicates: int, seed: int
) -> tuple[dict, dict]:
    rows = allk["allKPredictions"]
    dates = sorted({str(row["date"]) for row in rows})
    date_index = np.asarray([dates.index(str(row["date"])) for row in rows], dtype=np.int64)
    teams = sorted({canonical(row[side], aliases) for row in rows for side in ("home", "away")})
    team_index = {team: i for i, team in enumerate(teams)}
    home_team_index = np.asarray([team_index[canonical(row["home"], aliases)] for row in rows], dtype=np.int64)
    away_team_index = np.asarray([team_index[canonical(row["away"], aliases)] for row in rows], dtype=np.int64)
    match_count = len(rows)
    target_count = 2 * match_count
    target_actual = np.empty(target_count, dtype=np.int64)
    target_goals = np.empty(target_count, dtype=np.float64)
    leagues = np.empty(target_count, dtype="U3")
    default_probabilities = np.empty((target_count, len(K_VALUES), 4), dtype=np.float64)
    strict_probabilities = np.empty_like(default_probabilities)
    fit_diagnostics = []

    for match_index, row in enumerate(rows):
        date = str(row["date"])
        comp = str(row["comp"])
        if comp not in ("J1", "J2", "J3"):
            raise ValueError(f"unknown league code in frozen match: {comp}")
        home = canonical(row["home"], {})
        away = canonical(row["away"], {})
        hg, ag = row.get("hg"), row.get("ag")
        if type(hg) is not int or type(ag) is not int or hg < 0 or ag < 0:
            raise ValueError(f"invalid frozen actual score at match {match_index}")
        target_actual[2 * match_index] = bin_goals(hg)
        target_actual[2 * match_index + 1] = bin_goals(ag)
        target_goals[2 * match_index] = hg
        target_goals[2 * match_index + 1] = ag
        leagues[2 * match_index:2 * match_index + 2] = comp
        by_k = row.get("byK")
        if not isinstance(by_k, dict):
            raise ValueError(f"missing frozen byK predictions at match {match_index}")
        for k_index, k in enumerate(K_VALUES):
            saved = by_k.get(str(float(k)))
            if not isinstance(saved, dict):
                raise ValueError(f"missing default k={k:g} predictions")
            default_probabilities[2 * match_index, k_index] = saved["home"]
            default_probabilities[2 * match_index + 1, k_index] = saved["away"]

    for date in dates:
        prior = [match for match in h2026 if match["date"] < date]
        if not prior:
            raise ValueError(f"empty expanding training history before {date}")
        if any(match["date"] >= date for match in prior):
            raise ValueError(f"same-day/future row leaked into training for {date}")
        match_indices = [i for i, row in enumerate(rows) if str(row["date"]) == date]
        for k_index, k in enumerate(K_VALUES):
            params, fit_team_index, info = fit_shrinkage_strict(prior, k)
            if not info["success"] or info["status"] != 0:
                raise RuntimeError(f"strict fit failed for date={date}, k={k:g}: {info}")
            max_gradient = float(info["maxAbsGradient"])
            if not np.isfinite(max_gradient) or max_gradient < 0:
                raise RuntimeError(f"invalid strict gradient for date={date}, k={k:g}")
            for match_index in match_indices:
                row = rows[match_index]
                home = canonical(row["home"], aliases)
                away = canonical(row["away"], aliases)
                ph, pa = predict_shrinkage(params, fit_team_index, home, away)
                strict_probabilities[2 * match_index, k_index] = ph
                strict_probabilities[2 * match_index + 1, k_index] = pa
            fit_diagnostics.append({
                "date": date,
                "k": k,
                "trainingRows": len(prior),
                "teamCount": len(fit_team_index),
                "fitSuccess": bool(info["success"]),
                "optimizerStatus": int(info["status"]),
                "optimizerName": info["optimizerName"],
                "optimizerOptions": info["optimizerOptions"],
                "optimizerIterations": int(info["nit"]),
                "maxAbsGradient": max_gradient,
                "gradientL2": float(info["gradientL2"]),
                "gradientDimension": int(info["gradientDimension"]),
            })

    if not np.isfinite(strict_probabilities).all():
        raise ValueError("strict probability tensor contains non-finite values")
    sums = strict_probabilities.sum(axis=2)
    max_simplex_error = float(np.max(np.abs(sums - 1.0)))
    if max_simplex_error > PROBABILITY_SUM_TOL:
        raise ValueError(f"strict probability simplex error exceeds tolerance: {max_simplex_error}")

    default_sums = default_probabilities.sum(axis=2)
    if float(np.max(np.abs(default_sums - 1.0))) > PROBABILITY_SUM_TOL:
        raise ValueError("frozen default all-k probabilities fail simplex validation")

    strict_data = build_bootstrap_data(
        rows, dates, date_index, leagues, teams, home_team_index, away_team_index,
        target_actual, target_goals, strict_probabilities
    )
    default_data = build_bootstrap_data(
        rows, dates, date_index, leagues, teams, home_team_index, away_team_index,
        target_actual, target_goals, default_probabilities
    )
    strict_report = stratified_audit.run_audit(
        strict_data, replicates=replicates, seed=seed
    )
    default_point_metrics = point_metrics(default_data)
    strict_point_metrics = point_metrics(strict_data)

    # The pinned all-k artifact is the independent default baseline. Verify
    # its point metrics before interpreting the strict-optimizer sensitivity.
    baseline_differences = {}
    for k in K_VALUES:
        key = str(float(k))
        metric_key = f"{k:g}"
        saved = allk["metricsByK"][key]["overall"]
        computed = default_point_metrics[metric_key]
        baseline_differences[key] = {
            metric: float(computed[metric] - saved[metric])
            for metric in ("brier", "logloss", "accuracy")
        }
    max_baseline_metric_diff = max(
        abs(value)
        for per_k in baseline_differences.values()
        for value in per_k.values()
    )
    if max_baseline_metric_diff > 2e-12:
        raise ValueError(f"default baseline does not reproduce pinned all-k metrics: {max_baseline_metric_diff}")

    max_probability_shift = float(np.max(np.abs(strict_probabilities - default_probabilities)))
    max_abs_gradient = max(x["maxAbsGradient"] for x in fit_diagnostics)
    within_tolerance = sum(x["maxAbsGradient"] <= 1e-6 for x in fit_diagnostics)
    strict_report["strictOptimizerSensitivity"] = {
        "strictOptimizerOptions": dict(STRICT_LBFGSB_OPTIONS),
        "fitCount": len(fit_diagnostics),
        "fitSuccessCount": sum(x["fitSuccess"] for x in fit_diagnostics),
        "fitsWithinMaxAbsGradient1eMinus6": within_tolerance,
        "fitsAboveMaxAbsGradient1eMinus6": len(fit_diagnostics) - within_tolerance,
        "maximumObservedMaxAbsGradient": max_abs_gradient,
        "fitDiagnostics": fit_diagnostics,
        "defaultBaselinePointMetricsByK": default_point_metrics,
        "strictPointMetricsByK": strict_point_metrics,
        "defaultBaselineMetricDifferencesFromPinnedAllK": baseline_differences,
        "maximumDefaultToStrictProbabilityShift": max_probability_shift,
        "maximumStrictSimplexError": max_simplex_error,
        "k6MinusK10": {
            "brier": float(strict_point_metrics["6"]["brier"] - strict_point_metrics["10"]["brier"]),
            "logloss": float(strict_point_metrics["6"]["logloss"] - strict_point_metrics["10"]["logloss"]),
            "accuracy": float(strict_point_metrics["6"]["accuracy"] - strict_point_metrics["10"]["accuracy"]),
        },
        "selectionAdjustedK6MinusK10": strict_report["selectionStability"]["selectionAdjusted"]["intervals"]["6"],
        "bestCandidateFrequency": strict_report["selectionStability"]["bestCandidateFrequency"],
        "interpretation": (
            "Retrospective optimizer sensitivity only. The strict optimizer does not replace "
            "the pinned default implementation or imply production adoption."
        ),
    }
    return strict_report, {
        "dates": dates,
        "matchCount": match_count,
        "teamCount": len(teams),
        "targetCount": target_count,
        "defaultProbabilities": default_probabilities,
        "strictProbabilities": strict_probabilities,
        "fitDiagnostics": fit_diagnostics,
        "maxSimplexError": max_simplex_error,
    }


def build_bootstrap_data(rows, dates, date_index, leagues, teams, home_idx, away_idx, actual, goals, probabilities):
    target_count = len(actual)
    onehot = np.eye(4, dtype=float)[actual]
    brier = np.square(probabilities - onehot[:, None, :]).sum(axis=2)
    chosen = probabilities[np.arange(target_count)[:, None], np.arange(len(K_VALUES))[None, :], actual[:, None]]
    logloss = -np.log(np.maximum(chosen, 1e-12))
    accuracy = (probabilities.argmax(axis=2) == actual[:, None]).astype(float)
    lambdas = -np.log(np.maximum(probabilities[:, :, 0], 1e-12))
    pearson = np.square(goals[:, None] - lambdas) / np.maximum(lambdas, 1e-12)
    return {
        "matches": len(rows),
        "dates": dates,
        "dateIndex": date_index,
        "leagues": leagues,
        "actual": actual,
        "probabilities": probabilities,
        "brier": brier,
        "logloss": logloss,
        "accuracy": accuracy,
        "pearson": pearson,
        "teams": teams,
        "homeTeamIndex": home_idx,
        "awayTeamIndex": away_idx,
    }


def point_metrics(data: dict) -> dict:
    return {
        f"{k:g}": {
            "brier": float(data["brier"][:, i].mean()),
            "logloss": float(data["logloss"][:, i].mean()),
            "accuracy": float(data["accuracy"][:, i].mean()),
        }
        for i, k in enumerate(K_VALUES)
    }


def run(args: argparse.Namespace) -> dict:
    allk_path = Path(args.allk_input)
    history_path = Path(args.history2026)
    aliases_path = Path(args.aliases)
    csv2025_path = Path(args.csv2025)
    allk = load_allk(allk_path)
    runtime_fingerprint = execution_environment_fingerprint(allk)
    _, h2026 = load_data(history_path, aliases_path, csv2025_path)
    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))["aliases"]
    if len(h2026) != int(allk["cohort"]["history2026Rows"]):
        raise ValueError(
            f"history row count differs from frozen all-k cohort: {len(h2026)} "
            f"!= {allk['cohort']['history2026Rows']}"
        )
    strict_report, details = reconstruct_strict_predictions(
        allk, h2026, aliases, args.bootstrap_replicates, args.bootstrap_seed
    )
    script_path = Path(__file__).resolve()
    model_path = Path(__file__).with_name("nested_validation.py")
    optimizer_path = Path(__file__).with_name("optimizer_gradient.py")
    strict_report["executionEnvironment"] = {
        "source": "allkInput.executionEnvironment (same CI job, immediately before strict refit)",
        "allkInputSha256": sha256_file(allk_path),
        **runtime_fingerprint,
    }
    strict_report["provenance"] = {
        "allkInput": {"path": str(allk_path), "sizeBytes": allk_path.stat().st_size, "sha256": sha256_file(allk_path)},
        "history2026": {"path": str(history_path), "sizeBytes": history_path.stat().st_size, "sha256": sha256_file(history_path)},
        "teamAliases": {"path": str(aliases_path), "sizeBytes": aliases_path.stat().st_size, "sha256": sha256_file(aliases_path)},
        "history2025Csv": {"path": str(csv2025_path), "sizeBytes": csv2025_path.stat().st_size, "sha256": sha256_file(csv2025_path)},
        "nestedValidationPy": {"path": str(model_path), "sha256": sha256_file(model_path), "gitBlobSha": git_blob_sha(model_path)},
        "optimizerGradientPy": {"path": str(optimizer_path), "sha256": sha256_file(optimizer_path)},
        "strictOptimizerSensitivityPy": {"path": str(script_path), "sha256": sha256_file(script_path), "gitBlobSha": git_blob_sha(script_path)},
        "execution": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "githubSha": __import__("os").environ.get("GITHUB_SHA"),
            "githubRunId": __import__("os").environ.get("GITHUB_RUN_ID"),
            "githubRunAttempt": __import__("os").environ.get("GITHUB_RUN_ATTEMPT"),
        },
    }
    strict_report["cohort"]["history2026Rows"] = len(h2026)
    strict_report["cohort"]["history2026DistinctDates"] = len({x["date"] for x in h2026})
    strict_report["experiment"] = "GOAL3 strict-L-BFGS-B optimizer sensitivity on frozen all-k outer cohort"
    strict_report["scope"] = "research-only retrospective sensitivity; not prospective adoption"
    return strict_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allk-input", default="artifacts/goal3_outer_allk_reproduction.json")
    parser.add_argument("--history2026", default="data/history.json")
    parser.add_argument("--aliases", default="data/team_aliases.json")
    parser.add_argument("--csv2025", default="verification/goal3_independent/2025.csv")
    parser.add_argument("--out", default="artifacts/goal3_strict_optimizer_sensitivity.json")
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261111)
    args = parser.parse_args()
    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "cohort": report["cohort"],
        "strictOptimizerSensitivity": report["strictOptimizerSensitivity"],
        "selectionAdjustedK6MinusK10": report["strictOptimizerSensitivity"]["selectionAdjustedK6MinusK10"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
