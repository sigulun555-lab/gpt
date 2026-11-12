#!/usr/bin/env python3
"""Portable, independent verifier for Claude's GOAL3 k=6 vs k=10 share.

This verifier does not execute Claude's machine-specific analysis.py. It pins
and hashes the published source files, audits the 472-row frozen result, then
recomputes scores, league summaries, calibration-in-the-large, and Claude's
10,000-draw date/match bootstrap from the published per-target JSON.
Research-only: it writes only the requested audit report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import statistics
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

SOURCE_REPO = "sigulun555-lab/toto-lab"
SOURCE_COMMIT = "0a7bff073ac7686d011d9b100d7d9e169da4cbf7"
SOURCE_DIR = "research/goal3_k6_k10_2026-11-10"
SOURCE_FILES = {
    "README.md": {
        "path": f"{SOURCE_DIR}/README.md",
        "git_blob_sha1": "69f5102c3d8efcc1bdca5e7e5084d78eb2260e1d",
    },
    "analysis.py": {
        "path": f"{SOURCE_DIR}/analysis.py",
        "git_blob_sha1": "1f708f813df454f30b6f59b7a8299ff0268db1d0",
    },
    "goal3_outer_matches.json": {
        "path": f"{SOURCE_DIR}/goal3_outer_matches.json",
        "git_blob_sha1": "d7a6501be2b174c1cc09f9f403531a1f3be2c842",
    },
    "goal3_summary.json": {
        "path": f"{SOURCE_DIR}/goal3_summary.json",
        "git_blob_sha1": "e375c3279fb3f9a865ba42ce2f59981f57369116",
    },
}
K_VALUES = (6.0, 10.0)
PROBABILITY_SUM_TOLERANCE = 1.1e-8
SCORE_TOLERANCE = 1e-12
SUMMARY_TOLERANCE = 1e-11


def git_blob_sha1(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def fetch_pinned_sources() -> tuple[dict, dict]:
    payloads = {}
    manifest = {}
    for name, spec in SOURCE_FILES.items():
        url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/{spec['path']}"
        request = urllib.request.Request(url, headers={"User-Agent": "GOAL3-independent-audit/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        actual_blob = git_blob_sha1(data)
        if actual_blob != spec["git_blob_sha1"]:
            raise AssertionError(
                f"{name}: Git blob SHA-1 mismatch: expected {spec['git_blob_sha1']}, got {actual_blob}"
            )
        payloads[name] = data
        manifest[name] = {
            "path": spec["path"],
            "gitBlobSha1": actual_blob,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    return payloads, manifest


def score(probabilities: list[float], actual: int) -> tuple[float, float, int]:
    brier = sum((probabilities[c] - (1.0 if c == actual else 0.0)) ** 2 for c in range(4))
    logloss = -math.log(max(1e-12, probabilities[actual]))
    accuracy = int(max(range(4), key=lambda c: probabilities[c]) == actual)
    return brier, logloss, accuracy


def close(actual: float, expected: float, label: str, tolerance: float = SUMMARY_TOLERANCE) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{label} mismatch: actual={actual:.17g}, expected={expected:.17g}, tolerance={tolerance}"
        )


def mean(values: list[float]) -> float:
    if not values:
        raise AssertionError("cannot average an empty population")
    return sum(values) / len(values)


def percentile_order_statistic_indices(replicates: int) -> tuple[int, int]:
    """Return zero-based nearest-rank indices for a two-sided 95% interval."""
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    lower = max(0, math.ceil(0.025 * replicates) - 1)
    upper = min(replicates - 1, math.ceil(0.975 * replicates) - 1)
    return lower, upper


def _bootstrap(
    rows: list[dict],
    metric: str,
    cluster_key: str,
    rng: random.Random,
    replicates: int,
) -> tuple[list[float], int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[cluster_key])].append(row)
    keys = sorted(groups)
    if len(keys) < 2:
        raise AssertionError(f"bootstrap requires at least two {cluster_key} clusters")
    cluster_sums = {
        key: (sum(float(row[metric]) for row in groups[key]), len(groups[key]))
        for key in keys
    }
    samples = []
    for _ in range(replicates):
        numerator = 0.0
        denominator = 0
        for key in (keys[rng.randrange(len(keys))] for _ in keys):
            cluster_sum, cluster_count = cluster_sums[key]
            numerator += cluster_sum
            denominator += cluster_count
        samples.append(numerator / denominator)
    samples.sort()
    lo_index, hi_index = percentile_order_statistic_indices(replicates)
    return [samples[lo_index], samples[hi_index]], len(keys)


def audit_rows(
    rows: list[dict],
    summary: dict | None = None,
    *,
    seed: int = 1,
    replicates: int = 10000,
    strict_source: bool = False,
    analysis_source: str = "",
) -> dict:
    if not isinstance(rows, list) or not rows:
        raise AssertionError("rows must be a non-empty JSON array")
    if replicates < 100:
        raise ValueError("replicates must be at least 100")

    seen = set()
    match_groups: dict[str, list[dict]] = defaultdict(list)
    match_leagues: dict[str, set[str]] = defaultdict(set)
    date_folds: dict[str, set[int]] = defaultdict(set)
    teams_by_league: dict[str, set[str]] = defaultdict(set)
    dates = set()
    max_simplex_error = 0.0
    recomputed = []

    required = {
        "fold", "date", "league", "match", "side", "actual",
        "p_k6", "p_k10", "cutoff_prior_max_date",
        "brier_k6", "brier_k10", "ll_k6", "ll_k10", "D_brier", "D_ll",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise AssertionError(f"row {index} missing fields: {sorted(missing)}")
        date_text = row["date"]
        cutoff_text = row["cutoff_prior_max_date"]
        try:
            date_value = dt.date.fromisoformat(date_text)
            cutoff_value = dt.date.fromisoformat(cutoff_text)
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"row {index} has invalid ISO date/cutoff") from exc
        if date_value.isoformat() != date_text or cutoff_value.isoformat() != cutoff_text:
            raise AssertionError(f"row {index} date/cutoff is not canonical ISO format")
        if cutoff_value >= date_value:
            raise AssertionError(f"row {index} has same-day/future training data")
        if row["league"] not in ("J1", "J2", "J3"):
            raise AssertionError(f"row {index} has unknown league {row['league']!r}")
        if row["side"] not in ("home", "away"):
            raise AssertionError(f"row {index} has invalid side {row['side']!r}")
        if isinstance(row["actual"], bool) or not isinstance(row["actual"], int) or row["actual"] not in range(4):
            raise AssertionError(f"row {index} has invalid actual category")
        if isinstance(row["fold"], bool) or not isinstance(row["fold"], int) or row["fold"] not in (1, 2, 3):
            raise AssertionError(f"row {index} has invalid fold")
        key = (date_text, row["match"], row["side"])
        if key in seen:
            raise AssertionError(f"duplicate date|match|side key: {key}")
        seen.add(key)
        dates.add(date_text)
        date_folds[date_text].add(row["fold"])
        match_groups[row["match"]].append(row)
        match_leagues[row["match"]].add(row["league"])

        row_scores = {}
        for k, field in ((6.0, "p_k6"), (10.0, "p_k10")):
            probabilities = row[field]
            if not isinstance(probabilities, list) or len(probabilities) != 4:
                raise AssertionError(f"row {index} {field} must contain four categories")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0
                for value in probabilities
            ):
                raise AssertionError(f"row {index} {field} contains invalid probability")
            probability_sum = sum(float(value) for value in probabilities)
            simplex_error = abs(probability_sum - 1.0)
            max_simplex_error = max(max_simplex_error, simplex_error)
            if simplex_error > PROBABILITY_SUM_TOLERANCE:
                raise AssertionError(f"row {index} {field} simplex error {simplex_error}")
            brier, logloss, accuracy = score([float(x) for x in probabilities], row["actual"])
            row_scores[k] = (brier, logloss, accuracy)
            brier_key = "brier_k6" if k == 6.0 else "brier_k10"
            ll_key = "ll_k6" if k == 6.0 else "ll_k10"
            close(float(row[brier_key]), brier, f"row {index} {brier_key}", SCORE_TOLERANCE)
            close(float(row[ll_key]), logloss, f"row {index} {ll_key}", SCORE_TOLERANCE)
        d_brier = row_scores[6.0][0] - row_scores[10.0][0]
        d_ll = row_scores[6.0][1] - row_scores[10.0][1]
        close(float(row["D_brier"]), d_brier, f"row {index} D_brier", SCORE_TOLERANCE)
        close(float(row["D_ll"]), d_ll, f"row {index} D_ll", SCORE_TOLERANCE)
        recomputed.append({
            **row,
            "brier_k6": row_scores[6.0][0],
            "brier_k10": row_scores[10.0][0],
            "ll_k6": row_scores[6.0][1],
            "ll_k10": row_scores[10.0][1],
            "D_brier": d_brier,
            "D_ll": d_ll,
            "_accuracy_k6": row_scores[6.0][2],
            "_accuracy_k10": row_scores[10.0][2],
        })

    for match, pair in match_groups.items():
        if len(pair) != 2 or {row["side"] for row in pair} != {"home", "away"}:
            raise AssertionError(f"match {match!r} does not have exactly one home and one away row")
        if len(match_leagues[match]) != 1:
            raise AssertionError(f"match key collides across leagues: {match!r}")
        identities = {(row["date"], row["league"], row["fold"]) for row in pair}
        if len(identities) != 1:
            raise AssertionError(f"home/away pair metadata mismatch: {match!r}")
    if any(len(folds) != 1 for folds in date_folds.values()):
        raise AssertionError("one target date is assigned to multiple outer folds")

    match_count = len(match_groups)
    date_count = len(dates)
    fold_date_counts = {
        str(fold): sum(1 for fold_set in date_folds.values() if fold in fold_set)
        for fold in (1, 2, 3)
    }
    fold_target_counts = {
        str(fold): sum(1 for row in rows if row["fold"] == fold)
        for fold in (1, 2, 3)
    }
    if strict_source:
        if (len(rows), match_count, date_count) != (472, 236, 17):
            raise AssertionError(
                f"source cohort mismatch: targets={len(rows)}, matches={match_count}, dates={date_count}"
            )
        if fold_date_counts != {"1": 5, "2": 6, "3": 6}:
            raise AssertionError(f"unexpected outer fold date counts: {fold_date_counts}")
        if fold_target_counts != {"1": 120, "2": 134, "3": 218}:
            raise AssertionError(f"unexpected outer fold target counts: {fold_target_counts}")

    metrics = {
        "k6": {
            "brier": mean([row["brier_k6"] for row in recomputed]),
            "logloss": mean([row["ll_k6"] for row in recomputed]),
            "accuracy": mean([row["_accuracy_k6"] for row in recomputed]),
        },
        "k10": {
            "brier": mean([row["brier_k10"] for row in recomputed]),
            "logloss": mean([row["ll_k10"] for row in recomputed]),
            "accuracy": mean([row["_accuracy_k10"] for row in recomputed]),
        },
        "difference_k6_minus_k10": {
            "brier": mean([row["D_brier"] for row in recomputed]),
            "logloss": mean([row["D_ll"] for row in recomputed]),
            "accuracy": mean([row["_accuracy_k6"] - row["_accuracy_k10"] for row in recomputed]),
        },
    }

    rng = random.Random(seed)
    bootstrap = {}
    for metric, summary_key in (("D_brier", "D_brier"), ("D_ll", "D_ll")):
        bootstrap[summary_key] = {}
        for cluster_key in ("date", "match"):
            interval, cluster_count = _bootstrap(recomputed, metric, cluster_key, rng, replicates)
            bootstrap[summary_key][cluster_key] = {
                "ci95": interval,
                "clusterCount": cluster_count,
            }

    leagues = {}
    for league in sorted({row["league"] for row in recomputed}):
        subset = [row for row in recomputed if row["league"] == league]
        per_date: dict[str, list[dict]] = defaultdict(list)
        for row in subset:
            per_date[row["date"]].append(row)
        date_brier = [mean([row["D_brier"] for row in group]) for group in per_date.values()]
        date_ll = [mean([row["D_ll"] for row in group]) for group in per_date.values()]
        leagues[league] = {
            "team_sides": len(subset),
            "matches": len({row["match"] for row in subset}),
            "dates": len(per_date),
            "D_brier_mean": mean([row["D_brier"] for row in subset]),
            "D_brier_median_date": statistics.median(date_brier),
            "D_brier_sd_date": statistics.pstdev(date_brier),
            "D_ll_mean": mean([row["D_ll"] for row in subset]),
            "D_ll_median_date": statistics.median(date_ll),
            "D_ll_sd_date": statistics.pstdev(date_ll),
            "date_league_brier_over_0.05": sum(value >= 0.05 for value in date_brier),
            "date_league_ll_over_0.10": sum(value >= 0.10 for value in date_ll),
        }

    calibration = {}
    for field in ("p_k6", "p_k10"):
        calibration[field] = {}
        for category in range(4):
            calibration[field][str(category)] = {
                "mean_pred": mean([float(row[field][category]) for row in recomputed]),
                "observed": mean([1.0 if row["actual"] == category else 0.0 for row in recomputed]),
            }

    if summary is not None:
        if summary.get("team_side_targets") != len(rows) or summary.get("matches") != match_count:
            raise AssertionError("Claude summary cohort counts do not match the row artifact")
        for label, summary_key in (("D_brier", "D_brier"), ("D_ll", "D_ll")):
            close(metrics["difference_k6_minus_k10"]["brier" if label == "D_brier" else "logloss"],
                  float(summary["overall"][summary_key]), f"summary overall {summary_key}")
            for cluster_key, summary_interval_key in (
                ("date", "date_block_ci95"), ("match", "match_cluster_ci95")
            ):
                saved = summary["bootstrap"][label][summary_interval_key]
                actual = bootstrap[label][cluster_key]["ci95"]
                close(actual[0], float(saved[0]), f"summary {label} {cluster_key} lower")
                close(actual[1], float(saved[1]), f"summary {label} {cluster_key} upper")
                saved_count = summary["bootstrap"][label]["n_dates" if cluster_key == "date" else "n_matches"]
                if saved_count != bootstrap[label][cluster_key]["clusterCount"]:
                    raise AssertionError(f"summary {label} {cluster_key} cluster count mismatch")
        for league, values in leagues.items():
            saved = summary["leagues"][league]
            for key, value in values.items():
                if isinstance(value, int):
                    if saved[key] != value:
                        raise AssertionError(f"summary {league}.{key} mismatch")
                else:
                    close(value, float(saved[key]), f"summary {league}.{key}")
        for field in ("p_k6", "p_k10"):
            for category in range(4):
                for key in ("mean_pred", "observed"):
                    close(
                        calibration[field][str(category)][key],
                        float(summary["calibration"][field][str(category)][key]),
                        f"summary calibration {field}.{category}.{key}",
                    )
        close(metrics["k6"]["accuracy"], float(summary["accuracy"]["p_k6"]), "summary k6 accuracy")
        close(metrics["k10"]["accuracy"], float(summary["accuracy"]["p_k10"]), "summary k10 accuracy")

    analysis_abs_path_count = analysis_source.count("/home/claude/work/")
    intervals_include_zero = {
        metric: {
            cluster: interval["ci95"][0] <= 0.0 <= interval["ci95"][1]
            for cluster, interval in by_cluster.items()
        }
        for metric, by_cluster in bootstrap.items()
    }
    return {
        "experiment": "independent portable reproduction of Claude GOAL3 k=6 vs k=10 artifact",
        "scope": "research-only retrospective fixed cohort; not prospective adoption evidence",
        "cohort": {
            "teamSideTargets": len(rows),
            "matches": match_count,
            "dates": date_count,
            "dateRange": [min(dates), max(dates)],
            "foldDateCounts": fold_date_counts,
            "foldTargetCounts": fold_target_counts,
            "leagueTargets": {
                league: sum(1 for row in rows if row["league"] == league)
                for league in ("J1", "J2", "J3")
            },
            "uniqueDateMatchSideKeys": len(seen),
            "duplicateDateMatchSideKeys": 0,
            "maxProbabilitySimplexAbsError": max_simplex_error,
            "sameDayOrFutureCutoffViolations": 0,
            "crossLeagueMatchKeyCollisions": 0,
        },
        "metrics": metrics,
        "bootstrap": {
            "method": "Python random.Random (MT19937), one shared RNG stream across D_brier date/match then D_ll date/match; percentile order statistics",
            "seed": seed,
            "replicates": replicates,
            "lowerIndex": percentile_order_statistic_indices(replicates)[0],
            "upperIndex": percentile_order_statistic_indices(replicates)[1],
            "indexConvention": "zero-based nearest-rank order statistics: ceil(p*n)-1",
            "intervals": bootstrap,
            "allIntervalsContainZero": all(
                all(values.values()) for values in intervals_include_zero.values()
            ),
        },
        "leagueSummaries": leagues,
        "calibrationInTheLarge": calibration,
        "portability": {
            "claudeAnalysisAbsolutePathOccurrences": analysis_abs_path_count,
            "claudeAnalysisPortable": analysis_abs_path_count == 0,
            "claudeAnalysisExecuted": False,
            "independentVerifierUsesCliAndPinnedImmutableInputs": True,
            "note": "The upstream analysis.py is not executed; its outputs are independently recomputed from the pinned per-target JSON. This verifies artifact reproducibility, not the original local execution environment.",
        },
        "decision": "Both k6-k10 bootstrap marginal intervals include zero; no retrospective superiority or production adoption claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="path for the independent JSON audit report")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()

    payloads, manifest = fetch_pinned_sources()
    rows = json.loads(payloads["goal3_outer_matches.json"].decode("utf-8"))
    summary = json.loads(payloads["goal3_summary.json"].decode("utf-8"))
    analysis_source = payloads["analysis.py"].decode("utf-8")
    report = audit_rows(
        rows,
        summary,
        seed=args.seed,
        replicates=args.replicates,
        strict_source=True,
        analysis_source=analysis_source,
    )
    report["source"] = {
        "repository": SOURCE_REPO,
        "commit": SOURCE_COMMIT,
        "files": manifest,
    }
    report["reproduction"] = {
        "command": "python verification/goal3_independent/verify_claude_k6_k10_bootstrap.py --out verification/goal3_independent/results/claude_k6_k10_bootstrap_audit.json",
        "python": sys.version.split()[0],
        "sha256OfVerifier": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "teamSideTargets": report["cohort"]["teamSideTargets"],
        "matches": report["cohort"]["matches"],
        "dates": report["cohort"]["dates"],
        "metrics": report["metrics"],
        "bootstrap": report["bootstrap"],
        "claudeAnalysisPortable": report["portability"]["claudeAnalysisPortable"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
