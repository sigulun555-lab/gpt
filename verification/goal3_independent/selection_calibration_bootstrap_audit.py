#!/usr/bin/env python3
"""Reproducible GOAL3 all-k selection and four-category calibration uncertainty audit.

Research-only. Uses the immutable all-k outer-reconstruction artifact and retains
all home/away team-sides from each sampled target date. This is retrospective
uncertainty analysis, not prospective adoption evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import defaultdict
from datetime import date as Date
from pathlib import Path

import numpy as np

K_VALUES = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0,
            15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)
LEAGUES = ("J1", "J2", "J3")
SCOPES = ("ALL", "J1", "J2", "J3")
REFERENCE_K = 10.0
LOGLOSS_FLOOR = 1e-12
METRICS = ("brier", "logloss", "accuracy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_targets(report: dict) -> tuple[list[dict], list[str]]:
    matches = report.get("allKPredictions")
    if not isinstance(matches, list) or not matches:
        raise ValueError("input has no allKPredictions")

    rows = []
    seen_matches = set()
    folds_by_date = defaultdict(set)
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("all-k match entry must be an object")
        date = match.get("date")
        league = match.get("comp")
        home, away = match.get("home"), match.get("away")
        if not isinstance(date, str) or league not in LEAGUES:
            raise ValueError("invalid date or league")
        try:
            parsed_date = Date.fromisoformat(date)
        except ValueError as exc:
            raise ValueError(f"invalid target date: {date!r}") from exc
        if parsed_date.isoformat() != date or parsed_date.year != 2026:
            raise ValueError(f"invalid target date: {date!r}")
        if not isinstance(home, str) or not home.strip() or not isinstance(away, str) or not away.strip():
            raise ValueError("missing team name")
        home, away = home.strip(), away.strip()
        if home == away:
            raise ValueError("same team listed as home and away")
        match_key = (date, league, home, away)
        if match_key in seen_matches:
            raise ValueError("duplicate match key: " + repr(match_key))
        seen_matches.add(match_key)

        fold = match.get("fold")
        if type(fold) is not int or fold not in (1, 2, 3):
            raise ValueError("invalid fold")
        folds_by_date[date].add(fold)

        by_k = match.get("byK")
        if not isinstance(by_k, dict):
            raise ValueError("missing byK predictions")
        for side, goal_key in (("home", "hg"), ("away", "ag")):
            goals = match.get(goal_key)
            if type(goals) is not int or goals < 0:
                raise ValueError("invalid observed goals")
            probs_by_k = {}
            for k in K_VALUES:
                key = f"{k:.1f}"
                if key not in by_k or side not in by_k[key]:
                    raise ValueError(f"missing k={key} {side} probabilities")
                p = np.asarray(by_k[key][side], dtype=float)
                if p.shape != (4,) or not np.isfinite(p).all():
                    raise ValueError("invalid four-category probability vector")
                if np.any(p < 0.0) or np.any(p > 1.0) or abs(float(p.sum()) - 1.0) > 1e-9:
                    raise ValueError("probability vector outside simplex")
                probs_by_k[k] = p
            rows.append({
                "date": date,
                "league": league,
                "match_key": match_key,
                "side": side,
                "goals": goals,
                "actual": min(goals, 3),
                "probabilities": probs_by_k,
            })

    dates = sorted({row["date"] for row in rows})
    if len(matches) != 236 or len(rows) != 472 or len(dates) != 17:
        raise ValueError("expected fixed cohort: 236 matches / 472 targets / 17 dates")
    if any(len(folds) != 1 for folds in folds_by_date.values()):
        raise ValueError("a target date crosses outer-fold boundaries")
    target_keys = [
        (r["date"], r["league"], r["match_key"], r["side"]) for r in rows
    ]
    if len(target_keys) != len(set(target_keys)):
        raise ValueError("duplicate team-side target key")
    return rows, dates


def score_sums_by_date(rows: list[dict], dates: list[str]):
    date_index = {date: i for i, date in enumerate(dates)}
    counts = np.zeros(len(dates), dtype=np.int64)
    sums = {metric: np.zeros((len(dates), len(K_VALUES)), dtype=float)
            for metric in METRICS}
    for row in rows:
        di = date_index[row["date"]]
        counts[di] += 1
        y = row["actual"]
        for ki, k in enumerate(K_VALUES):
            p = row["probabilities"][k]
            sums["brier"][di, ki] += float(np.square(p - np.eye(4)[y]).sum())
            sums["logloss"][di, ki] += float(-np.log(np.clip(p[y], LOGLOSS_FLOOR, 1.0)))
            sums["accuracy"][di, ki] += float(int(np.argmax(p) == y))
    return counts, sums


def circular_moving_block_draws(replicates: int, n_dates: int, seed: int,
                                block_length: int = 2) -> np.ndarray:
    block_count = math.ceil(n_dates / block_length)
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, n_dates, size=(replicates, block_count))
    sampled = np.empty((replicates, n_dates), dtype=np.int16)
    for block in range(block_count):
        for offset in range(block_length):
            col = block * block_length + offset
            if col < n_dates:
                sampled[:, col] = (starts[:, block] + offset) % n_dates
    return sampled


def selection_audit(rows: list[dict], dates: list[str], replicates: int,
                    seed: int) -> dict:
    counts, sums = score_sums_by_date(rows, dates)
    sampled = circular_moving_block_draws(replicates, len(dates), seed, 2)
    sampled_counts = counts[sampled].sum(axis=1)
    if np.any(sampled_counts <= 0):
        raise ValueError("bootstrap replicate has no sampled targets")

    boot = {}
    point = {}
    for metric in METRICS:
        point[metric] = sums[metric].sum(axis=0) / counts.sum()
        boot[metric] = sums[metric][sampled].sum(axis=1) / sampled_counts[:, None]

    frequencies = {}
    for metric, direction in (("brier", "min"), ("logloss", "min"),
                              ("accuracy", "max")):
        winners = np.argmin(boot[metric], axis=1) if direction == "min" else np.argmax(boot[metric], axis=1)
        frequencies[metric] = {
            f"{k:g}": float(np.mean(winners == i))
            for i, k in enumerate(K_VALUES)
        }

    reference_index = K_VALUES.index(REFERENCE_K)
    candidate_indices = [i for i in range(len(K_VALUES)) if i != reference_index]
    contrast_columns = []
    contrast_meta = []
    for metric in ("brier", "logloss"):
        for ki in candidate_indices:
            delta = boot[metric][:, ki] - boot[metric][:, reference_index]
            observed = float(point[metric][ki] - point[metric][reference_index])
            se = float(np.std(delta, ddof=1))
            if not np.isfinite(se) or se <= 0:
                raise ValueError(f"invalid bootstrap SE for {metric}, k={K_VALUES[ki]}")
            contrast_columns.append((delta - observed) / se)
            contrast_meta.append((metric, ki, observed, se))
    max_abs_t = np.max(np.abs(np.column_stack(contrast_columns)), axis=1)
    critical = float(np.quantile(max_abs_t, 0.95, method="linear"))

    simultaneous = {}
    for metric, ki, observed, se in contrast_meta:
        k = K_VALUES[ki]
        simultaneous.setdefault(f"{k:g}", {})[metric] = {
            "differenceKMinusK10": observed,
            "bootstrapSE": se,
            "simultaneous95Band": [observed - critical * se, observed + critical * se],
            "nominalPercentile95CI": [
                float(np.quantile(
                    boot[metric][:, ki] - boot[metric][:, reference_index],
                    0.025, method="linear")),
                float(np.quantile(
                    boot[metric][:, ki] - boot[metric][:, reference_index],
                    0.975, method="linear")),
            ],
        }

    k6_index = K_VALUES.index(6.0)
    return {
        "method": {
            "resamplingUnit": "circular moving blocks of consecutive sorted target dates",
            "blockLengthDates": 2,
            "blocksPerReplicate": math.ceil(len(dates) / 2),
            "sampledDatesPerReplicate": len(dates),
            "weighting": "target-count weighted across sampled date clusters",
            "rng": "NumPy PCG64",
            "seed": seed,
            "replicates": replicates,
            "quantile": "linear percentile",
            "selectionFamily": "16 candidate k values; minimum Brier and minimum LogLoss; accuracy maximum is descriptive",
            "simultaneousFamily": "15 non-reference k contrasts x 2 primary metrics (Brier, LogLoss), each against k=10",
        },
        "pointMetricsByK": {
            f"{k:g}": {metric: float(point[metric][i]) for metric in METRICS}
            for i, k in enumerate(K_VALUES)
        },
        "bestCandidateFrequency": frequencies,
        "selectionAdjusted": {
            "referenceK": REFERENCE_K,
            "studentizedMaxAbsTCriticalValue": critical,
            "intervals": simultaneous,
            "k6MinusK10": {
                metric: simultaneous["6"][metric]
                for metric in ("brier", "logloss")
            },
        },
        "unadjustedK6MinusK10Bootstrap": {
            metric: {
                "difference": float(point[metric][k6_index] - point[metric][reference_index]),
                "percentile95CI": [
                    float(np.quantile(
                        boot[metric][:, k6_index] - boot[metric][:, reference_index],
                        0.025, method="linear")),
                    float(np.quantile(
                        boot[metric][:, k6_index] - boot[metric][:, reference_index],
                        0.975, method="linear")),
                ],
            }
            for metric in METRICS
        },
    }


def calibration_family(rows: list[dict], dates: list[str], sampled: np.ndarray,
                       scope: str, k: float) -> dict:
    date_index = {date: i for i, date in enumerate(dates)}
    n_by_date = np.zeros(len(dates), dtype=np.int64)
    pred_by_date = np.zeros((len(dates), 4), dtype=float)
    obs_by_date = np.zeros((len(dates), 4), dtype=float)
    selected = [
        row for row in rows
        if scope == "ALL" or row["league"] == scope
    ]
    for row in selected:
        di = date_index[row["date"]]
        n_by_date[di] += 1
        pred_by_date[di] += row["probabilities"][k]
        obs_by_date[di, row["actual"]] += 1.0
    n_total = int(n_by_date.sum())
    if n_total == 0:
        raise ValueError(f"empty calibration scope {scope}")

    pred_total = pred_by_date.sum(axis=0) / n_total
    obs_total = obs_by_date.sum(axis=0) / n_total
    point_gaps = (pred_total - obs_total) * 100.0

    sampled_n = n_by_date[sampled].sum(axis=1)
    if np.any(sampled_n <= 0):
        raise ValueError(f"bootstrap replicate has no targets for {scope}")
    sampled_pred = pred_by_date[sampled].sum(axis=1) / sampled_n[:, None]
    sampled_obs = obs_by_date[sampled].sum(axis=1) / sampled_n[:, None]
    boot_gaps = (sampled_pred - sampled_obs) * 100.0
    ses = boot_gaps.std(axis=0, ddof=1)
    if np.any(~np.isfinite(ses)) or np.any(ses <= 0):
        raise ValueError(f"invalid calibration bootstrap SE for {scope}, k={k}")

    return {
        "scope": scope,
        "k": k,
        "n": n_total,
        "pointGapsPp": point_gaps,
        "bootstrapSE": ses,
        "bootstrapGaps": boot_gaps,
    }


def calibration_audit(rows: list[dict], dates: list[str], replicates: int,
                      seed: int) -> dict:
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = rng.integers(0, len(dates), size=(replicates, len(dates)))
    raw = {}
    for scope in SCOPES:
        for k in (6.0, 10.0):
            raw[(scope, k)] = calibration_family(
                rows, dates, sampled, scope, k
            )

    families = {
        "k6_by_league": [(league, 6.0) for league in LEAGUES],
        "k10_by_league": [(league, 10.0) for league in LEAGUES],
        "k6_all": [("ALL", 6.0)],
        "k10_all": [("ALL", 10.0)],
        "global_32": [(scope, k) for scope in SCOPES for k in (6.0, 10.0)],
    }
    family_results = {}
    for family_name, scopes in families.items():
        columns = []
        metadata = []
        for scope, k in scopes:
            item = raw[(scope, k)]
            for cat in range(4):
                gap = item["bootstrapGaps"][:, cat]
                point = float(item["pointGapsPp"][cat])
                se = float(item["bootstrapSE"][cat])
                columns.append((gap - point) / se)
                metadata.append((scope, k, cat, point, se, gap, item["n"]))
        critical = float(np.quantile(
            np.max(np.abs(np.column_stack(columns)), axis=1),
            0.95, method="linear"
        ))
        entries = {}
        for scope, k, cat, point, se, gap, n in metadata:
            entries.setdefault(scope, {}).setdefault(f"{k:g}", []).append({
                "category": cat,
                "n": n,
                "gapPp": point,
                "bootstrapSE": se,
                "nominalPercentile95CI": [
                    float(np.quantile(gap, 0.025, method="linear")),
                    float(np.quantile(gap, 0.975, method="linear")),
                ],
                "simultaneous95Band": [
                    point - critical * se,
                    point + critical * se,
                ],
            })
        family_results[family_name] = {
            "comparisonCount": len(metadata),
            "studentizedMaxAbsTCriticalValue": critical,
            "intervals": entries,
        }

    return {
        "method": {
            "resamplingUnit": "target date; all matches and home/away team-sides retained together",
            "rng": "NumPy PCG64",
            "seed": seed,
            "replicates": replicates,
            "quantile": "linear percentile",
            "weighting": "target-count weighted within each scope",
            "families": {
                "k6_by_league": "J1/J2/J3 x categories 0/1/2/3+ (12 intervals)",
                "k10_by_league": "J1/J2/J3 x categories 0/1/2/3+ (12 intervals)",
                "k6_all": "ALL x categories 0/1/2/3+ (4 intervals)",
                "k10_all": "ALL x categories 0/1/2/3+ (4 intervals)",
                "global_32": "ALL/J1/J2/J3 x k=6/k=10 x categories 0/1/2/3+ (32 intervals)",
            },
        },
        "families": family_results,
    }


def audit(report: dict, replicates: int = 50_000, seed: int = 20261105) -> dict:
    if replicates < 1_000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    rows, dates = flatten_targets(report)
    selection = selection_audit(rows, dates, replicates, seed)
    calibration = calibration_audit(rows, dates, replicates, seed)
    return {
        "experiment": "GOAL3 all-k selection stability and four-category calibration uncertainty",
        "scope": "research-only retrospective fixed 2026-only outer cohort; no prospective adoption claim",
        "cohort": {
            "matches": 236,
            "teamSideTargets": 472,
            "dates": len(dates),
            "dateRange": [dates[0], dates[-1]],
            "leagueTargets": {
                league: sum(row["league"] == league for row in rows)
                for league in LEAGUES
            },
        },
        "inputProvenance": report.get("provenance", {}),
        "auditEnvironment": {"python": platform.python_version(), "numpy": np.__version__},
        "selectionStability": selection,
        "fourCategoryCalibration": calibration,
        "limitations": [
            "Only 17 target-date clusters are available; bootstrap tails are uncertain.",
            "The circular moving-block bootstrap treats adjacent sorted target dates as adjacent blocks despite unequal calendar gaps.",
            "Repeated teams across different target dates are not separately clustered.",
            "The history snapshot fetchedAt is retrospective and does not prove first-known result availability at each historical forecast date.",
            "Selection frequencies describe this fixed retrospective cohort and do not establish prospective superiority.",
            "Calibration bands are descriptive uncertainty intervals, not a formal calibration test or a basis for league-specific correction.",
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
    report = json.loads(input_path.read_text(encoding="utf-8"))
    result = audit(report, args.replicates, args.seed)
    result["inputSha256"] = sha256_file(input_path)
    result["codeSha256"] = sha256_file(Path(__file__))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "inputSha256": result["inputSha256"],
        "codeSha256": result["codeSha256"],
        "auditEnvironment": result["auditEnvironment"],
        "cohort": result["cohort"],
        "selectionAdjustedK6MinusK10": result["selectionStability"]["selectionAdjusted"]["k6MinusK10"],
        "calibrationCriticalValues": {
            key: value["studentizedMaxAbsTCriticalValue"]
            for key, value in result["fourCategoryCalibration"]["families"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
