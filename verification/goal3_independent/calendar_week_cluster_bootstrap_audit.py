#!/usr/bin/env python3
"""Calendar-aware ISO-week cluster sensitivity for retrospective GOAL3 k audits.

This supplements (does not replace) the pre-existing sorted-target-date moving
block bootstrap. All home/away targets and matches in a sampled ISO week move
together. Two designs are reported: iid ISO-week resampling and circular
two-consecutive-week moving-block sensitivity. Retrospective only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import date as Date
from pathlib import Path

import numpy as np

K_VALUES = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0,
            15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)
REFERENCE_K = 10.0
CANDIDATE_K = 6.0
SCOPES = ("ALL", "J1", "J2", "J3")
LEAGUES = ("J1", "J2", "J3")
LOGLOSS_FLOOR = 1e-12
MIN_COMMON_VALID_RATE = 0.95


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def flatten_targets(report: dict) -> tuple[list[dict], list[str], list[str]]:
    matches = report.get("allKPredictions")
    if not isinstance(matches, list) or len(matches) != 236:
        raise ValueError("expected fixed 236-match allKPredictions cohort")
    rows: list[dict] = []
    seen_matches = set()
    teams = {league: set() for league in LEAGUES}
    dates = set()
    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            raise ValueError(f"match {index} must be an object")
        raw_date, league = match.get("date"), match.get("comp")
        home, away = match.get("home"), match.get("away")
        if not isinstance(raw_date, str) or league not in LEAGUES:
            raise ValueError(f"invalid date or league at match {index}")
        try:
            parsed = Date.fromisoformat(raw_date)
        except ValueError as exc:
            raise ValueError(f"invalid date at match {index}: {raw_date!r}") from exc
        if parsed.isoformat() != raw_date or parsed.year != 2026:
            raise ValueError(f"noncanonical/non-2026 date at match {index}")
        if (not isinstance(home, str) or not home.strip()
                or not isinstance(away, str) or not away.strip()):
            raise ValueError(f"missing team at match {index}")
        home, away = home.strip(), away.strip()
        if home == away:
            raise ValueError(f"home/away collision at match {index}")
        match_key = (raw_date, league, home, away)
        if match_key in seen_matches:
            raise ValueError(f"duplicate match {match_key!r}")
        seen_matches.add(match_key)
        dates.add(raw_date)
        teams[league].update((home, away))
        by_k = match.get("byK")
        if not isinstance(by_k, dict):
            raise ValueError(f"missing byK at match {index}")
        for side, goal_field in (("home", "hg"), ("away", "ag")):
            goals = match.get(goal_field)
            if type(goals) is not int or goals < 0:
                raise ValueError(f"invalid observed goals at match {index}/{side}")
            probabilities = {}
            for k in K_VALUES:
                key = f"{k:.1f}"
                try:
                    p = np.asarray(by_k[key][side], dtype=float)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"missing k={key} {side} probabilities") from exc
                if p.shape != (4,) or not np.isfinite(p).all():
                    raise ValueError(f"invalid probability vector k={key}/{side}")
                if np.any(p < 0.0) or np.any(p > 1.0) or abs(float(p.sum()) - 1.0) > 1e-9:
                    raise ValueError(f"probability vector outside simplex k={key}/{side}")
                probabilities[k] = p
            week = parsed.isocalendar()
            rows.append({
                "date": raw_date, "week": (week.year, week.week),
                "league": league,
                "team": (league, home if side == "home" else away),
                "match": match_key, "side": side, "goals": goals,
                "actual": min(goals, 3), "probabilities": probabilities,
            })
    if len(rows) != 472 or len(dates) != 17:
        raise ValueError(f"expected 472 targets/17 dates, got {len(rows)}/{len(dates)}")
    for league in LEAGUES:
        if len(teams[league]) != 20:
            raise ValueError(f"expected 20 teams in {league}, got {len(teams[league])}")
    week_keys = sorted({row["week"] for row in rows})
    if len(week_keys) != 8:
        raise ValueError(f"expected 8 ISO-week clusters, got {len(week_keys)}")
    week_labels = [f"{year}-W{week:02d}" for year, week in week_keys]
    return rows, sorted(dates), week_labels


def make_week_draws(n_weeks: int, replicates: int, seed: int, design: str) -> np.ndarray:
    if n_weeks < 2 or replicates < 1:
        raise ValueError("need at least 2 weeks and 1 replicate")
    rng = np.random.Generator(np.random.PCG64(seed))
    if design == "iid_iso_week":
        return rng.integers(0, n_weeks, size=(replicates, n_weeks), dtype=np.int16)
    if design == "circular_two_week_blocks":
        block_count = math.ceil(n_weeks / 2)
        starts = rng.integers(0, n_weeks, size=(replicates, block_count), dtype=np.int16)
        draws = np.empty((replicates, n_weeks), dtype=np.int16)
        for block in range(block_count):
            for offset in range(2):
                column = block * 2 + offset
                if column < n_weeks:
                    draws[:, column] = (starts[:, block] + offset) % n_weeks
        return draws
    raise ValueError(f"unknown design: {design}")


def common_valid_mask(counts_by_scope: np.ndarray, draws: np.ndarray) -> np.ndarray:
    if counts_by_scope.ndim != 2 or draws.ndim != 2:
        raise ValueError("counts_by_scope and draws must be matrices")
    sampled = counts_by_scope[:, draws].sum(axis=2)
    return np.all(sampled > 0, axis=0)


def max_t_intervals(point: np.ndarray, boot: np.ndarray, labels: list[str]) -> dict:
    point, boot = np.asarray(point, dtype=float), np.asarray(boot, dtype=float)
    if boot.ndim != 2 or point.shape != (boot.shape[1],):
        raise ValueError("point/boot shape mismatch")
    if not np.isfinite(point).all() or not np.isfinite(boot).all():
        raise ValueError("non-finite values in max-t family")
    se = boot.std(axis=0, ddof=1)
    if np.any(~np.isfinite(se)) or np.any(se <= 0):
        raise ValueError("zero/non-finite bootstrap SE in max-t family")
    critical = float(np.quantile(
        np.max(np.abs((boot - point) / se), axis=1), 0.95, method="linear"
    ))
    intervals = {}
    for i, label in enumerate(labels):
        values = boot[:, i]
        intervals[label] = {
            "estimate": float(point[i]), "bootstrapSE": float(se[i]),
            "simultaneous95Band": [
                float(point[i] - critical * se[i]), float(point[i] + critical * se[i])
            ],
            "nominalPercentile95CI": [
                float(np.quantile(values, 0.025, method="linear")),
                float(np.quantile(values, 0.975, method="linear")),
            ],
        }
    return {"criticalValue": critical, "intervals": intervals}


def score_sufficient(rows: list[dict], week_labels: list[str]):
    wi = {label: i for i, label in enumerate(week_labels)}
    counts = np.zeros(len(week_labels), dtype=np.int64)
    sums = {m: np.zeros((len(week_labels), len(K_VALUES)), dtype=float)
            for m in ("brier", "logloss", "accuracy")}
    eye = np.eye(4)
    for row in rows:
        label = f"{row['week'][0]}-W{row['week'][1]:02d}"
        w = wi[label]
        counts[w] += 1
        y = row["actual"]
        for ki, k in enumerate(K_VALUES):
            p = row["probabilities"][k]
            sums["brier"][w, ki] += float(np.square(p - eye[y]).sum())
            sums["logloss"][w, ki] += float(-np.log(max(float(p[y]), LOGLOSS_FLOOR)))
            sums["accuracy"][w, ki] += float(int(np.argmax(p) == y))
    return counts, sums


def score_audit(counts: np.ndarray, sums: dict, draws: np.ndarray) -> dict:
    sampled_n = counts[draws].sum(axis=1)
    if np.any(sampled_n <= 0):
        raise ValueError("score bootstrap replicate has no sampled targets")
    point = {m: v.sum(axis=0) / counts.sum() for m, v in sums.items()}
    boot = {m: v[draws].sum(axis=1) / sampled_n[:, None] for m, v in sums.items()}
    frequencies = {}
    for metric, direction in (("brier", "min"), ("logloss", "min"), ("accuracy", "max")):
        winners = np.argmin(boot[metric], axis=1) if direction == "min" else np.argmax(boot[metric], axis=1)
        frequencies[metric] = {f"{k:g}": float(np.mean(winners == i))
                               for i, k in enumerate(K_VALUES)}
    ref = K_VALUES.index(REFERENCE_K)
    labels, point_cols, boot_cols = [], [], []
    for metric in ("brier", "logloss"):
        for i, k in enumerate(K_VALUES):
            if i == ref:
                continue
            labels.append(f"{metric}:k={k:g}-k=10")
            point_cols.append(float(point[metric][i] - point[metric][ref]))
            boot_cols.append(boot[metric][:, i] - boot[metric][:, ref])
    family = max_t_intervals(np.asarray(point_cols), np.column_stack(boot_cols), labels)
    k6 = K_VALUES.index(CANDIDATE_K)
    focal = {}
    for metric in ("brier", "logloss"):
        delta = boot[metric][:, k6] - boot[metric][:, ref]
        focal[metric] = {
            "differenceK6MinusK10": float(point[metric][k6] - point[metric][ref]),
            "bootstrapSE": float(delta.std(ddof=1)),
            "nominalPercentile95CI": [
                float(np.quantile(delta, 0.025, method="linear")),
                float(np.quantile(delta, 0.975, method="linear")),
            ],
            "selectionAdjustedSimultaneous95Band":
                family["intervals"][f"{metric}:k=6-k=10"]["simultaneous95Band"],
        }
    return {
        "pointMetricsByK": {
            f"{k:g}": {m: float(point[m][i]) for m in ("brier", "logloss", "accuracy")}
            for i, k in enumerate(K_VALUES)
        },
        "bestKFrequency": frequencies,
        "selectionAdjustedFamily": {
            "referenceK": REFERENCE_K, "familySize": len(labels),
            "studentizedMaxAbsTCriticalValue": family["criticalValue"],
            "contrasts": family["intervals"],
        },
        "k6MinusK10": focal,
    }


def calibration_sufficient(rows: list[dict], week_labels: list[str]):
    wi = {label: i for i, label in enumerate(week_labels)}
    counts = np.zeros((len(SCOPES), len(week_labels)), dtype=np.int64)
    pred = np.zeros((len(SCOPES), len(week_labels), 4, 2), dtype=float)
    obs = np.zeros((len(SCOPES), len(week_labels), 4), dtype=float)
    kidx = (K_VALUES.index(6.0), K_VALUES.index(10.0))
    for row in rows:
        w = wi[f"{row['week'][0]}-W{row['week'][1]:02d}"]
        for si in (0, SCOPES.index(row["league"])):
            counts[si, w] += 1
            obs[si, w, row["actual"]] += 1.0
            for local, global_i in enumerate(kidx):
                pred[si, w, :, local] += row["probabilities"][K_VALUES[global_i]]
    return counts, pred, obs


def calibration_audit(counts: np.ndarray, pred: np.ndarray, obs: np.ndarray,
                      draws: np.ndarray) -> dict:
    valid = common_valid_mask(counts, draws)
    valid_n, valid_rate = int(valid.sum()), float(valid.mean())
    points, boots, labels, grouped = [], [], [], {}
    for si, scope in enumerate(SCOPES):
        n = int(counts[si].sum())
        point_gap = (pred[si].sum(axis=0) / n - obs[si].sum(axis=0)[:, None] / n) * 100.0
        sn = counts[si, draws].sum(axis=1)[valid]
        ps = pred[si, draws].sum(axis=1)[valid]
        os = obs[si, draws].sum(axis=1)[valid]
        gaps = (ps / sn[:, None, None] - os[:, :, None] / sn[:, None, None]) * 100.0
        grouped[scope] = {"n": n}
        for ki, k in enumerate((6.0, 10.0)):
            grouped[scope][f"k={k:g}"] = []
            for cat in range(4):
                label = f"{scope}:k={k:g}:category={cat}"
                points.append(float(point_gap[cat, ki]))
                boots.append(gaps[:, cat, ki])
                labels.append(label)
                grouped[scope][f"k={k:g}"].append(
                    {"category": cat, "gapPp": float(point_gap[cat, ki])}
                )
    base = {
        "estimand": "marginal category calibration-in-the-large; mean predicted minus observed frequency (percentage points)",
        "commonValidReplicates": valid_n, "requestedReplicates": int(draws.shape[0]),
        "commonValidRate": valid_rate, "minimumRequiredRate": MIN_COMMON_VALID_RATE,
        "pointGaps": grouped,
    }
    if valid_rate < MIN_COMMON_VALID_RATE:
        return {**base, "status": "insufficient_common_valid_replicates",
                "simultaneousBands": None}
    family = max_t_intervals(np.asarray(points), np.column_stack(boots), labels)
    for scope in SCOPES:
        for k in (6.0, 10.0):
            for entry in grouped[scope][f"k={k:g}"]:
                label = f"{scope}:k={k:g}:category={entry['category']}"
                entry.update(family["intervals"][label])
    return {
        **base, "status": "computed", "simultaneousFamilySize": len(labels),
        "studentizedMaxAbsTCriticalValue": family["criticalValue"],
        "simultaneousBands": family["intervals"],
        "notAssessed": [
            "probability-bin reliability", "calibration slope/intercept",
            "joint multinomial calibration", "calibration conditional on match context",
        ],
    }


def poisson_sufficient(rows: list[dict], week_labels: list[str]) -> np.ndarray:
    wi = {label: i for i, label in enumerate(week_labels)}
    stats = np.zeros((len(SCOPES), len(week_labels), 2, 3), dtype=float)
    for row in rows:
        w = wi[f"{row['week'][0]}-W{row['week'][1]:02d}"]
        for si in (0, SCOPES.index(row["league"])):
            for ki, k in enumerate((6.0, 10.0)):
                p0 = float(row["probabilities"][k][0])
                if not 0.0 < p0 < 1.0:
                    raise ValueError(f"p(goal=0) must be in (0,1), got {p0}")
                lam = -math.log(p0)
                residual = (row["goals"] - lam) / math.sqrt(lam)
                stats[si, w, ki, 0] += 1.0
                stats[si, w, ki, 1] += residual
                stats[si, w, ki, 2] += residual * residual
    return stats


def poisson_audit(stats: np.ndarray, draws: np.ndarray) -> dict:
    counts = stats[:, :, :, 0].transpose(0, 2, 1).reshape(len(SCOPES) * 2, stats.shape[1])
    valid = common_valid_mask(counts, draws)
    valid_n, valid_rate = int(valid.sum()), float(valid.mean())
    points, boots, labels = [], [], []
    for si, scope in enumerate(SCOPES):
        for ki, k in enumerate((6.0, 10.0)):
            total = stats[si, :, ki, :].sum(axis=0)
            points.append(float(total[2] / total[0]))
            sampled = stats[si, draws, ki, :][valid].sum(axis=1)
            boots.append(sampled[:, 2] / sampled[:, 0])
            labels.append(f"{scope}:k={k:g}")
    base = {
        "commonValidReplicates": valid_n, "requestedReplicates": int(draws.shape[0]),
        "commonValidRate": valid_rate, "minimumRequiredRate": MIN_COMMON_VALID_RATE,
        "definition": "lambda=-log(P(goal=0)); Pearson residual=(goals-lambda)/sqrt(lambda); dispersion=mean(residual^2)",
        "reference": 1.0,
    }
    if valid_rate < MIN_COMMON_VALID_RATE:
        return {**base, "status": "insufficient_common_valid_replicates",
                "simultaneousBands": None}
    family = max_t_intervals(np.asarray(points), np.column_stack(boots), labels)
    paired = {}
    for si, scope in enumerate(SCOPES):
        delta = boots[si * 2] - boots[si * 2 + 1]
        paired[scope] = {
            "k6MinusK10Dispersion": float(points[si * 2] - points[si * 2 + 1]),
            "percentile95CI": [
                float(np.quantile(delta, 0.025, method="linear")),
                float(np.quantile(delta, 0.975, method="linear")),
            ],
        }
    return {
        **base, "status": "computed", "simultaneousFamilySize": len(labels),
        "studentizedMaxAbsTCriticalValue": family["criticalValue"],
        "simultaneousBands": family["intervals"], "pairedK6MinusK10": paired,
        "limitations": [
            "Eight ISO-week clusters only; bootstrap tails are uncertain.",
            "This is conditional Pearson residual dispersion, not raw-goal VMR.",
            "Descriptive only; not a formal goodness-of-fit or model-selection test.",
        ],
    }


def run_design(rows: list[dict], weeks: list[str], replicates: int,
               seed: int, design: str) -> dict:
    draws = make_week_draws(len(weeks), replicates, seed, design)
    score_counts, score_sums = score_sufficient(rows, weeks)
    cal_counts, cal_pred, cal_obs = calibration_sufficient(rows, weeks)
    residual_stats = poisson_sufficient(rows, weeks)
    return {
        "method": {
            "resamplingUnit": "ISO calendar week; all matches and both team-side targets move together",
            "design": design, "weeks": weeks, "weekClusterCount": len(weeks),
            "replicates": replicates, "rng": "NumPy PCG64", "seed": seed,
            "targetCountWeighting": "aggregate score sums divided by sampled target count",
            "calendarGapsPreserved": True, "quantile": "linear percentile",
        },
        "selection": score_audit(score_counts, score_sums, draws),
        "fourCategoryMarginalCalibration": calibration_audit(cal_counts, cal_pred, cal_obs, draws),
        "conditionalPoissonResidual": poisson_audit(residual_stats, draws),
    }


def audit(report: dict, replicates: int = 50_000, seed: int = 20261108,
          input_sha256: str | None = None, code_sha256: str | None = None) -> dict:
    if replicates < 1_000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    rows, dates, weeks = flatten_targets(report)
    parsed = [Date.fromisoformat(value) for value in dates]
    gaps = [(parsed[i] - parsed[i - 1]).days for i in range(1, len(parsed))]
    week_counts = Counter(f"{r['week'][0]}-W{r['week'][1]:02d}" for r in rows)
    designs = {
        "iid_iso_week": run_design(rows, weeks, replicates, seed, "iid_iso_week"),
        "circular_two_week_blocks": run_design(
            rows, weeks, replicates, seed + 1, "circular_two_week_blocks"
        ),
    }
    return {
        "experiment": "GOAL3 calendar-aware ISO-week cluster bootstrap sensitivity",
        "scope": "research-only retrospective fixed 2026 cohort; sensitivity supplement, not prospective adoption evidence",
        "provenance": {
            "inputSha256": input_sha256, "codeSha256": code_sha256,
            "python": platform.python_version(), "numpy": np.__version__,
        },
        "cohort": {
            "matches": 236, "teamSideTargets": 472, "targetDates": len(dates),
            "dateRange": [dates[0], dates[-1]], "isoWeekClusters": len(weeks),
            "targetsByIsoWeek": {w: int(week_counts[w]) for w in weeks},
            "dateGapDays": gaps, "maxDateGapDays": max(gaps),
            "gapsAtLeastThreeDays": sum(g >= 3 for g in gaps),
        },
        "designs": designs,
        "limitations": [
            "Only eight ISO-week clusters are available; bootstrap tails are uncertain.",
            "The iid-week design assumes exchangeable weeks and does not preserve serial dependence.",
            "The circular two-week block design preserves one adjacent-week lag but wraps the period endpoints.",
            "Neither design separately clusters repeated teams across different weeks.",
            "Retrospective source availability does not prove forecasts were generated before historical deadlines.",
            "Marginal calibration does not assess probability-bin reliability or joint multinomial calibration.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    input_path = Path(args.input)
    input_bytes = input_path.read_bytes()
    report = json.loads(input_bytes.decode("utf-8"))
    code_bytes = Path(__file__).read_bytes()
    result = audit(report, args.replicates, args.seed,
                   input_sha256=sha256_bytes(input_bytes),
                   code_sha256=sha256_bytes(code_bytes))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {}
    for name, design in result["designs"].items():
        summary[name] = {
            "k6MinusK10": design["selection"]["k6MinusK10"],
            "calibrationStatus": design["fourCategoryMarginalCalibration"]["status"],
            "calibrationValidRate": design["fourCategoryMarginalCalibration"]["commonValidRate"],
            "poissonStatus": design["conditionalPoissonResidual"]["status"],
            "poissonValidRate": design["conditionalPoissonResidual"]["commonValidRate"],
        }
    print(json.dumps({
        "cohort": result["cohort"], "inputSha256": result["provenance"]["inputSha256"],
        "codeSha256": result["provenance"]["codeSha256"], "designSummary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
