#!/usr/bin/env python3
"""Descriptive leave-one-ISO-week-out influence audit for GOAL3 k=6 vs k=10.

Research-only. Uses the frozen paired outer-match CSV and does not refit models.
The analysis identifies concentration of retrospective score differences by
calendar week; it is not a confidence interval or prospective adoption test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date as Date
from pathlib import Path

METRICS = ("brier", "logloss", "accuracy")
EXPECTED_MATCHES = 236
EXPECTED_TARGETS = 472
EXPECTED_DATES = 17
EXPECTED_WEEKS = 8
LOGLOSS_FLOOR = 1e-12
SIMPLEX_TOLERANCE = 2e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int(value: str, field: str, line: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line}: invalid integer {field}={value!r}") from exc
    return result


def parse_probabilities(row: dict[str, str], prefix: str, line: int) -> tuple[float, ...]:
    try:
        p = tuple(float(row[f"{prefix}_p{i}"]) for i in range(4))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"line {line}: invalid {prefix} probability vector") from exc
    if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in p):
        raise ValueError(f"line {line}: {prefix} probabilities outside [0,1]")
    if abs(sum(p) - 1.0) > SIMPLEX_TOLERANCE:
        raise ValueError(f"line {line}: {prefix} simplex error {sum(p)-1.0:+.9g}")
    return p


def load_rows(path: Path) -> tuple[list[dict], dict]:
    required = {
        "date", "comp", "fold", "home", "away", "hg", "ag",
        "side", "actual",
        *(f"k6_p{i}" for i in range(4)),
        *(f"k10_p{i}" for i in range(4)),
    }
    rows: list[dict] = []
    match_sides: dict[tuple, set[str]] = defaultdict(set)
    match_meta: dict[tuple, tuple] = {}
    target_keys = set()
    dates = set()
    weeks = set()
    teams = {league: set() for league in ("J1", "J2", "J3")}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"CSV missing required columns: {missing}")
        for line, raw in enumerate(reader, start=2):
            date_text, comp = raw["date"], raw["comp"]
            try:
                parsed = Date.fromisoformat(date_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line}: invalid ISO date {date_text!r}") from exc
            if parsed.isoformat() != date_text or parsed.year != 2026:
                raise ValueError(f"line {line}: noncanonical/non-2026 date")
            if comp not in teams:
                raise ValueError(f"line {line}: unsupported competition {comp!r}")
            home, away = raw["home"].strip(), raw["away"].strip()
            if not home or not away or home == away:
                raise ValueError(f"line {line}: invalid home/away identity")
            side = raw["side"]
            if side not in ("home", "away"):
                raise ValueError(f"line {line}: invalid side {side!r}")
            fold = parse_int(raw["fold"], "fold", line)
            if fold not in (1, 2, 3):
                raise ValueError(f"line {line}: invalid fold {fold}")
            hg, ag = parse_int(raw["hg"], "hg", line), parse_int(raw["ag"], "ag", line)
            if hg < 0 or ag < 0:
                raise ValueError(f"line {line}: negative observed goals")
            actual = parse_int(raw["actual"], "actual", line)
            expected_actual = min(hg if side == "home" else ag, 3)
            if actual != expected_actual:
                raise ValueError(f"line {line}: actual category disagrees with goals")
            match = (date_text, comp, home, away)
            target = (*match, side)
            if target in target_keys:
                raise ValueError(f"line {line}: duplicate target {target!r}")
            target_keys.add(target)
            meta = (fold, hg, ag)
            if match in match_meta and match_meta[match] != meta:
                raise ValueError(f"line {line}: inconsistent match metadata")
            match_meta[match] = meta
            match_sides[match].add(side)
            teams[comp].update((home, away))
            dates.add(date_text)
            iso = parsed.isocalendar()
            week = f"{iso.year}-W{iso.week:02d}"
            weeks.add(week)
            rows.append({
                "date": date_text, "week": week, "comp": comp, "fold": fold,
                "match": match, "side": side, "actual": actual,
                "p6": parse_probabilities(raw, "k6", line),
                "p10": parse_probabilities(raw, "k10", line),
            })
    if not rows:
        raise ValueError("CSV has no targets")
    if any(sides != {"home", "away"} for sides in match_sides.values()):
        raise ValueError("every match must contain exactly one home and one away target")
    if len(rows) != EXPECTED_TARGETS or len(match_meta) != EXPECTED_MATCHES:
        raise ValueError(
            f"fixed cohort mismatch: {len(match_meta)} matches / {len(rows)} targets"
        )
    if len(dates) != EXPECTED_DATES or len(weeks) != EXPECTED_WEEKS:
        raise ValueError(f"fixed cohort mismatch: {len(dates)} dates / {len(weeks)} ISO weeks")
    if any(len(teams[league]) != 20 for league in teams):
        raise ValueError("expected exactly 20 distinct teams in each J1/J2/J3 league")
    folds_by_date: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        folds_by_date[row["date"]].add(row["fold"])
    if any(len(v) != 1 for v in folds_by_date.values()):
        raise ValueError("a target date crosses outer-fold boundaries")
    cohort = {
        "matches": len(match_meta),
        "teamSideTargets": len(rows),
        "dates": len(dates),
        "isoWeeks": len(weeks),
        "weeks": sorted(weeks),
        "targetsByIsoWeek": {
            w: sum(r["week"] == w for r in rows) for w in sorted(weeks)
        },
        "maxSimplexAbsError": max(
            max(abs(sum(r["p6"]) - 1.0), abs(sum(r["p10"]) - 1.0)) for r in rows
        ),
    }
    return rows, cohort


def score(p: tuple[float, ...], actual: int) -> tuple[float, float, float]:
    brier = sum((p[i] - (1.0 if i == actual else 0.0)) ** 2 for i in range(4))
    logloss = -math.log(max(p[actual], LOGLOSS_FLOOR))
    accuracy = 1.0 if max(range(4), key=lambda i: p[i]) == actual else 0.0
    return brier, logloss, accuracy


def metric_means(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot score an empty target set")
    totals = [0.0, 0.0, 0.0]
    for row in rows:
        s6, s10 = score(row["p6"], row["actual"]), score(row["p10"], row["actual"])
        for i in range(3):
            totals[i] += s6[i] - s10[i]
    return {metric: totals[i] / len(rows) for i, metric in enumerate(METRICS)}


def audit(rows: list[dict], cohort: dict, input_sha: str, code_sha: str) -> dict:
    overall = metric_means(rows)
    weeks = sorted({r["week"] for r in rows})
    by_week = []
    loo = []
    for week in weeks:
        wk_rows = [r for r in rows if r["week"] == week]
        wk_diff = metric_means(wk_rows)
        loo_rows = [r for r in rows if r["week"] != week]
        loo_diff = metric_means(loo_rows)
        brier_sum = wk_diff["brier"] * len(wk_rows)
        logloss_sum = wk_diff["logloss"] * len(wk_rows)
        by_week.append({
            "isoWeek": week,
            "targets": len(wk_rows),
            "matches": len({r["match"] for r in wk_rows}),
            "meanDifferenceK6MinusK10": wk_diff,
            "brierDifferenceSum": brier_sum,
            "loglossDifferenceSum": logloss_sum,
            "shareOfOverallBrierDifferenceSum": (
                brier_sum / (overall["brier"] * len(rows))
                if overall["brier"] != 0 else None
            ),
            "shareOfOverallLoglossDifferenceSum": (
                logloss_sum / (overall["logloss"] * len(rows))
                if overall["logloss"] != 0 else None
            ),
        })
        loo.append({
            "excludedIsoWeek": week,
            "excludedTargets": len(wk_rows),
            "excludedMatches": len({r["match"] for r in wk_rows}),
            "remainingTargets": len(loo_rows),
            "differenceK6MinusK10": loo_diff,
            "signReversal": {
                metric: (
                    overall[metric] < 0 < loo_diff[metric]
                    or overall[metric] > 0 > loo_diff[metric]
                )
                for metric in METRICS
            },
        })
    ranges = {
        metric: [
            min(item["differenceK6MinusK10"][metric] for item in loo),
            max(item["differenceK6MinusK10"][metric] for item in loo),
        ]
        for metric in METRICS
    }
    reversals = {
        metric: sum(item["signReversal"][metric] for item in loo)
        for metric in METRICS
    }
    return {
        "schemaVersion": 1,
        "experiment": "GOAL3 fixed-prediction leave-one-ISO-week-out influence audit",
        "scope": "retrospective descriptive influence diagnostic; no refit, no bootstrap, no prospective inference",
        "provenance": {
            "inputCsvSha256": input_sha,
            "scriptSha256": code_sha,
        },
        "cohort": cohort,
        "metricDefinition": {
            "difference": "score(k=6) - score(k=10); negative favors k=6",
            "brier": "four-category multiclass Brier sum, averaged over team-side targets",
            "logloss": "natural log loss with probability floor 1e-12",
            "accuracy": "argmax accuracy difference",
        },
        "overallDifferenceK6MinusK10": overall,
        "byIsoWeek": by_week,
        "leaveOneIsoWeekOut": {
            "differenceRanges": ranges,
            "signReversalCounts": reversals,
            "results": loo,
        },
        "limitations": [
            "Only eight ISO-week clusters are present; leave-one-week-out is descriptive and cannot provide a reliable cluster-level sampling distribution.",
            "Predictions are held fixed; no model refit or hyperparameter reselection is performed after excluding a week.",
            "The cohort is retrospective and does not establish historical prediction-generation time or prospective adoption eligibility.",
            "The two team-side targets within each match are kept together for exclusion and cohort validation, but point metrics remain target-weighted as predeclared.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")
    rows, cohort = load_rows(args.csv)
    result = audit(rows, cohort, sha256_file(args.csv), sha256_file(Path(__file__)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
