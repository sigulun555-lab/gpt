#!/usr/bin/env python3
"""Descriptive GOAL3 k=6 vs k=10 paired score audit by league and date.

Uses the frozen all-k outer prediction JSON. Positive k6-k10 differences mean
k=6 scores worse. This is a retrospective descriptive audit, not adoption evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

K6, K10 = "6.0", "10.0"
BRIER_WARNING, LOGLOSS_WARNING = 0.05, 0.10  # provisional triggers only
PROB_FLOOR = 1e-12


def score(probabilities, goals):
    if isinstance(goals, bool) or not isinstance(goals, int) or goals < 0:
        raise ValueError(f"invalid goals: {goals!r}")
    if len(probabilities) != 4:
        raise ValueError("expected four category probabilities")
    p = [float(x) for x in probabilities]
    if any(not math.isfinite(x) or x < 0 or x > 1 for x in p):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if abs(sum(p) - 1.0) > 1e-6:
        raise ValueError("probabilities must sum to one")
    y = min(goals, 3)
    brier = sum((p[i] - (i == y)) ** 2 for i in range(4))
    logloss = -math.log(max(p[y], PROB_FLOOR))
    accuracy = int(max(range(4), key=lambda i: p[i]) == y)
    return brier, logloss, accuracy


def summary(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "sampleSd": None,
                "positiveCount": 0, "min": None, "max": None}
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "sampleSd": statistics.stdev(values) if len(values) > 1 else None,
            "positiveCount": sum(x > 0 for x in values),
            "min": min(values), "max": max(values)}


def build_report(document, input_sha256):
    matches = document.get("allKPredictions")
    if not isinstance(matches, list) or not matches:
        raise ValueError("allKPredictions must be a non-empty list")
    targets, seen = [], set()
    for row in matches:
        key = tuple(str(row[k]) for k in ("date", "comp", "home", "away"))
        if key in seen:
            raise ValueError(f"duplicate match: {key}")
        seen.add(key)
        by_k = row.get("byK", {})
        if K6 not in by_k or K10 not in by_k:
            raise ValueError("each match needs both k=6 and k=10")
        for side, goals in (("home", row["hg"]), ("away", row["ag"])):
            a, b = score(by_k[K6][side], goals), score(by_k[K10][side], goals)
            targets.append({"date": key[0], "league": key[1], "match": key,
                            "brier": a[0]-b[0], "logloss": a[1]-b[1],
                            "accuracy": a[2]-b[2]})

    league_groups, league_dates, all_dates = defaultdict(list), defaultdict(list), defaultdict(list)
    for x in targets:
        league_groups[x["league"]].append(x)
        league_dates[(x["league"], x["date"])].append(x)
        all_dates[x["date"]].append(x)

    def date_rows(groups):
        out = []
        for key, group in sorted(groups.items()):
            date = key[-1] if isinstance(key, tuple) else key
            out.append({"date": date,
                        "matches": len({x["match"] for x in group}),
                        "teamSideTargets": len(group),
                        **{metric: statistics.mean(x[metric] for x in group)
                           for metric in ("brier", "logloss", "accuracy")}})
        return out

    by_league = {}
    for league, group in sorted(league_groups.items()):
        days = date_rows({k: v for k, v in league_dates.items() if k[0] == league})
        by_league[league] = {
            "matches": len({x["match"] for x in group}),
            "teamSideTargets": len(group), "evaluationDates": len(days),
            "targetLevel": {m: summary([x[m] for x in group]) for m in ("brier", "logloss", "accuracy")},
            "dateLevel": {
                **{m: summary([x[m] for x in days]) for m in ("brier", "logloss", "accuracy")},
                "brierWarningExceedCount": sum(x["brier"] > BRIER_WARNING for x in days),
                "logLossWarningExceedCount": sum(x["logloss"] > LOGLOSS_WARNING for x in days),
                "dailyRows": days,
            },
        }
    overall_days = date_rows(all_dates)
    return {
        "method": {
            "scope": "GOAL3 retrospective all-k outer cohort only",
            "comparison": "k=6 minus k=10; positive means k=6 is worse",
            "brier": "four-category squared-error sum per team-side, then mean",
            "logLoss": "natural logarithm; probability floor 1e-12",
            "dateAggregation": "mean paired team-side difference within league/date",
            "warningThresholds": {"brierStrictlyGreaterThan": BRIER_WARNING,
                                  "logLossStrictlyGreaterThan": LOGLOSS_WARNING,
                                  "status": "provisional descriptive triggers, not adoption criteria"},
            "inference": "descriptive only; no league-level inferential claim",
        },
        "provenance": {"inputSha256": input_sha256, "experiment": document.get("experiment"),
                       "cohort": document.get("cohort", {})},
        "totals": {"matches": len(seen), "teamSideTargets": len(targets), "dates": len(all_dates)},
        "overallDateLevel": {
            **{m: summary([x[m] for x in overall_days]) for m in ("brier", "logloss", "accuracy")},
            "brierWarningExceedCount": sum(x["brier"] > BRIER_WARNING for x in overall_days),
            "logLossWarningExceedCount": sum(x["logloss"] > LOGLOSS_WARNING for x in overall_days),
            "dailyRows": overall_days,
        },
        "byLeague": by_league,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raw = Path(args.input).read_bytes()
    report = build_report(json.loads(raw), hashlib.sha256(raw).hexdigest())
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "inputSha256": report["provenance"]["inputSha256"],
                      **report["totals"], "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
