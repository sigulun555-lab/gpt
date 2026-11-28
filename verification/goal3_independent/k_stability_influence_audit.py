#!/usr/bin/env python3
"""Descriptive leave-one-cluster-out influence audit for GOAL3 k=6 vs k=10.

Predictions are held fixed; no model refit is performed. Negative Brier and
LogLoss differences favor k=6; positive accuracy differences favor k=6.
This is influence sensitivity, not a confidence interval or hypothesis test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import date as Date
from pathlib import Path
import numpy as np

LEAGUES = ("J1", "J2", "J3")
METRICS = ("brier", "logloss", "accuracy")


def _score(p, y):
    one = np.zeros(4, dtype=float)
    one[y] = 1.0
    return (
        float(np.square(p - one).sum()),
        float(-math.log(max(float(p[y]), 1e-12))),
        float(int(np.argmax(p) == y)),
    )


def build_rows(report):
    matches = report.get("allKPredictions")
    if not isinstance(matches, list) or len(matches) != 236:
        raise ValueError("expected fixed 236-match allKPredictions cohort")
    rows, seen, dates = [], set(), set()
    teams = {league: set() for league in LEAGUES}
    for m in matches:
        if not isinstance(m, dict):
            raise ValueError("match entry must be an object")
        date, comp = m.get("date"), m.get("comp")
        if not isinstance(date, str) or comp not in LEAGUES:
            raise ValueError("invalid date or league")
        try:
            parsed = Date.fromisoformat(date)
        except ValueError as exc:
            raise ValueError(f"invalid date {date!r}") from exc
        if parsed.isoformat() != date or parsed.year != 2026:
            raise ValueError(f"noncanonical/non-2026 date {date!r}")
        home, away = m.get("home"), m.get("away")
        if not isinstance(home, str) or not home.strip() or not isinstance(away, str) or not away.strip():
            raise ValueError("missing team name")
        home, away = home.strip(), away.strip()
        if home == away:
            raise ValueError("home and away teams collide")
        match_key = (date, comp, home, away)
        if match_key in seen:
            raise ValueError(f"duplicate match {match_key}")
        seen.add(match_key)
        dates.add(date)
        teams[comp].update((home, away))
        for side, team, opponent, field in (
            ("home", home, away, "hg"), ("away", away, home, "ag")
        ):
            goals = m.get(field)
            if type(goals) is not int or goals < 0:
                raise ValueError("observed goals must be nonnegative integers")
            y = min(goals, 3)
            scores = {}
            for k in ("6.0", "10.0"):
                try:
                    p = np.asarray(m["byK"][k][side], dtype=float)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"missing probability vector: k={k}, side={side}") from exc
                if p.shape != (4,) or not np.isfinite(p).all():
                    raise ValueError("probability vector must have four finite values")
                if (p < 0).any() or (p > 1).any() or abs(float(p.sum()) - 1.0) > 1e-9:
                    raise ValueError("probability vector outside four-category simplex")
                scores[k] = _score(p, y)
            rows.append({
                "date": date, "league": comp, "team": (comp, team),
                "opponent": (comp, opponent), "match": match_key,
                "d_brier": scores["6.0"][0] - scores["10.0"][0],
                "d_logloss": scores["6.0"][1] - scores["10.0"][1],
                "d_accuracy": scores["6.0"][2] - scores["10.0"][2],
            })
    if len(rows) != 472 or len(dates) != 17:
        raise ValueError(f"expected 472 targets/17 dates, got {len(rows)}/{len(dates)}")
    for league in LEAGUES:
        if len(teams[league]) != 20:
            raise ValueError(f"expected 20 unique teams in {league}, got {len(teams[league])}")
    return rows


def _loo(rows, groups, kind, full):
    records = []
    for name, excluded_indices in groups.items():
        excluded = set(excluded_indices)
        remain = [r for i, r in enumerate(rows) if i not in excluded]
        if not remain:
            raise ValueError(f"leave-one-{kind}-out leaves no targets: {name}")
        rec = {"cluster": name, "removedTargets": len(excluded), "remainingTargets": len(remain)}
        for metric in METRICS:
            rec[f"d_{metric}"] = float(np.mean([r[f"d_{metric}"] for r in remain]))
            rec[f"influence_{metric}"] = rec[f"d_{metric}"] - full[metric]
        records.append(rec)
    out = {"clusterCount": len(records), "records": records, "signFlipCounts": {}}
    for metric in ("brier", "logloss"):
        key = f"d_{metric}"
        values = [r[key] for r in records]
        if full[metric] < 0:
            flips = sum(v >= 0 for v in values)
        elif full[metric] > 0:
            flips = sum(v <= 0 for v in values)
        else:
            flips = None
        out["signFlipCounts"][metric] = flips
        out[f"range_{metric}"] = [float(min(values)), float(max(values))]
        out[f"maxAbsInfluence_{metric}"] = float(max(abs(r[f"influence_{metric}"]) for r in records))
        out[f"mostInfluential_{metric}"] = sorted(
            records, key=lambda r: abs(r[f"influence_{metric}"]), reverse=True
        )[:5]
    out["range_accuracy"] = [
        float(min(r["d_accuracy"] for r in records)),
        float(max(r["d_accuracy"] for r in records)),
    ]
    return out


def audit(report):
    rows = build_rows(report)
    full = {metric: float(np.mean([r[f"d_{metric}"] for r in rows])) for metric in METRICS}
    dates, teams, matches = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        dates[row["date"]].append(i)
        teams[row["team"]].append(i)
        teams[row["opponent"]].append(i)
        matches[row["match"]].append(i)
    def keystr(k):
        return "|".join(k) if isinstance(k, tuple) else str(k)
    return {
        "experiment": "GOAL3 retrospective fixed-prediction k=6 vs k=10 influence sensitivity",
        "cohort": {"matches": 236, "teamSideTargets": 472, "dates": 17,
                   "teamsByLeague": {"J1": 20, "J2": 20, "J3": 20}},
        "definition": {
            "difference": "score(k=6) - score(k=10)",
            "brier": "four-category squared-error sum per team-side target (0..2), then mean",
            "logloss": "natural logarithm; probability floor 1e-12",
            "accuracy": "correct(k=6) - correct(k=10)",
            "interpretation": "negative Brier/LogLoss and positive accuracy favor k=6",
        },
        "overall": full,
        "leaveOneDateOut": _loo(rows, {keystr(k): v for k, v in dates.items()}, "date", full),
        "leaveOneTeamOut": _loo(rows, {keystr(k): v for k, v in teams.items()}, "team", full),
        "leaveOneMatchOut": _loo(rows, {keystr(k): v for k, v in matches.items()}, "match", full),
        "limitations": [
            "Predictions are held fixed; no model refit follows exclusions.",
            "Leave-one-cluster-out values are descriptive influence diagnostics, not CIs or tests.",
            "Retrospective first-known result availability remains unproven.",
            "This cohort is not prospective adoption evidence.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw = Path(args.input).read_bytes()
    result = audit(json.loads(raw.decode("utf-8")))
    result["provenance"] = {
        "inputSha256": hashlib.sha256(raw).hexdigest(),
        "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "cohort": result["cohort"], "overall": result["overall"],
        "dateSignFlips": result["leaveOneDateOut"]["signFlipCounts"],
        "teamSignFlips": result["leaveOneTeamOut"]["signFlipCounts"],
        "matchSignFlips": result["leaveOneMatchOut"]["signFlipCounts"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
