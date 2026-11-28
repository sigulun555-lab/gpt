#!/usr/bin/env python3
"""Research-only GOAL3 k=6 vs k=10 league/date-cluster audit.

Input is the immutable all-k outer reconstruction JSON. Differences are
k6-k10; negative values favor k=6. This is retrospective descriptive analysis.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
from collections import defaultdict
from datetime import date as Date
from pathlib import Path
import numpy as np

SCOPES = ("ALL", "J1", "J2", "J3")
METRICS = ("brier", "logloss", "error_rate")


def score(p, y):
    return {
        "brier": float(np.square(p - np.eye(4)[y]).sum()),
        "logloss": float(-np.log(max(float(p[y]), 1e-12))),
        "error_rate": float(1 - int(np.argmax(p) == y)),
    }


def build_targets(report):
    matches = report.get("allKPredictions")
    if not isinstance(matches, list) or not matches:
        raise ValueError("input has no allKPredictions")
    rows, seen = [], set()
    for m in matches:
        if not isinstance(m, dict):
            raise ValueError("all-k match entry must be an object")
        date, comp = m.get("date"), m.get("comp")
        if not isinstance(date, str) or comp not in ("J1", "J2", "J3"):
            raise ValueError("invalid date or competition")
        try:
            parsed_date = Date.fromisoformat(date)
        except ValueError as exc:
            raise ValueError(f"invalid target date: {date!r}") from exc
        if parsed_date.isoformat() != date or parsed_date.year != 2026:
            raise ValueError(f"invalid target date: {date!r}")
        home, away = m.get("home"), m.get("away")
        if not isinstance(home, str) or not home.strip() or not isinstance(away, str) or not away.strip():
            raise ValueError("missing team name")
        home, away = home.strip(), away.strip()
        if home == away:
            raise ValueError("same team listed as home and away")
        mk = (date, comp, home, away)
        if mk in seen:
            raise ValueError(f"duplicate match key: {mk}")
        seen.add(mk)
        for side, goal_key in (("home", "hg"), ("away", "ag")):
            goals = m.get(goal_key)
            if type(goals) is not int or goals < 0:
                raise ValueError("invalid observed goals")
            probs = {}
            for k in ("6.0", "10.0"):
                p = np.asarray(m["byK"][k][side], dtype=float)
                if p.shape != (4,) or not np.isfinite(p).all():
                    raise ValueError("invalid probability vector")
                if (p < 0).any() or (p > 1).any() or abs(float(p.sum()) - 1) > 1e-9:
                    raise ValueError("probability vector outside simplex")
                probs[k] = p
            rows.append({"date": date, "comp": comp, "match": mk,
                         "side": side, "y": min(goals, 3), "p": probs})
    if len(rows) != 2 * len(matches):
        raise AssertionError("team-side expansion mismatch")
    keys = [(r["date"], r["comp"], r["match"], r["side"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate target key")
    return rows


def audit(report, replicates=50000, seed=20261105, input_sha256=None, code_sha256=None, code_git_blob_sha=None):
    if replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    rows = build_targets(report)
    dates = sorted({r["date"] for r in rows})
    matches = report["allKPredictions"]
    if len(matches) != 236 or len(rows) != 472 or len(dates) != 17:
        raise ValueError("expected fixed cohort: 236 matches / 472 targets / 17 dates")
    grouped = defaultdict(list)
    for r in rows:
        a, b = score(r["p"]["6.0"], r["y"]), score(r["p"]["10.0"], r["y"])
        grouped[(r["date"], r["comp"])].append({m: a[m] - b[m] for m in METRICS})
    rng = np.random.Generator(np.random.PCG64(seed))
    result = {
        "cohort": {"matches": len(matches), "teamSideTargets": len(rows),
                   "dates": len(dates), "dateRange": [dates[0], dates[-1]]},
        "bootstrap": {"method": "date-cluster resampling; retain all team-sides per date; target-count weighted",
                      "replicates": replicates, "rng": "NumPy PCG64", "seed": seed,
                      "quantile": "linear", "loglossClip": 1e-12},
        "byScope": {}, "calibration": {},
        "provenance": {
            "inputSha256": input_sha256,
            "codeSha256": code_sha256,
            "codeGitBlobSha": code_git_blob_sha,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "githubSha": os.environ.get("GITHUB_SHA"),
            "githubRef": os.environ.get("GITHUB_REF"),
            "workflowRunId": os.environ.get("GITHUB_RUN_ID"),
            "workflowRunAttempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        },
    }
    for scope in SCOPES:
        used_dates, daily, counts, target_diffs = [], {m: [] for m in METRICS}, [], {m: [] for m in METRICS}
        for date in dates:
            vals = [v for (dt, comp), arr in grouped.items()
                    if dt == date and (scope == "ALL" or comp == scope) for v in arr]
            if not vals:
                continue
            used_dates.append(date)
            counts.append(len(vals))
            for metric in METRICS:
                v = np.asarray([x[metric] for x in vals], dtype=float)
                daily[metric].append(float(v.mean()))
                target_diffs[metric].extend(v.tolist())
        out = {"dates": len(used_dates), "targets": int(sum(counts)), "dateDetails": []}
        for i, date in enumerate(used_dates):
            out["dateDetails"].append({"date": date, "targets": counts[i],
                                       **{m: daily[m][i] for m in METRICS}})
        # Use identical sampled date-cluster indices for Brier and LogLoss
        # within each scope, preserving the predeclared paired resampling.
        shared_picks = rng.integers(0, len(used_dates), size=(replicates, len(used_dates)))
        shared_draws_sha256 = hashlib.sha256(
            np.asarray(shared_picks, dtype="<i8").tobytes()
        ).hexdigest()
        for metric in METRICS:
            vals = np.asarray(target_diffs[metric], dtype=float)
            day = np.asarray(daily[metric], dtype=float)
            weights = np.asarray(counts, dtype=float)
            boot = np.empty(replicates)
            for j, ix in enumerate(shared_picks):
                boot[j] = np.average(day[ix], weights=weights[ix])
            warning = 0.05 if metric == "brier" else (0.10 if metric == "logloss" else None)
            out[metric] = {
                "meanTargetWeightedK6MinusK10": float(vals.mean()),
                "medianTargetDifference": float(np.median(vals)),
                "sdTargetDifference": float(np.std(vals, ddof=1)),
                "minTargetDifference": float(vals.min()), "maxTargetDifference": float(vals.max()),
                "positiveTargetCount": int((vals > 0).sum()),
                "positiveDateCount": int((day > 0).sum()),
                "warningThreshold": warning,
                "datesExceedingWarning": int((day > warning).sum()) if warning is not None else None,
                "dateClusterBootstrap95CI": [float(x) for x in np.quantile(boot, [0.025, 0.975], method="linear")],
                "bootstrapDrawsSha256": shared_draws_sha256,
                "dateMean": float(day.mean()), "dateMedian": float(np.median(day)),
                "dateSd": float(np.std(day, ddof=1)), "dateMin": float(day.min()), "dateMax": float(day.max()),
            }
        result["byScope"][scope] = out
        selected = [r for r in rows if scope == "ALL" or r["comp"] == scope]
        result["calibration"][scope] = {}
        for k in ("6.0", "10.0"):
            result["calibration"][scope][k] = []
            for cat in range(4):
                pred = float(np.mean([r["p"][k][cat] for r in selected]))
                obs = float(np.mean([r["y"] == cat for r in selected]))
                result["calibration"][scope][k].append({
                    "category": cat, "n": len(selected), "meanPredicted": pred,
                    "observedRate": obs, "differencePp": 100 * (pred - obs),
                })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    input_path = Path(a.input)
    input_bytes = input_path.read_bytes()
    code_bytes = Path(__file__).read_bytes()
    code_blob_header = b"blob " + str(len(code_bytes)).encode("ascii") + b"\0"
    result = audit(
        json.loads(input_bytes), a.replicates, a.seed,
        input_sha256=hashlib.sha256(input_bytes).hexdigest(),
        code_sha256=hashlib.sha256(code_bytes).hexdigest(),
        code_git_blob_sha=hashlib.sha1(code_blob_header + code_bytes).hexdigest(),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cohort": result["cohort"],
        "scopes": {s: {m: result["byScope"][s][m] for m in METRICS} for s in SCOPES}},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
