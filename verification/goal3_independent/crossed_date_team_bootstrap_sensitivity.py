#!/usr/bin/env python3
"""Exploratory crossed date/team-cluster sensitivity audit for GOAL3 k-sweep.

This does not replace the pre-registered prospective round-cluster bootstrap.
It resamples both target dates and team identities to stress-test the
retrospective 236-match / 472-team-side all-k reconstruction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import date
import numpy as np

K_VALUES = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0,
            15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)
LOGLOSS_FLOOR = 1e-12


def build_arrays(report: dict, strict: bool = True) -> dict:
    matches = report.get("allKPredictions")
    if not isinstance(matches, list) or not matches:
        raise ValueError("input has no allKPredictions")
    if any(not isinstance(m.get("date"), str) or not m.get("date") for m in matches):
        raise ValueError("invalid target date")
    if any(not isinstance(t, str) or not t for m in matches for t in (m.get("home"), m.get("away"))):
        raise ValueError("invalid team name")
    dates = sorted({m["date"] for m in matches})
    teams = sorted({t for m in matches for t in (m["home"], m["away"])})
    date_index = {d: i for i, d in enumerate(dates)}
    team_index = {t: i for i, t in enumerate(teams)}
    match_keys, target_keys, rows, fold_by_date = set(), set(), [], {}
    for m in matches:
        date, comp, home, away = m.get("date"), m.get("comp"), m.get("home"), m.get("away")
        if comp not in ("J1", "J2", "J3") or not isinstance(date, str):
            raise ValueError("invalid date or competition")
        if not isinstance(home, str) or not home or not isinstance(away, str) or not away or home == away:
            raise ValueError("invalid home/away team")
        mk = (date, comp, home, away)
        if mk in match_keys:
            raise ValueError(f"duplicate match key: {mk}")
        match_keys.add(mk)
        fold = m.get("fold")
        if strict and not isinstance(fold, int):
            raise ValueError("missing outer fold")
        if isinstance(fold, int):
            if date in fold_by_date and fold_by_date[date] != fold:
                raise ValueError("a target date crosses outer-fold boundaries")
            fold_by_date[date] = fold
        by_k = m.get("byK")
        if not isinstance(by_k, dict):
            raise ValueError("missing byK")
        for side, team, opponent, goal_key in (
            ("home", home, away, "hg"), ("away", away, home, "ag")
        ):
            goals = m.get(goal_key)
            if not isinstance(goals, int) or goals < 0:
                raise ValueError("invalid observed goals")
            key = (*mk, side)
            if key in target_keys:
                raise ValueError(f"duplicate target key: {key}")
            target_keys.add(key)
            probs = []
            for k in K_VALUES:
                vec = np.asarray(by_k.get(f"{k:.1f}", {}).get(side), dtype=float)
                if vec.shape != (4,) or not np.isfinite(vec).all():
                    raise ValueError(f"invalid probability vector for k={k}, {side}")
                if np.any(vec < 0) or np.any(vec > 1) or abs(float(vec.sum()) - 1.0) > 1e-9:
                    raise ValueError(f"probability vector outside simplex for k={k}, {side}")
                probs.append(vec)
            rows.append((date_index[date], team_index[team], team_index[opponent],
                         comp, min(goals, 3), np.asarray(probs, dtype=float)))
    if len(rows) != 2 * len(matches):
        raise ValueError("home/away target expansion mismatch")
    if strict and (len(matches), len(rows), len(dates), len(teams)) != (236, 472, 17, 60):
        raise ValueError("expected fixed cohort 236 matches / 472 targets / 17 dates / 60 teams")
    date_ids = np.asarray([r[0] for r in rows], dtype=np.int16)
    team_ids = np.asarray([r[1] for r in rows], dtype=np.int16)
    opponent_ids = np.asarray([r[2] for r in rows], dtype=np.int16)
    leagues = np.asarray([r[3] for r in rows], dtype=object)
    y = np.asarray([r[4] for r in rows], dtype=np.int8)
    p = np.stack([r[5] for r in rows])
    return {"matches": len(matches), "dates": dates, "teams": teams,
            "date_ids": date_ids, "team_ids": team_ids,
            "opponent_ids": opponent_ids, "leagues": leagues, "y": y, "p": p}


def _scores(arrays: dict) -> dict:
    y, p = arrays["y"], arrays["p"]
    one = np.eye(4, dtype=float)[y]
    brier = np.square(p - one[:, None, :]).sum(axis=2)
    selected = p[np.arange(len(y))[:, None], np.arange(len(K_VALUES))[None, :], y[:, None]]
    logloss = -np.log(np.clip(selected, LOGLOSS_FLOOR, 1.0))
    accuracy = (p.argmax(axis=2) == y[:, None]).astype(float)
    return {"brier": brier, "logloss": logloss, "accuracy": accuracy}


def _weights(rng, arrays, reps, mode):
    nd, nt = len(arrays["dates"]), len(arrays["teams"])
    dcount = rng.multinomial(nd, np.full(nd, 1.0 / nd), size=reps)
    tcount = rng.multinomial(nt, np.full(nt, 1.0 / nt), size=reps)
    w = dcount[:, arrays["date_ids"]].astype(float) * tcount[:, arrays["team_ids"]]
    if mode == "date_x_home_away_team_product":
        w *= tcount[:, arrays["opponent_ids"]]
    elif mode != "date_x_target_team":
        raise ValueError(f"unknown bootstrap mode: {mode}")
    return w


def _bootstrap_scores(arrays, scores, reps, seed, mode, chunk=500):
    rng = np.random.Generator(np.random.PCG64(seed))
    nk = scores["brier"].shape[1]
    sims = {metric: np.empty((reps, nk), dtype=float) for metric in scores}
    min_mass = math.inf
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        weights = _weights(rng, arrays, count, mode)
        denom = weights.sum(axis=1)
        if np.any(denom <= 0):
            raise ValueError(f"empty overall bootstrap replicate in {mode}")
        min_mass = min(min_mass, float(denom.min()))
        for metric, values in scores.items():
            sims[metric][start:start + count] = (weights @ values) / denom[:, None]
    point = {metric: values.mean(axis=0) for metric, values in scores.items()}
    out = {"method": mode, "rng": "NumPy PCG64", "seed": seed,
           "replicates": reps, "quantile": "linear percentile",
           "minimumWeightedTargetMass": min_mass,
           "bestKFrequency": {}, "k6MinusK10": {}}
    i6, i10 = K_VALUES.index(6.0), K_VALUES.index(10.0)
    for metric, values in sims.items():
        winner = values.argmax(axis=1) if metric == "accuracy" else values.argmin(axis=1)
        out["bestKFrequency"][metric] = {
            f"{k:g}": float(np.mean(winner == i)) for i, k in enumerate(K_VALUES)
        }
        delta = values[:, i6] - values[:, i10]
        out["k6MinusK10"][metric] = {
            "pointDifference": float(point[metric][i6] - point[metric][i10]),
            "percentile95CI": [float(x) for x in np.quantile(delta, [0.025, 0.975], method="linear")],
        }
    reference_index = K_VALUES.index(10.0)
    contrasts, meta = [], []
    for metric in ("brier", "logloss"):
        values = sims[metric]
        for ki, k in enumerate(K_VALUES):
            if ki == reference_index:
                continue
            delta = values[:, ki] - values[:, reference_index]
            observed = float(point[metric][ki] - point[metric][reference_index])
            se = float(delta.std(ddof=1))
            if not np.isfinite(se) or se <= 0:
                raise ValueError(f"invalid crossed-bootstrap SE for {metric}, k={k:g}")
            contrasts.append((delta - observed) / se)
            meta.append((metric, k, observed, se))
    critical = float(np.quantile(np.max(np.abs(np.column_stack(contrasts)), axis=1),
                                 0.95, method="linear"))
    k6_adjusted = {}
    for metric, k, observed, se in meta:
        if k == 6.0:
            k6_adjusted[metric] = {
                "differenceK6MinusK10": observed,
                "simultaneous95Band": [observed - critical * se, observed + critical * se],
            }
    out["selectionAdjusted"] = {
        "family": "15 non-reference k contrasts x 2 primary metrics against k=10",
        "maxAbsTCriticalValue": critical,
        "k6MinusK10": k6_adjusted,
    }
    return out



def _calendar_moving_block_bootstrap(arrays, scores, reps, seed, block_length, chunk=500):
    parsed = [date.fromisoformat(value) for value in arrays["dates"]]
    first, last = min(parsed), max(parsed)
    n_calendar_days = (last - first).days + 1
    date_offsets = np.asarray([(value - first).days for value in parsed], dtype=np.int16)
    block_count = math.ceil(n_calendar_days / block_length)
    rng = np.random.Generator(np.random.PCG64(seed))
    nk = scores["brier"].shape[1]
    sims = {metric: np.empty((reps, nk), dtype=float) for metric in scores}
    min_mass = math.inf
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        starts = rng.integers(0, n_calendar_days, size=(count, block_count))
        day_counts = np.zeros((count, n_calendar_days), dtype=np.int16)
        for block in range(block_count):
            for offset in range(block_length):
                column = block * block_length + offset
                if column < n_calendar_days:
                    day_counts[np.arange(count), (starts[:, block] + offset) % n_calendar_days] += 1
        weights = day_counts[:, date_offsets[arrays["date_ids"]]].astype(float)
        denom = weights.sum(axis=1)
        if np.any(denom <= 0):
            raise ValueError(f"empty calendar-block bootstrap replicate for {block_length} days")
        min_mass = min(min_mass, float(denom.min()))
        for metric, values in scores.items():
            sims[metric][start:start + count] = (weights @ values) / denom[:, None]
    point = {metric: values.mean(axis=0) for metric, values in scores.items()}
    i6, i10 = K_VALUES.index(6.0), K_VALUES.index(10.0)
    out = {"blockLengthCalendarDays": block_length, "calendarSpanDays": n_calendar_days,
           "dateRange": [first.isoformat(), last.isoformat()], "rng": "NumPy PCG64",
           "seed": seed, "replicates": reps, "quantile": "linear percentile",
           "minimumWeightedTargetMass": min_mass, "k6MinusK10": {}, "bestKFrequency": {}}
    for metric, values in sims.items():
        delta = values[:, i6] - values[:, i10]
        winner = values.argmax(axis=1) if metric == "accuracy" else values.argmin(axis=1)
        out["k6MinusK10"][metric] = {
            "pointDifference": float(point[metric][i6] - point[metric][i10]),
            "percentile95CI": [float(x) for x in np.quantile(delta, [0.025, 0.975], method="linear")],
        }
        out["bestKFrequency"][metric] = {
            f"{k:g}": float(np.mean(winner == i)) for i, k in enumerate(K_VALUES)
        }
    contrasts, meta = [], []
    for metric in ("brier", "logloss"):
        for ki, k in enumerate(K_VALUES):
            if k == 10.0:
                continue
            delta = sims[metric][:, ki] - sims[metric][:, i10]
            observed = float(point[metric][ki] - point[metric][i10])
            se = float(delta.std(ddof=1))
            if not np.isfinite(se) or se <= 0:
                raise ValueError(f"invalid calendar-bootstrap SE for {metric}, k={k:g}")
            contrasts.append((delta - observed) / se)
            meta.append((metric, k, observed, se))
    critical = float(np.quantile(np.max(np.abs(np.column_stack(contrasts)), axis=1),
                                 0.95, method="linear"))
    out["selectionAdjusted"] = {"family": "15 non-reference k contrasts x 2 primary metrics against k=10",
                                "maxAbsTCriticalValue": critical, "k6MinusK10": {}}
    for metric, k, observed, se in meta:
        if k == 6.0:
            out["selectionAdjusted"]["k6MinusK10"][metric] = {
                "differenceK6MinusK10": observed,
                "simultaneous95Band": [observed - critical * se, observed + critical * se],
            }
    return out

def _calibration_and_poisson(arrays, reps, seed, chunk=500):
    rng = np.random.Generator(np.random.PCG64(seed))
    y, p, leagues = arrays["y"], arrays["p"], arrays["leagues"]
    present_scopes = ["ALL"] + [s for s in ("J1", "J2", "J3") if np.any(leagues == s)]
    ks = (6.0, 10.0)
    kidx = {k: K_VALUES.index(k) for k in ks}
    masks = {scope: (np.ones(len(y), dtype=bool) if scope == "ALL" else leagues == scope)
             for scope in present_scopes}
    cal_meta, cal_point = [], []
    for scope in present_scopes:
        mask = masks[scope]
        for k in ks:
            for cat in range(4):
                gap = p[mask, kidx[k], cat].mean() - np.mean(y[mask] == cat)
                cal_meta.append((scope, k, cat, mask))
                cal_point.append(float(gap * 100.0))
    cal_point = np.asarray(cal_point)
    pois = {}
    for k in ks:
        lam = -np.log(np.clip(p[:, kidx[k], 0], 1e-15, 1.0))
        if np.any(~np.isfinite(lam)) or np.any(lam <= 0):
            raise ValueError(f"non-positive Poisson mean for k={k:g}")
        pois[k] = np.square((y.astype(float) - lam) / np.sqrt(lam))
    pois_meta, pois_point = [], []
    for scope in present_scopes:
        mask = masks[scope]
        for k in ks:
            pois_meta.append((scope, k, mask))
            pois_point.append(float(pois[k][mask].mean()))
    pois_point = np.asarray(pois_point)
    cal_boot = np.empty((reps, len(cal_meta)), dtype=float)
    pois_boot = np.empty((reps, len(pois_meta)), dtype=float)
    min_scope_mass = math.inf
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        weights = _weights(rng, arrays, count, "date_x_target_team")
        for j, (scope, k, cat, mask) in enumerate(cal_meta):
            ww = weights[:, mask]
            den = ww.sum(axis=1)
            if np.any(den <= 0):
                raise ValueError(f"empty calibration replicate for {scope}")
            min_scope_mass = min(min_scope_mass, float(den.min()))
            pred, obs = p[mask, kidx[k], cat], (y[mask] == cat).astype(float)
            cal_boot[start:start + count, j] = ((ww @ pred - ww @ obs) / den) * 100.0
        for j, (scope, k, mask) in enumerate(pois_meta):
            ww = weights[:, mask]
            den = ww.sum(axis=1)
            if np.any(den <= 0):
                raise ValueError(f"empty Poisson replicate for {scope}")
            pois_boot[start:start + count, j] = (ww @ pois[k][mask]) / den
    cal_se = cal_boot.std(axis=0, ddof=1)
    cal_t = (cal_boot - cal_point) / cal_se
    cal_critical = float(np.quantile(np.max(np.abs(cal_t), axis=1), 0.95, method="linear"))
    cal_intervals = []
    for j, (scope, k, cat, mask) in enumerate(cal_meta):
        cal_intervals.append({"scope": scope, "k": k, "category": cat, "n": int(mask.sum()),
                              "gapPp": float(cal_point[j]),
                              "simultaneous95Band": [float(cal_point[j] - cal_critical * cal_se[j]),
                                                      float(cal_point[j] + cal_critical * cal_se[j])]})
    pois_se = pois_boot.std(axis=0, ddof=1)
    pois_t = (pois_boot - pois_point) / pois_se
    pois_critical = float(np.quantile(np.max(np.abs(pois_t), axis=1), 0.95, method="linear"))
    pois_intervals = []
    for j, (scope, k, mask) in enumerate(pois_meta):
        pois_intervals.append({"scope": scope, "k": k, "n": int(mask.sum()),
                               "dispersionIndex": float(pois_point[j]),
                               "joint95Band": [float(pois_point[j] - pois_critical * pois_se[j]),
                                               float(pois_point[j] + pois_critical * pois_se[j])]})
    return {
        "method": "date x target-team crossed cluster bootstrap; target weights are date multiplicity times own-team multiplicity",
        "seed": seed, "replicates": reps, "rng": "NumPy PCG64", "quantile": "linear",
        "minimumWeightedScopeMass": min_scope_mass,
        "calibration": {"familySize": len(cal_meta), "maxTcritical": cal_critical,
                        "zeroIncludedCount": sum(x["simultaneous95Band"][0] <= 0 <= x["simultaneous95Band"][1] for x in cal_intervals),
                        "intervals": cal_intervals},
        "poisson": {"familySize": len(pois_meta), "maxTcritical": pois_critical,
                    "referenceValue": 1.0,
                    "referenceOneIncludedCount": sum(x["joint95Band"][0] <= 1 <= x["joint95Band"][1] for x in pois_intervals),
                    "intervals": pois_intervals},
    }


def audit(report, replicates=50000, seed=20261106, strict=True):
    if replicates < 1000:
        raise ValueError("at least 1000 replicates are required")
    arrays = build_arrays(report, strict=strict)
    scores = _scores(arrays)
    date_team = _bootstrap_scores(arrays, scores, replicates, seed, "date_x_target_team")
    date_dyad = _bootstrap_scores(arrays, scores, replicates, seed, "date_x_home_away_team_product")
    calendar_7 = _calendar_moving_block_bootstrap(arrays, scores, replicates, seed, 7)
    calendar_14 = _calendar_moving_block_bootstrap(arrays, scores, replicates, seed, 14)
    cal_pois = _calibration_and_poisson(arrays, replicates, seed)
    return {
        "experiment": "GOAL3 exploratory crossed date/team-cluster sensitivity; not prospective adoption evidence",
        "cohort": {"matches": arrays["matches"], "teamSideTargets": len(arrays["y"]),
                   "dates": len(arrays["dates"]), "teams": len(arrays["teams"]),
                   "dateRange": [arrays["dates"][0], arrays["dates"][-1]]},
        "pointMetricsByK": {
            f"{k:g}": {metric: float(scores[metric][:, i].mean()) for metric in scores}
            for i, k in enumerate(K_VALUES)
        },
        "crossedBootstrap": {"date_x_target_team": date_team,
                             "date_x_home_away_team_product": date_dyad},
        "calendarMovingBlockBootstrap": {"7_day": calendar_7, "14_day": calendar_14},
        "dateTeamCalibrationAndPoisson": cal_pois,
        "limitations": [
            "Exploratory sensitivity only; not the pre-registered prospective round-cluster bootstrap.",
            "Only 17 target dates and 60 teams are available; bootstrap tails remain uncertain.",
            "Date and team resampling dimensions are treated as crossed clusters; this does not model all opponent/network dependence.",
            "The dyadic product-weight variant is reported for overall score/selection only; league subgroup draws can be empty and are not used there.",
            "History is community-source and has one shared fetchedAt timestamp; first-known historical availability is not established.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()
    raw = Path(args.input).read_bytes()
    report = json.loads(raw.decode("utf-8"))
    result = audit(report, args.replicates, args.seed, strict=True)
    result["provenance"] = {"inputSha256": hashlib.sha256(raw).hexdigest()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cohort": result["cohort"],
                      "dateTeam": result["crossedBootstrap"]["date_x_target_team"]["k6MinusK10"],
                      "dateDyad": result["crossedBootstrap"]["date_x_home_away_team_product"]["k6MinusK10"],
                      "calibration": {"maxT": result["dateTeamCalibrationAndPoisson"]["calibration"]["maxTcritical"],
                                      "zeroIncluded": result["dateTeamCalibrationAndPoisson"]["calibration"]["zeroIncludedCount"]},
                      "poisson": {"maxT": result["dateTeamCalibrationAndPoisson"]["poisson"]["maxTcritical"],
                                  "oneIncluded": result["dateTeamCalibrationAndPoisson"]["poisson"]["referenceOneIncludedCount"]}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
