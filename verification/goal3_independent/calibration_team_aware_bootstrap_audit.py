#!/usr/bin/env python3
"""Research-only GOAL3 date/team pigeonhole calibration sensitivity.

A replicate samples 17 dates with replacement and samples 20 teams with
replacement within each league. Each match weight is date multiplicity times
both endpoint-team multiplicities; that same weight is assigned to home and
away targets. This is an exploratory multiway pigeonhole bootstrap, not an
exact randomization distribution or prospective evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import date as Date
from pathlib import Path

import numpy as np


LEAGUES = ("J1", "J2", "J3")
KS = (6.0, 10.0)
CATEGORIES = (0, 1, 2, 3)
BINS = (0.0, .1, .2, .3, .4, .5, .6, 1.0000000001)
MIN_BIN_N = 20
EPS = 1e-6
GRAD_TOL = 1e-6
MIN_EIGEN = 1e-8
MAX_COEF = 50.0
MIN_VALID = .95


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()



def flatten_matches(report):
    matches = report.get("allKPredictions")
    if not isinstance(matches, list) or len(matches) != 236:
        raise ValueError("expected exactly 236 matches")
    dates = sorted({m.get("date") for m in matches if isinstance(m, dict)})
    if len(dates) != 17:
        raise ValueError("expected exactly 17 target dates")
    teams = {
        league: sorted({
            team.strip() for m in matches if isinstance(m, dict) and m.get("comp") == league
            for team in (m.get("home"), m.get("away"))
            if isinstance(team, str) and team.strip()
        }) for league in LEAGUES
    }
    if any(len(teams[x]) != 20 for x in LEAGUES):
        raise ValueError("expected exactly 20 teams in each league")
    rows, seen = [], set()
    for i, m in enumerate(matches):
        day, league = m.get("date"), m.get("comp")
        try:
            parsed = Date.fromisoformat(day)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid date at match {i}") from exc
        if parsed.isoformat() != day or parsed.year != 2026 or league not in LEAGUES:
            raise ValueError(f"invalid date/league at match {i}")
        home, away = m.get("home"), m.get("away")
        if not isinstance(home, str) or not isinstance(away, str) or not home.strip() or not away.strip():
            raise ValueError(f"missing team at match {i}")
        home, away = home.strip(), away.strip()
        key = (day, league, home, away)
        if key in seen or home == away or home not in teams[league] or away not in teams[league]:
            raise ValueError(f"duplicate or invalid match at {i}")
        seen.add(key)
        by_k = m.get("byK")
        if not isinstance(by_k, dict):
            raise ValueError(f"missing byK at match {i}")
        for side, goal_key in (("home", "hg"), ("away", "ag")):
            goals = m.get(goal_key)
            if type(goals) is not int or goals < 0:
                raise ValueError(f"invalid goals at match {i}/{side}")
            probs = {}
            for k in KS:
                v = np.asarray(by_k.get(f"{k:.1f}", {}).get(side), dtype=float)
                if v.shape != (4,) or not np.isfinite(v).all() or np.any(v < 0) or np.any(v > 1) or abs(v.sum()-1) > 1e-9:
                    raise ValueError(f"invalid probability vector at match {i}/{side}/k={k}")
                probs[k] = v
            rows.append({"date": day, "league": league, "actual": min(goals, 3), "probabilities": probs})
    if len(rows) != 472:
        raise ValueError("expected 472 team-side targets")
    return matches, rows, dates, teams
            raise ValueError("duplicate or invalid match identity")
        seen.add(key)
        match_rows.append({"date": day, "league": league, "home": home, "away": away})
    return match_rows, rows, dates, teams


def draw_cluster_weights(matches, dates, teams, n, rng):
    di = {d: i for i, d in enumerate(dates)}
    ti = {lg: {t: i for i, t in enumerate(ts)} for lg, ts in teams.items()}
    dd = rng.integers(0, len(dates), size=(n, len(dates)))
    dm = np.column_stack([(dd == j).sum(axis=1) for j in range(len(dates))]).astype(np.int16)
    tm = {}
    for lg, ts in teams.items():
        draws = rng.integers(0, len(ts), size=(n, len(ts)))
        tm[lg] = np.column_stack([(draws == j).sum(axis=1) for j in range(len(ts))]).astype(np.int16)
    mw = np.empty((n, len(matches)), dtype=float)
    for j, m in enumerate(matches):
        lg = m["league"]
        mw[:, j] = dm[:, di[m["date"]]] * tm[lg][:, ti[lg][m["home"]]] * tm[lg][:, ti[lg][m["away"]]]
    return dm, tm, mw


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def fit_weighted_logistic_batch(x, y, w, initial=None, max_iter=100, tol=1e-7):
    b = w.shape[0]
    theta = np.tile(np.asarray(initial if initial is not None else [0.0, 1.0]), (b, 1)).astype(float)
    positive = np.isfinite(w).all(axis=1) & (w.sum(axis=1) > 0)
    failed = np.zeros(b, bool)
    near_step = np.zeros(b, bool)
    line_fail = np.zeros(b, bool)
    iterations = np.zeros(b, np.int16)

    def loss(t):
        z = t[:, 0, None] + t[:, 1, None] * x[None, :]
        return np.sum(w * (np.logaddexp(0.0, z) - y[None, :] * z), axis=1)

    for _ in range(max_iter):
        z = theta[:, 0, None] + theta[:, 1, None] * x[None, :]
        p = sigmoid(z)
        r = w * (p - y[None, :])
        g0, g1 = r.sum(axis=1), (r * x[None, :]).sum(axis=1)
        gn = np.maximum(np.abs(g0), np.abs(g1))
        need = positive & ~failed & (gn >= tol)
        if not need.any():
            break
        q = w * p * (1.0 - p)
        h00, h01, h11 = q.sum(axis=1), (q * x[None, :]).sum(axis=1), (q * (x*x)[None, :]).sum(axis=1)
        disc = np.sqrt(np.maximum((h00-h11)**2 + 4*h01*h01, 0.0))
        eig = (h00 + h11 - disc) / 2.0
        det = h00*h11 - h01*h01
        ok = need & np.isfinite(eig) & (eig > 1e-12) & np.isfinite(det) & (det > 1e-18)
        near_step |= need & ~ok
        failed |= need & ~ok
        d0, d1 = np.zeros(b), np.zeros(b)
        d0[ok] = (h11[ok]*g0[ok] - h01[ok]*g1[ok]) / det[ok]
        d1[ok] = (-h01[ok]*g0[ok] + h00[ok]*g1[ok]) / det[ok]
        old = loss(theta)
        scale, accepted = np.ones(b), ~ok
        for _ in range(30):
            cand = theta.copy()
            cand[:, 0] -= scale*d0
            cand[:, 1] -= scale*d1
            new = loss(cand)
            accept = ok & np.isfinite(new) & (new <= old + 1e-12)
            theta[accept] = cand[accept]
            accepted |= accept
            pending = ok & ~accepted
            if not pending.any():
                break
            scale[pending] *= .5
        line_fail |= ok & ~accepted
        failed |= ok & ~accepted
        iterations[need] += 1

    z = theta[:, 0, None] + theta[:, 1, None] * x[None, :]
    p = sigmoid(z)
    r = w * (p - y[None, :])
    gn = np.maximum(np.abs(r.sum(axis=1)), np.abs((r*x[None, :]).sum(axis=1)))
    q = w*p*(1-p)
    h00, h01, h11 = q.sum(axis=1), (q*x[None, :]).sum(axis=1), (q*(x*x)[None, :]).sum(axis=1)
    disc = np.sqrt(np.maximum((h00-h11)**2 + 4*h01*h01, 0.0))
    eig = (h00+h11-disc)/2.0
    maxcoef = np.max(np.abs(theta), axis=1)
    finite = np.isfinite(theta).all(axis=1) & np.isfinite(gn) & np.isfinite(eig)
    valid = positive & ~failed & finite & (gn <= GRAD_TOL) & (eig >= MIN_EIGEN) & (maxcoef <= MAX_COEF)
    diag = {"gradient": gn, "eigen": eig, "maxcoef": maxcoef, "iterations": iterations,
            "positive": positive, "nearStep": near_step, "lineFail": line_fail, "finite": finite}
    return theta, valid, diag


def ess(w):
    s = w.sum(axis=1)
    return s*s / np.maximum((w*w).sum(axis=1), 1.0)


def qsummary(x):
    return {"p05": float(np.quantile(x, .05)), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95))}


def audit(report, input_sha, code_sha, replicates=5000, seed=20261113, batch_size=5000):
    if replicates < 1000 or batch_size < 50:
        raise ValueError("replicates >=1000 and batch_size >=50 required")
    matches, rows, dates, teams = prepare(report)
    target_match = np.repeat(np.arange(len(matches)), 2)
    ycat = np.asarray([r["actual"] for r in rows], float)
    pcat = {(int(k), c): np.asarray([r["probabilities"][k][c] for r in rows], float)
            for k in KS for c in CATEGORIES}
    xcat = {key: np.log(np.clip(p, EPS, 1-EPS) / (1-np.clip(p, EPS, 1-EPS)))
            for key, p in pcat.items()}
    ybin = {c: (ycat == c).astype(float) for c in CATEGORIES}

    point = {}
    for key in sorted(xcat):
        t, v, _ = fit_weighted_logistic_batch(xcat[key], ybin[key[1]], np.ones((1, len(rows))))
        if not v[0]:
            raise ValueError(f"unstable unweighted point fit: {key}")
        point[key] = t[0]

    # Fixed probability-bin sufficient statistics aggregated to match level.
    bin_meta, bin_n, bin_p, bin_y = [], [], [], []
    for k in KS:
        for c in CATEGORIES:
            p, y = pcat[(int(k), c)], ybin[c]
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                mask = (p >= lo) & (p < hi)
                n = int(mask.sum())
                item = {"k": int(k), "category": c, "bin": [lo, min(hi, 1.0)], "n": n}
                if n < MIN_BIN_N:
                    item["ciStatus"] = "empty_bin" if n == 0 else "not_computed_sparse_bin"
                else:
                    item.update({
                        "meanPredictedProbability": float(p[mask].mean()),
                        "observedFrequency": float(y[mask].mean()),
                        "gapPercentagePoints": float((p[mask].mean()-y[mask].mean())*100),
                        "ciStatus": "pending",
                    })
                    bin_n.append(np.bincount(target_match[mask], minlength=len(matches)).astype(float))
                    bin_p.append(np.bincount(target_match[mask], weights=p[mask], minlength=len(matches)))
                    bin_y.append(np.bincount(target_match[mask], weights=y[mask], minlength=len(matches)))
                bin_meta.append(item)
    nmat = np.column_stack(bin_n) if bin_n else np.empty((len(matches), 0))
    pmat = np.column_stack(bin_p) if bin_p else np.empty((len(matches), 0))
    ymat = np.column_stack(bin_y) if bin_y else np.empty((len(matches), 0))
    eligible = [x for x in bin_meta if x["ciStatus"] == "pending"]

    params = {key: np.full((replicates, 2), np.nan) for key in sorted(xcat)}
    fit_valid = {key: np.zeros(replicates, bool) for key in sorted(xcat)}
    diag_arrays = {key: {name: np.full(replicates, np.nan) for name in
                         ("gradient", "eigen", "maxcoef")} for key in sorted(xcat)}
    reason_arrays = {key: {name: np.zeros(replicates, bool) for name in
                           ("positive", "nearStep", "lineFail", "finite")} for key in sorted(xcat)}
    bin_gap = np.full((replicates, len(eligible)), np.nan)
    bin_ok = np.zeros((replicates, len(eligible)), bool)
    date_ess, match_ess, zero_frac = np.empty(replicates), np.empty(replicates), np.empty(replicates)
    team_ess = {lg: np.empty(replicates) for lg in LEAGUES}

    rng = np.random.Generator(np.random.PCG64(seed))
    start = 0
    while start < replicates:
        n = min(batch_size, replicates-start)
        dm, tm, mw = draw_cluster_weights(matches, dates, teams, n, rng)
        sl = slice(start, start+n)
        date_ess[sl], match_ess[sl] = ess(dm), ess(mw)
        zero_frac[sl] = (mw == 0).mean(axis=1)
        for lg in LEAGUES:
            team_ess[lg][sl] = ess(tm[lg])
        tw = mw[:, target_match]
        for key in sorted(xcat):
            t, v, d = fit_weighted_logistic_batch(xcat[key], ybin[key[1]], tw, initial=point[key])
            params[key][sl], fit_valid[key][sl] = t, v
            for dest, src in (("gradient", "gradient"), ("eigen", "eigen"), ("maxcoef", "maxcoef")):
                diag_arrays[key][dest][sl] = d[src]
            for name in reason_arrays[key]:
                reason_arrays[key][name][sl] = d[name]
        if eligible:
            den = mw @ nmat
            pred, obs = mw @ pmat, mw @ ymat
            ok = den > 0
            gaps = np.full_like(den, np.nan)
            gaps[ok] = (pred[ok]/den[ok] - obs[ok]/den[ok])*100
            bin_gap[sl], bin_ok[sl] = gaps, ok
        start += n

    keys = sorted(fit_valid)
    common_fit = np.column_stack([fit_valid[k] for k in keys]).all(axis=1)
    if common_fit.mean() < MIN_VALID:
        raise ValueError(f"common valid fit fraction {common_fit.mean():.6f} < {MIN_VALID}")
    fit_z, fit_meta = [], []
    for key in keys:
        vals = params[key][common_fit]
        for j, name in enumerate(("intercept", "slope")):
            se = float(vals[:, j].std(ddof=1))
            if not np.isfinite(se) or se <= 0:
                raise ValueError(f"invalid SE for {key}/{name}")
            fit_z.append((vals[:, j]-point[key][j])/se)
            fit_meta.append((key, j, name, se, vals[:, j]))
    fit_crit = float(np.quantile(np.max(np.abs(np.column_stack(fit_z)), axis=1), .95, method="linear"))
    per_arm = {}
    for key, j, name, se, vals in fit_meta:
        k, c = key
        estimate = float(point[key][j])
        per_arm[f"k{k}_category{c}_{name}"] = {
            "k": k, "category": c, "parameter": name, "pointEstimate": estimate,
            "idealValue": 0.0 if j == 0 else 1.0, "bootstrapSE": se,
            "percentile95CI": np.quantile(vals, [.025, .975], method="linear").tolist(),
            "simultaneous95Band": [estimate-fit_crit*se, estimate+fit_crit*se],
        }

    paired_z, paired_meta = [], []
    for c in CATEGORIES:
        for j, name in ((0, "intercept"), (1, "slope")):
            vals = params[(6, c)][common_fit, j] - params[(10, c)][common_fit, j]
            estimate = float(point[(6, c)][j] - point[(10, c)][j])
            se = float(vals.std(ddof=1))
            if not np.isfinite(se) or se <= 0:
                raise ValueError(f"invalid paired SE for category {c}/{name}")
            paired_z.append((vals-estimate)/se)
            paired_meta.append((c, name, estimate, se, vals))
    paired_crit = float(np.quantile(np.max(np.abs(np.column_stack(paired_z)), axis=1), .95, method="linear"))
    paired = {}
    for c, name, estimate, se, vals in paired_meta:
        paired[f"category{c}_{name}"] = {
            "differenceK6MinusK10": estimate, "bootstrapSE": se,
            "percentile95CI": np.quantile(vals, [.025, .975], method="linear").tolist(),
            "simultaneous95Band": [estimate-paired_crit*se, estimate+paired_crit*se],
        }

    if eligible:
        common_bin = bin_ok.all(axis=1)
        n_common = int(common_bin.sum())
        if common_bin.mean() >= MIN_VALID:
            vals = bin_gap[common_bin]
            points = np.asarray([x["gapPercentagePoints"] for x in eligible])
            ses = vals.std(axis=0, ddof=1)
            if np.any(~np.isfinite(ses)) or np.any(ses <= 0):
                raise ValueError("invalid probability-bin bootstrap SE")
            bin_crit = float(np.quantile(np.max(np.abs((vals-points)/ses), axis=1), .95, method="linear"))
            for j, item in enumerate(eligible):
                item["bootstrapReplicatesValid"] = n_common
                item["bootstrapSE"] = float(ses[j])
                item["percentile95CI"] = np.quantile(vals[:, j], [.025, .975], method="linear").tolist()
                item["simultaneous95Band"] = [float(points[j]-bin_crit*ses[j]), float(points[j]+bin_crit*ses[j])]
                item["ciStatus"] = "computed"
        else:
            bin_crit = None
            for j, item in enumerate(eligible):
                item["bootstrapReplicatesValid"] = int(bin_ok[:, j].sum())
                item["ciStatus"] = "insufficient_common_nonempty_replicates"
    else:
        common_bin, n_common, bin_crit = np.ones(replicates, bool), replicates, None

    fit_diag = {}
    for key in keys:
        d, v = diag_arrays[key], fit_valid[key]
        fit_diag[f"k{key[0]}_category{key[1]}"] = {
            "validReplicates": int(v.sum()), "invalidReplicates": int((~v).sum()),
            "minimumHessianEigenvalueAmongValid": float(np.min(d["eigen"][v])) if v.any() else None,
            "maximumAbsoluteCoefficientAmongValid": float(np.max(d["maxcoef"][v])) if v.any() else None,
            "maximumGradientInfinityNormAmongValid": float(np.max(d["gradient"][v])) if v.any() else None,
            "invalidReasons": {
                "noPositiveWeight": int((~reason_arrays[key]["positive"]).sum()),
                "nearSingularDuringFit": int(reason_arrays[key]["nearStep"].sum()),
                "lineSearchFailed": int(reason_arrays[key]["lineFail"].sum()),
                "nonFinite": int((~reason_arrays[key]["finite"]).sum()),
                "finalNearSingularHessian": int((d["eigen"] < MIN_EIGEN).sum()),
                "gradientNotConverged": int((d["gradient"] > GRAD_TOL).sum()),
                "coefficientMagnitudeOverLimit": int((d["maxcoef"] > MAX_COEF).sum()),
            },
        }

    return {
        "experiment": "GOAL3 team-aware match-pair-preserving calibration bootstrap sensitivity",
        "cohort": {"matches": 236, "teamSideTargets": 472, "dates": 17,
                   "dateRange": [dates[0], dates[-1]],
                   "teamsByLeague": {lg: len(teams[lg]) for lg in LEAGUES}},
        "method": {
            "resampling": "independent pigeonhole draws of target dates and teams sampled within each league",
            "matchWeight": "date multiplicity * home-team multiplicity * away-team multiplicity",
            "pairPreservation": "one match weight is assigned to both home and away targets",
            "rng": "NumPy PCG64", "seed": seed, "replicates": replicates, "batchSize": batch_size,
            "fitMethod": "unpenalized vectorized Newton logistic with backtracking line search",
            "fitValidity": {"gradientInfinityNormAtMost": GRAD_TOL,
                            "minimumWeightedHessianEigenvalueAtLeast": MIN_EIGEN,
                            "maximumAbsoluteCoefficientAtMost": MAX_COEF,
                            "commonValidReplicateFractionAtLeast": MIN_VALID,
                            "invalidFitsExcludedFromCommonFamilies": True},
            "families": {"perArm": "8 one-vs-rest fits x intercept/slope (16 parameters)",
                         "paired": "4 categories x intercept/slope differences (8 contrasts)",
                         "bins": "all eligible non-empty fixed probability bins"},
            "probabilityBins": list(BINS), "minimumBinTargetsForCI": MIN_BIN_N,
        },
