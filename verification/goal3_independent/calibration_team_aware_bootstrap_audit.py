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
