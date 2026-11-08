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
