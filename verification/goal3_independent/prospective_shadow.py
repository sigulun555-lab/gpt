#!/usr/bin/env python3
"""Create a paired, deadline-safe GOAL3 shrinkage-MLE shadow forecast.

This runner is research-only. It never changes production predictions.
Both k arms use the same 2026-only history, fixtures, cutoff and model code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from nested_validation import canonical, fit_shrinkage, predict_shrinkage  # noqa: E402

K_ARMS = (6.0, 10.0)
VALID_LEAGUES = {"J1", "J2", "J3"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include timezone: {value}")
    return parsed.astimezone(timezone.utc)


def load_json(path: Path):
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw), raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(ROOT / "data" / "history.json"))
    ap.add_argument("--predictions", default=str(ROOT / "data" / "predictions.json"))
    ap.add_argument("--aliases", default=str(ROOT / "data" / "team_aliases.json"))
    ap.add_argument("--out", required=True, help="New output JSON path; existing files are never overwritten")
    args = ap.parse_args()

    history_path, predictions_path, aliases_path = map(Path, (args.history, args.predictions, args.aliases))
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"Append-only guard: refusing to overwrite {out_path}")

    history_doc, history_raw = load_json(history_path)
    predictions, predictions_raw = load_json(predictions_path)
    aliases_doc, aliases_raw = load_json(aliases_path)

    if predictions.get("marketType") not in (None, "totoGOAL3"):
        raise ValueError("Prediction snapshot is not totoGOAL3")
    if predictions.get("status") != "predictions_generated" or not predictions.get("totoTargetsConfirmed"):
        raise ValueError("Official GOAL3 targets are not confirmed in the snapshot")
    round_no = predictions.get("round")
    if not isinstance(round_no, int):
        raise ValueError("Missing official round number")
    generated_at = predictions.get("generatedAt")
    deadline = predictions.get("deadline")
    if not generated_at or not deadline:
        raise ValueError("Missing generatedAt/deadline")
    as_of = parse_time(generated_at)
    deadline_dt = parse_time(deadline)
    recorded_at = datetime.now(timezone.utc)
    if not as_of < deadline_dt or not recorded_at < deadline_dt:
        raise ValueError("Refusing to create a forecast at or after the official deadline")
    source_fetched = predictions.get("totoSource", {}).get("lastFetchedAt")
    if not source_fetched or parse_time(source_fetched) > as_of:
        raise ValueError("Official target snapshot was fetched after the declared forecast time")

    fixtures = [
        p for p in predictions.get("predictions", [])
        if p.get("marketType") == "totoGOAL3" and p.get("totoRound") == round_no
    ]
    if len(fixtures) != 3 or len({p.get("number") for p in fixtures}) != 3:
        raise ValueError(f"Expected exactly 3 unique GOAL3 fixtures, got {len(fixtures)}")
    if any(not p.get("home") or not p.get("away") or not p.get("date") for p in fixtures):
        raise ValueError("Fixture is missing team or date")
    if any(p.get("totoTargetConfirmed") is not True for p in fixtures):
        raise ValueError("At least one fixture is not officially confirmed")

    aliases = aliases_doc.get("aliases", {})
    if not isinstance(aliases, dict):
        raise ValueError("Invalid team alias map")

    # Strict as-of filter: use only J1/J2/J3 results played before each target
    # date and already present in the source snapshot by generatedAt.
    source_rows = history_doc.get("history", [])
    prepared = []
    for row in source_rows:
        if row.get("competition") not in VALID_LEAGUES:
            continue
        if not isinstance(row.get("home_goals"), int) or not isinstance(row.get("away_goals"), int):
            continue
        if not row.get("date") or not row.get("home") or not row.get("away"):
            continue
        fetched_at = row.get("fetchedAt")
        if not fetched_at:
            raise ValueError("History row lacks fetchedAt; cannot prove as-of eligibility")
        if parse_time(fetched_at) > as_of:
            continue
        prepared.append({
            "date": row["date"],
            "home": canonical(row["home"], aliases),
            "away": canonical(row["away"], aliases),
            "hg": int(row["home_goals"]),
            "ag": int(row["away_goals"]),
            "sourceFetchedAt": fetched_at,
        })

    if not prepared:
        raise ValueError("No eligible historical results in the frozen input snapshot")

    fixture_results = []
    fit_diagnostics = []
    for fixture in sorted(fixtures, key=lambda p: p["number"]):
        target_date = fixture["date"]
        prior = [m for m in prepared if m["date"] < target_date]
        if not prior:
            raise ValueError(f"No strictly earlier training history for fixture {fixture['number']}")
        if any(m["date"] >= target_date for m in prior):
            raise AssertionError("Temporal leakage guard failed")
        home = canonical(fixture["home"], aliases)
        away = canonical(fixture["away"], aliases)
        arms = {}
        for k in K_ARMS:
            params, team_index, info = fit_shrinkage(prior, k)
            ph, pa = predict_shrinkage(params, team_index, home, away)
            if abs(sum(ph) - 1.0) > 1e-9 or abs(sum(pa) - 1.0) > 1e-9:
                raise AssertionError("Goal-category probabilities do not sum to one")
            n_teams = len(team_index)
            home_eta = float(params[0] + params[1] +
                (params[2 + team_index[home]] if home in team_index else 0.0) -
                (params[2 + n_teams + team_index[away]] if away in team_index else 0.0))
            away_eta = float(params[0] +
                (params[2 + team_index[away]] if away in team_index else 0.0) -
                (params[2 + n_teams + team_index[home]] if home in team_index else 0.0))
            arms[str(int(k))] = {
                "homeGoalProbabilities": ph,
                "awayGoalProbabilities": pa,
                "homeExpectedGoals": math.exp(max(-20.0, min(20.0, home_eta))),
                "awayExpectedGoals": math.exp(max(-20.0, min(20.0, away_eta))),
                "fit": info,
            }
            fit_diagnostics.append({
                "fixtureNumber": fixture["number"], "k": int(k),
                "trainingRows": len(prior), "trainingCutoffExclusive": target_date,
                "fitSuccess": bool(info["success"]), "optimizerIterations": int(info["nit"]),
            })
        fixture_results.append({
            "number": fixture["number"], "date": target_date,
            "home": fixture["home"], "away": fixture["away"],
            "arms": arms,
        })

    code_path = Path(__file__)
    model_path = HERE / "nested_validation.py"
    record = {
        "schemaVersion": 1,
        "experiment": "GOAL3_PROSPECTIVE_SHADOW_K6_VS_K10",
        "round": round_no,
        "forecastAsOf": generated_at,
        "recordedAt": recorded_at.isoformat(),
        "deadline": deadline,
        "status": "forecast_created_before_deadline",
        "productionImpact": "none",
        "model": {
            "family": "independent shrinkage MLE with Poisson goal marginals",
            "trainingWindow": "2026_only",
            "penalty": "single L2 coefficient; k is the same coefficient as lambda",
            "arms": [6, 10],
            "otherAdjustments": "none; no market blend, Dixon-Coles or draw boost",
        },
        "inputs": {
            "historySha256": sha256_text(history_raw),
            "officialPredictionsSha256": sha256_text(predictions_raw),
            "aliasesSha256": sha256_text(aliases_raw),
            "historyEligibleRows": len(prepared),
            "historyMaxFetchedAt": max(m["sourceFetchedAt"] for m in prepared),
        },
        "code": {
            "runnerSha256": sha256_text(code_path.read_text(encoding="utf-8")),
            "modelSha256": sha256_text(model_path.read_text(encoding="utf-8")),
            "snapshotPipelineSha256": sha256_text((ROOT / "scripts" / "auto_pipeline.js").read_text(encoding="utf-8")),
        },
        "fitDiagnostics": fit_diagnostics,
        "fixtures": fixture_results,
        "evaluation": {
            "target": "home and away goals categorized as 0,1,2,3+",
            "primaryMetrics": ["brier", "logloss"],
            "minimumEvidence": {"rounds": 10, "matches": 30},
            "adoptionRule": "paired round/date-block bootstrap 95% CI upper bound below zero for both Brier and LogLoss; no material J1/J2/J3 deterioration; no increased fit failures or missing forecasts",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidental replacement of a prior forecast.
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "status": record["status"], "round": round_no,
        "fixtures": len(fixture_results), "teamTargets": len(fixture_results) * 2,
        "fitCount": len(fit_diagnostics),
        "out": str(out_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
