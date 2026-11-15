#!/usr/bin/env python3
"""Reproduce the agreed retrospective 17-date paired k=6 vs k=10 bootstrap."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np

EXPECTED_INNER_SHA256 = "3baf2baa9b6dc9fe948acd0588777245de05982cc29d3d759124073342faa477"
SOURCE_ARTIFACT_ID = 14175799907
SOURCE_RUN_ID = 43752228196
SOURCE_ZIP_SHA256 = "3283b0bb33f7e3c3194d6911415cc6942914d35305cf6cac044f739cbc034e2d"

def flatten_matches(payload: dict) -> list[dict]:
    rows = []
    for match in payload["allKPredictions"]:
        match_id = "|".join((match["date"], match["comp"], match["home"], match["away"]))
        for side in ("home", "away"):
            goals = int(match["hg"] if side == "home" else match["ag"])
            rows.append({"date": match["date"], "comp": match["comp"], "matchId": match_id,
                         "side": side, "actual": min(goals, 3),
                         "k6": match["byK"]["6.0"][side], "k10": match["byK"]["10.0"][side]})
    return rows

def score(rows: list[dict], arm: str) -> dict:
    probabilities = np.asarray([row[arm] for row in rows], dtype=float)
    actual = np.asarray([row["actual"] for row in rows], dtype=int)
    if probabilities.shape != (len(rows), 4):
        raise ValueError("Each team-side target must have four probabilities.")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Invalid probability value.")
    if np.any(np.abs(probabilities.sum(axis=1) - 1.0) > 2e-6):
        raise ValueError("Probability vector does not sum to one.")
    one_hot = np.eye(4)[actual]
    return {
        "brier": float(np.sum((probabilities - one_hot) ** 2, axis=1).mean()),
        "logLoss": float(-np.log(np.maximum(1e-12, probabilities[np.arange(len(actual)), actual])).mean()),
        "accuracy": float((probabilities.argmax(axis=1) == actual).mean()),
        "n": len(rows),
    }

def percentile_linear(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile from an empty sample.")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)

def compute(input_path: str | Path, replicates: int = 10_000, seed: int = 1) -> dict:
    raw = Path(input_path).read_bytes()
    input_sha = hashlib.sha256(raw).hexdigest()
    if input_sha != EXPECTED_INNER_SHA256:
        raise ValueError(f"Unexpected all-k artifact SHA256: {input_sha}")
    payload = json.loads(raw)
    rows = flatten_matches(payload)
    dates = sorted({row["date"] for row in rows})
    match_ids = {row["matchId"] for row in rows}
    if len(rows) != 472 or len(match_ids) != 236 or len(dates) != 17:
        raise ValueError(f"Unexpected cohort: rows={len(rows)}, matches={len(match_ids)}, dates={len(dates)}")
    per_match_sides = {}
    for row in rows:
        per_match_sides.setdefault(row["matchId"], set()).add(row["side"])
    if any(sides != {"home", "away"} for sides in per_match_sides.values()):
        raise ValueError("Each match must retain both team-side targets.")
    groups = {date: [row for row in rows if row["date"] == date] for date in dates}
    point_k6, point_k10 = score(rows, "k6"), score(rows, "k10")
    point_difference = {m: point_k6[m] - point_k10[m] for m in ("brier", "logLoss", "accuracy")}
    rng = random.Random(seed)
    brier_differences, logloss_differences, accuracy_differences = [], [], []
    for _ in range(replicates):
        sampled_dates = rng.choices(dates, k=len(dates))
        sample = [row for date in sampled_dates for row in groups[date]]
        a, b = score(sample, "k6"), score(sample, "k10")
        brier_differences.append(a["brier"] - b["brier"])
        logloss_differences.append(a["logLoss"] - b["logLoss"])
        accuracy_differences.append(a["accuracy"] - b["accuracy"])
    ci = {
        "brier": [percentile_linear(brier_differences, .025), percentile_linear(brier_differences, .975)],
        "logLoss": [percentile_linear(logloss_differences, .025), percentile_linear(logloss_differences, .975)],
        "accuracy": [percentile_linear(accuracy_differences, .025), percentile_linear(accuracy_differences, .975)],
    }
    return {
        "schemaVersion": 1,
        "source": {"artifactId": SOURCE_ARTIFACT_ID, "workflowRunId": SOURCE_RUN_ID,
                   "artifactZipSha256": SOURCE_ZIP_SHA256, "artifactInnerJsonSha256": input_sha,
                   "experiment": payload.get("experiment")},
        "cohort": {"matches": len(match_ids), "teamSideTargets": len(rows), "dates": len(dates),
                   "dateRange": [dates[0], dates[-1]],
                   "dateClusterSizes": {date: len(groups[date]) for date in dates}},
        "pointEstimates": {"k6": point_k6, "k10": point_k10, "differenceK6MinusK10": point_difference},
        "bootstrap": {
            "resamplingUnit": "date cluster", "clustersPerReplicate": len(dates),
            "replicates": replicates, "seed": seed,
            "rng": "Python random.Random (MT19937); random.Random(seed).choices(sorted_dates, k=17)",
            "pairing": "k=6 and k=10 use the same sampled date clusters in every replicate",
            "clusterPreservation": "all matches and both team-side rows for each sampled date are retained together",
            "statistic": "D = score(k=6) - score(k=10); D < 0 favors k=6",
            "brierDefinition": "mean over team-side targets of the sum of four squared one-hot probability errors",
            "logLossDefinition": "mean negative log probability of the observed 0/1/2/3+ category; probability floor 1e-12",
            "confidenceInterval": "95% percentile interval with linear interpolation",
            "differenceIntervals": ci,
        },
        "interpretation": {
            "brierIntervalIncludesZero": ci["brier"][0] <= 0 <= ci["brier"][1],
            "logLossIntervalIncludesZero": ci["logLoss"][0] <= 0 <= ci["logLoss"][1],
            "accuracyIntervalIncludesZero": ci["accuracy"][0] <= 0 <= ci["accuracy"][1],
            "conclusion": "No statistically established superiority of k=6 over k=10 in this retrospective 17-date cluster sensitivity analysis.",
            "deploymentDecision": "Do not change production model or market weights; k=10 remains the research baseline and k=6 remains a candidate.",
            "scope": "Retrospective date-cluster sensitivity only; not the prospective round-level primary analysis.",
        },
        "limitations": [
            "The 17 target dates do not carry official toto round IDs, so a round-cluster primary analysis is not computed here.",
            "The source history snapshot contains 429 rows with one shared fetchedAt timestamp; this is not match-specific first-observed provenance.",
            "The team alias map was last updated 2026-10-22; 12 of the 17 target dates precede that alias-map update.",
            "The result evaluates frozen predictions from the pinned all-k artifact; it does not regenerate the nested model or establish prospective performance.",
        ],
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    result = compute(args.input, args.replicates, args.seed)
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    if result != expected:
        raise SystemExit("Date-cluster bootstrap result differs from the committed expected JSON.")
    print(json.dumps({"status": "PASS", "rows": result["cohort"]["teamSideTargets"],
                      "matches": result["cohort"]["matches"], "dates": result["cohort"]["dates"],
                      "replicates": result["bootstrap"]["replicates"],
                      "brierCI": result["bootstrap"]["differenceIntervals"]["brier"],
                      "logLossCI": result["bootstrap"]["differenceIntervals"]["logLoss"]},
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
