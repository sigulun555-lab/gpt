#!/usr/bin/env python3
"""Read-only historical parity audit: Claude round-1666 standalone lambda=10 vs GPT lambda=10.

This is an implementation comparison on one immutable historical input snapshot.
It is not a prospective forecast, not a scoring experiment, and not evidence of
production suitability. The saved Claude record did not contain its input hash;
the 2026-10-27 snapshot is a candidate identified from the last gpt history.json
commit before predictionConfirmedAt and must not be described as proven original
input unless the rounded saved predictions reproduce.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "verification/goal3_independent/inputs/round1666_history_snapshot_2026-10-27.json"
REFERENCE = ROOT / "verification/goal3_independent/inputs/round1666_claude_predictions_reference.jsonl"
CLAUDE_SCRIPT = ROOT / "verification/goal3_independent/vendor/claude_round1666_goal3.py"
CLAUDE_INSTRUMENTED_SCRIPT = ROOT / "verification/goal3_independent/vendor/claude_round1666_goal3_instrumented.py"
GPT_SCRIPT = ROOT / "verification/goal3_independent/nested_validation.py"
ALIASES_PATH = ROOT / "data/team_aliases.json"

EXPECTED_GIT_BLOBS = {
    "history": "023d826f568bfd6fff21a38833b9e31458fe056e",
    "claude_script": "558d4ea89088c19409a56e45ca6d9d6e7dbb87d5",
    "claude_instrumented_script": "7d05c54836d203c0aef60a0459f592a7613bae02",
    "reference_predictions": "9c58f1ada8500f0d19c1e93a1017151f2995083e",
    "gpt_nested_validation": "2b364ae31673b1f7471958084c416ca1ad8901f8",
    "team_aliases": "5de22584524ff4c07ddac1d27f49afbbbc5f88ab",
}
HISTORY_SOURCE_COMMIT = "988700300b65784e56e18adcf729a06aa833a2c8"
CLAUDE_SOURCE_COMMIT = "420923c1909ece11e4bc6f9b0fc9841a4bf4e15a"
FIXTURES = (("水戸", "C大阪"), ("横浜FM", "広島"), ("名古屋", "浦和"))
PREDICTION_CONFIRMED_AT = "2026-10-28T00:26:14Z"
PROB_KEYS = ("0", "1", "2", "3")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def load_gpt_module():
    spec = importlib.util.spec_from_file_location("goal3_nested_validation", GPT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load GPT nested_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(name: str, aliases: dict[str, str]) -> str:
    normalized = unicodedata.normalize("NFKC", str(name)).strip()
    return aliases.get(normalized, normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_path = Path(args.out)

    paths = {
        "history": HISTORY,
        "claude_script": CLAUDE_SCRIPT,
        "claude_instrumented_script": CLAUDE_INSTRUMENTED_SCRIPT,
        "reference_predictions": REFERENCE,
        "gpt_nested_validation": GPT_SCRIPT,
        "team_aliases": ALIASES_PATH,
    }
    blob_hashes = {key: git_blob_sha1(path) for key, path in paths.items()}
    instrumented_text = CLAUDE_INSTRUMENTED_SCRIPT.read_text(encoding="utf-8")
    instrumentation_lines = (
        '                    "expected_goals_full": lam_goals,\n',
        '                    "p0_full": probs[0], "p1_full": probs[1], "p2_full": probs[2], "p3plus_full": probs[3],\n',
    )
    stripped_instrumented = instrumented_text
    for line in instrumentation_lines:
        if stripped_instrumented.count(line) != 1:
            raise RuntimeError("unexpected Claude output instrumentation patch")
        stripped_instrumented = stripped_instrumented.replace(line, "", 1)
    if stripped_instrumented != CLAUDE_SCRIPT.read_text(encoding="utf-8"):
        raise RuntimeError("instrumented Claude copy differs outside output-only precision fields")
    for key, expected in EXPECTED_GIT_BLOBS.items():
        if blob_hashes[key] != expected:
            raise RuntimeError(f"Git blob SHA mismatch for {key}: {blob_hashes[key]} != {expected}")

    history_obj = json.loads(HISTORY.read_text(encoding="utf-8"))
    raw_history = history_obj.get("history")
    if not isinstance(raw_history, list):
        raise RuntimeError("history.json has no history array")
    rows = [
        row for row in raw_history
        if row.get("competition") in ("J1", "J2", "J3")
        and isinstance(row.get("home_goals"), int)
        and isinstance(row.get("away_goals"), int)
    ]
    dates = sorted({row["date"] for row in rows})
    teams = sorted({row["home"] for row in rows} | {row["away"] for row in rows})
    fetched_at = sorted({row.get("fetchedAt") for row in rows})
    if (len(rows), len(dates), len(teams)) != (380, 34, 60):
        raise RuntimeError(
            f"unexpected historical snapshot shape: rows={len(rows)} dates={len(dates)} teams={len(teams)}"
        )
    if len(fetched_at) != 1 or not fetched_at[0]:
        raise RuntimeError("history snapshot does not have one unique fetchedAt")
    fetched_dt = datetime.fromisoformat(fetched_at[0].replace("Z", "+00:00"))
    confirmed_dt = datetime.fromisoformat(PREDICTION_CONFIRMED_AT.replace("Z", "+00:00"))
    if fetched_dt > confirmed_dt:
        raise RuntimeError("candidate history snapshot was fetched after Claude prediction time")

    aliases = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))["aliases"]
    gpt = load_gpt_module()
    gpt_prior = []
    for row in rows:
        gpt_prior.append({
            "date": row["date"],
            "home": canonical(row["home"], aliases),
            "away": canonical(row["away"], aliases),
            "hg": int(row["home_goals"]),
            "ag": int(row["away_goals"]),
            "year": 2026,
        })
    # Match the current GPT nested-validation preprocessing exactly.
    gpt_prior.sort(key=lambda row: (row["date"], row["home"], row["away"]))

    with tempfile.TemporaryDirectory(prefix="goal3-round1666-parity-") as tmp:
        claude_out = Path(tmp) / "claude_predictions.json"
        proc = subprocess.run(
            [
                sys.executable, str(CLAUDE_INSTRUMENTED_SCRIPT),
                "--history", str(HISTORY),
                "--out", str(claude_out),
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "vendored Claude script failed\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr
            )
        fit_match = re.search(r"fit success\s+(True|False)\s+nit\s+(\d+)", proc.stdout)
        if not fit_match or fit_match.group(1) != "True":
            raise RuntimeError("Claude optimizer success could not be confirmed from stdout")
        claude_nit = int(fit_match.group(2))
        claude_rows = json.loads(claude_out.read_text(encoding="utf-8"))

    reference_rows = [
        json.loads(line)
        for line in REFERENCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(claude_rows) != 6 or len(reference_rows) != 6:
        raise RuntimeError("expected exactly six Claude and six saved reference targets")

    gpt_params, gpt_idx, gpt_fit = gpt.fit_shrinkage(gpt_prior, 10.0)
    if not gpt_fit.get("success"):
        raise RuntimeError("GPT L-BFGS-B fit failed: " + str(gpt_fit))

    target_results = []
    saved_diffs = []
    implementation_diffs = []
    expected_goal_diffs = []
    for i, (home_raw, away_raw) in enumerate(FIXTURES):
        home = canonical(home_raw, aliases)
        away = canonical(away_raw, aliases)
        gpt_home, gpt_away = gpt.predict_shrinkage(gpt_params, gpt_idx, home, away)
        for side_index, side in enumerate(("home", "away")):
            c_row = claude_rows[i * 2 + side_index]
            ref_row = reference_rows[i * 2 + side_index]
            gpt_probs = np.asarray(gpt_home if side == "home" else gpt_away, dtype=float)
            claude_probs_rounded = np.asarray([float(c_row["p0"]), float(c_row["p1"]), float(c_row["p2"]), float(c_row["p3plus"])], dtype=float)
            claude_probs_full = np.asarray([float(c_row["p0_full"]), float(c_row["p1_full"]), float(c_row["p2_full"]), float(c_row["p3plus_full"])], dtype=float)
            ref_probs = np.asarray([float(ref_row["probabilities"][key]) for key in PROB_KEYS], dtype=float)
            if c_row.get("side") != side:
                raise RuntimeError(f"Claude output order/side mismatch at target {i * 2 + side_index}")
            if ref_row.get("isHome") != (side == "home"):
                raise RuntimeError(f"saved reference order/side mismatch at target {i * 2 + side_index}")
            if not np.isfinite(gpt_probs).all() or not np.isfinite(claude_probs_full).all():
                raise RuntimeError("non-finite probabilities")
            if abs(float(gpt_probs.sum()) - 1.0) > 1e-9:
                raise RuntimeError("GPT full-precision probability vector failed normalization guard")
            if abs(float(claude_probs_full.sum()) - 1.0) > 1e-9:
                raise RuntimeError("Claude instrumented full-precision probability vector failed normalization guard")
            # The archived/reference fields are independently rounded to 4dp.
            if abs(float(claude_probs_rounded.sum()) - 1.0) > 0.00025:
                raise RuntimeError("Claude rounded probability vector exceeds 4dp rounding tolerance")

            saved_diff = float(np.max(np.abs(claude_probs_rounded - ref_probs)))
            impl_diff = float(np.max(np.abs(gpt_probs - claude_probs_full)))
            saved_diffs.append(saved_diff)
            implementation_diffs.append(impl_diff)
            gpt_lambda = -math.log(max(float(gpt_probs[0]), 1e-300))
            claude_lambda = float(c_row["expected_goals_full"])
            claude_lambda_rounded = float(c_row["expected_goals"])
            expected_goal_diffs.append(abs(gpt_lambda - claude_lambda))
            target_results.append({
                "match": f"{home_raw} vs {away_raw}",
                "side": side,
                "referenceTeam": ref_row["team"],
                "claudeScriptTeam": c_row["team"],
                "gptCanonicalTeam": home if side == "home" else away,
                "claudeExpectedGoalsFullPrecision": claude_lambda,
                "claudeExpectedGoalsRounded3dp": claude_lambda_rounded,
                "gptExpectedGoalsFromP0": gpt_lambda,
                "expectedGoalsAbsDifference": abs(gpt_lambda - claude_lambda),
                "savedReferenceMaxAbsProbabilityDifference": saved_diff,
                "gptVsClaudeMaxAbsProbabilityDifference": impl_diff,
                "claudeProbabilitiesFullPrecision": claude_probs_full.tolist(),
                "claudeProbabilitiesRounded4dp": claude_probs_rounded.tolist(),
                "gptProbabilitiesFullPrecision": gpt_probs.tolist(),
                "savedReferenceProbabilitiesRounded4dp": ref_probs.tolist(),
            })

    max_saved_diff = max(saved_diffs)
    max_impl_diff = max(implementation_diffs)
    max_lambda_diff = max(expected_goal_diffs)
    if max_saved_diff > 0.00005:
        raise RuntimeError(
            f"Claude script does not reproduce saved predictions to 4dp: max diff={max_saved_diff}"
        )

    report = {
        "experiment": "GOAL3 round1666 historical standalone implementation parity audit",
        "scope": "research-only implementation comparison; no performance scoring and no prospective forecast",
        "sourceCommits": {
            "historySnapshotSourceCommit": HISTORY_SOURCE_COMMIT,
            "claudeScriptSourceCommit": CLAUDE_SOURCE_COMMIT,
            "gptResearchHeadAtAuditDesign": "5841e90db57d9c59ff1b205608ce976850b861bd",
        },
        "inputProvenance": {
            key: {
                "path": str(path.relative_to(ROOT)),
                "gitBlobSha1": blob_hashes[key],
                "sha256": sha256_file(path),
            }
            for key, path in paths.items()
        },
        "cohort": {
            "matches": len(rows),
            "dates": len(dates),
            "dateRange": [dates[0], dates[-1]],
            "teams": len(teams),
            "competitionRows": {
                league: sum(row["competition"] == league for row in rows)
                for league in ("J1", "J2", "J3")
            },
            "uniqueFetchedAt": fetched_at,
            "predictionConfirmedAt": PREDICTION_CONFIRMED_AT,
            "fetchedAtBeforePredictionConfirmedAt": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "instrumentationPatchVerified": True,
        "optimizerAndPreprocessingDifferences": {
            "claude": {
                "method": "L-BFGS-B",
                "options": {"gtol": 1e-10},
                "preprocessing": "sort by (date, competition); no alias canonicalization",
                "clippingGradient": "uses clipped residual without zeroing derivative outside +/-20",
                "fullPrecisionOutputInstrumentation": "output-only fields; exact original code restored by removing two added JSON lines",
                "probabilityTransform": "scipy.stats.poisson.pmf for 0/1/2; 3+ is 1-sum",
                "fitSuccess": True,
                "iterations": claude_nit,
            },
            "gpt": {
                "method": "L-BFGS-B",
                "options": {"maxiter": 300, "default gtol": 1e-5},
                "preprocessing": "NFKC + team aliases; sort by (date, home, away)",
                "clippingGradient": "chain rule through clip; derivative zero outside +/-20",
                "probabilityTransform": "closed-form Poisson 0/1/2; 3+ is residual mass",
                "fitSuccess": bool(gpt_fit.get("success")),
                "iterations": int(gpt_fit.get("nit", -1)),
                "status": int(gpt_fit.get("status", -1)),
                "message": str(gpt_fit.get("message", "")),
            },
        },
        "savedClaudePredictionReproduction": {
            "targets": 6,
            "maxAbsProbabilityDifference": max_saved_diff,
            "reproducedToFourDecimalPlaces": max_saved_diff <= 0.00005,
            "interpretation": (
                "The candidate 380-match snapshot reproduces the saved Claude probabilities "
                "to their stored four-decimal precision; the original record still lacks an input hash, "
                "so exact historical input identity is strongly supported but not cryptographically proven."
            ),
        },
        "implementationComparison": {
            "targets": 6,
            "maxAbsProbabilityDifference": max_impl_diff,
            "maxExpectedGoalsAbsDifference": max_lambda_diff,
            "equivalentWithin1e-6": max_impl_diff <= 1e-6,
            "targetResults": target_results,
            "interpretation": (
                "This comparison uses unrounded probabilities from an output-only instrumented copy of Claude's exact script. "
                "Any remaining difference is numerical/optimization/preprocessing, not the 4dp archive rounding. "
                "Keep code-lineage and output comparisons separate; do not double-score the same specification "
                "as independent model families, but do not substitute one implementation's predictions for the other."
            ),
        },
        "decision": {
            "round1666StandaloneClaudeScriptReproduced": True,
            "gptClaudeImplementationEquivalentWithin1e-6": max_impl_diff <= 1e-6,
            "productionAdoptionAuthorized": False,
            "prospectiveEvidence": False,
            "historicalInputHashOriginallyRecorded": False,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "cohort": report["cohort"],
        "claudeIterations": claude_nit,
        "gptIterations": int(gpt_fit.get("nit", -1)),
        "savedClaudePredictionMaxAbsDiff": max_saved_diff,
        "gptVsClaudeMaxAbsDiff": max_impl_diff,
        "gptVsClaudeEquivalentWithin1e-6": max_impl_diff <= 1e-6,
        "reportSha256": sha256_file(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
