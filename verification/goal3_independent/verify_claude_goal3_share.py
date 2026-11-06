#!/usr/bin/env python3
"""Independent, read-only audit of Claude's GOAL3 all-k and version-share artifacts.

Pinned external sources are fetched by immutable commit SHA. This script only
writes a JSON report under the research PR workspace; it never edits forecasts,
production code, main, market weights, or existing shadow artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOTO_REPO = "sigulun555-lab/toto-lab"
ALLK_COMMIT = "17da09beedee57c571038c8204cd51112fb1d959"
VERSIONS_COMMIT = "7401bc2f640a22f7f0cac36fca541ca127cfb91b"

ALLK_FILES = {
    "goal3_outer_allk.json": {
        "path": "docs/goal3-allk-share-20261106/goal3_outer_allk.json",
        "sha256": "67f2bf3fb65fd26ba32da992771d984eecd49c592dbb329598b4dc0de96068f6",
        "git_blob_sha1": "eed46735e1624e04a8796e4d0f56ac0778a6d430",
    },
    "diag.py": {
        "path": "docs/goal3-allk-share-20261106/diag.py",
        "sha256": "ea304d838477ae320c81eb6bfcb41a1795c4ec57cbe3e4b010c2c2ba8a59b702",
        "git_blob_sha1": "17ae85ea37798a6ad6c9d100ce72299407bf2aaa",
    },
    "diag2.py": {
        "path": "docs/goal3-allk-share-20261106/diag2.py",
        "sha256": "319829f46ac55f73d02a0ac3ed46033191800de32839dbbd6c0b563017f399c9",
        "git_blob_sha1": "8a3db593a8d4fc374c2fcefb5eea2497a665d0b1",
    },
    "analyze.py": {
        "path": "docs/goal3-allk-share-20261106/analyze.py",
        "sha256": "ada3057bfa3981e575e14b734f01aa8985b3b9b7e9e3cae3be305ad5a2fef225",
        "git_blob_sha1": "2a6edcc921b26d272e3e6eaf3acaafe6816153cb",
    },
    "goal3_outer_matches_compact.csv": {
        "path": "docs/goal3-allk-share-20261106/goal3_outer_matches_compact.csv",
        "sha256": "975f4697cfae5db761ad1168636edb0191b9fdeb91be3cd8ee2768c26bca0efc",
        "git_blob_sha1": "7f8c48413c2ceab1003e9da6d93636aad1d75c46",
    },
}

VERSION_FILES = {
    "README.md": ("research/goal3_versions/README.md", "d33b8174292f277d5974913341df24051e7a4d3c"),
    "goal3_versions.py": ("research/goal3_versions/goal3_versions.py", "f1a96cf624bd53aa1aa635d44cabc63d99e5bc3d"),
    "test_versions.py": ("research/goal3_versions/test_versions.py", "dc2781bb282a5ef827a51da1c94690b8176dbbd0"),
}

K_VALUES = ["0.5", "1", "2", "3", "4", "6", "8", "10", "15", "20", "30", "50", "75", "100", "150", "200"]
LOGLOSS_FLOOR = 1e-12


def git_blob_sha1(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def fetch_pinned(commit: str, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{TOTO_REPO}/{commit}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "GOAL3-research-audit/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def fetch_and_verify(name: str, commit: str, spec: dict) -> bytes:
    data = fetch_pinned(commit, spec["path"])
    got_sha256 = hashlib.sha256(data).hexdigest()
    got_blob = git_blob_sha1(data)
    if got_sha256 != spec["sha256"]:
        raise AssertionError(f"{name}: SHA-256 mismatch: expected {spec['sha256']}, got {got_sha256}")
    if got_blob != spec["git_blob_sha1"]:
        raise AssertionError(f"{name}: Git blob SHA-1 mismatch: expected {spec['git_blob_sha1']}, got {got_blob}")
    return data


def fetch_version_file(name: str, path: str, expected_blob: str) -> bytes:
    data = fetch_pinned(VERSIONS_COMMIT, path)
    got_blob = git_blob_sha1(data)
    if got_blob != expected_blob:
        raise AssertionError(f"version source {name}: Git blob mismatch: expected {expected_blob}, got {got_blob}")
    return data


def read_csv_bytes(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def read_csv_path(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def category(goal: int) -> int:
    return min(3, max(0, int(goal)))


def score(probabilities: list[float], actual: int) -> tuple[float, float, int]:
    brier = sum((float(probabilities[c]) - (1.0 if c == actual else 0.0)) ** 2 for c in range(4))
    logloss = -math.log(max(LOGLOSS_FLOOR, float(probabilities[actual])))
    accuracy = int(max(range(4), key=lambda c: float(probabilities[c])) == actual)
    return brier, logloss, accuracy


def mean_metrics(rows: list[dict], k: str, indices: list[int] | None = None) -> dict:
    use = indices if indices is not None else list(range(len(rows)))
    if not use:
        raise AssertionError("empty metric population")
    vals = [score(rows[i]["P"][k], int(rows[i]["actual"])) for i in use]
    n = len(vals)
    return {
        "n": n,
        "brier": sum(x[0] for x in vals) / n,
        "logloss": sum(x[1] for x in vals) / n,
        "accuracy": sum(x[2] for x in vals) / n,
    }


def percentile_pair(values: list[float]) -> list[float]:
    ordered = sorted(values)
    # Match analyze.py's explicit 10,000-replicate order-statistic indices.
    return [ordered[249], ordered[9749]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    downloaded = {
        name: fetch_and_verify(name, ALLK_COMMIT, spec)
        for name, spec in ALLK_FILES.items()
    }
    version_bytes = {
        name: fetch_version_file(name, path, blob)
        for name, (path, blob) in VERSION_FILES.items()
    }

    allk_rows = json.loads(downloaded["goal3_outer_allk.json"].decode("utf-8"))
    compact_rows = read_csv_bytes(downloaded["goal3_outer_matches_compact.csv"])
    gpt_csv_path = ROOT / "verification" / "goal3_independent" / "goal3_outer_matches.csv"
    gpt_rows = read_csv_path(gpt_csv_path)

    if len(allk_rows) != 472 or len(compact_rows) != 472 or len(gpt_rows) != 472:
        raise AssertionError(f"target row count mismatch: allk={len(allk_rows)}, Claude CSV={len(compact_rows)}, GPT CSV={len(gpt_rows)}")

    # Claude's compact CSV must match the GPT fixed outer CSV exactly, field by field.
    csv_mismatches = []
    for i, (claude_row, gpt_row) in enumerate(zip(compact_rows, gpt_rows)):
        if claude_row != gpt_row:
            csv_mismatches.append({"row": i + 2, "claude": claude_row, "gpt": gpt_row})
            if len(csv_mismatches) >= 3:
                break
    if csv_mismatches:
        raise AssertionError(f"Claude compact CSV differs from GPT fixed outer CSV: {csv_mismatches[:1]}")

    if len(allk_rows) != len(compact_rows):
        raise AssertionError("all-k JSON and compact CSV row counts differ")

    k_keys = sorted(allk_rows[0]["P"].keys(), key=float)
    if k_keys != K_VALUES:
        raise AssertionError(f"unexpected k candidates: {k_keys}")

    target_rows = []
    for i, (r, c) in enumerate(zip(allk_rows, compact_rows)):
        if (r["date"] != c["date"] or r["comp"] != c["comp"] or int(r["fold"]) != int(c["fold"])
                or r["side"] != c["side"] or int(r["actual"]) != int(c["actual"])):
            raise AssertionError(f"all-k JSON / compact CSV identity mismatch at target row {i + 2}")
        if int(r["actual"]) != category(int(r["g"])):
            raise AssertionError(f"binned actual does not match raw goals at row {i + 2}")
        p6 = r["P"]["6"]
        p10 = r["P"]["10"]
        for probs in r["P"].values():
            if len(probs) != 4 or any((not math.isfinite(float(x)) or float(x) < 0.0 or float(x) > 1.0) for x in probs):
                raise AssertionError(f"invalid probability vector at row {i + 2}")
            if abs(sum(float(x) for x in probs) - 1.0) > 1e-9:
                raise AssertionError(f"probability sum error at row {i + 2}")
        for j in range(4):
            if abs(float(p6[j]) - float(c[f"k6_p{j}"])) > 5.1e-7:
                raise AssertionError(f"k=6 JSON / CSV probability mismatch at row {i + 2}")
            if abs(float(p10[j]) - float(c[f"k10_p{j}"])) > 5.1e-7:
                raise AssertionError(f"k=10 JSON / CSV probability mismatch at row {i + 2}")
        target_rows.append({
            **r,
            "P": {k: [float(x) for x in r["P"][k]] for k in k_keys},
            "actual": int(r["actual"]),
            "g": int(r["g"]),
            "fold": int(r["fold"]),
            "target_team": c["home"] if c["side"] == "home" else c["away"],
            "home": c["home"],
            "away": c["away"],
        })

    match_groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(compact_rows):
        key = (r["date"], r["comp"], r["home"], r["away"])
        match_groups[key].append(i)
    if len(match_groups) != 236:
        raise AssertionError(f"expected 236 unique matches, got {len(match_groups)}")
    for key, ix in match_groups.items():
        sides = [compact_rows[i]["side"] for i in ix]
        if len(ix) != 2 or sorted(sides) != ["away", "home"]:
            raise AssertionError(f"match does not have exactly one home and away target: {key} -> {sides}")

    dates = sorted({r["date"] for r in target_rows})
    fold_counts = {str(f): sum(r["fold"] == f for r in target_rows) for f in sorted({r["fold"] for r in target_rows})}
    side_counts = {s: sum(c["side"] == s for c in compact_rows) for s in ("home", "away")}
    comp_counts = {c: sum(r["comp"] == c for r in target_rows) for c in ("J1", "J2", "J3")}
    if len(dates) != 17 or fold_counts != {"1": 120, "2": 134, "3": 218}:
        raise AssertionError(f"unexpected date/fold counts: dates={len(dates)}, folds={fold_counts}")

    metrics = {k: mean_metrics(target_rows, k) for k in k_keys}
    calibration = {}
    for k in ("6", "10"):
        cells = []
        for cat in range(4):
            cells.append({
                "category": cat,
                "mean_predicted": sum(r["P"][k][cat] for r in target_rows) / len(target_rows),
                "observed": sum(int(r["actual"] == cat) for r in target_rows) / len(target_rows),
            })
        calibration[k] = cells

    # Date- and match-cluster bootstrap of fixed k=6 minus k=10.
    diffs = {}
    for i, r in enumerate(target_rows):
        b6, l6, _ = score(r["P"]["6"], r["actual"])
        b10, l10, _ = score(r["P"]["10"], r["actual"])
        diffs[i] = (b6 - b10, l6 - l10)
    by_date: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(target_rows):
        by_date[r["date"]].append(i)
    rng = random.Random(1)
    date_b, date_l = [], []
    for _ in range(10_000):
        sample_dates = [rng.choice(dates) for _ in dates]
        ix = [i for d in sample_dates for i in by_date[d]]
        date_b.append(sum(diffs[i][0] for i in ix) / len(ix))
        date_l.append(sum(diffs[i][1] for i in ix) / len(ix))

    match_keys = sorted(match_groups.keys(), key=lambda x: (x[0], x[2], x[3]))
    by_match_target = {key: [i for i, r in enumerate(target_rows)
                             if (r["date"], r["comp"], r["home"], r["away"]) == key]
                       for key in match_keys}
    rng = random.Random(1)
    match_b, match_l = [], []
    for _ in range(10_000):
        sample_matches = [rng.choice(match_keys) for _ in match_keys]
        ix = [i for key in sample_matches for i in by_match_target[key]]
        match_b.append(sum(diffs[i][0] for i in ix) / len(ix))
        match_l.append(sum(diffs[i][1] for i in ix) / len(ix))

    # LOO sensitivity: sign of fixed k=6 - k=10 and selected best k.
    date_loo = {"brier_negative": 0, "logloss_negative": 0, "k6_best_brier": 0, "k6_best_logloss": 0}
    for d in dates:
        ix = [i for i, r in enumerate(target_rows) if r["date"] != d]
        m = {k: mean_metrics(target_rows, k, ix) for k in k_keys}
        date_loo["brier_negative"] += int(m["6"]["brier"] < m["10"]["brier"])
        date_loo["logloss_negative"] += int(m["6"]["logloss"] < m["10"]["logloss"])
        date_loo["k6_best_brier"] += int(min(k_keys, key=lambda k: m[k]["brier"]) == "6")
        date_loo["k6_best_logloss"] += int(min(k_keys, key=lambda k: m[k]["logloss"]) == "6")
    teams = sorted({r["target_team"] for r in target_rows})
    team_loo = {"brier_negative": 0, "logloss_negative": 0, "k6_best_brier": 0, "k6_best_logloss": 0}
    for team in teams:
        ix = [i for i, r in enumerate(target_rows) if r["target_team"] != team]
        m = {k: mean_metrics(target_rows, k, ix) for k in k_keys}
        team_loo["brier_negative"] += int(m["6"]["brier"] < m["10"]["brier"])
        team_loo["logloss_negative"] += int(m["6"]["logloss"] < m["10"]["logloss"])
        team_loo["k6_best_brier"] += int(min(k_keys, key=lambda k: m[k]["brier"]) == "6")
        team_loo["k6_best_logloss"] += int(min(k_keys, key=lambda k: m[k]["logloss"]) == "6")

    # Conditional Pearson dispersion from the Poisson mean implied by p(0), separate from raw var/mean.
    pearson = {}
    raw_dispersion = {}
    for k in ("6", "10"):
        pearson[k] = {}
        for comp in ("ALL", "J1", "J2", "J3"):
            group = [r for r in target_rows if comp == "ALL" or r["comp"] == comp]
            ss = 0.0
            for r in group:
                lam = -math.log(max(LOGLOSS_FLOOR, r["P"][k][0]))
                ss += ((r["g"] - lam) ** 2) / max(LOGLOSS_FLOOR, lam)
            pearson[k][comp] = {"n": len(group), "pearson_dispersion": ss / len(group)}
    for comp in ("ALL", "J1", "J2", "J3"):
        group = [r["g"] for r in target_rows if comp == "ALL" or r["comp"] == comp]
        mean_g = sum(group) / len(group)
        var_g = sum((x - mean_g) ** 2 for x in group) / max(1, len(group) - 1)
        raw_dispersion[comp] = {"n": len(group), "mean_goals": mean_g, "sample_variance": var_g,
                                "raw_variance_to_mean": var_g / mean_g if mean_g else None}

    # Import the exact version runner from Claude's pinned source commit.
    version_path = ROOT / "verification" / "goal3_independent" / "results" / "_claude_goal3_versions.py"
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_bytes(version_bytes["goal3_versions.py"])
    spec = importlib.util.spec_from_file_location("claude_goal3_versions", version_path)
    version_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version_module)

    # Rebuild the exact 2026-only canonicalized history used by GPT nested_validation.
    nv_path = ROOT / "verification" / "goal3_independent" / "nested_validation.py"
    nv_spec = importlib.util.spec_from_file_location("nested_validation_for_claude_parity", nv_path)
    nv = importlib.util.module_from_spec(nv_spec)
    nv_spec.loader.exec_module(nv)
    history_path = ROOT / "data" / "history.json"
    aliases_path = ROOT / "data" / "team_aliases.json"
    csv_2025_path = ROOT / "verification" / "goal3_independent" / "2025.csv"
    h2025, h2026 = nv.load_data(str(history_path), str(aliases_path), str(csv_2025_path))
    expected_input_blobs = {
        "history.json": (history_path, "f5cc8c044758e45b1e679e4ce796e68b3f27a854"),
        "team_aliases.json": (aliases_path, "5de22584524ff4c07ddac1d27f49afbbbc5f88ab"),
        "2025.csv": (csv_2025_path, "847d324c5a6168d02efe328f8173418a951af720"),
        "nested_validation.py": (nv_path, "7ac8f83da420fa577b45e9c5c021ce46387f584e"),
        "goal3_outer_matches.csv": (gpt_csv_path, "56892251679b8774a9e1b7ccc4b5beb42e660e36"),
    }
    for label, (path, expected_blob) in expected_input_blobs.items():
        actual_blob = git_blob_sha1(path.read_bytes())
        if actual_blob != expected_blob:
            raise AssertionError(f"GPT input blob mismatch for {label}: expected {expected_blob}, got {actual_blob}")
    if len(h2026) != 429:
        raise AssertionError(f"pinned GPT PR history expected 429 valid 2026 rows, got {len(h2026)}")

    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))["aliases"]
    raw_history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
    h2026_by_key = {}
    for m in raw_history:
        if (m.get("competition") in ("J1", "J2", "J3")
                and isinstance(m.get("home_goals"), int) and isinstance(m.get("away_goals"), int)
                and str(m.get("date", "")).startswith("2026-")):
            key = (m["date"], m["competition"], nv.canonical(m["home"], aliases), nv.canonical(m["away"], aliases))
            if key in h2026_by_key:
                raise AssertionError(f"duplicate canonicalized history key: {key}")
            h2026_by_key[key] = {"hg": int(m["home_goals"]), "ag": int(m["away_goals"])}
    if len(h2026_by_key) != len(h2026):
        raise AssertionError("duplicate 2026 history match key")
    target_keys = sorted(match_groups.keys(), key=lambda x: (x[0], x[2], x[3]))
    if not set(target_keys).issubset(h2026_by_key):
        raise AssertionError("outer target fixtures are not present in pinned 2026 history")

    max_v05_diff = 0.0
    max_v05_csv_diff = 0.0
    v05_fit_failures = 0
    for d in dates:
        prior = [r for r in h2026 if r["date"] < d]
        params = version_module.fit(prior, 6.0)
        for key in [x for x in target_keys if x[0] == d]:
            hist_match = h2026_by_key[key]
            home_p, away_p = version_module.predict_side(params, key[2], key[3])
            side_probs = {"home": home_p, "away": away_p}
            for side in ("home", "away"):
                row_index = next(i for i, c in enumerate(compact_rows)
                                 if (c["date"], c["comp"], c["home"], c["away"], c["side"]) == (*key, side))
                target = target_rows[row_index]
                if int(hist_match["hg"]) != int(target["g"]) if side == "home" else int(hist_match["ag"]) != int(target["g"]):
                    raise AssertionError(f"target goals differ from pinned history for {key} {side}")
                p = side_probs[side]
                max_v05_diff = max(max_v05_diff, max(abs(float(a) - float(b)) for a, b in zip(p, target["P"]["6"])))
                csv_p = [float(compact_rows[row_index][f"k6_p{j}"]) for j in range(4)]
                max_v05_csv_diff = max(max_v05_csv_diff, max(abs(float(a) - b) for a, b in zip(p, csv_p)))
    if max_v05_diff > 1e-8 or max_v05_csv_diff > 5.1e-7:
        raise AssertionError(f"Claude v0.5 k=6 parity failed: max JSON diff={max_v05_diff}, CSV diff={max_v05_csv_diff}")

    # Review test portability and what the existing v0/v1 assertion actually tests.
    version_test_text = version_bytes["test_versions.py"].decode("utf-8")
    hardcoded_paths = [line.strip() for line in version_test_text.splitlines()
                       if "/home/claude/" in line or "/tmp/claude-" in line]
    version_source = version_bytes["goal3_versions.py"].decode("utf-8")
    test3_uses_canonical_gpt_loader = "nv.load_data(" in version_test_text and "rows_outer =" in version_test_text
    version_uses_alias_loader = "team_aliases" in version_source or "canonical(" in version_source
    v0_v1_same_lambda = (version_module.VERSIONS["v0"]["lam"] == version_module.VERSIONS["v1"]["lam"])
    version_test_portable = len(hardcoded_paths) == 0

    report = {
        "scope": "GOAL3 research-only; no production, main, market weight, prediction log, or existing shadow changes",
        "pinned_sources": {
            "allk_commit": ALLK_COMMIT,
            "versions_commit": VERSIONS_COMMIT,
            "files": {
                name: {
                    "path": spec["path"],
                    "sha256": spec["sha256"],
                    "git_blob_sha1": spec["git_blob_sha1"],
                    "verified": True,
                    "bytes": len(downloaded[name]),
                } for name, spec in ALLK_FILES.items()
            },
            "version_files": {
                name: {
                    "path": path,
                    "git_blob_sha1": blob,
                    "sha256": hashlib.sha256(version_bytes[name]).hexdigest(),
                    "verified": True,
                    "bytes": len(version_bytes[name]),
                } for name, (path, blob) in VERSION_FILES.items()
            },
        },
        "data_integrity": {
            "allk_rows": len(allk_rows),
            "compact_csv_rows": len(compact_rows),
            "gpt_fixed_csv_rows": len(gpt_rows),
            "claude_vs_gpt_csv_exact_row_mismatches": len(csv_mismatches),
            "json_vs_compact_identity_rows": len(allk_rows),
            "json_vs_compact_k6_k10_max_abs_diff": {
                "k6": max(abs(float(r["P"]["6"][j]) - float(c[f"k6_p{j}"]))
                          for r, c in zip(allk_rows, compact_rows) for j in range(4)),
                "k10": max(abs(float(r["P"]["10"][j]) - float(c[f"k10_p{j}"]))
                           for r, c in zip(allk_rows, compact_rows) for j in range(4)),
            },
            "matches": len(match_groups),
            "dates": len(dates),
            "fold_target_counts": fold_counts,
            "side_target_counts": side_counts,
            "league_target_counts": comp_counts,
            "k_candidates": [float(k) for k in k_keys],
            "probability_sum_max_abs_error": max(abs(sum(r["P"][k]) - 1.0) for r in target_rows for k in k_keys),
        },
        "all_k_metrics": metrics,
        "four_category_calibration": calibration,
        "k_stability": {
            "date_loo": {**date_loo, "n_omissions": len(dates)},
            "team_side_loo": {**team_loo, "n_omissions": len(teams)},
        },
        "fixed_k6_minus_k10_bootstrap": {
            "reps": 10_000,
            "seed": 1,
            "date_cluster_percentile_indices": [249, 9749],
            "date_cluster_brier_95": percentile_pair(date_b),
            "date_cluster_logloss_95": percentile_pair(date_l),
            "match_cluster_percentile_indices": [249, 9749],
            "match_cluster_brier_95": percentile_pair(match_b),
            "match_cluster_logloss_95": percentile_pair(match_l),
            "sign_convention": "negative k6-k10 means k=6 lower loss",
        },
        "poisson_residual_review": {
            "conditional_pearson_dispersion_from_lambda_minus_log_p0": pearson,
            "unadjusted_raw_count_variance_to_mean_not_a_conditional_poisson_residual": raw_dispersion,
            "note": "diag2.py's var/mean is an unconditional count-dispersion summary; it is not a model-based Pearson residual statistic.",
        },
        "version_implementation_review": {
            "v05_k6_max_abs_diff_vs_claude_allk_json": max_v05_diff,
            "v05_k6_max_abs_diff_vs_compact_csv": max_v05_csv_diff,
            "v05_reproduced_with_gpt_nested_validation_2026_history": True,
            "gpt_history_git_blob_sha1": git_blob_sha1(history_path.read_bytes()),
            "gpt_aliases_git_blob_sha1": git_blob_sha1(aliases_path.read_bytes()),
            "gpt_2025_csv_git_blob_sha1": git_blob_sha1(csv_2025_path.read_bytes()),
            "nested_validation_git_blob_sha1": git_blob_sha1(nv_path.read_bytes()),
            "raw_goal3_versions_loader_canonicalizes_aliases": version_uses_alias_loader,
            "test3_bypasses_goal3_versions_load_rows": test3_uses_canonical_gpt_loader,
            "test_versions_has_absolute_claude_paths": bool(hardcoded_paths),
            "test_versions_hardcoded_path_lines": hardcoded_paths,
            "test_versions_portable": version_test_portable,
            "v0_v1_same_lambda_and_shared_fit_path": v0_v1_same_lambda,
            "interpretation": "v0/v1 are one specification, not two independent model families. The separate round1666 parity workflow tests Claude's original implementation; this audit tests v0.5 against the fixed all-k outputs.",
            "v2_market_weight": version_module.MARKET_WEIGHT,
            "v2_executed": False,
            "v2_note": "0.7 is only a research constant carried over from toto13; no GOAL3 market blend or production weight was changed or validated.",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(out),
        "allk_rows": len(allk_rows),
        "matches": len(match_groups),
        "dates": len(dates),
        "claude_vs_gpt_compact_csv_exact": len(csv_mismatches) == 0,
        "v05_k6_max_abs_diff": max_v05_diff,
        "date_bootstrap_brier_95": report["fixed_k6_minus_k10_bootstrap"]["date_cluster_brier_95"],
        "date_bootstrap_logloss_95": report["fixed_k6_minus_k10_bootstrap"]["date_cluster_logloss_95"],
        "match_bootstrap_brier_95": report["fixed_k6_minus_k10_bootstrap"]["match_cluster_brier_95"],
        "match_bootstrap_logloss_95": report["fixed_k6_minus_k10_bootstrap"]["match_cluster_logloss_95"],
        "portable_version_test": version_test_portable,
        "raw_dispersion_is_not_pearson_residual": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
