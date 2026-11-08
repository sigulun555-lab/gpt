#!/usr/bin/env python3
"""Independent, read-only audit of Claude's GOAL3 all-k and version-share artifacts.

Pinned external sources are fetched by immutable commit SHA. This script only
writes a JSON report under the research PR workspace; it never edits forecasts,
production code, main, market weights, or existing shadow artifacts.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOTO_REPO = "sigulun555-lab/toto-lab"
GPT_REPO = "sigulun555-lab/gpt"
CLAUDE_DOCUMENTED_GPT_COMMIT = "2adf65a20247dd30c5b86ae95c22bd75980cda89"
CLAUDE_DOCUMENTED_GPT_FILES = {
    "history.json": "data/history.json",
    "team_aliases.json": "data/team_aliases.json",
    "2025.csv": "verification/goal3_independent/2025.csv",
    "nested_validation.py": "verification/goal3_independent/nested_validation.py",
}
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


def fetch_pinned_gpt(commit: str, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{GPT_REPO}/{commit}/{path}"
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
    # Claude README documents this exact GPT PR #19 dependency snapshot.
    # The original generation manifest omitted these hashes, so this verifies
    # the documented snapshot but cannot prove which local bytes were consumed.
    documented_gpt_bytes = {
        name: fetch_pinned_gpt(CLAUDE_DOCUMENTED_GPT_COMMIT, path)
        for name, path in CLAUDE_DOCUMENTED_GPT_FILES.items()
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
    team_side_loo = {"brier_negative": 0, "logloss_negative": 0, "k6_best_brier": 0, "k6_best_logloss": 0}
    team_match_loo = {"brier_negative": 0, "logloss_negative": 0, "k6_best_brier": 0, "k6_best_logloss": 0}
    for team in teams:
        # Target-team LOO removes only the omitted team's own team-side rows.
        ix_side = [i for i, r in enumerate(target_rows) if r["target_team"] != team]
        m_side = {k: mean_metrics(target_rows, k, ix_side) for k in k_keys}
        team_side_loo["brier_negative"] += int(m_side["6"]["brier"] < m_side["10"]["brier"])
        team_side_loo["logloss_negative"] += int(m_side["6"]["logloss"] < m_side["10"]["logloss"])
        team_side_loo["k6_best_brier"] += int(min(k_keys, key=lambda k: m_side[k]["brier"]) == "6")
        team_side_loo["k6_best_logloss"] += int(min(k_keys, key=lambda k: m_side[k]["logloss"]) == "6")
        # Match-involvement LOO removes both target rows from every match involving the team.
        ix_match = [i for i, r in enumerate(target_rows) if team not in (r["home"], r["away"])]
        m_match = {k: mean_metrics(target_rows, k, ix_match) for k in k_keys}
        team_match_loo["brier_negative"] += int(m_match["6"]["brier"] < m_match["10"]["brier"])
        team_match_loo["logloss_negative"] += int(m_match["6"]["logloss"] < m_match["10"]["logloss"])
        team_match_loo["k6_best_brier"] += int(min(k_keys, key=lambda k: m_match[k]["brier"]) == "6")
        team_match_loo["k6_best_logloss"] += int(min(k_keys, key=lambda k: m_match[k]["logloss"]) == "6")

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

    # Preserve Claude's exact pinned source bytes, then create a separate
    # research-only copy that closes file handles without changing model logic.
    version_path = ROOT / "verification" / "goal3_independent" / "results" / "_claude_goal3_versions.py"
    portable_version_path = ROOT / "verification" / "goal3_independent" / "results" / "_claude_goal3_versions_portable.py"
    version_path.parent.mkdir(parents=True, exist_ok=True)
    original_version_bytes = version_bytes["goal3_versions.py"]
    version_path.write_bytes(original_version_bytes)
    original_version_source = original_version_bytes.decode("utf-8")
    portable_version_source = original_version_source
    file_handle_patches = [
        (
            "load_rows_history",
            '    H = json.load(open(history_path, encoding="utf-8"))["history"]\n',
            '    with open(history_path, encoding="utf-8") as f:\n        H = json.load(f)["history"]\n',
        ),
        (
            "cli_fixtures",
            '    fixtures = [tuple(x) for x in json.load(open(a.fixtures, encoding="utf-8"))]\n',
            '    with open(a.fixtures, encoding="utf-8") as f:\n        fixtures = [tuple(x) for x in json.load(f)]\n',
        ),
        (
            "cli_market",
            '    market_raw = json.load(open(a.market, encoding="utf-8")) if a.market else None\n',
            '    if a.market:\n        with open(a.market, encoding="utf-8") as f:\n            market_raw = json.load(f)\n    else:\n        market_raw = None\n',
        ),
        (
            "cli_output",
            '    json.dump(result, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)\n',
            '    with open(a.out, "w", encoding="utf-8") as f:\n        json.dump(result, f, ensure_ascii=False, indent=2)\n',
        ),
    ]
    applied_file_handle_patches = []
    for patch_name, old_text, new_text in file_handle_patches:
        if portable_version_source.count(old_text) != 1:
            raise AssertionError(f"expected exactly one pinned source occurrence for {patch_name}")
        portable_version_source = portable_version_source.replace(old_text, new_text, 1)
        applied_file_handle_patches.append(patch_name)
    portable_version_path.write_text(portable_version_source, encoding="utf-8")
    portable_version_sha256 = hashlib.sha256(portable_version_source.encode("utf-8")).hexdigest()

    original_spec = importlib.util.spec_from_file_location("claude_goal3_versions_original", version_path)
    original_version_module = importlib.util.module_from_spec(original_spec)
    original_spec.loader.exec_module(original_version_module)
    portable_spec = importlib.util.spec_from_file_location("claude_goal3_versions_portable", portable_version_path)
    version_module = importlib.util.module_from_spec(portable_spec)
    portable_spec.loader.exec_module(version_module)

    # Rebuild the exact 2026-only canonicalized history used by GPT nested_validation.
    nv_path = ROOT / "verification" / "goal3_independent" / "nested_validation.py"
    nv_spec = importlib.util.spec_from_file_location("nested_validation_for_claude_parity", nv_path)
    nv = importlib.util.module_from_spec(nv_spec)
    nv_spec.loader.exec_module(nv)
    history_path = ROOT / "data" / "history.json"
    aliases_path = ROOT / "data" / "team_aliases.json"
    csv_2025_path = ROOT / "verification" / "goal3_independent" / "2025.csv"
    h2025, h2026 = nv.load_data(str(history_path), str(aliases_path), str(csv_2025_path))

    # Re-run the loader from the exact GPT PR #19 dependency commit documented
    # by Claude. This tests whether the later fail-closed loader changed the
    # actual 2025/2026 training population, independently of the model core.
    documented_nv_path = ROOT / "verification" / "goal3_independent" / "results" / "_claude_documented_nested_validation.py"
    documented_nv_path.write_bytes(documented_gpt_bytes["nested_validation.py"])
    documented_nv_spec = importlib.util.spec_from_file_location("claude_documented_nested_validation", documented_nv_path)
    documented_nv = importlib.util.module_from_spec(documented_nv_spec)
    documented_nv_spec.loader.exec_module(documented_nv)
    documented_h2025, documented_h2026 = documented_nv.load_data(
        str(history_path), str(aliases_path), str(csv_2025_path)
    )
    old_nested_text = documented_gpt_bytes["nested_validation.py"].decode("utf-8")
    current_nested_text = nv_path.read_text(encoding="utf-8")
    old_core = old_nested_text.split("def bin_goals", 1)[1].split("\ndef main()", 1)[0]
    current_core = current_nested_text.split("def bin_goals", 1)[1].split("\ndef main()", 1)[0]
    documented_loader_equivalence = {
        "documented_commit": CLAUDE_DOCUMENTED_GPT_COMMIT,
        "2025_rows": len(documented_h2025),
        "2026_rows": len(documented_h2026),
        "2025_rows_exact_match": documented_h2025 == h2025,
        "2026_rows_exact_match": documented_h2026 == h2026,
        "strict_loader_population_unchanged": documented_h2025 == h2025 and documented_h2026 == h2026,
        "nested_validation_full_file_identical": old_nested_text == current_nested_text,
        "nested_validation_model_core_identical": old_core == current_core,
    }
    if not documented_loader_equivalence["strict_loader_population_unchanged"]:
        raise AssertionError(f"documented dependency loader population differs from PR #20: {documented_loader_equivalence}")
    if not documented_loader_equivalence["nested_validation_model_core_identical"]:
        raise AssertionError("documented nested_validation fit/predict core differs from current PR #20 code")
    expected_input_blobs = {
        "history.json": (history_path, "f5cc8c044758e45b1e679e4ce796e68b3f27a854"),
        "team_aliases.json": (aliases_path, "5de22584524ff4c07ddac1d27f49afbbbc5f88ab"),
        "2025.csv": (csv_2025_path, "847d324c5a6168d02efe328f8173418a951af720"),
        "nested_validation.py": (nv_path, "58d07e711faf2cfa152dae2902707c3f57a2fb7c"),
        "goal3_outer_matches.csv": (gpt_csv_path, "56892251679b8774a9e1b7ccc4b5beb42e660e36"),
    }
    for label, (path, expected_blob) in expected_input_blobs.items():
        actual_blob = git_blob_sha1(path.read_bytes())
        if actual_blob != expected_blob:
            raise AssertionError(f"GPT input blob mismatch for {label}: expected {expected_blob}, got {actual_blob}")
    if len(h2026) != 429:
        raise AssertionError(f"pinned GPT PR history expected 429 valid 2026 rows, got {len(h2026)}")

    current_input_paths = {
        "history.json": history_path,
        "team_aliases.json": aliases_path,
        "2025.csv": csv_2025_path,
        "nested_validation.py": nv_path,
    }
    independent_input_manifest = {}
    documented_dependency_manifest = {}
    for name, path in current_input_paths.items():
        current_bytes = path.read_bytes()
        documented_bytes = documented_gpt_bytes[name]
        independent_input_manifest[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": len(current_bytes),
            "sha256": hashlib.sha256(current_bytes).hexdigest(),
            "git_blob_sha1": git_blob_sha1(current_bytes),
        }
        documented_dependency_manifest[name] = {
            "path": CLAUDE_DOCUMENTED_GPT_FILES[name],
            "bytes": len(documented_bytes),
            "sha256": hashlib.sha256(documented_bytes).hexdigest(),
            "git_blob_sha1": git_blob_sha1(documented_bytes),
            "matches_pr20_current_bytes": documented_bytes == current_bytes,
        }
    documented_dependency_manifest["nested_validation.py"]["core_from_bin_goals_identical"] = documented_loader_equivalence["nested_validation_model_core_identical"]
    documented_dependency_manifest["nested_validation.py"]["documented_loader_population_unchanged"] = documented_loader_equivalence["strict_loader_population_unchanged"]

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
                if int(hist_match["hg"] if side == "home" else hist_match["ag"]) != int(target["g"]):
                    raise AssertionError(f"target goals differ from pinned history for {key} {side}")
                p = side_probs[side]
                max_v05_diff = max(max_v05_diff, max(abs(float(a) - float(b)) for a, b in zip(p, target["P"]["6"])))
                csv_p = [float(compact_rows[row_index][f"k6_p{j}"]) for j in range(4)]
                max_v05_csv_diff = max(max_v05_csv_diff, max(abs(float(a) - b) for a, b in zip(p, csv_p)))
    if max_v05_diff > 1e-8 or max_v05_csv_diff > 5.1e-7:
        raise AssertionError(f"Claude v0.5 k=6 parity failed: max JSON diff={max_v05_diff}, CSV diff={max_v05_csv_diff}")

    # Also exercise the version script's own CLI loader, not only test_versions.py's
    # injected canonicalized rows. Its labels are raw source names and its sort key
    # differs from nested_validation; verify that this path remains equivalent here.
    with warnings.catch_warnings(record=True) as original_loader_warnings:
        warnings.simplefilter("always", ResourceWarning)
        original_raw_version_rows = original_version_module.load_rows(str(history_path))
        gc.collect()
    with warnings.catch_warnings(record=True) as portable_loader_warnings:
        warnings.simplefilter("always", ResourceWarning)
        raw_version_rows = version_module.load_rows(str(history_path))
        gc.collect()
    if original_raw_version_rows != raw_version_rows:
        raise AssertionError("file-handle-only portability patch changed load_rows output")
    resource_warning_review = {
        "original_pinned_load_rows_resource_warning_count": sum(
            issubclass(item.category, ResourceWarning) for item in original_loader_warnings
        ),
        "patched_portable_load_rows_resource_warning_count": sum(
            issubclass(item.category, ResourceWarning) for item in portable_loader_warnings
        ),
        "original_and_patched_rows_exact_match": original_raw_version_rows == raw_version_rows,
        "upstream_source_modified": False,
    }
    if resource_warning_review["patched_portable_load_rows_resource_warning_count"] != 0:
        raise AssertionError("portable loader still emits ResourceWarning")
    raw_by_key = {}
    canonical_to_raw_names: dict[str, set[str]] = defaultdict(set)
    for r in raw_version_rows:
        ch = nv.canonical(r["home"], aliases)
        ca = nv.canonical(r["away"], aliases)
        canonical_to_raw_names[ch].add(r["home"])
        canonical_to_raw_names[ca].add(r["away"])
        key = (r["date"], r["competition"], ch, ca)
        if key in raw_by_key:
            raise AssertionError(f"duplicate raw-loader canonicalized key: {key}")
        raw_by_key[key] = r
    alias_collision_groups = {k: sorted(v) for k, v in canonical_to_raw_names.items() if len(v) > 1}
    if len(raw_version_rows) != len(h2026) or set(raw_by_key) != set(h2026_by_key):
        raise AssertionError("goal3_versions.load_rows population differs from pinned nested_validation 2026 history")
    raw_loader_max_abs_diff = 0.0
    for d in dates:
        prior_raw = [r for r in raw_version_rows if r["date"] < d]
        raw_params = version_module.fit(prior_raw, 6.0)
        for key in [x for x in target_keys if x[0] == d]:
            raw_match = raw_by_key[key]
            raw_home_p, raw_away_p = version_module.predict_side(raw_params, raw_match["home"], raw_match["away"])
            for side, probs in (("home", raw_home_p), ("away", raw_away_p)):
                row_index = next(i for i, cc in enumerate(compact_rows)
                                 if (cc["date"], cc["comp"], cc["home"], cc["away"], cc["side"]) == (*key, side))
                ref = target_rows[row_index]["P"]["6"]
                raw_loader_max_abs_diff = max(raw_loader_max_abs_diff,
                                              max(abs(float(a) - float(b)) for a, b in zip(probs, ref)))
    if raw_loader_max_abs_diff > 1e-8:
        raise AssertionError(f"goal3_versions.load_rows function parity exceeded 1e-8: {raw_loader_max_abs_diff}")

    # Exercise the actual portable module CLI on the full pinned 429-row history
    # and one real outer fixture. The separate unittest CLI smoke test uses
    # synthetic temporary inputs; this closes the real-data end-to-end gap.
    cli_key = target_keys[0]
    cli_raw_match = raw_by_key[cli_key]
    cli_prior_count = sum(r["date"] < cli_key[0] for r in raw_version_rows)
    with tempfile.TemporaryDirectory(prefix="goal3-real-cli-") as tmp:
        tmp_root = Path(tmp)
        cli_fixtures_path = tmp_root / "fixtures.json"
        cli_output_path = tmp_root / "cli_output.json"
        cli_fixtures_path.write_text(
            json.dumps([[cli_raw_match["home"], cli_raw_match["away"]]], ensure_ascii=False),
            encoding="utf-8",
        )
        cli_command = [
            sys.executable, str(portable_version_path),
            "--history", str(history_path),
            "--target-date", cli_key[0],
            "--fixtures", str(cli_fixtures_path),
            "--versions", "v0.5",
            "--out", str(cli_output_path),
        ]
        cli_env = os.environ.copy()
        cli_env["PYTHONWARNINGS"] = "always::ResourceWarning"
        cli_process = subprocess.run(
            cli_command, capture_output=True, text=True, env=cli_env, check=False
        )
        if cli_process.returncode != 0:
            raise AssertionError(
                "real-history portable CLI failed: "
                f"returncode={cli_process.returncode}; stderr={cli_process.stderr[-4000:]}"
            )
        if "ResourceWarning" in cli_process.stderr:
            raise AssertionError(f"real-history portable CLI emitted ResourceWarning: {cli_process.stderr}")
        if not cli_output_path.is_file():
            raise AssertionError("real-history portable CLI did not create its output")
        cli_payload = json.loads(cli_output_path.read_text(encoding="utf-8"))
        cli_version = cli_payload.get("versions", {}).get("v0.5", {})
        cli_fixtures = cli_version.get("fixtures", [])
        if cli_version.get("status") != "ok" or len(cli_fixtures) != 1:
            raise AssertionError(f"unexpected real-history CLI payload: {cli_version}")
        cli_fixture = cli_fixtures[0]
        if (cli_fixture.get("home"), cli_fixture.get("away")) != (
            cli_raw_match["home"], cli_raw_match["away"]
        ):
            raise AssertionError("real-history CLI fixture identity changed")
        cli_expected_params = version_module.fit(
            [r for r in raw_version_rows if r["date"] < cli_key[0]], 6.0
        )
        cli_expected_home, cli_expected_away = version_module.predict_side(
            cli_expected_params, cli_raw_match["home"], cli_raw_match["away"]
        )
        cli_max_abs_diff = max(
            max(abs(float(a) - float(b)) for a, b in zip(cli_fixture["home_probs"], cli_expected_home)),
            max(abs(float(a) - float(b)) for a, b in zip(cli_fixture["away_probs"], cli_expected_away)),
        )
        if cli_max_abs_diff > 1e-12:
            raise AssertionError(f"real-history CLI/function probability mismatch: {cli_max_abs_diff}")
        if cli_payload.get("history_sha256") != hashlib.sha256(history_path.read_bytes()).hexdigest():
            raise AssertionError("real-history CLI input SHA-256 mismatch")
        if int(cli_payload.get("history_rows_used", -1)) != cli_prior_count:
            raise AssertionError("real-history CLI strict prior-row count mismatch")
        cli_real_history_reproduction = {
            "status": "passed",
            "targetDate": cli_key[0],
            "fixture": {"home": cli_raw_match["home"], "away": cli_raw_match["away"]},
            "historyRowsLoaded": len(raw_version_rows),
            "historyRowsUsedStrictlyBeforeTargetDate": cli_prior_count,
            "historySha256": cli_payload["history_sha256"],
            "portableModuleSha256": portable_version_sha256,
            "maxAbsProbabilityDiffVsInProcessReference": cli_max_abs_diff,
            "resourceWarningCount": 0,
            "execution": "subprocess CLI on the pinned full history and one actual outer fixture",
        }

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
        "generation_provenance": {
            "claude_original_generation_manifest": {
                "status": "not_provided",
                "missing_input_hashes": ["data/history.json SHA-256", "data/team_aliases.json SHA-256",
                                         "verification/goal3_independent/nested_validation.py SHA-256"],
                "documented_dependency_commit": CLAUDE_DOCUMENTED_GPT_COMMIT,
                "execution_linkage_proven": False,
                "note": "The Claude README documents this GPT PR #19 dependency commit. These hashes verify that commit and compare it with PR #20, but the original run manifest is absent, so the exact local bytes consumed at generation time are not proven.",
            },
            "claude_documented_dependency_snapshot": documented_dependency_manifest,
            "documented_loader_reproduction": documented_loader_equivalence,
            "gpt_independent_reproduction_manifest": {
                "status": "verified",
                "inputs": independent_input_manifest,
                "allk_json_sha256": hashlib.sha256(downloaded["goal3_outer_allk.json"]).hexdigest(),
                "compact_csv_sha256": hashlib.sha256(downloaded["goal3_outer_matches_compact.csv"]).hexdigest(),
                "reproduction": "independent per-date 2026-only fit and probability comparison against all 472 all-k targets and compact CSV",
                "v05_k6_max_abs_diff": max_v05_diff,
                "raw_load_rows_k6_max_abs_diff": raw_loader_max_abs_diff,
            },
        },
        "all_k_metrics": metrics,
        "four_category_calibration": calibration,
        "k_stability": {
            "date_loo": {**date_loo, "n_omissions": len(dates)},
            "team_match_loo": {**team_match_loo, "n_omissions": len(teams), "definition": "remove both team-side rows from every match involving the omitted team"},
            "team_side_loo": {**team_side_loo, "n_omissions": len(teams), "definition": "remove only target rows whose predicted team is the omitted team"},
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
            "raw_loader_rows": len(raw_version_rows),
            "raw_loader_alias_collision_groups": alias_collision_groups,
            "raw_loader_match_population_matches_nested_validation": set(raw_by_key) == set(h2026_by_key),
            "raw_loader_k6_max_abs_diff_vs_allk_json": raw_loader_max_abs_diff,
            "raw_loader_requires_fixture_labels_to_match_raw_history_names": True,
            "load_rows_function_reproduction": {
                "status": "passed",
                "rows": len(raw_version_rows),
                "canonicalized_population_matches_nested_validation": set(raw_by_key) == set(h2026_by_key),
                "max_abs_probability_diff": raw_loader_max_abs_diff,
            },
            "real_history_cli_reproduction": cli_real_history_reproduction,
            "gpt_portable_harness_path": "tests/test_goal3_claude_versions_portable.py",
            "portable_module_path": str(portable_version_path.relative_to(ROOT)),
            "portable_module_sha256": portable_version_sha256,
            "portable_runtime_patch": {
                "status": "applied_to_research_copy_only",
                "upstream_source_sha256": hashlib.sha256(original_version_bytes).hexdigest(),
                "upstream_source_git_blob_sha1": git_blob_sha1(original_version_bytes),
                "patched_copy_sha256": portable_version_sha256,
                "file_handle_patches": applied_file_handle_patches,
                "upstream_source_modified": False,
                "model_math_or_data_filter_changed": False,
            },
            "resource_warning_review": resource_warning_review,
            "original_claude_test_file_remains_unmodified": True,
            "original_claude_test4_calls_run_version_v2": False,
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
