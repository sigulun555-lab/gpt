#!/usr/bin/env python3
"""Same-host OpenBLAS core-dispatch sensitivity for the GOAL3 research cohort.

The parent imports only stdlib. Each worker is a fresh interpreter, so
OPENBLAS_CORETYPE is set before NumPy/SciPy load their native libraries.
Diagnostic only: no model, prediction, threshold, or shadow artifact is changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TARGETS = {
    "Haswell": {"avx2", "fma"},
    "SkylakeX": {"avx2", "fma", "avx512f", "avx512cd", "avx512dq", "avx512bw", "avx512vl"},
}
GATE = 1e-6
COHORT = {"matches": 236, "teamSideTargets": 472, "dates": 17}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_coretype(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def parse_cpu_flags(text):
    """Intersection across processor records; never assume heterogeneous cores."""
    records = re.split(r"\n\s*\n", text.strip())
    per_cpu = []
    for record in records:
        for line in record.splitlines():
            if line.lower().startswith(("flags", "features")) and ":" in line:
                per_cpu.append(set(line.split(":", 1)[1].strip().lower().split()))
                break
    if not per_cpu:
        return set()
    flags = set(per_cpu[0])
    for item in per_cpu[1:]:
        flags.intersection_update(item)
    return flags


def read_cpu_flags():
    try:
        return parse_cpu_flags(Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return set()


def architecture_matches(threadpools, requested):
    pools = [x for x in threadpools if str(x.get("internal_api", "")).lower() == "openblas"]
    return bool(pools) and all(
        normalize_coretype(x.get("architecture")) == normalize_coretype(requested) for x in pools
    )


def summarize_pair(left, right):
    if left.get("sourceSha256") != right.get("sourceSha256"):
        raise ValueError("controlled workers used different input SHA-256 values")
    left_runtime, right_runtime = left.get("runtime", {}), right.get("runtime", {})
    for key in ("python", "numpy", "scipy", "threadpoolctl"):
        if left_runtime.get(key) != right_runtime.get(key):
            raise ValueError(f"controlled workers used different {key} versions")
    for key in ("openblasVersions", "openblasThreadCounts"):
        if sorted(left_runtime.get(key, [])) != sorted(right_runtime.get(key, [])):
            raise ValueError(f"controlled workers used different OpenBLAS {key}")

    per_k = {}
    def bucket(k):
        label = format(float(k), ".15g")
        return per_k.setdefault(label, {
            "fitCount": 0, "gradientGateFlipCount": 0,
            "maximumGradientAbsoluteDifference": 0.0,
            "maximumParameterAbsoluteDifference": 0.0,
            "exactParameterHashMatches": 0,
            "probabilityCellCount": 0, "probabilityCellDifferenceCount": 0,
            "maximumProbabilityAbsoluteDifference": 0.0,
        })

    lf = {(x["date"], float(x["k"])): x for x in left.get("fitDiagnostics", [])}
    rf = {(x["date"], float(x["k"])): x for x in right.get("fitDiagnostics", [])}
    common = sorted(set(lf) & set(rf))
    flips, max_g, max_parameter, max_probability = [], 0.0, 0.0, 0.0
    param_equal = msg_changes = iter_changes = status_changes = success_changes = 0
    for key in common:
        a, b = lf[key], rf[key]
        slot = bucket(key[1])
        slot["fitCount"] += 1
        ga, gb = float(a["maxAbsGradient"]), float(b["maxAbsGradient"])
        gradient_delta = abs(ga - gb)
        max_g = max(max_g, gradient_delta)
        slot["maximumGradientAbsoluteDifference"] = max(
            slot["maximumGradientAbsoluteDifference"], gradient_delta
        )
        if (ga <= GATE) != (gb <= GATE):
            flips.append({"date": key[0], "k": key[1], "leftGradient": ga, "rightGradient": gb})
            slot["gradientGateFlipCount"] += 1
        pa, pb = a["parameters"], b["parameters"]
        if len(pa) != len(pb):
            raise ValueError(f"parameter dimension mismatch at {key}")
        if pa:
            delta = max(abs(float(x) - float(y)) for x, y in zip(pa, pb))
            max_parameter = max(max_parameter, delta)
            slot["maximumParameterAbsoluteDifference"] = max(
                slot["maximumParameterAbsoluteDifference"], delta
            )
        same_hash = a["parameterSha256"] == b["parameterSha256"]
        param_equal += int(same_hash)
        slot["exactParameterHashMatches"] += int(same_hash)
        msg_changes += int(a["message"] != b["message"])
        iter_changes += int(a["iterations"] != b["iterations"])
        status_changes += int(a["status"] != b["status"])
        success_changes += int(a["success"] != b["success"])

    lp = {(int(x["matchIndex"]), float(x["k"])): x for x in left.get("predictions", [])}
    rp = {(int(x["matchIndex"]), float(x["k"])): x for x in right.get("predictions", [])}
    common_pred = sorted(set(lp) & set(rp))
    cells = differing = 0
    for key in common_pred:
        slot = bucket(key[1])
        for side in ("home", "away"):
            av, bv = lp[key][side], rp[key][side]
            if len(av) != 4 or len(bv) != 4:
                raise ValueError(f"invalid four-category probability vector at {key}/{side}")
            for x, y in zip(av, bv):
                delta = abs(float(x) - float(y))
                cells += 1
                differing += int(delta > 0.0)
                max_probability = max(max_probability, delta)
                slot["probabilityCellCount"] += 1
                slot["probabilityCellDifferenceCount"] += int(delta > 0.0)
                slot["maximumProbabilityAbsoluteDifference"] = max(
                    slot["maximumProbabilityAbsoluteDifference"], delta
                )
    return {
        "leftMode": left.get("mode"), "rightMode": right.get("mode"),
        "commonFitCount": len(common), "commonPredictionMatchKCount": len(common_pred),
        "gradientGateFlipCount": len(flips), "gradientGateFlips": flips,
        "maximumGradientAbsoluteDifference": max_g,
        "maximumParameterAbsoluteDifference": max_parameter,
        "maximumProbabilityAbsoluteDifference": max_probability,
        "inputSha256Match": True, "runtimeVersionAndLibraryMatch": True,
        "exactParameterHashMatches": param_equal,
        "optimizerMessageChangeCount": msg_changes,
        "iterationCountChangeCount": iter_changes,
        "optimizerStatusChangeCount": status_changes,
        "optimizerSuccessChangeCount": success_changes,
        "probabilityCellCount": cells, "probabilityCellDifferenceCount": differing,
        "perK": {key: per_k[key] for key in sorted(per_k, key=float)},
    }

def worker(args):
    # Keep all numerical imports below process startup: core type is selected
    # by the parent environment before these modules load.
    import numpy as np
    import scipy
    import threadpoolctl
    from nested_validation import canonical, load_data, predict_shrinkage
    from optimizer_gradient import fit_shrinkage_strict

    controller = threadpoolctl.threadpool_limits(limits=1)
    try:
        pools = threadpoolctl.threadpool_info()
        openblas = [x for x in pools if str(x.get("internal_api", "")).lower() == "openblas"]
        requested = args.coretype if args.mode == "forced" else None
        if not openblas:
            result = {"mode": args.label, "status": "unsupported_blas_backend",
                      "reason": "threadpoolctl found no OpenBLAS library", "threadpools": pools}
            Path(args.out).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            return 0
        if requested and not architecture_matches(pools, requested):
            result = {"mode": args.label, "status": "unverified_dispatch",
                      "reason": "requested OPENBLAS_CORETYPE was not confirmed by every loaded OpenBLAS library",
                      "requestedCoreType": requested, "environmentCoreType": os.environ.get("OPENBLAS_CORETYPE"),
                      "threadpools": pools}
            Path(args.out).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            return 0
        if any(int(x.get("num_threads", 1)) != 1 for x in openblas):
            result = {"mode": args.label, "status": "unverified_thread_count",
                      "reason": "one-thread control was not honored by every OpenBLAS library",
                      "threadpools": pools}
            Path(args.out).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            return 0

        allk_path, history_path = Path(args.allk_input), Path(args.history2026)
        aliases_path, csv_path = Path(args.aliases), Path(args.csv2025)
        allk = json.loads(allk_path.read_text(encoding="utf-8"))
        for key, expected in COHORT.items():
            if allk.get("cohort", {}).get(key) != expected:
                raise ValueError(f"unexpected cohort {key}: {allk.get('cohort', {}).get(key)!r}")
        rows = allk.get("allKPredictions")
        ks = [float(x) for x in allk.get("kCandidates", [])]
        if not isinstance(rows, list) or len(rows) != COHORT["matches"] or len(ks) != 16:
            raise ValueError("all-k input does not match the frozen 236-match / 16-k cohort")
        _, h2026 = load_data(history_path, aliases_path, csv_path)
        if len(h2026) != int(allk["cohort"].get("history2026Rows", -1)):
            raise ValueError("history row count differs from frozen all-k provenance")
        aliases = json.loads(aliases_path.read_text(encoding="utf-8"))["aliases"]
        dates = sorted({str(x["date"]) for x in rows})
        if len(dates) != COHORT["dates"]:
            raise ValueError("unexpected target-date count")

        fits, predictions = [], []
        for date in dates:
            prior = [x for x in h2026 if x["date"] < date]
            if not prior or any(x["date"] >= date for x in prior):
                raise ValueError(f"invalid strict earlier-date slice for {date}")
            indices = [i for i, row in enumerate(rows) if str(row["date"]) == date]
            for k in ks:
                params, team_index, info = fit_shrinkage_strict(prior, k)
                theta = np.asarray(params, dtype="<f8")
                if not np.isfinite(theta).all():
                    raise ValueError(f"non-finite fit parameters for {date}, k={k}")
                fits.append({
                    "date": date, "k": k, "success": bool(info["success"]),
                    "status": int(info["status"]), "iterations": int(info["nit"]),
                    "message": str(info["message"]),
                    "maxAbsGradient": float(info["maxAbsGradient"]),
                    "parameterSha256": hashlib.sha256(theta.tobytes()).hexdigest(),
                    "parameters": [float(x) for x in theta],
                })
                for index in indices:
                    row = rows[index]
                    home = canonical(row["home"], aliases)
                    away = canonical(row["away"], aliases)
                    ph, pa = predict_shrinkage(params, team_index, home, away)
                    hv, av = [float(x) for x in ph], [float(x) for x in pa]
                    for vector in (hv, av):
                        if len(vector) != 4 or not all(np.isfinite(vector)) or abs(sum(vector) - 1.0) > 1e-12:
                            raise ValueError(f"invalid probability vector at match={index}, k={k}")
                    predictions.append({"matchIndex": index, "k": k, "home": hv, "away": av})

        result = {
            "mode": args.label, "status": "complete", "requestedCoreType": requested,
            "environmentCoreType": os.environ.get("OPENBLAS_CORETYPE"),
            "fitCount": len(fits), "fitSuccessCount": sum(x["success"] for x in fits),
            "withinGradientGateCount": sum(x["maxAbsGradient"] <= GATE for x in fits),
            "maximumGradient": max(x["maxAbsGradient"] for x in fits),
            "predictionMatchKCount": len(predictions),
            "runtime": {
                "python": platform.python_version(), "platform": platform.platform(),
                "machine": platform.machine(), "processor": platform.processor(),
                "numpy": np.__version__, "scipy": scipy.__version__,
                "threadpoolctl": threadpoolctl.__version__, "threadpools": pools,
                "openblasCoreTypes": [x.get("architecture") for x in openblas],
                "openblasVersions": [x.get("version") for x in openblas],
                "openblasThreadCounts": [x.get("num_threads") for x in openblas],
                "cpuFlags": sorted(read_cpu_flags()),
            },
            "sourceSha256": {
                "allk": sha256_file(allk_path), "history2026": sha256_file(history_path),
                "aliases": sha256_file(aliases_path), "csv2025": sha256_file(csv_path),
            },
            "fitDiagnostics": fits, "predictions": predictions,
        }
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        return 0
    finally:
        controller.restore_original_limits()


def run_worker(script, label, coretype, args, output):
    env = dict(os.environ)
    env.update({
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    mode = "forced" if coretype else "native"
    if coretype:
        env["OPENBLAS_CORETYPE"] = coretype
    else:
        env.pop("OPENBLAS_CORETYPE", None)
    cmd = [sys.executable, str(script), "--worker", "--mode", mode, "--label", label,
           "--allk-input", str(args.allk_input), "--history2026", str(args.history2026),
           "--aliases", str(args.aliases), "--csv2025", str(args.csv2025), "--out", str(output)]
    if coretype:
        cmd += ["--coretype", coretype]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=args.timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return {"mode": label, "status": "worker_timeout"}
    if not output.exists():
        return {"mode": label, "status": "worker_failed", "returnCode": p.returncode,
                "stderrTail": (p.stderr or "")[-2000:]}
    result = json.loads(output.read_text(encoding="utf-8"))
    if p.returncode != 0:
        result["returnCode"] = p.returncode
        result["stderrTail"] = (p.stderr or "")[-2000:]
        if result.get("status") == "complete":
            result["status"] = "worker_failed"
    return result


def parent_run(args):
    flags = read_cpu_flags()
    results = {}
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="goal3-openblas-dispatch-") as temp:
        root = Path(temp)
        results["native"] = run_worker(script, "native", None, args, root / "native.json")
        for target in ("Haswell", "SkylakeX"):
            missing = sorted(TARGETS[target] - flags)
            if missing:
                results[target] = {"mode": target, "status": "skipped_hardware_flags",
                                   "missingRequiredCpuFlags": missing}
            else:
                results[target] = run_worker(script, target, target, args, root / f"{target}.json")
        comparisons = {}
        for left, right in (("native", "Haswell"), ("native", "SkylakeX"), ("Haswell", "SkylakeX")):
            a, b = results[left], results[right]
            if a.get("status") == "complete" and b.get("status") == "complete":
                comparisons[f"{left}_vs_{right}"] = summarize_pair(a, b)
            else:
                comparisons[f"{left}_vs_{right}"] = {
                    "status": "not_comparable", "leftStatus": a.get("status"), "rightStatus": b.get("status")
                }
        complete = all(results[x].get("status") == "complete" and results[x].get("fitCount") == 272
                       for x in ("native", "Haswell", "SkylakeX"))
        any_complete = any(x.get("status") == "complete" for x in results.values())
        status = "complete" if complete else ("partial" if any_complete else "unsupported")
        summaries = {}
        for name, item in results.items():
            summaries[name] = {k: item.get(k) for k in (
                "status", "fitCount", "fitSuccessCount", "withinGradientGateCount",
                "maximumGradient", "predictionMatchKCount", "requestedCoreType",
                "environmentCoreType", "runtime", "sourceSha256", "missingRequiredCpuFlags",
                "reason", "returnCode", "stderrTail"
            ) if k in item}
        return {
            "schemaVersion": 1, "status": status,
            "purpose": "Controlled same-host OpenBLAS kernel-dispatch sensitivity; diagnostic only",
            "gradientGate": GATE, "cohort": COHORT, "cpuFlagsIntersection": sorted(flags),
            "requiredFlagsByTarget": {k: sorted(v) for k, v in TARGETS.items()},
            "modeSummaries": summaries, "comparisons": comparisons,
            "sourceSha256": {
                "allk": sha256_file(args.allk_input), "history2026": sha256_file(args.history2026),
                "aliases": sha256_file(args.aliases), "csv2025": sha256_file(args.csv2025),
                "script": sha256_file(script),
            },
            "interpretation": (
                "Complete means native, Haswell and SkylakeX were all verified on one host. "
                "Partial/unsupported is not evidence that CPU dispatch is harmless. "
                "The 1e-6 gate and all model outputs remain unchanged."
            ),
            "openblasDocumentation": "https://www.openmathlib.org/OpenBLAS/docs/faq/",
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=("native", "forced"))
    p.add_argument("--label", default="native")
    p.add_argument("--coretype", choices=tuple(TARGETS))
    p.add_argument("--allk-input", required=True)
    p.add_argument("--history2026", required=True)
    p.add_argument("--aliases", required=True)
    p.add_argument("--csv2025", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--timeout-seconds", type=int, default=900)
    args = p.parse_args()
    if args.worker:
        if args.mode == "forced" and not args.coretype:
            p.error("forced worker requires --coretype")
        try:
            return worker(args)
        except Exception as exc:
            Path(args.out).write_text(json.dumps({
                "mode": args.label, "status": "worker_error",
                "errorType": type(exc).__name__, "error": str(exc),
            }, sort_keys=True), encoding="utf-8")
            return 2
    report = parent_run(args)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "modeStatuses": {k: v["status"] for k, v in report["modeSummaries"].items()},
                      "comparisons": report["comparisons"], "output": args.out}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
