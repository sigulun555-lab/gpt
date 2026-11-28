#!/usr/bin/env python3
"""Independent exploratory probability-bin and one-vs-rest calibration audit for GOAL3.

Research-only. Reads the frozen all-k outer reconstruction and does not refit
the football model. All uncertainty resampling is by target date, retaining
every match and both team-sides on each sampled date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import date as Date
from pathlib import Path

import numpy as np

K_VALUES = (6.0, 10.0)
CATEGORIES = (0, 1, 2, 3)
PROBABILITY_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0000000001)
MIN_BIN_TARGETS_FOR_CI = 20
LOGIT_EPS = 1e-6


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def flatten(report: dict) -> tuple[list[dict], list[str]]:
    matches = report.get("allKPredictions")
    if not isinstance(matches, list) or len(matches) != 236:
        raise ValueError("expected exactly 236 all-k matches")
    rows: list[dict] = []
    seen = set()
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("match must be an object")
        date = match.get("date")
        league = match.get("comp")
        home, away = match.get("home"), match.get("away")
        if not isinstance(date, str):
            raise ValueError("invalid 2026 target date")
        try:
            parsed_date = Date.fromisoformat(date)
        except ValueError as exc:
            raise ValueError("invalid 2026 target date") from exc
        if parsed_date.isoformat() != date or parsed_date.year != 2026:
            raise ValueError("invalid 2026 target date")
        if league not in ("J1", "J2", "J3"):
            raise ValueError("invalid league")
        if not isinstance(home, str) or not home.strip() or not isinstance(away, str) or not away.strip():
            raise ValueError("missing team")
        if home.strip() == away.strip():
            raise ValueError("home and away collide")
        key = (date, league, home.strip(), away.strip())
        if key in seen:
            raise ValueError("duplicate match")
        seen.add(key)
        by_k = match.get("byK")
        if not isinstance(by_k, dict):
            raise ValueError("missing byK")
        for side, goal_field in (("home", "hg"), ("away", "ag")):
            goals = match.get(goal_field)
            if type(goals) is not int or goals < 0:
                raise ValueError("invalid observed goal count")
            actual = min(goals, 3)
            probs = {}
            for k in K_VALUES:
                p = np.asarray(by_k.get(f"{k:.1f}", {}).get(side), dtype=float)
                if p.shape != (4,) or not np.isfinite(p).all():
                    raise ValueError("invalid probability vector")
                if np.any(p < 0) or np.any(p > 1) or abs(float(p.sum()) - 1.0) > 1e-9:
                    raise ValueError("probability simplex violation")
                probs[k] = p
            rows.append({"date": date, "league": league, "actual": actual, "probabilities": probs})
    dates = sorted({r["date"] for r in rows})
    if len(rows) != 472 or len(dates) != 17:
        raise ValueError("expected 472 team-sides across 17 dates")
    return rows, dates


def make_date_multiplicities(dates: list[str], replicates: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, len(dates), size=(replicates, len(dates)))
    multiplicities = np.zeros((replicates, len(dates)), dtype=np.int16)
    for j in range(len(dates)):
        multiplicities[:, j] = np.sum(draws == j, axis=1)
    return multiplicities


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def batch_logistic_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                       initial: np.ndarray | None = None,
                       max_iter: int = 80, tol: float = 1e-7):
    """Vectorized Newton fit of logit Pr(Y=1)=alpha+beta*logit(p)."""
    b_count = weights.shape[0]
    theta = np.tile(np.asarray(initial if initial is not None else [0.0, 1.0], dtype=float), (b_count, 1))
    valid = np.ones(b_count, dtype=bool)

    def loss(th):
        z = th[:, 0, None] + th[:, 1, None] * x[None, :]
        return np.sum(weights * (np.logaddexp(0.0, z) - y[None, :] * z), axis=1)

    for _ in range(max_iter):
        z = theta[:, 0, None] + theta[:, 1, None] * x[None, :]
        pr = sigmoid(z)
        resid = weights * (pr - y[None, :])
        g0 = resid.sum(axis=1)
        g1 = (resid * x[None, :]).sum(axis=1)
        grad = np.maximum(np.abs(g0), np.abs(g1))
        done = grad < tol
        if np.all(done | ~valid):
            break
        q = weights * pr * (1.0 - pr)
        h00 = q.sum(axis=1)
        h01 = (q * x[None, :]).sum(axis=1)
        h11 = (q * (x * x)[None, :]).sum(axis=1)
        det = h00 * h11 - h01 * h01
        good = valid & ~done & np.isfinite(det) & (np.abs(det) > 1e-14)
        d0 = np.zeros(b_count)
        d1 = np.zeros(b_count)
        d0[good] = (h11[good] * g0[good] - h01[good] * g1[good]) / det[good]
        d1[good] = (-h01[good] * g0[good] + h00[good] * g1[good]) / det[good]
        valid &= (done | good)
        old_loss = loss(theta)
        scale = np.ones(b_count)
        accepted = done | ~valid
        for _ls in range(24):
            candidate = theta.copy()
            candidate[:, 0] -= scale * d0
            candidate[:, 1] -= scale * d1
            new_loss = loss(candidate)
            accept_now = valid & (new_loss <= old_loss + 1e-10)
            accepted |= accept_now
            theta[accept_now] = candidate[accept_now]
            need = valid & ~accepted
            if not np.any(need):
                break
            scale[need] *= 0.5
        valid &= accepted
        if not np.any(valid):
            break

    z = theta[:, 0, None] + theta[:, 1, None] * x[None, :]
    pr = sigmoid(z)
    resid = weights * (pr - y[None, :])
    grad = np.maximum(np.abs(resid.sum(axis=1)), np.abs((resid * x[None, :]).sum(axis=1)))
    valid &= np.isfinite(theta).all(axis=1) & (grad < 1e-5)
    return theta, valid, grad


def calibration_fit_and_bootstrap(rows, dates, multiplicities, k, category):
    p = np.asarray([r["probabilities"][k][category] for r in rows], dtype=float)
    y = np.asarray([int(r["actual"] == category) for r in rows], dtype=float)
    date_idx = np.asarray([dates.index(r["date"]) for r in rows], dtype=int)
    clipped = np.clip(p, LOGIT_EPS, 1.0 - LOGIT_EPS)
    x = np.log(clipped / (1.0 - clipped))
    point, point_valid, point_grad = batch_logistic_fit(x, y, np.ones((1, len(rows))))
    if not point_valid[0]:
        raise ValueError(f"point calibration fit failed for k={k}, category={category}")
    boot_weights = multiplicities[:, date_idx].astype(float)
    boot, valid, grad = batch_logistic_fit(x, y, boot_weights, initial=point[0])
    vals = boot[valid]
    if len(vals) < 0.95 * len(boot):
        raise ValueError(f"fewer than 95% valid date-bootstrap calibration fits for k={k}, category={category}")
    return {
        "n": len(rows),
        "events": int(y.sum()),
        "intercept": float(point[0, 0]),
        "slope": float(point[0, 1]),
        "pointGradientInfinityNorm": float(point_grad[0]),
        "bootstrapReplicates": int(len(boot)),
        "validBootstrapReplicates": int(valid.sum()),
        "invalidBootstrapReplicates": int((~valid).sum()),
        "interceptPercentile95CI": np.quantile(vals[:, 0], [0.025, 0.975], method="linear").tolist(),
        "slopePercentile95CI": np.quantile(vals[:, 1], [0.025, 0.975], method="linear").tolist(),
        "bootstrapSE": {"intercept": float(vals[:, 0].std(ddof=1)), "slope": float(vals[:, 1].std(ddof=1))},
        "_bootstrapParameters": boot,
        "_valid": valid,
    }


def probability_bin_audit(rows, dates, multiplicities):
    date_index = {d: i for i, d in enumerate(dates)}
    results = []
    eligible_boot = []
    for k in K_VALUES:
        for category in CATEGORIES:
            p = np.asarray([r["probabilities"][k][category] for r in rows], dtype=float)
            y = np.asarray([int(r["actual"] == category) for r in rows], dtype=float)
            di = np.asarray([date_index[r["date"]] for r in rows], dtype=int)
            for lo, hi in zip(PROBABILITY_BINS[:-1], PROBABILITY_BINS[1:]):
                mask = (p >= lo) & (p < hi)
                n = int(mask.sum())
                if n == 0:
                    results.append({
                        "k": int(k), "category": category,
                        "bin": [lo, min(hi, 1.0)], "n": 0,
                        "meanPredictedProbability": None,
                        "observedFrequency": None,
                        "gapPercentagePoints": None,
                        "ciStatus": "empty_bin",
                    })
                    continue
                n_by_date = np.bincount(di[mask], minlength=len(dates)).astype(float)
                p_by_date = np.bincount(di[mask], weights=p[mask], minlength=len(dates))
                y_by_date = np.bincount(di[mask], weights=y[mask], minlength=len(dates))
                mean_p = float(p[mask].mean())
                obs = float(y[mask].mean())
                gap = (mean_p - obs) * 100.0
                item = {
                    "k": int(k), "category": category,
                    "bin": [lo, min(hi, 1.0)], "n": n,
                    "meanPredictedProbability": mean_p,
                    "observedFrequency": obs,
                    "gapPercentagePoints": gap,
                    "ciStatus": "not_computed_sparse_bin" if n < MIN_BIN_TARGETS_FOR_CI else "pending",
                }
                if n >= MIN_BIN_TARGETS_FOR_CI:
                    bn = multiplicities @ n_by_date
                    bp = multiplicities @ p_by_date
                    by = multiplicities @ y_by_date
                    valid = bn > 0
                    if valid.mean() >= 0.95:
                        gaps = (bp[valid] / bn[valid] - by[valid] / bn[valid]) * 100.0
                        se = float(gaps.std(ddof=1))
                        item["bootstrapReplicatesValid"] = int(valid.sum())
                        item["bootstrapSE"] = se
                        item["gapPercentile95CI"] = np.quantile(gaps, [0.025, 0.975], method="linear").tolist()
                        item["ciStatus"] = "computed"
                        eligible_boot.append((item, gaps, se, gap))
                    else:
                        item["bootstrapReplicatesValid"] = int(valid.sum())
                        item["ciStatus"] = "insufficient_nonempty_bootstrap_replicates"
                results.append(item)

    if eligible_boot:
        z_cols = [(gaps - point) / se for _, gaps, se, point in eligible_boot if se > 0]
        if z_cols:
            critical = float(np.quantile(np.max(np.abs(np.column_stack(z_cols)), axis=1), 0.95, method="linear"))
            for item, _, se, point in eligible_boot:
                item["simultaneous95Band"] = [point - critical * se, point + critical * se]
            family_count = len(eligible_boot)
        else:
            critical, family_count = None, 0
    else:
        critical, family_count = None, 0
    return results, {
        "family": "all non-empty fixed probability bins with at least 20 targets across k=6/10 and categories 0/1/2/3+",
        "comparisonCount": family_count,
        "studentizedMaxAbsTCriticalValue": critical,
        "note": "simultaneous max-|t| band; bins below 20 targets have point estimates only",
    }


def audit(report: dict, input_sha: str, script_sha: str,
          replicates: int = 5000, seed: int = 20261112) -> dict:
    if replicates < 200:
        raise ValueError("at least 200 replicates required outside unit tests")
    rows, dates = flatten(report)
    multiplicities = make_date_multiplicities(dates, replicates, seed)
    calibration = {}
    for k in K_VALUES:
        for category in CATEGORIES:
            calibration[(int(k), category)] = calibration_fit_and_bootstrap(
                rows, dates, multiplicities, k, category
            )

    calibration_columns = []
    calibration_meta = []
    for k in (6, 10):
        for category in CATEGORIES:
            item = calibration[(k, category)]
            vals = item["_bootstrapParameters"]
            point = np.asarray([item["intercept"], item["slope"]])
            for parameter_index, parameter_name in ((0, "intercept"), (1, "slope")):
                se = float(vals[:, parameter_index].std(ddof=1))
                if not np.isfinite(se) or se <= 0:
                    raise ValueError(f"invalid bootstrap SE for {parameter_name}, k={k}, category={category}")
                calibration_columns.append((vals[:, parameter_index] - point[parameter_index]) / se)
                calibration_meta.append((k, category, parameter_name, point[parameter_index], se))
    calibration_critical = float(np.quantile(
        np.max(np.abs(np.column_stack(calibration_columns)), axis=1), 0.95, method="linear"
    ))
    calibration_simultaneous = {}
    for k, category, parameter_name, point, se in calibration_meta:
        calibration_simultaneous[(k, category, parameter_name)] = [
            float(point - calibration_critical * se),
            float(point + calibration_critical * se),
        ]

    calibration_public = []
    paired_columns = []
    paired_meta = []
    paired_intercepts = {}
    paired_slopes = {}
    for category in CATEGORIES:
        a, b = calibration[(6, category)], calibration[(10, category)]
        valid = a["_valid"] & b["_valid"]
        if int(valid.sum()) < 0.95 * replicates:
            raise ValueError(f"fewer than 95% paired calibration fits for category={category}")
        da = a["_bootstrapParameters"][valid, 0] - b["_bootstrapParameters"][valid, 0]
        db = a["_bootstrapParameters"][valid, 1] - b["_bootstrapParameters"][valid, 1]
        for parameter_name, values in (("intercept", da), ("slope", db)):
            point = a[parameter_name] - b[parameter_name]
            se = float(values.std(ddof=1))
            if not np.isfinite(se) or se <= 0:
                raise ValueError(f"invalid paired calibration bootstrap SE for {parameter_name}, category={category}")
            paired_columns.append((values - point) / se)
            paired_meta.append((category, parameter_name, point, se, values))
        paired_intercepts[str(category)] = np.quantile(da, [0.025, 0.975], method="linear").tolist()
        paired_slopes[str(category)] = np.quantile(db, [0.025, 0.975], method="linear").tolist()
        for k in (6, 10):
            item = {key: value for key, value in calibration[(k, category)].items()
                    if not key.startswith("_")}
            item.update({
                "k": k, "category": category,
                "idealIntercept": 0.0, "idealSlope": 1.0,
                "interceptSimultaneous95Band": calibration_simultaneous[(k, category, "intercept")],
                "slopeSimultaneous95Band": calibration_simultaneous[(k, category, "slope")],
                "interpretation": "one-vs-rest logistic recalibration; exploratory",
            })
            calibration_public.append(item)

    paired_critical = float(np.quantile(
        np.max(np.abs(np.column_stack(paired_columns)), axis=1), 0.95, method="linear"
    ))
    paired_simultaneous = {"interceptDifference": {}, "slopeDifference": {}}
    for category, parameter_name, point, se, values in paired_meta:
        paired_simultaneous[parameter_name + "Difference"][str(category)] = {
            "differenceK6MinusK10": float(point),
            "percentile95CI": np.quantile(values, [0.025, 0.975], method="linear").tolist(),
            "simultaneous95Band": [float(point - paired_critical * se), float(point + paired_critical * se)],
        }

    bins, bin_family = probability_bin_audit(rows, dates, multiplicities)
    return {
        "experiment": "GOAL3 exploratory probability-bin reliability and one-vs-rest logistic calibration",
        "scope": "overall cohort only; k=6 and k=10; categories 0/1/2/3+",
        "cohort": {"matches": 236, "teamSideTargets": 472, "dates": 17,
                   "dateRange": [dates[0], dates[-1]]},
        "method": {
            "probabilityBins": list(PROBABILITY_BINS),
            "minimumBinTargetsForCI": MIN_BIN_TARGETS_FOR_CI,
            "binCI": "95% percentile interval using target-date cluster bootstrap",
            "binSimultaneousFamily": bin_family,
            "logisticModel": "one-vs-rest logit Pr(Y=category)=intercept+slope*logit(predicted category probability)",
            "idealCalibration": {"intercept": 0.0, "slope": 1.0},
            "logisticBootstrap": "date-cluster percentile 95% intervals; same date draws paired across k and categories",
            "logisticSimultaneousFamily": {
                "family": "8 one-vs-rest fits x intercept/slope (16 parameters)",
                "studentizedMaxAbsTCriticalValue": calibration_critical,
                "comparisonCount": len(calibration_meta),
            },
            "pairedK6MinusK10SimultaneousFamily": {
                "family": "4 categories x intercept/slope differences (8 contrasts)",
                "studentizedMaxAbsTCriticalValue": paired_critical,
                "comparisonCount": len(paired_meta),
            },
            "rng": "NumPy PCG64", "seed": seed, "replicates": replicates,
            "quantile": "linear percentile",
            "resamplingUnit": "target date; all matches and both home/away team-sides retained",
            "notAssessed": [
                "joint multinomial recalibration",
                "calibration conditional on match covariates",
                "sharpness/resolution as a decision metric",
                "prospective calibration on official completed GOAL3 rounds",
            ],
        },
        "calibrationInterceptSlope": calibration_public,
        "pairedK6MinusK10BootstrapCI": {
            "interceptDifference": paired_intercepts,
            "slopeDifference": paired_slopes,
            "simultaneousBands": paired_simultaneous,
            "note": "unadjusted percentile intervals and separate max-|t| simultaneous bands; exploratory",
        },
        "probabilityBinReliability": bins,
        "limitations": [
            "Only 17 retrospective target-date clusters; bootstrap intervals are exploratory and may understate dependence from teams recurring across dates.",
            "Bins with fewer than 20 targets have point estimates only; sparse bins are not interpreted.",
            "One-vs-rest logistic calibration treats each category separately; the four binary fits are not a joint multinomial recalibration.",
            "The single retrospective history fetchedAt timestamp does not establish first-known availability for each historical target date.",
            "No prospective GOAL3 shadow outcomes are included.",
        ],
        "inputSha256": input_sha,
        "codeSha256": script_sha,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = json.loads(raw)
    result = audit(report, sha256_bytes(raw), sha256_bytes(Path(__file__).read_bytes()),
                   replicates=args.replicates, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "inputSha256": result["inputSha256"],
                      "codeSha256": result["codeSha256"], "cohort": result["cohort"],
                      "replicates": args.replicates,
                      "validCalibrationFits": sum(x["validBootstrapReplicates"] for x in result["calibrationInterceptSlope"]),
                      "calibrationFits": len(result["calibrationInterceptSlope"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
