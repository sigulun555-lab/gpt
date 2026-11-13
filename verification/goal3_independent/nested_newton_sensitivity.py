#!/usr/bin/env python3
"""Independent exact-Newton sensitivity for the GOAL3 nested validation grids.

Research-only: reuses the frozen nested protocol/data selection/scoring, while
replacing only L-BFGS-B with an independently implemented exact-Hessian
Newton solver for the same clipped penalized-Poisson objective. Existing
SciPy artifacts are read-only comparison baselines and are never overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize, root
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import nested_validation as nv

FIT_DIAGNOSTICS: list[dict] = []
DESIGN_CACHE: dict[str, tuple] = {}
SOLVER_TOLERANCE = 1e-6  # Match the protocol's strict gradient diagnostic gate
MAX_ITERATIONS = 300
TRUST_EXACT_MAX_ITERATIONS = 1000

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _design(prior: list[dict]):
    """Build/cache the sparse design matrix for home/away score observations."""
    payload = "\n".join("|".join(str(m[k]) for k in ("date","home","away","hg","ag")) for m in prior).encode("utf-8")
    key = hashlib.sha256(payload).hexdigest()
    if key in DESIGN_CACHE:
        return DESIGN_CACHE[key]
    teams = sorted({m["home"] for m in prior} | {m["away"] for m in prior})
    if not teams:
        return None
    idx = {team:i for i,team in enumerate(teams)}
    n, count = len(teams), len(prior)
    homes = np.fromiter((idx[m["home"]] for m in prior), dtype=np.int64, count=count)
    aways = np.fromiter((idx[m["away"]] for m in prior), dtype=np.int64, count=count)
    hg = np.fromiter((m["hg"] for m in prior), dtype=np.float64, count=count)
    ag = np.fromiter((m["ag"] for m in prior), dtype=np.float64, count=count)
    # Rows: all home observations followed by all away observations.
    hr = np.repeat(np.arange(count, dtype=np.int64), 4)
    hc = np.column_stack((np.zeros(count,dtype=np.int64),np.ones(count,dtype=np.int64),2+homes,2+n+aways)).ravel()
    hv = np.tile(np.array([1.0,1.0,1.0,-1.0]), count)
    ar = count + np.repeat(np.arange(count, dtype=np.int64), 3)
    ac = np.column_stack((np.zeros(count,dtype=np.int64),2+aways,2+n+homes)).ravel()
    av = np.tile(np.array([1.0,1.0,-1.0]), count)
    matrix = coo_matrix(
        (np.concatenate((hv,av)), (np.concatenate((hr,ar)), np.concatenate((hc,ac)))),
        shape=(2*count, 2+2*n),
    ).tocsr()
    design = (teams, idx, matrix, np.concatenate((hg,ag)), n)
    DESIGN_CACHE[key] = design
    return design

def _objective_gradient_hessian(x: np.ndarray, design, lam_pen: float, need_hessian: bool=True):
    teams, idx, matrix, observed, n = design
    raw = np.asarray(matrix @ x, dtype=np.float64)
    eta = np.clip(raw, -20.0, 20.0)
    active = (raw > -20.0) & (raw < 20.0)
    rate = np.exp(eta)
    penalty = x[2:]
    value = float(np.sum(rate-observed*eta) + lam_pen*np.dot(penalty,penalty))
    residual = (rate-observed)*active
    gradient = np.asarray(matrix.T @ residual, dtype=np.float64).ravel()
    gradient[2:] += 2.0*lam_pen*penalty
    if not need_hessian:
        return value, gradient, None
    weighted = matrix.multiply((rate*active)[:,None])
    hessian = (matrix.T @ weighted).toarray()
    hessian[2:,2:] += (2.0*lam_pen)*np.eye(2*n)
    return value, gradient, (hessian+hessian.T)*0.5

def fit_shrinkage_exact_newton(prior: list[dict], lam_pen: float):
    """Fit the same objective with exact Hessian Newton and trust-exact polishing."""
    design = _design(prior)
    if design is None:
        return None, {}, {"success":True,"status":0,"nit":0,"message":"empty prior fallback",
                          "optimizerFit":False,"maxAbsGradient":None,"gradientL2":None,"gradientDimension":0}
    teams, idx, _, _, n = design
    x = np.zeros(2+2*n, dtype=np.float64)
    x[0] = np.log(1.3)
    damping_used, accepted_steps, message = 0.0, 0, ""
    fallback_reason = None
    fallback_info = None

    for iteration in range(MAX_ITERATIONS+1):
        value, gradient, hessian = _objective_gradient_hessian(x, design, lam_pen, True)
        max_abs = float(np.max(np.abs(gradient)))
        if max_abs <= SOLVER_TOLERANCE:
            message = f"exact Newton converged: max_abs_gradient<={SOLVER_TOLERANCE:g}"
            break
        if iteration == MAX_ITERATIONS:
            fallback_reason = f"manual Newton iteration limit reached at gradient {max_abs:.12g}"
            break

        direction, last_error = None, None
        for damping in (0.0,1e-12,1e-10,1e-8,1e-6,1e-4):
            try:
                h = hessian if damping == 0.0 else hessian+damping*np.eye(hessian.shape[0])
                candidate = solve(h, -gradient, assume_a="pos", check_finite=False)
                if np.isfinite(candidate).all() and float(np.dot(gradient,candidate)) < 0.0:
                    direction = candidate
                    damping_used = max(damping_used,damping)
                    break
            except Exception as exc:
                last_error = repr(exc)
        if direction is None:
            fallback_reason = f"manual Newton could not find descent direction: {last_error}"
            break

        slope = float(np.dot(gradient,direction))
        step, accepted = 1.0, False
        for _ in range(60):
            trial = x+step*direction
            trial_value,_,_ = _objective_gradient_hessian(trial,design,lam_pen,False)
            if np.isfinite(trial_value) and trial_value <= value+1e-4*step*slope:
                x = trial
                accepted = True
                accepted_steps += 1
                break
            step *= 0.5
        if not accepted:
            final_value, final_gradient, _ = _objective_gradient_hessian(x,design,lam_pen,True)
            final_max = float(np.max(np.abs(final_gradient)))
            if final_max <= SOLVER_TOLERANCE:
                message = f"exact Newton converged after line-search stagnation: {final_max:.12g}"
                break
            fallback_reason = f"manual Newton Armijo line search stagnated at gradient {final_max:.12g}, objective {final_value:.12g}"
            break
    else:
        fallback_reason = "manual Newton loop ended without convergence"

    _, current_gradient, _ = _objective_gradient_hessian(x,design,lam_pen,False)
    current_max = float(np.max(np.abs(current_gradient)))
    if current_max > SOLVER_TOLERANCE:
        start_x = x.copy()

        def trust_fun(z):
            value, grad, _ = _objective_gradient_hessian(z,design,lam_pen,False)
            return value, grad

        def trust_hess(z):
            return _objective_gradient_hessian(z,design,lam_pen,True)[2]

        trust_result = minimize(
            trust_fun, start_x, jac=True, hess=trust_hess, method="trust-exact",
            options={"gtol":SOLVER_TOLERANCE*0.1,"maxiter":TRUST_EXACT_MAX_ITERATIONS},
        )
        trust_x = np.asarray(trust_result.x,dtype=np.float64)
        _, trust_gradient, _ = _objective_gradient_hessian(trust_x,design,lam_pen,False)
        trust_max = float(np.max(np.abs(trust_gradient)))
        root_hybrid_result = None
        root_hybrid_info = None
        root_lm_result = None
        root_lm_info = None
        bfgs_result = None
        bfgs_info = None
        root_candidates = []
        if trust_max > SOLVER_TOLERANCE:
            # Root-finding acts on the analytic gradient itself, not tiny
            # objective differences. This helps when objective decrease is
            # below float64 resolution near a stationary point.
            root_hybrid_result = root(
                lambda z: _objective_gradient_hessian(z,design,lam_pen,False)[1],
                trust_x,
                jac=lambda z: _objective_gradient_hessian(z,design,lam_pen,True)[2],
                method="hybr",
                options={"xtol":1e-12,"maxfev":5000},
            )
            root_hybrid_x = np.asarray(root_hybrid_result.x,dtype=np.float64)
            root_hybrid_value, root_hybrid_gradient, _ = _objective_gradient_hessian(root_hybrid_x,design,lam_pen,False)
            root_hybrid_max = float(np.max(np.abs(root_hybrid_gradient)))
            root_hybrid_info = {
                "successFlag":bool(root_hybrid_result.success),
                "status":int(root_hybrid_result.status),
                "functionEvaluations":int(root_hybrid_result.nfev),
                "message":str(root_hybrid_result.message),
                "objective":float(root_hybrid_value),
                "maxAbsGradient":root_hybrid_max,
            }
            root_candidates.append(("root-hybr",root_hybrid_x))
            if root_hybrid_max > SOLVER_TOLERANCE:
                root_lm_result = root(
                    lambda z: _objective_gradient_hessian(z,design,lam_pen,False)[1],
                    root_hybrid_x,
                    jac=lambda z: _objective_gradient_hessian(z,design,lam_pen,True)[2],
                    method="lm",
                    options={"ftol":1e-14,"xtol":1e-14,"gtol":1e-14,"maxiter":5000},
                )
                root_lm_x = np.asarray(root_lm_result.x,dtype=np.float64)
                root_lm_value, root_lm_gradient, _ = _objective_gradient_hessian(root_lm_x,design,lam_pen,False)
                root_lm_max = float(np.max(np.abs(root_lm_gradient)))
                root_lm_info = {
                    "successFlag":bool(root_lm_result.success),
                    "status":int(root_lm_result.status),
                    "functionEvaluations":int(root_lm_result.nfev),
                    "message":str(root_lm_result.message),
                    "objective":float(root_lm_value),
                    "maxAbsGradient":root_lm_max,
                }
                root_candidates.append(("root-lm",root_lm_x))
            best_root = min(
                root_candidates,
                key=lambda item:float(np.max(np.abs(_objective_gradient_hessian(item[1],design,lam_pen,False)[1]))),
            )
            best_root_gradient = _objective_gradient_hessian(best_root[1],design,lam_pen,False)[1]
            if float(np.max(np.abs(best_root_gradient))) > SOLVER_TOLERANCE:
                # Last independent gradient-based polish. The optimizer success
                # flag is not trusted; the original gradient is always rechecked.
                bfgs_result = minimize(
                    trust_fun, best_root[1], jac=True, method="BFGS",
                    options={"gtol":SOLVER_TOLERANCE*0.01,"maxiter":2000,"disp":False},
                )
                bfgs_x = np.asarray(bfgs_result.x,dtype=np.float64)
                bfgs_value, bfgs_gradient, _ = _objective_gradient_hessian(bfgs_x,design,lam_pen,False)
                bfgs_info = {
                    "successFlag":bool(bfgs_result.success),
                    "status":int(bfgs_result.status),
                    "nit":int(bfgs_result.nit),
                    "message":str(bfgs_result.message),
                    "objective":float(bfgs_value),
                    "maxAbsGradient":float(np.max(np.abs(bfgs_gradient))),
                }
        candidates = [("manual-newton",start_x),("trust-exact",trust_x),*root_candidates]
        if bfgs_result is not None:
            candidates.append(("bfgs-polish",np.asarray(bfgs_result.x,dtype=np.float64)))
        scored = []
        for candidate_name,candidate_x in candidates:
            candidate_value,candidate_gradient,_ = _objective_gradient_hessian(candidate_x,design,lam_pen,False)
            candidate_max = float(np.max(np.abs(candidate_gradient)))
            scored.append((candidate_name,candidate_x,float(candidate_value),candidate_max))
        eligible = [item for item in scored if item[3] <= SOLVER_TOLERANCE]
        selected = min(eligible,key=lambda item:item[2]) if eligible else min(scored,key=lambda item:item[3])
        selected_name,selected_x,selected_value,selected_max = selected
        x = selected_x.copy()
        fallback_info = {
            "successFlag":bool(trust_result.success),
            "status":int(trust_result.status),
            "nit":int(trust_result.nit),
            "message":str(trust_result.message),
            "initialGradient":current_max,
            "fallbackReason":fallback_reason,
            "rootHybrid":root_hybrid_info,
            "rootLm":root_lm_info,
            "bfgsPolish":bfgs_info,
            "selectedCandidate":selected_name,
            "selectedObjective":selected_value,
            "selectedMaxAbsGradient":selected_max,
        }
        message = f"trust-exact polish after {fallback_reason}; selected {selected_name}; trust={trust_result.message}"

    final_value, final_gradient, _ = _objective_gradient_hessian(x,design,lam_pen,False)
    max_abs = float(np.max(np.abs(final_gradient)))
    grad_l2 = float(np.linalg.norm(final_gradient))
    if not np.isfinite(x).all() or not np.isfinite(final_value) or max_abs > SOLVER_TOLERANCE:
        raise RuntimeError(
            f"exact-Hessian solver final validation failed; lambda={lam_pen:g}; "
            f"objective={final_value}; gradient={max_abs}; trustExact={fallback_info}"
        )
    manual_iterations = int(iteration)
    trust_iterations = int(fallback_info["nit"]) if fallback_info else 0
    bfgs_iterations = int(((fallback_info or {}).get("bfgsPolish") or {}).get("nit",0) or 0)
    root_function_evaluations = sum(
        int((fallback_info or {}).get(key,{}).get("functionEvaluations",0) or 0)
        for key in ("rootHybrid","rootLm") if (fallback_info or {}).get(key)
    )
    # SciPy root reports function evaluations (nfev), not iterations (nit).
    # Keep these diagnostics separate rather than adding unlike units.
    total_iterations = manual_iterations + trust_iterations + bfgs_iterations
    info = {
        "success":True,"status":0,"nit":total_iterations,"totalIterations":total_iterations,
        "rootFunctionEvaluations":root_function_evaluations,
        "message":message,"optimizerFit":True,
        "maxAbsGradient":max_abs,"gradientL2":grad_l2,"gradientDimension":int(final_gradient.size),
        "optimizerName":"independent exact-Hessian Newton with Armijo, trust-exact, gradient-root, and BFGS polish",
        "optimizerOptions":{
            "manualMaxIterations":MAX_ITERATIONS,"trustExactMaxIterations":TRUST_EXACT_MAX_ITERATIONS,
            "maxAbsGradientTolerance":SOLVER_TOLERANCE,
            "dampingSchedule":[0.0,1e-12,1e-10,1e-8,1e-6,1e-4],
            "armijoConstant":1e-4,"maxBacktrackingSteps":60,
            "trustExactMethod":"SciPy trust-exact with the same analytic Hessian",
            "gradientRootMethods":"SciPy root hybr then lm with the analytic Hessian as the Jacobian of the original objective gradient",
            "bfgsPolishMethod":"SciPy full BFGS with the same analytic objective gradient; final original-objective gradient is rechecked",
        },
        "objective":final_value,"acceptedNewtonSteps":accepted_steps,"maximumDampingUsed":damping_used,
        "manualNewtonIterations":manual_iterations,
        "trustExactFallbackUsed":fallback_info is not None,
        "rootHybridUsed":bool(fallback_info and fallback_info.get("rootHybrid") is not None),
        "rootLmUsed":bool(fallback_info and fallback_info.get("rootLm") is not None),
        "bfgsPolishUsed":bool(fallback_info and fallback_info.get("bfgsPolish") is not None),
        "trustExactFallback":fallback_info,
    }
    FIT_DIAGNOSTICS.append({"lambda":float(lam_pen),**info})
    return x,idx,info

def _comparison(exact: dict, baseline: dict) -> dict:
    folds=[]
    for ef,bf in zip(exact["outerFolds"],baseline["outerFolds"]):
        es,bs=ef["selected"],bf["selected"]
        em,bm=ef["outerMetrics"]["shrinkageMLE"],bf["outerMetrics"]["shrinkageMLE"]
        folds.append({
            "fold":ef["fold"],"testStart":ef["testStart"],"testEnd":ef["testEnd"],
            "exactNewtonSelected":{"window":es["window"],"lambda":es["lambda"]},
            "scipyLBFGSBSelected":{"window":bs["window"],"lambda":bs["lambda"]},
            "selectedConfigurationMatches":es["window"]==bs["window"] and es["lambda"]==bs["lambda"],
            "exactNewtonOuterMetrics":em,"scipyLBFGSBOuterMetrics":bm,
            "metricDifferencesExactMinusLBFGSB":{m:float(em[m]-bm[m]) for m in ("brier","logloss","accuracy")},
        })
    return {"baselineMethod":"original nested_validation.py SciPy L-BFGS-B",
            "interpretation":"Solver sensitivity only; neither solver output is prospective evidence.",
            "folds":folds,"selectedConfigurationMatchesAllFolds":all(x["selectedConfigurationMatches"] for x in folds)}

def _fallback_summary(fit_diagnostics: list[dict]) -> dict:
    def fallback(item):
        return item.get("trustExactFallback") or {}
    def nested(item, key):
        return fallback(item).get(key) or {}
    return {
        "trustExactFallbackFitCount":sum(bool(x.get("trustExactFallbackUsed")) for x in fit_diagnostics),
        "gradientRootHybridFitCount":sum(bool(x.get("rootHybridUsed")) for x in fit_diagnostics),
        "gradientRootHybridOptimizerSuccessCount":sum(bool(nested(x,"rootHybrid").get("successFlag")) for x in fit_diagnostics),
        "gradientRootHybridGradientGatePassCount":sum(
            bool(nested(x,"rootHybrid")) and float(nested(x,"rootHybrid").get("maxAbsGradient",float("inf"))) <= SOLVER_TOLERANCE
            for x in fit_diagnostics
        ),
        "gradientRootLmFitCount":sum(bool(x.get("rootLmUsed")) for x in fit_diagnostics),
        "gradientRootLmOptimizerSuccessCount":sum(bool(nested(x,"rootLm").get("successFlag")) for x in fit_diagnostics),
        "gradientRootLmGradientGatePassCount":sum(
            bool(nested(x,"rootLm")) and float(nested(x,"rootLm").get("maxAbsGradient",float("inf"))) <= SOLVER_TOLERANCE
            for x in fit_diagnostics
        ),
        "bfgsPolishFitCount":sum(bool(x.get("bfgsPolishUsed")) for x in fit_diagnostics),
    }


def run_grid(grid: list[float], label: str, baseline_path: Path, output_path: Path,
             history_path: Path, aliases_path: Path, csv2025_path: Path) -> dict:
    global FIT_DIAGNOSTICS
    FIT_DIAGNOSTICS=[]
    old_fit,old_argv=nv.fit_shrinkage,sys.argv[:]
    try:
        nv.fit_shrinkage=fit_shrinkage_exact_newton
        sys.argv=[str(HERE/"nested_validation.py"),"--history2026",str(history_path),
                  "--aliases",str(aliases_path),"--csv2025",str(csv2025_path),
                  "--lambda-grid",",".join(format(x,".15g") for x in grid),"--out",str(output_path)]
        nv.main()
    finally:
        nv.fit_shrinkage=old_fit
        sys.argv=old_argv
    result=json.loads(output_path.read_text(encoding="utf-8"))
    baseline=json.loads(baseline_path.read_text(encoding="utf-8"))
    expected=int(baseline["optimizerDiagnostics"]["totalOptimizerFitCount"])
    if len(FIT_DIAGNOSTICS)!=expected:
        raise AssertionError(f"{label}: exact Newton fit count {len(FIT_DIAGNOSTICS)} != expected {expected}")
    if any(not x["success"] or x["maxAbsGradient"]>SOLVER_TOLERANCE for x in FIT_DIAGNOSTICS):
        raise AssertionError(f"{label}: exact Newton fit failed {SOLVER_TOLERANCE:g} gradient tolerance")
    result["solverSensitivity"]={
        "solver":"independent exact-Hessian Newton with Armijo, trust-exact, analytic-gradient root (hybr/lm), and optional BFGS polish",
        "gridLabel":label,"fitCount":len(FIT_DIAGNOSTICS),
        "successfulFits":sum(x["success"] for x in FIT_DIAGNOSTICS),
        "maxAbsGradient":max(x["maxAbsGradient"] for x in FIT_DIAGNOSTICS),
        "maxGradientL2":max(x["gradientL2"] for x in FIT_DIAGNOSTICS),
        "maxIterations":max(x["nit"] for x in FIT_DIAGNOSTICS),
        "maxRootFunctionEvaluations":max(x.get("rootFunctionEvaluations",0) for x in FIT_DIAGNOSTICS),
        "totalRootFunctionEvaluations":sum(x.get("rootFunctionEvaluations",0) for x in FIT_DIAGNOSTICS),
        "iterationCountDefinition":"manual Newton + trust-exact + BFGS iterations; SciPy root function evaluations are reported separately",
        "totalAcceptedNewtonSteps":sum(x["acceptedNewtonSteps"] for x in FIT_DIAGNOSTICS),
        "maxDampingUsed":max(x["maximumDampingUsed"] for x in FIT_DIAGNOSTICS),
        **_fallback_summary(FIT_DIAGNOSTICS),
        "gradientTolerance":SOLVER_TOLERANCE,"strictGateComparison":"same 1e-6 max-absolute-gradient threshold as the protocol diagnostic gate","fitDiagnostics":FIT_DIAGNOSTICS,
        "solverScriptSha256":sha256_file(Path(__file__).resolve()),
        "comparisonToOriginalSciPy":_comparison(result,baseline),
        "scope":"research-only nested solver sensitivity; protocol, data, windows, dates, selection rule and scoring unchanged",
    }
    output_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--history2026",type=Path,default=Path("data/history.json"))
    p.add_argument("--aliases",type=Path,default=Path("data/team_aliases.json"))
    p.add_argument("--csv2025",type=Path,default=Path("verification/goal3_independent/2025.csv"))
    p.add_argument("--predeclared-baseline",type=Path,required=True)
    p.add_argument("--expanded-baseline",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,default=Path("artifacts/nested-newton"))
    a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    pre=run_grid([1.0,3.0,10.0,30.0,100.0],"predeclared-5lambda",a.predeclared_baseline,
                 a.out_dir/"goal3_nested_newton_predeclared_5lambda.json",a.history2026,a.aliases,a.csv2025)
    exp=run_grid(list(nv.DEFAULT_LAMBDAS),"exploratory-expanded-16lambda",a.expanded_baseline,
                 a.out_dir/"goal3_nested_newton_expanded_16lambda.json",a.history2026,a.aliases,a.csv2025)
    print(json.dumps({
        "predeclared":{"fitCount":pre["solverSensitivity"]["fitCount"],
                       "maxAbsGradient":pre["solverSensitivity"]["maxAbsGradient"],
                       "comparison":pre["solverSensitivity"]["comparisonToOriginalSciPy"]},
        "expanded":{"fitCount":exp["solverSensitivity"]["fitCount"],
                    "maxAbsGradient":exp["solverSensitivity"]["maxAbsGradient"],
                    "comparison":exp["solverSensitivity"]["comparisonToOriginalSciPy"]},
        "warning":"Five-value grid remains predeclared; 16-value grid remains exploratory. Do not infer prospective adoption."
    },ensure_ascii=False))

if __name__=="__main__":
    main()
