"""Independent optimizer and analytic-gradient diagnostics for GOAL3 research.

The pinned nested_validation implementation remains unchanged for retrospective
parity. Prospective v1.5 uses the same penalized Poisson objective with fixed,
stricter L-BFGS-B options; this module records an independent analytic gradient
at the returned parameter vector so success status alone is not mistaken for
stationarity.
"""
import math

import numpy as np
from scipy.optimize import minimize


STRICT_LBFGSB_OPTIONS = {
    "maxiter": 10000,
    "maxfun": 200000,
    "ftol": 0.0,
    "gtol": 1e-8,
    "maxls": 100,
    "maxcor": 50,
}


def objective_gradient(matches, params, team_index, lam_pen):
    if not matches:
        raise ValueError("gradient diagnostics require non-empty training matches")
    n = len(team_index)
    x = np.asarray(params, dtype=float)
    if x.shape != (2 + 2 * n,) or not np.isfinite(x).all():
        raise ValueError("invalid parameter vector for gradient diagnostics")
    homes = np.asarray([team_index[m["home"]] for m in matches], dtype=np.int64)
    aways = np.asarray([team_index[m["away"]] for m in matches], dtype=np.int64)
    home_goals = np.asarray([m["hg"] for m in matches], dtype=float)
    away_goals = np.asarray([m["ag"] for m in matches], dtype=float)
    mu, home_adv = x[0], x[1]
    attack = x[2:2 + n]
    defense = x[2 + n:2 + 2 * n]

    eta_h_raw = mu + home_adv + attack[homes] - defense[aways]
    eta_a_raw = mu + attack[aways] - defense[homes]
    eta_h = np.clip(eta_h_raw, -20.0, 20.0)
    eta_a = np.clip(eta_a_raw, -20.0, 20.0)
    lambda_h = np.exp(eta_h)
    lambda_a = np.exp(eta_a)

    residual_h = (lambda_h - home_goals) * ((eta_h_raw > -20.0) & (eta_h_raw < 20.0))
    residual_a = (lambda_a - away_goals) * ((eta_a_raw > -20.0) & (eta_a_raw < 20.0))
    attack_grad = np.zeros(n, dtype=float)
    defense_grad = np.zeros(n, dtype=float)
    np.add.at(attack_grad, homes, residual_h)
    np.add.at(defense_grad, aways, -residual_h)
    np.add.at(attack_grad, aways, residual_a)
    np.add.at(defense_grad, homes, -residual_a)
    attack_grad += 2.0 * float(lam_pen) * attack
    defense_grad += 2.0 * float(lam_pen) * defense
    return np.concatenate((
        np.asarray([residual_h.sum() + residual_a.sum(), residual_h.sum()], dtype=float),
        attack_grad,
        defense_grad,
    ))


def gradient_diagnostics(matches, params, team_index, lam_pen):
    gradient = objective_gradient(matches, params, team_index, lam_pen)
    if not np.isfinite(gradient).all():
        return {"maxAbsGradient": None, "gradientL2": None, "gradientDimension": int(gradient.size)}
    return {
        "maxAbsGradient": float(np.max(np.abs(gradient))),
        "gradientL2": float(np.linalg.norm(gradient)),
        "gradientDimension": int(gradient.size),
    }


def fit_shrinkage_strict(matches, lam_pen):
    if not matches:
        raise ValueError("strict shrinkage fit requires non-empty training matches")
    team_list = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    team_index = {team: i for i, team in enumerate(team_list)}
    n = len(team_list)
    homes = np.asarray([team_index[m["home"]] for m in matches], dtype=np.int64)
    aways = np.asarray([team_index[m["away"]] for m in matches], dtype=np.int64)
    home_goals = np.asarray([m["hg"] for m in matches], dtype=float)
    away_goals = np.asarray([m["ag"] for m in matches], dtype=float)
    x0 = np.zeros(2 + 2 * n, dtype=float)
    x0[0] = math.log(1.3)

    def objective(x):
        mu, home_adv = x[0], x[1]
        attack = x[2:2 + n]
        defense = x[2 + n:2 + 2 * n]
        eta_h_raw = mu + home_adv + attack[homes] - defense[aways]
        eta_a_raw = mu + attack[aways] - defense[homes]
        eta_h = np.clip(eta_h_raw, -20.0, 20.0)
        eta_a = np.clip(eta_a_raw, -20.0, 20.0)
        lambda_h = np.exp(eta_h)
        lambda_a = np.exp(eta_a)
        value = (
            np.sum(lambda_h - home_goals * eta_h)
            + np.sum(lambda_a - away_goals * eta_a)
            + float(lam_pen) * (np.sum(attack ** 2) + np.sum(defense ** 2))
        )
        gradient = objective_gradient(matches, x, team_index, lam_pen)
        return float(value), gradient

    result = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        options=dict(STRICT_LBFGSB_OPTIONS),
    )
    diagnostics = gradient_diagnostics(matches, result.x, team_index, lam_pen)
    info = {
        "success": bool(result.success),
        "status": int(result.status),
        "nit": int(result.nit),
        "message": str(result.message),
        "optimizerName": "L-BFGS-B",
        "optimizerOptions": dict(STRICT_LBFGSB_OPTIONS),
        **diagnostics,
    }
    return result.x, team_index, info
