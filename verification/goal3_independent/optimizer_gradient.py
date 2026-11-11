"""Independent analytic gradient diagnostics for the pinned nested_validation fit.

This module does not change optimizer settings or model predictions. It evaluates
the gradient of the exact penalized Poisson objective at the returned parameter
vector so a successful L-BFGS-B status is not mistaken for stationarity.
"""
import numpy as np


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
