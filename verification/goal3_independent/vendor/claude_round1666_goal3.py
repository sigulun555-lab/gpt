import json, math, argparse, os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

# Exact script used to generate round 1666's saved totoGOAL3 predictions
# (predictions/round_1666_Claude_goal3.jsonl, commit dfe0a4f). Not
# walk-forward: it fits once on all available 2026 history to predict the
# real future fixtures, since there is no "future" data to hold out for an
# actual upcoming round.
#
# --history defaults to the gpt repo's data/history.json, cloned as a
# sibling directory of this repo:
#   git clone https://github.com/sigulun555-lab/gpt ../gpt
parser = argparse.ArgumentParser()
parser.add_argument("--history", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "gpt", "data", "history.json"))
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "goal3_predictions.json"))
args = parser.parse_args()

H = json.load(open(args.history))["history"]
rows = [{"date": h["date"], "competition": h["competition"], "home": h["home"], "away": h["away"],
         "hg": int(h["home_goals"]), "ag": int(h["away_goals"])}
        for h in H if h["competition"] in ("J1","J2","J3")]
rows = sorted(rows, key=lambda r: (r["date"], r["competition"]))
teams = sorted({r["home"] for r in rows} | {r["away"] for r in rows})
ix = {t:i for i,t in enumerate(teams)}; n = len(teams)
print("teams", n, "rows", len(rows))

LAM = 10.0
def objgrad(p):
    att, dfn, mu, h = p[:n], p[n:2*n], p[2*n], p[2*n+1]
    hi = np.array([ix[r["home"]] for r in rows]); ai = np.array([ix[r["away"]] for r in rows])
    yh = np.array([r["hg"] for r in rows], float); ya = np.array([r["ag"] for r in rows], float)
    lh = np.clip(mu+h+att[hi]-dfn[ai], -20, 20); la = np.clip(mu+att[ai]-dfn[hi], -20, 20)
    L = np.exp(lh); R = np.exp(la)
    f = -((yh*lh-L).sum() + (ya*la-R).sum()) + LAM*((att**2).sum()+(dfn**2).sum())
    eh = L-yh; ea = R-ya
    g = np.zeros_like(p)
    np.add.at(g[:n], hi, eh); np.add.at(g[:n], ai, ea)
    np.add.at(g[n:2*n], ai, -eh); np.add.at(g[n:2*n], hi, -ea)
    g[2*n] = eh.sum()+ea.sum(); g[2*n+1] = eh.sum()
    g[:2*n] += 2*LAM*p[:2*n]
    return f, g

p0 = np.zeros(2*n+2); p0[2*n] = math.log(1.3)
r = minimize(objgrad, p0, jac=True, method="L-BFGS-B", options={"gtol":1e-10})
print("fit success", r.success, "nit", r.nit)
att, dfn, mu, h = r.x[:n], r.x[n:2*n], r.x[2*n], r.x[2*n+1]

matches = [("水戸","C大阪"), ("横浜FM","広島"), ("名古屋","浦和")]
out = []
for home, away in matches:
    i, j = ix[home], ix[away]
    lh = math.exp(mu + h + att[i] - dfn[j])
    la = math.exp(mu + att[j] - dfn[i])
    for team, lam_goals, side in [(home, lh, "home"), (away, la, "away")]:
        pk = poisson.pmf([0,1,2], lam_goals)
        p3plus = 1 - pk.sum()
        probs = [float(pk[0]), float(pk[1]), float(pk[2]), float(p3plus)]
        pred = int(np.argmax(probs))
        label = "3" if pred==3 else str(pred)
        out.append({"match": f"{home} vs {away}", "team": team, "side": side,
                    "expected_goals": round(lam_goals,3),
                    "p0": round(probs[0],4), "p1": round(probs[1],4), "p2": round(probs[2],4), "p3plus": round(probs[3],4),
                    "prediction": label, "confidence": round(probs[pred],4)})
json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
for o in out:
    print(f"{o['match']:20s} {o['team']:8s}({o['side']:4s}) 期待得点={o['expected_goals']:.2f}  "
          f"P(0)={o['p0']:.3f} P(1)={o['p1']:.3f} P(2)={o['p2']:.3f} P(3+)={o['p3plus']:.3f}  "
          f"予想={o['prediction']}  確信度={o['confidence']:.3f}")
