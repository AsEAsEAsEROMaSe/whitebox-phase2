"""Аналіз frontier.py: MSE кожного члена, афінна LOO-суміш по дампах, флопи, МК за тією самою ціною.
МК: семпл = прямий прохід 16 шарів 1024² = 2·n²·L флопів; MSE_MC(F) = avg_var·c/F; F_MC(MSE) = avg_var·c/MSE.
🔒 пломби: reg/dense проти збережених 23.09 (extrap_i.npz); МК на повному бюджеті проти «MC baseline» грейдера 1.13e-6."""
import json, glob, numpy as np
n, L, B = 1024, 16, 2.199023255552e12; c = 2.0 * n * n * L
fs = sorted(glob.glob("/home/trim/wb2data/frontier_*.npz")); D = [np.load(f) for f in fs]
idx = [int(f.split("_")[-1].split(".")[0]) for f in fs]
tags = [k for k in D[0].files if k not in ("gt", "avg_var", "bank_mean")]
FL = json.load(open("/home/trim/wblogs/frontier_flops.json"))
av = np.array([float(d["avg_var"]) for d in D]); avm = av.mean()
mse = lambda p, d: float(np.mean((p - d["gt"])**2))
print("дампи", idx, "· avg_var сер %.4e (%.3e…%.3e) · семпл МК %.3e флопів" % (avm, av.min(), av.max(), c))
print("🔒 МК на повному бюджеті грейдера: %.3e (грейдер, 50 публічних: 1.13e-6)" % (avm * c / B))
print("🔒 банк mean[-1] проти gt: max|Δ| %.2e" % max(float(np.abs(d["bank_mean"] - d["gt"]).max()) for d in D))
for t in ("reg", "dense"):
    old = [np.load("/home/trim/wb2data/extrap_%d.npz" % i)[t] for i in idx]
    print("🔒 %s проти 23.09: max|Δp| %.2e · MSE %.4e проти %.4e" % (t, max(float(np.abs(o - d[t]).max()) for o, d in zip(old, D)),
          np.mean([mse(d[t], d) for d in D]), np.mean([mse(o, d) for o, d in zip(old, D)])))
def mc(F, m): return avm * c / F, avm * c / m
base = np.mean([mse(d["reg"], d) for d in D])
rows = []
for t in tags:
    m = np.mean([mse(d[t], d) for d in D]); F = FL.get(t, np.nan); rows.append((t, m, F))
def loo(members, ridge=0.0):
    """афінна суміш p = reg + Σ w_k (P_k − reg), фіт на решті дампів (LS по всіх нейронах), тест на відкладеному."""
    others = [t for t in members if t != "reg"]; res = []
    for j in range(len(D)):
        X = np.concatenate([np.stack([D[k][t] - D[k]["reg"] for t in others], 1) for k in range(len(D)) if k != j])
        y = np.concatenate([D[k]["gt"] - D[k]["reg"] for k in range(len(D)) if k != j])
        wv = np.linalg.solve(X.T @ X + ridge * np.trace(X.T @ X) / len(others) * np.eye(len(others)), X.T @ y)
        p = D[j]["reg"] + np.stack([D[j][t] - D[j]["reg"] for t in others], 1) @ wv
        res.append(mse(p, D[j]))
    return float(np.mean(res)), res
mix = [("суміш reg+dense", ["reg", "dense"]), ("суміш 7 (як 23.09)", ["reg", "dense", "k211", "regen", "B2", "wk22", "no31", "trunc"]),
       ("суміш усіх", tags)]
for name, mem in mix:
    m, per = loo(mem); F = sum(FL.get(t, np.nan) for t in mem); rows.append((name, m, F))
    print("   %s: LOO по дампах %s" % (name, " ".join("%.2e" % x for x in per)))
print("\n%-22s %11s %8s %11s %8s %13s %13s" % ("метод", "MSE", "×reg", "флопи", "×бюдж", "МК за ціною", "МК треба ×"))
for t, m, F in rows:
    mF, need = mc(F, m)
    print("%-22s %11.4e %8.3f %11.3e %8.3f %13.3e %13.1f" % (t, m, base / m, F, F / B, mF, need / F))
print("\nдовідково (оракули, не методи; 8 дампів, записи 13.10/19.09): точні парні кумулянти K=3 1.085e-9 · істинні κ21+κ31 у reg 8.08e-9")
for name, m in (("оракул пар K=3", 1.085e-9), ("оракул κ21+κ31", 8.08e-9)):
    print("   %s: МК треба %.3e флопів = %.0f бюджетів" % (name, avm * c / m, avm * c / m / B))
