"""С115-ж СТАДІЯ 4: ЗАСТОСУВАННЯ ранг-1 поправки `cp ← cp + c·outer(mp,mp)`.

Форма ВИВЕДЕНА зі стадії 1 (δ log s = 0.0012·t² ⟹ δC = 0.0024·mp mpᵀ),
а не вгадана; стадія 3 міряє її прямо на банку істинної C.

Гейт узгодженості `C`: ранг-1 чіпає діагональ І позадіагональ РАЗОМ ⟹
проходить за побудовою (на відміну від скалярного κ, який чіпає лише масштаб).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 оптимум c > 0 (наша C ЗАНИЖЕНА вздовж середнього)                  80%
  P2 ранг-1 б'є понейронний exp-фіт стадії 2                            55%
  P3 оптимум c МЕНШИЙ за зміряний у стадії 3 («поправка СТИСНЕНА»)      70%
"""
import numpy as np, pyarrow.parquet as pq, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def run(W, c, kstart, mode="ours"):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None; rng = np.random.default_rng(7)
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        mp = m @ w
        if c != 0.0 and i >= kstart:
            u = mp if mode == "ours" else rng.permutation(mp)   # 🔒 контроль
            cp = cp + c*np.outer(u, u)
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 24
CS = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                         else "0,0.0006,0.0012,0.0024,0.0048")]
KST = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else "1")]
keys = [(ks, c, md) for ks in KST for c in CS for md in ("ours", "shuf")
        if not (c == 0.0 and (ks != KST[0] or md != "ours"))]
acc = {k: [] for k in keys}
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    tr = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)[15]
    for k in keys: acc[k].append(cerr(run(W, k[1], k[0], k[2]), tr))
    log("%2d %s" % (j, tb["mlp_name"][j].as_py()))
base = acc[(KST[0], 0.0, "ours")]; b = np.median(base)
print("\n%-6s %-9s %-5s %10s %8s %8s" % ("kstart", "c", "режим", "похибка%", "×база", "знак"))
for k in keys:
    v = np.median(acc[k]); sgn = sum(1 for x, y in zip(acc[k], base) if x < y)
    print("%-6d %-9.5f %-5s %10.4f %8.4f %5d/%d" % (k[0], k[1], k[2], v*100, b/v, sgn, len(base)))
json.dump({"%d_%g_%s" % k: list(map(float, v)) for k, v in acc.items()},
          open("/home/trim/wblogs/sig4.json", "w"))
log("ГОТОВО")
