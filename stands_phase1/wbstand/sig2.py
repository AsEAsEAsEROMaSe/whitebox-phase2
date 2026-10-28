"""С115-ж СТАДІЯ 2: ЗАСТОСУВАННЯ поправки `s ← s·exp(λ·δ̂(|t|))` при i>KSIG.

Узгодженість `C` (закон CLAUDE.md): множимо не лише діагональ, а ВСЮ `cp`
через outer(f,f) ⟹ кореляції `ρ` НЕ міняються, міняється лише масштаб.
Це понейронне узагальнення `_KAPPA` (яка є f = const).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 оптимум λ лежить МІЖ 0 і 1 («поправка СТИСНЕНА», 6 підтверджень)   75%
  P2 виграш у косинусній похибці ×1.03…×1.12                            45%
  P3 виграш НУЛЬ (λ*=0)                                                 30%
  P4 крива по λ гладка з одним максимумом                               85%
"""
import numpy as np, pyarrow.parquet as pq, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

COEF = {int(k): np.array(v) for k, v in json.load(open("/home/trim/wblogs/sig1_coef.json")).items()}
def basis(t): return np.stack([np.ones_like(t), t, t*t, np.exp(-0.5*t*t)], 1)

def run(W, lam, kstart):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w
        if lam != 0.0 and i >= kstart:
            t0 = np.abs(mp/s)
            f = np.exp(lam*(basis(t0) @ COEF[i]))
            cp = cp*np.outer(f, f); s = s*f
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
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
KST = [9, 1]
acc = {}
for ks in KST:
    for L in LAMS: acc[(ks, L)] = []
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    tr = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)[15]
    for ks in KST:
        for L in LAMS:
            if L == 0.0 and ks != KST[0]: acc[(ks, L)] = acc[(KST[0], 0.0)]; continue
            acc[(ks, L)].append(cerr(run(W, L, ks), tr))
    log("%2d %s" % (j, tb["mlp_name"][j].as_py()))
b = np.median(acc[(KST[0], 0.0)])
print("\n%-6s %-5s %10s %8s %8s" % ("kstart", "lam", "похибка%", "×база", "знак"))
for ks in KST:
    for L in LAMS:
        v = np.median(acc[(ks, L)])
        sgn = sum(1 for x, y in zip(acc[(ks, L)], acc[(KST[0], 0.0)]) if x < y)
        print("%-6d %-5.2f %10.4f %8.4f %5d/%d" % (ks, L, v*100, b/v, sgn, len(acc[(ks, L)])))
json.dump({"%d_%g" % k: list(map(float, v)) for k, v in acc.items()},
          open("/home/trim/wblogs/sig2.json", "w"))
log("ГОТОВО")
