"""С115-ж СТАДІЯ 8: ОСТАННІЙ КОНТРОЛЬ — чи оживає ранг-1 при ПЕРЕПІДБОРІ партнерів.

Наш власний закон (С114/С115-г): «константа, що балансує два ефекти, зсувається,
коли покращуєш один із них». `κ` (глобальний множник на cp) і `_LAM_FEAT`
(_shape_corr) підбирались ПРИ НАШІЙ хибній C. Скануємо їх РАЗОМ із `c`.
Якщо оптимум усе одно при c=0 на ВСІХ зрізах — канал закрито не «за підбором».
"""
import numpy as np, pyarrow.parquet as pq, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def run(W, c, kap, lf, kstart=9):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*kap
        mp = m @ w
        if c != 0.0 and i >= kstart: cp = cp + c*np.outer(mp, mp)
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if lf != 0.0: m = m - lf*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
CS = [0.0, 0.0004, 0.001, 0.002]
KAPS = [KAPPA, 0.998, 0.995, 0.990]
LFS = [0.35, 0.20, 0.50]
keys = [(c, k, 0.35) for c in CS for k in KAPS] + [(c, KAPPA, l) for c in CS for l in LFS[1:]]
acc = {k: [] for k in keys}
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    tr = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)[15]
    for k in keys: acc[k].append(cerr(run(W, *k), tr))
    log("%2d" % j)
b = np.median(acc[(0.0, KAPPA, 0.35)])
print("\n%-9s %-8s %-6s %10s %8s" % ("c", "kappa", "lfeat", "похибка%", "×база"))
for k in keys:
    v = np.median(acc[k])
    print("%-9.5f %-8.4f %-6.2f %10.4f %8.4f" % (k[0], k[1], k[2], v*100, b/v))
json.dump({str(k): list(map(float,v)) for k,v in acc.items()}, open("/home/trim/wblogs/sig8.json","w"))
log("ГОТОВО")
