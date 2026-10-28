"""С115-ж СТАДІЯ 5: ДІАГНОЗ негативу стадії 2. Три гіпотези, всі в одному прогоні.

  H1 КОНКУРЕНЦІЯ: `_shape_corr` (LAM_FEAT=0.35) уже поглинає це відхилення
     ⟹ при LAM_FEAT=0 поправка `s` МУСИТЬ допомагати.                    45%
  H2 НАКОПИЧЕННЯ: коефіцієнти фітовані на НЕПОПРАВЛЕНОМУ ланцюгу ⟹ при
     застосуванні на 7 шарах поспіль подвійний облік. Однократне
     застосування (лише шар 15) МУСИТЬ допомагати.                       35%
  H3 БАЛАНС: похибка `s` вже скорочується у виході, часткова поправка
     ЛАМАЄ скорочення ⟹ нуль в обох контролях.                           20%
"""
import numpy as np, pyarrow.parquet as pq, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
COEF = {int(k): np.array(v) for k, v in json.load(open("/home/trim/wblogs/sig1_coef.json")).items()}
def basis(t): return np.stack([np.ones_like(t), t, t*t, np.exp(-0.5*t*t)], 1)

def run(W, lam, lfeat, only=None):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20)); mp = m @ w
        ok = (i >= 9) if only is None else (i == only)
        if lam != 0.0 and ok:
            f = np.exp(lam*(basis(np.abs(mp/s)) @ COEF[i]))
            cp = cp*np.outer(f, f); s = s*f
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if lfeat != 0.0: m = m - lfeat*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
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
KEYS = ([("H1", L, 0.0, None) for L in (0.0, 0.25, 0.5, 1.0)] +
        [("H2", L, 0.35, 15) for L in (0.25, 0.5, 1.0)] +
        [("H2b", L, 0.35, 9) for L in (0.5, 1.0)] +
        [("B", 0.0, 0.35, None)])
acc = {k: [] for k in KEYS}
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    tr = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)[15]
    for k in KEYS: acc[k].append(cerr(run(W, k[1], k[2], k[3]), tr))
    log("%2d" % j)
print("\n%-4s %-6s %-7s %-5s %10s %8s %8s" % ("гіп", "lam", "lfeat", "шар", "похибка%", "×своя", "знак"))
for tag, ref in (("H1", ("H1", 0.0, 0.0, None)), ("H2", ("B", 0.0, 0.35, None)),
                 ("H2b", ("B", 0.0, 0.35, None)), ("B", ("B", 0.0, 0.35, None))):
    pass
for k in KEYS:
    r = ("H1", 0.0, 0.0, None) if k[0] == "H1" else ("B", 0.0, 0.35, None)
    b = np.median(acc[r]); v = np.median(acc[k])
    sgn = sum(1 for x, y in zip(acc[k], acc[r]) if x < y)
    print("%-4s %-6.2f %-7.2f %-5s %10.4f %8.4f %5d/%d" %
          (k[0], k[1], k[2], str(k[3]), v*100, b/v, sgn, len(acc[k])))
json.dump({str(k): list(map(float, v)) for k, v in acc.items()}, open("/home/trim/wblogs/sig5.json", "w"))
log("ГОТОВО")
