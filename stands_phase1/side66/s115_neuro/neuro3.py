"""КОНТРОЛЬ до neuro2: шум істини чи НЕУЗГОДЖЕНІСТЬ (m,C)?

  ctrl_k : вкидаємо НАШЕ ЖИТТЄВЕ m + синтетичний шум ТІЄЇ САМОЇ величини,
           що й шум істини датасету (√(Var_j/4e5)).  Значення НАШЕ ⟹ пара
           (m,C) лишається узгодженою, змінюється ЛИШЕ шум.
  cons_k : вкидаємо істинне m І перебудовуємо C так, щоб дисперсія лишалась
           нашою (тобто чіпаємо РІВНО один канал, без вторинного зсуву).

  Якщо ctrl ≈ base ⟹ шум ні до чого, винна НЕУЗГОДЖЕНІСТЬ.
  Якщо ctrl ≈ one  ⟹ увесь ефект — шум істини.
"""
import numpy as np, pyarrow.parquet as pq, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def run(W, T, mode, k=-1, rng=None):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        dd0 = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m           # наша дисперсія relu
        if i == k:
            if mode == "one":  m = T[i].copy()
            elif mode == "ctrl": m = m + rng.normal(0.0, np.sqrt(np.maximum(dd0,0)/4e5))
            elif mode == "cons": m = T[i].copy()            # дисперсію лишаємо dd0
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        dd = dd0 if mode == "cons" else (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = C + np.diag(dd - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
KS = [4, 8, 12]
acc = {"base": []}
for k in KS:
    for md in ("one", "ctrl", "cons"): acc["%s%d" % (md, k)] = []
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)
    tr = T[15]; rng = np.random.default_rng(4242+j)
    acc["base"].append(cerr(run(W, T, "base"), tr))
    for k in KS:
        for md in ("one", "ctrl", "cons"):
            acc["%s%d" % (md, k)].append(cerr(run(W, T, md, k, rng), tr))
    log("%2d %s" % (j, tb["mlp_name"][j].as_py()))
b = np.median(acc["base"])
print("\n%-8s %9s %8s" % ("режим", "похибка%", "×база"))
for kk in ["base"]+["%s%d" % (m_, k) for k in KS for m_ in ("one", "ctrl", "cons")]:
    v = np.median(acc[kk]); print("%-8s %9.4f %8.3f" % (kk, v*100, b/v))
json.dump({k: list(map(float, v)) for k, v in acc.items()}, open("/home/trim/wblogs/neuro3.json", "w"))
log("ГОТОВО")
