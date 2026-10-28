"""С115-е-2: ВКИДАННЯ ІСТИННОГО `m` — де народжується похибка (зони руху).

Істина `all_layer_means` уже в датасеті ⟹ одноканальний оракул на `m` коштує 0.
🚨 істина ШУМНА: 0.231%(ш0) -> 0.047%(ш15) ⟹ вкидання МІЛКО може ШКОДИТИ
   (пастка early100 фази 1: «шум оракула домінував»).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  Q1 вкидання на ОДНОМУ мілкому шарі (k<=2) не допоможе або зашкодить   80%
  Q2 вкидання на КОЖНОМУ шарі (лишається лише похибка КРОКУ) дасть
     ×2..×5 у косинусній похибці виходу                                 55%
  Q3 приріст росте з глибиною k                                         70%
"""
import numpy as np, pyarrow.parquet as pq, sys, json, time
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def closure_inj(W, T, mode, k=-1):
    """mode: 'base' | 'one' (істинне m ПІСЛЯ шару k) | 'all' (кожен шар)"""
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]
        cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0:
            m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        if mode == "all" or (mode == "one" and i == k):
            m = T[i].copy()                      # ← вкидаємо ІСТИНУ
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = C + np.diag(dd - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300)
    e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
KS = [0, 1, 2, 4, 6, 8, 10, 12, 14]
acc = {"base": [], "all": []}; [acc.setdefault("one%d" % k, []) for k in KS]
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)
    tr = T[15]
    acc["base"].append(cerr(closure_inj(W, T, "base"), tr))
    acc["all"].append(cerr(closure_inj(W, T, "all"), tr))
    for k in KS: acc["one%d" % k].append(cerr(closure_inj(W, T, "one", k), tr))
    log("%2d %-22s base %.4f%%  all %.4f%%" % (j, tb["mlp_name"][j].as_py(),
        acc["base"][-1]*100, acc["all"][-1]*100))
b = np.median(acc["base"])
print("\n%-8s %9s %8s" % ("режим", "похибка%", "×база"))
for kk in ["base"]+["one%d" % k for k in KS]+["all"]:
    v = np.median(acc[kk]); print("%-8s %9.4f %8.3f" % (kk, v*100, b/v))
json.dump({k: list(map(float, v)) for k, v in acc.items()}, open("/home/trim/wblogs/neuro2.json", "w"))
log("ГОТОВО")
