"""КРИВА ВІДГУКУ: скільки дає ЧАСТКОВЕ покращення `m` на шарі k.
   m_mix = (1-mu)*m_наше + mu*m_істинне, дисперсія ЛИШАЄТЬСЯ нашою (режим cons).
   Це «курс обміну»: d(похибка виходу) / d(точність m на шарі k).
ПЕРЕДБАЧЕННЯ: крива ВИПУКЛА (малий mu дає мало) 60% · лінійна 30% · увігнута 10%
"""
import numpy as np, pyarrow.parquet as pq, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def run(W, T, ks=(), mu=0.0):
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
        dd0 = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        if i in ks: m = (1.0-mu)*m + mu*T[i]
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd0 - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MUS = [0.11, 0.25, 0.5, 0.75, 1.0]; KS = [4, 8, 12]
ALL = tuple(range(15))
acc = {"base": []}
for k in KS:
    for mu in MUS: acc["k%d_mu%.2f" % (k, mu)] = []
for mu in MUS: acc["ALL_mu%.2f" % mu] = []
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)
    tr = T[15]
    acc["base"].append(cerr(run(W, T), tr))
    for k in KS:
        for mu in MUS: acc["k%d_mu%.2f" % (k, mu)].append(cerr(run(W, T, (k,), mu), tr))
    for mu in MUS: acc["ALL_mu%.2f" % mu].append(cerr(run(W, T, ALL, mu), tr))
    log("%2d %s  base %.4f%%" % (j, tb["mlp_name"][j].as_py(), acc["base"][-1]*100))
b = np.median(acc["base"])
print("\n%-6s" % "mu" + "".join("%12s" % ("шар %d" % k) for k in KS) + "%12s" % "ВСІ шари")
print("%-6s" % "0.00" + "".join("%12.3f" % 1.0 for _ in KS) + "%12.3f" % 1.0)
for mu in MUS:
    print("%-6.2f" % mu + "".join("%12.3f" % (b/np.median(acc["k%d_mu%.2f" % (k, mu)])) for k in KS)
          + "%12.3f" % (b/np.median(acc["ALL_mu%.2f" % mu])))
print("\nбаза %.4f%%   (× = у ПОХИБЦІ; у MSE квадрат)" % (b*100))
json.dump({k: list(map(float, v)) for k, v in acc.items()}, open("/home/trim/wblogs/neuro4.json", "w"))
log("ГОТОВО")
