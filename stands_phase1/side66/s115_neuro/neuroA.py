"""ЯКА КОМПОНЕНТА ВИБІРКОВОГО ШУМУ ВБИВАЄ РЕМОНТ?
   e = SM[i] - T[i];  g = T[i]/‖T[i]‖;  e = e_par*g + e_orth
   real   : T + e            (усе)
   orth   : T + e_orth       (знято компоненту вздовж сигналу)
   par    : T + e_par*g      (лише вона)
   Якщо orth ≈ переміш ⟹ винна МАСШТАБНА компонента, і вона ЗНІМАЄТЬСЯ:
   `_ai`-подібне нормування вибіркового середнього коштує ~0 флопів.
"""
import sys, numpy as np, pyarrow.parquet as pq, json
sys.path.insert(0, "/home/trim/gpupkg"); import cupy as cp
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
NPAIR = 3952
exec(open("/home/trim/wbstand/neuro7.py").read().split("def run(")[0].split("NPAIR = 3952")[1])
def run(W, TG, mu=0.0):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; q = w.T @ C @ w
        if i > KSIG: q = q*KAPPA
        s = np.sqrt(np.maximum(np.diag(q), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(q, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        dd0 = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        if mu > 0.0: m = (1.0-mu)*m + mu*TG[i]
        C = q*np.outer(Ph, Ph) + 0.5*(q*q)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd0 - np.diag(C))
    return last
def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); MUS = [0.03, 0.06, 0.11, 0.18]
KEYS = ("real", "orth", "par", "shuf")
acc = {"base": []}
for mu in MUS:
    for k in KEYS: acc["%s%.2f" % (k, mu)] = []
PAR = []
for j in range(12):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)
    SM = sample_means(W, 555+j); tr = T[15]; rng = np.random.default_rng(31+j)
    E = SM - T
    G = np.array([T[i]/np.sqrt((T[i]*T[i]).sum()) for i in range(16)])
    PA = np.array([(E[i]@G[i]) for i in range(16)])
    TG = {"real": SM,
          "orth": np.array([T[i] + E[i] - PA[i]*G[i] for i in range(16)]),
          "par":  np.array([T[i] + PA[i]*G[i] for i in range(16)]),
          "shuf": np.array([T[i] + E[i][rng.permutation(1024)] for i in range(16)])}
    PAR.append([float(abs(PA[i])/np.sqrt((E[i]*E[i]).sum())*32.0) for i in (4, 8, 12, 15)])
    acc["base"].append(cerr(run(W, None), tr))
    for mu in MUS:
        for k in KEYS: acc["%s%.2f" % (k, mu)].append(cerr(run(W, TG[k], mu), tr))
    log("%2d готово" % j)
b = np.median(acc["base"])
print("\n%-6s" % "mu" + "".join("%10s" % k.upper() for k in KEYS) + "   (× до бази)")
for mu in MUS:
    print("%-6.2f" % mu + "".join("%10.3f" % (b/np.median(acc["%s%.2f" % (k, mu)])) for k in KEYS))
P = np.array(PAR)
print("\nпроєкція на сигнал / очікувана для ізотропного (шари 4,8,12,15):")
print("   " + "  ".join("×%.2f" % v for v in np.median(P, axis=0)))
json.dump({k: list(map(float, v)) for k, v in acc.items()}, open("/home/trim/wblogs/neuroA.json", "w"))
log("ГОТОВО")
