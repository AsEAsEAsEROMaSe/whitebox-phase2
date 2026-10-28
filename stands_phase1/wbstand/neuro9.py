"""ВИРІШАЛЬНИЙ КОНТРОЛЬ: чи вбиває ремонт саме КРОС-НЕЙРОННА кореляція шуму?
   real   : m <- (1-mu)m + mu*SM[i]                      (справжній шум)
   shufE  : m <- (1-mu)m + mu*(T[i] + ПЕРЕМІШАНЕ(SM-T))  ті самі маргінали,
            кроснейронну структуру ЗНИЩЕНО
   Якщо shufE відновлює виграш ⟹ винна саме структура.
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
acc = {"base": []}
for mu in MUS:
    for k in ("real", "shufE", "gaussE"): acc["%s%.2f" % (k, mu)] = []
SPEC = []
for j in range(12):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)
    SM = sample_means(W, 555+j); tr = T[15]
    rng = np.random.default_rng(31+j)
    E = SM - T
    SH = np.array([T[i] + E[i][rng.permutation(1024)] for i in range(16)])
    GA = np.array([T[i] + rng.normal(0, E[i].std(), 1024) for i in range(16)])
    # наскільки шум «колективний»: частка енергії у топ-16 модах W_{i+1}^T
    for i in (8, 12):
        pr = E[i] @ W[i+1] if i+1 < 16 else E[i]
        SPEC.append(float(np.sqrt((pr*pr).sum())/np.sqrt((E[i]*E[i]).sum()*(W[i+1]**2).sum()/1024)))
    acc["base"].append(cerr(run(W, None), tr))
    for mu in MUS:
        acc["real%.2f" % mu].append(cerr(run(W, SM, mu), tr))
        acc["shufE%.2f" % mu].append(cerr(run(W, SH, mu), tr))
        acc["gaussE%.2f" % mu].append(cerr(run(W, GA, mu), tr))
    log("%2d готово" % j)
b = np.median(acc["base"])
print("\n%-6s %9s %9s %9s   (× до бази)" % ("mu", "РЕАЛЬНИЙ", "ПЕРЕМІШ", "ГАУСІВ"))
for mu in MUS:
    print("%-6.2f %9.3f %9.3f %9.3f" % (mu, b/np.median(acc["real%.2f" % mu]),
          b/np.median(acc["shufE%.2f" % mu]), b/np.median(acc["gaussE%.2f" % mu])))
print("\nпідсилення шуму при переносі через W (1.0 = ізотропний): %.3f ± %.3f" % (np.mean(SPEC), np.std(SPEC)))
json.dump({k: list(map(float, v)) for k, v in acc.items()}, open("/home/trim/wblogs/neuro9.json", "w"))
log("ГОТОВО")
