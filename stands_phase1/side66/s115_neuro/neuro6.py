"""РЕАЛІСТИЧНИЙ РЕМОНТ: замикання підправляється ВИБІРКОВИМ середнім (з шумом).
   m <- (1-mu)*m_замик + mu*(m_істинне + eps),  eps ~ шум при N=7904 (наш сабміт)
   ⟹ це те, що МОЖНА зробити в оцінювачі: `xm` уже рахується в семплюванні.
   Дисперсія лишається НАШОЮ (закон cons із neuro3).

🔒 КОНТРОЛЬ: mu_shuf — той самий шум, але m_істинне ПЕРЕМІШАНЕ по нейронах
   ⟹ якщо виграш лишається, він від згладжування, а не від інформації.

ПЕРЕДБАЧЕННЯ: оптимум mu у 0.05..0.20 · виграш ×1.10..1.30 у похибці   55%
              нуль або шкода (шум з'їдає)                               35%
              >×1.30                                                    10%
"""
import numpy as np, pyarrow.parquet as pq, sys, json
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
NS = 7904.0

def run(W, T, mu=0.0, rng=None, shuf=False, noise=True):
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
        if mu > 0.0:
            tg = T[i].copy()
            if shuf: tg = tg[rng.permutation(n)]
            if noise: tg = tg + rng.normal(0.0, np.sqrt(np.maximum(dd0, 0)/NS))
            m = (1.0-mu)*m + mu*tg
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd0 - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MUS = [0.03, 0.06, 0.11, 0.18, 0.30, 0.50]
acc = {"base": []}
for mu in MUS: acc["mu%.2f" % mu] = []; acc["shuf%.2f" % mu] = []
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)
    tr = T[15]
    acc["base"].append(cerr(run(W, T), tr))
    for mu in MUS:
        acc["mu%.2f" % mu].append(cerr(run(W, T, mu, np.random.default_rng(90+j)), tr))
        acc["shuf%.2f" % mu].append(cerr(run(W, T, mu, np.random.default_rng(90+j), shuf=True), tr))
    log("%2d %s" % (j, tb["mlp_name"][j].as_py()))
b = np.median(acc["base"])
print("\n%-6s %10s %9s | %10s %9s" % ("mu", "похибка%", "×база", "ПЕРЕМІШ%", "×база"))
print("%-6s %10.4f %9.3f |" % ("0.00", b*100, 1.0))
for mu in MUS:
    v = np.median(acc["mu%.2f" % mu]); q = np.median(acc["shuf%.2f" % mu])
    print("%-6.2f %10.4f %9.3f | %10.4f %9.3f" % (mu, v*100, b/v, q*100, b/q))
w = min(MUS, key=lambda m_: np.median(acc["mu%.2f" % m_]))
sg = sum(1 for x, y in zip(acc["mu%.2f" % w], acc["base"]) if x < y)
print("\nзнаковий при mu=%.2f: %d/%d" % (w, sg, len(acc["base"])))
json.dump({k: list(map(float, v)) for k, v in acc.items()}, open("/home/trim/wblogs/neuro6.json", "w"))
log("ГОТОВО")
