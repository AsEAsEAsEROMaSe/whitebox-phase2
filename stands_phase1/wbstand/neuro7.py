"""ЧЕСНИЙ РЕМОНТ: замикання правиться СПРАВЖНІМ вибірковим середнім.
   n_pairs=3952 (рівно як у сабміті), вибілювання + антитетика — як в оцінювачі.
   ⟹ шум КОРЕЛЬОВАНИЙ між нейронами (усі бачать ті самі z), на відміну від neuro6.

🔒 КОНТРОЛЬ «ЧУЖА»: середні з ІНШОЇ мережі — мусить зламати.
ПЕРЕДБАЧЕННЯ: кореляція шуму З'ЇСТЬ частину виграшу, лишиться ×1.05..1.15  50%
              виграш збережеться (×1.15+)                                  25%
              зникне зовсім                                                25%
"""
import sys, os, time, numpy as np, pyarrow.parquet as pq, json
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
NPAIR = 3952

def sample_means(W, seed):
    """повертає (16,1024) вибіркових середніх активацій — як в оцінювачі."""
    n = W.shape[1]
    rng = cp.random.default_rng(seed)
    z = rng.standard_normal((NPAIR, n), dtype=cp.float32)
    cov = (z.T @ z) / NPAIR                                  # вибілювання
    L = cp.linalg.cholesky(cov.astype(cp.float64) + 1e-9*cp.eye(n))
    wh = cp.linalg.inv(L).T.astype(cp.float32)
    y = (z @ wh) @ cp.asarray(W[0], cp.float32)
    a = cp.concatenate([cp.maximum(y, 0), cp.maximum(-y, 0)], 0)   # антитетика
    out = [cp.asnumpy(a.mean(0, dtype=cp.float64))]
    for i in range(1, W.shape[0]):
        a = cp.maximum(a @ cp.asarray(W[i], cp.float32), 0)
        out.append(cp.asnumpy(a.mean(0, dtype=cp.float64)))
    del a, z; cp.get_default_memory_pool().free_all_blocks()
    return np.array(out)

def run(W, SM, mu=0.0):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        dd0 = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        if mu > 0.0: m = (1.0-mu)*m + mu*SM[i]
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd0 - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MUS = [0.03, 0.06, 0.11, 0.18, 0.30]
acc = {"base": []}
for mu in MUS: acc["mu%.2f" % mu] = []; acc["for%.2f" % mu] = []
Ws, Ts, SMs = [], [], []
for j in range(min(NN, tb.num_rows)):
    Ws.append(np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024))
    Ts.append(np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024))
    SMs.append(sample_means(Ws[-1], 555+j))
    log("семпли %2d %s" % (j, tb["mlp_name"][j].as_py()))
for j in range(len(Ws)):
    W, T, tr = Ws[j], Ts[j], Ts[j][15]
    acc["base"].append(cerr(run(W, None), tr))
    for mu in MUS:
        acc["mu%.2f" % mu].append(cerr(run(W, SMs[j], mu), tr))
        acc["for%.2f" % mu].append(cerr(run(W, SMs[(j+1) % len(Ws)], mu), tr))
    log("%2d готово" % j)
b = np.median(acc["base"])
print("\n%-6s %10s %9s | %10s %9s" % ("mu", "похибка%", "×база", "ЧУЖА%", "×база"))
print("%-6s %10.4f %9.3f |" % ("0.00", b*100, 1.0))
for mu in MUS:
    v = np.median(acc["mu%.2f" % mu]); q = np.median(acc["for%.2f" % mu])
    print("%-6.2f %10.4f %9.3f | %10.4f %9.3f" % (mu, v*100, b/v, q*100, b/q))
w = min(MUS, key=lambda m_: np.median(acc["mu%.2f" % m_]))
print("\nзнаковий при mu=%.2f: %d/%d" % (w, sum(1 for x, y in zip(acc["mu%.2f" % w], acc["base"]) if x < y), len(acc["base"])))
json.dump({k: list(map(float, v)) for k, v in acc.items()}, open("/home/trim/wblogs/neuro7.json", "w"))
log("ГОТОВО")
