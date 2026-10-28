"""С115-к: БІСЕКЦІЯ між стендом (×2.387) і оцінювачем (×0.77).

Відтворюємо в стенді ПОВНИЙ конвеєр оцінювача — замикання + аналітичний перенос
`v` + масштаб `_ai` + бленд `_W_BLEND` — і додаємо ту саму регресійну CV.
  якщо стенд ПОВТОРИТЬ ×0.77  ⟹ помилка в ПОСТАНОВЦІ (щось у конвеєрі гасить CV)
  якщо стенд ПОКАЖЕ виграш    ⟹ помилка в КОДІ оцінювача

Метрика — ЯКІСТЬ СЕРЕДНЬОГО (косинусна похибка), не бал.
"""
import numpy as np, pyarrow.parquet as pq, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
INV = 0.3989422804014327
NPAIR = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
NREP  = int(sys.argv[3]) if len(sys.argv) > 3 else 6
RIDGE, WB, ASHR = 0.03, 0.820, 0.5

def closure(W):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); MS = []
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        MS.append(m.copy())
        if i == d-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS

def sample(W, z):
    d = W.shape[0]
    cov = (z.T @ z)/z.shape[0]
    wz = np.linalg.inv(np.linalg.cholesky(cov)).T
    y = (z @ wz) @ W[0]
    x = np.concatenate([np.maximum(y,0.0), np.maximum(-y,0.0)],0).astype(np.float32)
    sg0 = np.sqrt(np.maximum((W[0]*W[0]).sum(0),1e-30))
    v = x.mean(0).astype(np.float64) - sg0*INV
    pen = None
    for i in range(1, d):
        p = x @ W[i].astype(np.float32)
        v = (v @ W[i]) * (p > 0.0).mean(0).astype(np.float64)
        x = np.maximum(p, 0.0)
        if i == d-2: pen = x
    return x, v, pen

def fitB(D, F, r):
    G = (D.T @ D)/D.shape[0]
    G = G + r*np.trace(G)/G.shape[0]*np.eye(G.shape[0])
    return np.linalg.solve(G, (D.T @ F)/D.shape[0])

def cerr(x, tr):
    a = (x@tr)/max((x*x).sum(),1e-300); e = a*x - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 6
KEYS = ["МК гола", "МК+CV", "МК−v", "МК−v+CV", "бленд(сабміт)", "бленд+CV", "бленд+CV,W0.65"]
acc = {k: [] for k in KEYS}
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16,1024,1024)
    tr = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16,1024)[15]
    MS = closure(W); mdeep = MS[15]; mpen = MS[14]
    for r in range(NREP):
        rng = np.random.default_rng(51000+100*j+r)
        z = rng.standard_normal((NPAIR,1024)).astype(np.float32)
        xf, v, pen = sample(W, z)
        f = xf.astype(np.float64); fm = f.mean(0)
        a = pen.astype(np.float64); am = a.mean(0)
        B = fitB(a-am, f-fm, RIDGE)
        corr = (am - mpen) @ B
        def blend(o):
            ai = (o @ mdeep)/max((mdeep*mdeep).sum(),1e-300)
            ai = 1.0 + ASHR*(ai - 1.0)
            return o + WB*(ai*mdeep - o)
        def blend65(o):
            ai = (o @ mdeep)/max((mdeep*mdeep).sum(),1e-300)
            ai = 1.0 + ASHR*(ai - 1.0)
            return o + 0.655*(ai*mdeep - o)
        acc["МК гола"].append(cerr(fm, tr))
        acc["МК+CV"].append(cerr(fm - corr, tr))
        acc["МК−v"].append(cerr(fm - v, tr))
        acc["МК−v+CV"].append(cerr(fm - v - corr, tr))
        acc["бленд(сабміт)"].append(cerr(blend(fm - v), tr))
        acc["бленд+CV"].append(cerr(blend(fm - v - corr), tr))
        acc["бленд+CV,W0.65"].append(cerr(blend65(fm - v - corr), tr))
    log("%2d" % j)
print("\n%-16s %10s %8s" % ("схема", "похибка%", "×своя база"))
for k in KEYS:
    base = acc["бленд(сабміт)"] if "бленд" in k else acc["МК гола"]
    v_ = np.median(acc[k]); print("%-16s %10.4f %8.4f" % (k, v_*100, np.median(base)/v_))
json.dump({k: list(map(float,v)) for k,v in acc.items()}, open("/home/trim/wblogs/cv5.json","w"))
log("ГОТОВО")
