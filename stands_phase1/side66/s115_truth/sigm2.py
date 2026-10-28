"""С115-з: РОЗДІЛИТИ КАНАЛИ. Виправлена `s` — ЛИШЕ у формулі `m`, НЕ в пропагації `C`.

Основа — два виміри цієї сесії:
  `C`/`s` як СТАН     немонотонний, часткове наближення ШКОДИТЬ   [sig2,4,7,8]
  `m`                 МОНОТОННИЙ, mu=0.11 дає ×1.069 (шар 12)     [neuro4]
У sig2 поправка йшла в ОБИДВА канали (через cp). Тут вона йде ЛИШЕ в `m`.

режими:
  mo_t  s_eff у `m` І в `t`   (t = mp/s_eff)
  mo_s  s_eff лише як множник φ, `t` зі СТАРОЇ `s`
  both  як sig2 (контроль, має відтворити ×0.18 при λ=1)

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 mo_* краще за both на кожному λ                              80%
  P2 знайдеться λ з виграшем (×>1.01)                             40%
  P3 усе одно монотонно гірше ⟹ винен не канал, а сам напрям      45%
"""
import numpy as np, pyarrow.parquet as pq, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
COEF = {int(k): np.array(v) for k, v in json.load(open("/home/trim/wblogs/sig1_coef.json")).items()}
def basis(t): return np.stack([np.ones_like(t), t, t*t, np.exp(-0.5*t*t)], 1)

def run(W, lam, mode, kstart=9):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20)); mp = m @ w
        se = s
        if lam != 0.0 and i >= kstart and mode != "base":
            f = np.exp(lam*(basis(np.abs(mp/s)) @ COEF[i]))
            if mode == "both":
                cp = cp*np.outer(f, f); s = s*f; se = s
            else:
                se = s*f                                  # ЛИШЕ у формулі `m`
        t = mp/(se if mode in ("both", "mo_t") else s)
        ph = phi(t); Ph = Phi(t)
        m = mp*Ph + (se if mode != "base" else s)*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        # 🔑 дисперсія і C — ЗАВЖДИ з НЕпоправленої `s` (крім режиму both)
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 24
LAMS = [0.0, 0.15, 0.3, 0.6, 1.0]
keys = [("base", 0.0)] + [(md, L) for md in ("mo_t", "mo_s", "both") for L in LAMS[1:]]
acc = {k: [] for k in keys}
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16,1024,1024)
    tr = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16,1024)[15]
    for k in keys: acc[k].append(cerr(run(W, k[1], k[0]), tr))
    log("%2d" % j)
b = np.median(acc[("base", 0.0)])
print("\n%-6s %-6s %10s %8s %8s" % ("режим", "lam", "похибка%", "×база", "знак"))
for k in keys:
    v = np.median(acc[k]); sgn = sum(1 for x,y in zip(acc[k], acc[("base",0.0)]) if x < y)
    print("%-6s %-6.2f %10.4f %8.4f %5d/%d" % (k[0], k[1], v*100, b/v, sgn, len(acc[k])))
json.dump({str(k): list(map(float,v)) for k,v in acc.items()}, open("/home/trim/wblogs/sigm2.json","w"))
log("ГОТОВО")
