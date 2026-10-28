"""С115-ж СТАДІЯ 6: ФІТ проти ОРАКУЛА в ОДНІЙ таблиці, на ТИХ САМИХ 8 мережах.

Розрив, який треба пояснити:
   оракул s_deep(8..15) = ×1.17    ·    наш фіт (R2=0.88) на 9..15 = ×0.18
Питання: винен ФІТ чи сама ІДЕЯ? Порівнюємо ПОШАРОВО і на тій самій вибірці.
Додатково міряємо, чи фіт РЕАЛЬНО наближає `s` до істини (метрика на s, не на виході).
"""
import numpy as np, pyarrow.parquet as pq, glob, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
COEF = {int(k): np.array(v) for k, v in json.load(open("/home/trim/wblogs/sig1_coef.json")).items()}
def basis(t): return np.stack([np.ones_like(t), t, t*t, np.exp(-0.5*t*t)], 1)

def run(W, S, mode, ks, lam=1.0, diag=None):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20)); mp = m @ w
        if mode != "base" and i in ks:
            if mode == "orac": f = S[i]/s
            else:              f = np.exp(lam*(basis(np.abs(mp/s)) @ COEF[i]))
            if diag is not None:
                diag.append((i, float(np.sqrt((np.log(S[i]/s)**2).mean())),
                                float(np.sqrt((np.log(S[i]/(s*f))**2).mean()))))
            cp = cp*np.outer(f, f); s = s*f
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ)
MODES = [("base", None, ())]
for L in (9, 12, 15):
    MODES += [("orac", "@%d" % L, (L,)), ("fit", "@%d" % L, (L,))]
MODES += [("orac", "deep", tuple(range(8, 16))), ("fit", "deep", tuple(range(9, 16))),
          ("fit", "deep .1", tuple(range(9, 16)))]
acc = {}; DG = []
for f in sorted(glob.glob("/home/trim/wblogs/truth115_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    j = [k for k in range(tb.num_rows) if tb["mlp_name"][k].as_py() == nm][0]
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    tr = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)[15]
    S = d["s"]
    for md, tag, ks in MODES:
        lam = 0.1 if tag == "deep .1" else 1.0
        dg = DG if (md == "fit" and tag == "deep") else None
        acc.setdefault((md, tag), []).append(cerr(run(W, S, md, ks, lam, dg), tr))
    log(nm)
b = np.median(acc[("base", None)])
print("\n%-6s %-8s %10s %8s %8s" % ("режим", "де", "похибка%", "×база", "знак"))
for md, tag, ks in MODES:
    v = np.median(acc[(md, tag)])
    sgn = sum(1 for x, y in zip(acc[(md, tag)], acc[("base", None)]) if x < y)
    print("%-6s %-8s %10.4f %8.4f %5d/%d" % (md, str(tag), v*100, b/v, sgn, len(acc[(md,tag)])))
print("\nчи наближає фіт `s` до істини (RMS log-відхилення):")
print("%-4s %10s %10s %8s" % ("ш", "ДО", "ПІСЛЯ", "×краще"))
for i in sorted(set(x[0] for x in DG)):
    a = np.array([x[1] for x in DG if x[0] == i]); c = np.array([x[2] for x in DG if x[0] == i])
    print("%-4d %10.4f %10.4f %8.3f" % (i, np.median(a)*100, np.median(c)*100,
                                        np.median(a)/np.median(c)))
log("ГОТОВО")
