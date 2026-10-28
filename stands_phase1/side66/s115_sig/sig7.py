"""С115-ж СТАДІЯ 7: САМОУЗГОДЖЕНИЙ ФІТ. Коефіцієнти фітуються на ПОПРАВЛЕНОМУ ланцюгу.

Діагноз стадії 6: фіт ідентичний оракулу на ОДНОМУ шарі (×0.743 проти ×0.741),
але розходиться на семи, бо COEF описує ланцюг БЕЗ поправки. Ітеруємо:
   ланцюг із COEF_k  →  залишок log(s_true/s_corr)  →  COEF_{k+1} = COEF_k + фіт(залишок)

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 збігається за 2-4 ітерації                                       60%
  P2 сходиться до ×1.05…×1.12 (наближається до оракула 1.123)         40%
  P3 розходиться / осцилює                                            35%
  P4 сходиться, але виграш НУЛЬ (×1.00)                               25%
"""
import numpy as np, pyarrow.parquet as pq, glob, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
def basis(t): return np.stack([np.ones_like(t), t, t*t, np.exp(-0.5*t*t)], 1)
KS = tuple(range(9, 16))
NPAR = 4

def chain(W, COEF, S=None, damp=1.0):
    """повертає m_вих і (якщо S) список (i, t_corr, залишок log)"""
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None; rec = []
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20)); mp = m @ w
        if COEF is not None and i in KS:
            tt = np.abs(mp/s); f = np.exp(damp*(basis(tt) @ COEF[i]))
            cp = cp*np.outer(f, f); s = s*f
        if S is not None and i in KS:
            rec.append((i, np.abs(mp/s), np.log(S[i]/s)))
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return last, rec

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ)
NETS = []
for f in sorted(glob.glob("/home/trim/wblogs/truth115_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    j = [k for k in range(tb.num_rows) if tb["mlp_name"][k].as_py() == nm][0]
    NETS.append((np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16,1024,1024),
                 np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16,1024)[15],
                 d["s"], nm))
log("мереж: %d" % len(NETS))

COEF = {i: np.zeros(NPAR) for i in KS}
DAMP = 0.7
hist = []
for it in range(7):
    errs = []; XS = {i: [] for i in KS}; YS = {i: [] for i in KS}
    for W, tr, S, nm in NETS:
        mo, rec = chain(W, COEF if it > 0 else None, S)
        errs.append(cerr(mo, tr))
        for i, tt, r in rec: XS[i].append(basis(tt)); YS[i].append(r)
    e = np.median(errs); hist.append(e)
    resid = np.median([np.sqrt((np.concatenate(YS[i])**2).mean()) for i in KS])
    log("iter %d  похибка %.4f%%  ×база %.4f  сер|залишок| %.4f%%"
        % (it, e*100, hist[0]/e, resid*100))
    for i in KS:
        X = np.concatenate(XS[i]); Y = np.concatenate(YS[i])
        COEF[i] = COEF[i] + DAMP*np.linalg.lstsq(X, Y, rcond=None)[0]
json.dump({str(i): list(map(float, COEF[i])) for i in KS},
          open("/home/trim/wblogs/sig7_coef.json", "w"))
print("\nісторія похибки: " + " ".join("%.4f" % (x*100) for x in hist))
log("ГОТОВО")
