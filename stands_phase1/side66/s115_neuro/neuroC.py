"""ПОВНИЙ ДВОМОМЕНТНИЙ ОРАКУЛ ФАЗИ 2 — з ІСТИННОЮ ПОВНОЮ `C`.
Закриває застереження: досі вкидалась лише ДІАГОНАЛЬ (sm_all ×8.99 у MSE).
Питання фази 1 (orac101): канали МНОЖАТЬСЯ, і `full` був ГІРШИЙ за `off`.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  C1 повна C дасть БІЛЬШЕ за діагональ (×3.3..×5 у похибці)      45%
  C2 приблизно те саме (позадіагональ не несе)                   30%
  C3 МЕНШЕ за діагональ (як `full`<`off` у фазі 1)               25%
"""
import sys, numpy as np, pyarrow.parquet as pq, glob, json
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def run(W, T, S, C4, use_s=0, use_m=0, use_off=0):
    """use_s: істинна ДІАГОНАЛЬ · use_off: істинна ПОВНА C (передактивації)"""
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; q = w.T @ C @ w
        if i > KSIG: q = q*KAPPA
        if use_off:
            q = C4[i].astype(np.float64).copy()          # ПОВНА істинна коваріація
        s = np.sqrt(np.maximum(np.diag(q), 1e-20))
        if use_s and not use_off:
            st = S[i]; q = q*np.outer(st/s, st/s); s = st.copy()
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(q, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        dd0 = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        if use_m: m = T[i].copy()
        C = q*np.outer(Ph, Ph) + 0.5*(q*q)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd0 - np.diag(C))
    return last
def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ)
fc = sorted(glob.glob("/home/trim/wblogs/truthC_*.npz"))
MODES = [("base", 0, 0, 0), ("diag", 1, 0, 0), ("FULLC", 0, 0, 1),
         ("m", 0, 1, 0), ("m+diag", 1, 1, 0), ("m+FULLC", 0, 1, 1)]
acc = {k: [] for k, _, _, _ in MODES}
for f in fc:
    dc = np.load(f, allow_pickle=True); nm = str(dc["name"])
    j = [k for k in range(24) if tb["mlp_name"][k].as_py() == nm][0]
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)
    C4 = dc["C"]; S = np.sqrt(np.maximum(np.array([np.diag(C4[i]) for i in range(16)]), 1e-20))
    tr = T[15]
    for k, us, um, uo in MODES:
        acc[k].append(cerr(run(W, T, S, C4, us, um, uo), tr))
    log("%-22s base %.4f%%  m+FULLC %.4f%%" % (nm, acc["base"][-1]*100, acc["m+FULLC"][-1]*100))
b = np.median(acc["base"])
print("\n%-9s %10s %9s %9s" % ("режим", "похибка%", "×похиб", "×MSE"))
for k, _, _, _ in MODES:
    v = np.median(acc[k]); print("%-9s %10.4f %9.3f %9.3f" % (k, v*100, b/v, (b/v)**2))
print("\nмереж: %d" % len(fc))
json.dump({k: list(map(float, v)) for k, v in acc.items()}, open("/home/trim/wblogs/neuroC.json", "w"))
log("ГОТОВО")
