"""σ-КАНАЛ І ПОВНИЙ ДВОМОМЕНТНИЙ ОРАКУЛ ФАЗИ 2.
Вкидаємо ІСТИННУ `s` (діагональ cp) і/або ІСТИННЕ `m`. Останній рядок = `allm`
фази 2 — число, яке вирішує, чи містить наш клас перше місце.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  S1 істинна `s` САМА покращує               45%   (фаза 1: діагональ ЛАМАЛА ×8.5)
  S2 істинна `s` сама ШКОДИТЬ (баланс)       45%
  S3 `s`+`m` разом дадуть ×3..×8 у похибці   55%   (фаза 1 allm = ×5.55)
"""
import numpy as np, pyarrow.parquet as pq, glob, sys, json
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def run(W, T, S, use_s=False, use_m=False, ks=None):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); last = None
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        if use_s and (ks is None or i in ks):
            st = S[i]
            cp = cp*np.outer(st/s, st/s)          # масштабуємо ВСЮ cp ⟹ кореляції цілі
            s = st.copy()
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        last = m.copy()
        if i == d-1: break
        dd0 = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        if use_m and (ks is None or i in ks): m = T[i].copy()
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd0 - np.diag(C))
    return last

def cerr(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
fs = sorted(glob.glob("/home/trim/wblogs/truth115_*.npz"))
MODES = [("base", 0, 0, None), ("s_all", 1, 0, None), ("m_all", 0, 1, None),
         ("sm_all", 1, 1, None), ("s_deep", 1, 0, tuple(range(8, 16))),
         ("s_shal", 1, 0, tuple(range(0, 8)))]
acc = {k: [] for k, _, _, _ in MODES}
for f in fs:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    j = [k for k in range(24) if tb["mlp_name"][k].as_py() == nm][0]
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)
    S = d["s"].astype(np.float64); tr = d["m"][15].astype(np.float64)   # ціль = GPU (менш шумна)
    for nmm, us, um, ks in MODES:
        acc[nmm].append(cerr(run(W, T if not um else d["m"].astype(np.float64), S, us, um, ks), tr))
    log("%-22s base %.4f%%  sm_all %.4f%%" % (nm, acc["base"][-1]*100, acc["sm_all"][-1]*100))
b = np.median(acc["base"])
print("\n%-8s %10s %9s %9s" % ("режим", "похибка%", "×похиб", "×MSE"))
for k, _, _, _ in MODES:
    v = np.median(acc[k]); print("%-8s %10.4f %9.3f %9.3f" % (k, v*100, b/v, (b/v)**2))
print("\nмереж: %d" % len(fs))
json.dump({k: list(map(float, v)) for k, v in acc.items()}, open("/home/trim/wblogs/neuro5.json", "w"))
log("ГОТОВО")
