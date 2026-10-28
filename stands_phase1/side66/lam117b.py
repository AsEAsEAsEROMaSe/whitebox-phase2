"""С117 — ЧИ ОПТИМАЛЬНИЙ `_LAM_FEAT` ПРИ УВІМКНЕНІЙ γ₃?

ПРЕЦЕДЕНТ: [С116] `_ALPHA` підбирався у конфігурації БЕЗ бленду і коштував
×1.039, коли його переміряли в робочій. Тут той самий підозрюваний:
`_LAM_FEAT = 0.35` підбирався ДО того, як з'явилась понейронна γ₃, а
`_shape_corr` містить `_FW[4]·tφ + _FW[5]·(t²−1)φ` — той самий базис.

[dbl117] LAM_FEAT 0 → 0.175 → 0.35 при точному δ дає MSE 1.055e-06 → 9.657e-07
→ 9.127e-06... тобто **МОНОТОННО КРАЩЕ, і 0.35 — КРАЙ СІТКИ.**
🚨 оптимум міг ніколи не скануватись вище.

Тут: спільний скан (LAM_FEAT × _G3_C) на двох діапазонах, γ₃ ІСТИННА
(рішення про напрямок; величину потім переміряти з шумом).

🔮 L1 (55%) оптимум LAM_FEAT > 0.35 ⟹ константа була в не тій конфігурації
🔮 L2 (35%) оптимум рівно 0.35…0.45 ⟹ вже майже оптимальна
🔮 L3 (20%) спільний оптимум зсуває і _G3_C помітно (>0.1)
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT = 3.3478454e-06, 1.4355576e-06, 0.1
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)

def closure(W, g3=None, c=0.0, LAY=(), lam=LAM_FEAT):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if g3 is not None and i in LAY and c != 0.0:
            m = m - c*(s*((g3[i]/6.0)*(t*ph)))
        if lam != 0.0: m = m - lam*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, T=d["relu_15"],
                  g3=np.stack([d["g3_%d" % L] for L in range(D)])))
lg("мереж %d" % len(B))
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
lg("база (LAM=0.35, без γ₃) %.6e" % BASE)

LAMS = [0.05, 0.15, 0.25, 0.30, 0.35]
CS = [0.30, 0.35, 0.40, 0.45, 0.55]
for nm, LAY in (("9..15", tuple(range(9, 16))),):
    print("\n── шари %s, АБСОЛЮТНА MSE ──" % nm)
    print("%9s | %s" % ("LAM_FEAT", " ".join("c=%.2f" % c for c in CS)))
    for lam in LAMS:
        row = [float(np.mean([cerr(closure(b["W"], b["g3"], c, LAY, lam), b["T"])
                              for b in B])) for c in CS]
        print("%9.2f | %s   min %.4e" % (lam, " ".join("%.4e" % x for x in row), min(row)))
        sys.stdout.flush()
lg("ГОТОВО")
print("DONE_LAM117B")
