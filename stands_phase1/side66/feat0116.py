"""ЧИ ПСУЄ `shape_corr` ЄДИНУ ТОЧНО ВІДОМУ ІСТИНУ ЗАДАЧІ?

Шар 0: вхід гаусів ТОЧНО ⟹ cp₀ = W₀ᵀW₀ точне, mp₀ = 0 точне,
       m₀ = s₀·φ(0) = s₀/√(2π) — ТОЧНА ІСТИНА (корпус: «λ-блок, єдине місце
       з відомою істиною»). Але оцінювач однаково віднімає LAM_FEAT·shape_corr.
[stransfer116] коефіцієнт передачі шару 1 = 0.6715 — НАЙБІЛЬШИЙ з усіх.
[gam116b] похибка mp на ш1 = 0.072..0.075% при похибці s = 0.034% (= шум банку).

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 Q1 (70%) вимкнення feat на шарі 0 ріже похибку mp₁ щонайменше вдвічі
 Q2 (40%) воно ж покращує КІНЦЕВУ косинусну похибку (може й зіпсувати:
          feat фітився разом з усім іншим ⟹ його «шкода» на ш0 могла бути
          скомпенсована на глибині)
🔒 ПЛОМБА: при feat OFF на ш0 замикальне m₀ мусить дорівнювати s₀/√(2π) біт-у-біт
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT, TRUE2 = 3.3478454e-06, 1.4355576e-06, 0.1, 0.909290
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)
rel = lambda a, b: 100*np.linalg.norm(a-b)/np.linalg.norm(b)

def closure(W, off=()):
    """off — множина шарів, де shape_corr НЕ прикладається"""
    m = np.zeros(NW); C = np.eye(NW); MP = []; MM = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w; MP.append(mp.copy())
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0 and i not in off:
            m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        MM.append(m.copy())
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m, np.stack(MP), np.stack(MM)

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, mp=np.stack([d["mp_%d" % L] for L in range(D)]),
                  rl=np.stack([d["relu_%d" % L] for L in range(D)]), T=d["relu_15"]))
print("банків %d" % len(B), flush=True)

# 🔒 ПЛОМБА: m₀ при feat OFF = s₀/√(2π)
W = B[0]["W"]; cp0 = W[0].T @ W[0]; s0 = np.sqrt(np.diag(cp0))
_, _, MM = closure(W, off=(0,))
print("🔒 ПЛОМБА m₀: max|m₀ − s₀/√(2π)| = %.3e  (мусить бути ~0)"
      % np.max(np.abs(MM[0] - s0*0.3989422804014327)))
print("   похибка m₀ проти банку: feat ON %.4f%% · feat OFF %.4f%%"
      % (rel(closure(W)[2][0], B[0]["rl"][0]), rel(MM[0], B[0]["rl"][0])))

MODES = [("база (feat скрізь)", ()), ("feat OFF@0", (0,)), ("feat OFF@0,1", (0,1)),
         ("feat OFF@0..3", (0,1,2,3)), ("feat OFF@0..7", tuple(range(8))),
         ("🔒 feat OFF СКРІЗЬ", tuple(range(D)))]
print("\n%-22s %11s %9s %12s %8s | %s" % ("режим","косинусна","похибка%","adj","×база","похибка mp по шарах %"))
BASE = None
for tag, off in MODES:
    ce, prof = [], []
    for b in B:
        m, MP, _ = closure(b["W"], off=off)
        ce.append(cerr(m, b["T"])); prof.append([rel(MP[L], b["mp"][L]) for L in (1,4,8,15)])
    c = float(np.mean(ce))
    if BASE is None: BASE = c
    e2 = c*(ANCH_E2/BASE); a = (1.0/(1.0/V_MC + 1.0/e2))*MULT
    pr = np.mean(prof, 0)
    print("%-22s %11.5e %8.4f%% %12.4e  ×%.4f | ш1 %.4f  ш4 %.4f  ш8 %.4f  ш15 %.4f"
          % (tag, c, 100*np.sqrt(c/TRUE2), a,
             ((1.0/(1.0/V_MC + 1.0/ANCH_E2))*MULT)/a, pr[0], pr[1], pr[2], pr[3]))
