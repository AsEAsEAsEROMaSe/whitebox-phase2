"""С117 — ЧИ РАХУЄМО МИ γ₃ ДВІЧІ (shape_corr уже містить `t·φ·s`)?

СВІДЧЕННЯ: у КОЖНОМУ скані оптимальна вага поправки `c*` = 0.40…0.60, ніколи
не 1.0, хоча поправка виведена ТОЧНО (`res117`: R²=0.999) і застосовується до
величини, яку вона описує. Стиснення на 40% = класична ознака, що частину
роботи вже роблять.

ПІДОЗРА: `_shape_corr` має `_FW[4]·(t·φ) + _FW[5]·((t²−1)·φ)`, помножені на `s`
— це РІВНО базисні функції γ₃ і γ₄ у формулі кроку. Тобто пошарове середнє
γ₃-поправки вже вбудоване в навчену поправку, і понейронна γ₃ додається ЗВЕРХУ.

ТЕСТ: вимкнути LAM_FEAT. Якщо підозра вірна:
  (1) `c*` поїде до ~1.0
  (2) виграш ВІД γ₃ (у ×) ЗРОСТЕ, бо зникне подвійний рахунок

🔮 P1 (65%) c* при LAM_FEAT=0 стає ≥0.85
🔮 P2 (50%) відносний виграш γ₃ при LAM_FEAT=0 більший, ніж при 0.35
🔮 P3 (35%) абсолютна похибка (LAM_FEAT=0 + γ₃ при c*) б'є (LAM_FEAT=0.35 + γ₃)
            ⟹ shape_corr можна ЗАМІНИТИ понейронною γ₃, а не доповнювати
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

def closure(W, dl=None, c=0.0, LAY=(), lam=LAM_FEAT):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if dl is not None and i in LAY and c != 0.0: m = m + c*s*dl[i]
        if lam != 0.0: m = m - lam*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz"))[:12]:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    dl = {}
    for L in range(D):
        mp, sd = d["mp_%d" % L], d["sd_%d" % L]
        t = mp/np.maximum(sd, 1e-300); ph = phi(t); Ph = Phi(t)
        dl[L] = (d["relu_%d" % L] - (mp*Ph + sd*ph))/np.maximum(sd, 1e-300)
    B.append(dict(name=nm, W=W, T=d["relu_15"], dl=dl))
lg("мереж %d" % len(B))

LAY = tuple(range(1, 16))
CS = [0.0, 0.30, 0.50, 0.70, 0.90, 1.10]
print("\n── ТОЧНИЙ δ на шарах 1..15, АБСОЛЮТНА MSE ──")
print("%10s | %s" % ("LAM_FEAT", " ".join("c=%.2f" % c for c in CS)))
best = {}
for lam in (0.0, 0.175, 0.35):
    row = [float(np.mean([cerr(closure(b["W"], b["dl"], c, LAY, lam), b["T"]) for b in B]))
           for c in CS]
    best[lam] = (CS[int(np.argmin(row))], min(row), row[0])
    print("%10.3f | %s" % (lam, " ".join("%.4e" % x for x in row)))
    sys.stdout.flush()
print("\n%10s %8s %14s %14s %9s" % ("LAM_FEAT", "c*", "MSE при c=0", "MSE при c*", "×виграш"))
for lam in (0.0, 0.175, 0.35):
    c, mn, m0 = best[lam]
    print("%10.3f %8.2f %14.4e %14.4e %9.4f" % (lam, c, m0, mn, m0/mn))
b0 = min(best[l][1] for l in best)
print("\nнайкраща абсолютна MSE серед усіх: %.4e (проти базової %.4e)"
      % (b0, best[0.35][2]))
lg("ГОТОВО")
print("DONE_DBL117")
