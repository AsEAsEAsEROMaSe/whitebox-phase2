"""С117 — ДЕ САМЕ СТАН НЕПРАВИЛЬНИЙ: mp · діагональ · КОРЕЛЯЦІЇ.

[res117] показав, що формула кроку виправляється на 99.9% через γ₃+γ₄ ⟹
похибка живе НЕ в середньому кроці. Лишається ПЕРЕНОС `C` (Мелер K=2),
якого жоден залишок кроку не чіпає.

`lad116` вкидав істинну ДІАГОНАЛЬ при НАШИХ кореляціях і діставав шкоду.
Тут уперше вкидається ІСТИННА ПОВНА `cp`.

Режими на шарі K (далі — наша рекурсія до 15):
   mp     істинне середнє передактивацій, наша cp
   diag   істинна діагональ через outer(g,g)  (режим lad116)
   full   істинна ПОВНА cp, наше mp
   mpdiag mp + diag                            (режим lad116)
   mpfull mp + full                            ← НЕЗМІРЯНА КЛІТИНКА

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 Q1 (70%) `full` сам по собі допомагає слабко (×1.0…×1.2) — узгоджено з С115
 Q2 (55%) `mpfull` дає СИНЕРГІЮ і б'є `mpdiag` у ≥×3 на K=12
 Q3 (40%) `mpfull` на K=12 дає ×3+ проти бази ⟹ рекурсія здорова, винен стан
 Q4 (60%) відносна похибка КОРЕЛЯЦІЙ росте з глибиною сильніше за похибку sd
"""
import sys, glob, time, os
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT = 3.3478454e-06, 1.4355576e-06, 0.1
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)

def closure(W, K=None, mode="", mpT=None, cpT=None, want_state=False):
    m = np.zeros(NW); C = np.eye(NW); ST = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        mp = m @ w
        if K is not None and i == K:
            if "full" in mode: cp_ = cpT[i].astype(np.float64)
            elif "diag" in mode:
                g = np.maximum(np.sqrt(np.maximum(np.diag(cpT[i]), 1e-20)), 1e-20) \
                    / np.maximum(np.sqrt(np.maximum(np.diag(cp_), 1e-20)), 1e-20)
                cp_ = cp_*np.outer(g, g)
            if "mp" in mode: mp = mpT[i]
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20))
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if want_state: ST.append((mp.copy(), cp_.copy()))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return (m, ST) if want_state else m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wbcov/cov_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, T=d["relu_%d" % (D-1)],
                  mp=[d["mp_%d" % L] for L in range(D)],
                  cp=[d["cp_%d" % L] for L in range(D)]))
lg("мереж %d" % len(B))

# ── 0. КАРТА ПОХИБОК СТАНУ ───────────────────────────────────────────────
print("\n── де саме наш стан не той (%d мереж, %%) ──" % len(B))
print("%4s %9s %9s %9s %9s" % ("шар", "mp", "sd", "корел R", "cp повна"))
for b in B[:1]: _, ST = closure(b["W"], want_state=True)
acc = np.zeros((D, 4))
for b in B:
    _, ST = closure(b["W"], want_state=True)
    for L in range(D):
        mpo, cpo = ST[L]; mpt, cpt = b["mp"][L], b["cp"][L].astype(np.float64)
        so = np.sqrt(np.maximum(np.diag(cpo), 1e-30)); st = np.sqrt(np.maximum(np.diag(cpt), 1e-30))
        Ro = cpo/np.outer(so, so); Rt = cpt/np.outer(st, st)
        acc[L] += [100*np.linalg.norm(mpo-mpt)/np.linalg.norm(mpt),
                   100*np.linalg.norm(so-st)/np.linalg.norm(st),
                   100*np.linalg.norm(Ro-Rt)/np.linalg.norm(Rt),
                   100*np.linalg.norm(cpo-cpt)/np.linalg.norm(cpt)]
acc /= len(B)
for L in range(D):
    print("%4d %9.4f %9.4f %9.4f %9.4f" % (L, acc[L,0], acc[L,1], acc[L,2], acc[L,3]))

# ── 1. ВКИДАННЯ ──────────────────────────────────────────────────────────
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
lg("база %.6e · adj %.4e" % (BASE, sc(BASE)))
KS = [4, 8, 11, 12, 13, 14, 15]
MODES = ["mp", "diag", "full", "mpdiag", "mpfull"]
print("\n── вкидання на шарі K, ×база у косинусній похибці ──")
print("%4s | %s" % ("K", " ".join("%8s" % m for m in MODES)))
for K in KS:
    row = []
    for md in MODES:
        v = float(np.mean([cerr(closure(b["W"], K, md, b["mp"], b["cp"]), b["T"]) for b in B]))
        row.append(BASE/v)
    print("%4d | %s" % (K, " ".join("%8.4f" % x for x in row)))
    sys.stdout.flush()

# ── 2. ВКИДАННЯ НА ВСІХ ШАРАХ ВІД K (канал НЕ дрейфує) ───────────────────
def closure_all(W, K, mode, mpT, cpT):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        mp = m @ w
        if i >= K:
            if "full" in mode: cp_ = cpT[i].astype(np.float64)
            elif "diag" in mode:
                g = np.maximum(np.sqrt(np.maximum(np.diag(cpT[i]), 1e-20)), 1e-20) \
                    / np.maximum(np.sqrt(np.maximum(np.diag(cp_), 1e-20)), 1e-20)
                cp_ = cp_*np.outer(g, g)
            if "mp" in mode: mp = mpT[i]
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20))
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

print("\n── вкидання на ВСІХ шарах >= K (канал більше не дрейфує) ──")
print("%4s | %s" % ("K", " ".join("%8s" % m for m in MODES)))
for K in [0, 4, 8, 12, 14]:
    row = []
    for md in MODES:
        v = float(np.mean([cerr(closure_all(b["W"], K, md, b["mp"], b["cp"]), b["T"]) for b in B]))
        row.append(BASE/v)
    print("%4d | %s" % (K, " ".join("%8.4f" % x for x in row)))
    sys.stdout.flush()

lg("ГОТОВО")
print("DONE_CINJ")
