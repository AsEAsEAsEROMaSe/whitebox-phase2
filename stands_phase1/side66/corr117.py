"""С117 — НАША ДІАГОНАЛЬ + ІСТИННІ КОРЕЛЯЦІЇ (клітинка, якої не було).

[cinj117] `mpdiag` (істинна діагональ + НАШІ кореляції) = ×0.387 на K=12,
          `mpfull` (усе істинне) = ×2.285  ⟹ змішування коштує ×5.9.
ЗВОРОТНЄ змішування — НАША діагональ + ІСТИННІ кореляції — не міряв ніхто.

НАВІЩО ЦЕ ВАЖЛИВО: з 6144 семплів, які ми ВЖЕ ганяємо, кореляції оцінюються
з шумом 1/√N = 1.28% при нашій похибці 22.7% ⟹ SNR = 17.7, найвищий у задачі.
Але шум діагоналі `√(2/N)` = 1.8% ГІРШИЙ за нашу похибку sd 2.9% лише в 1.6×,
а передача вибіркового шуму ~1 проти нашої 0.063 [С116] ⟹ діагональ брати НЕ
можна. Отже питання рівно одне: чи працює ЛИШЕ кореляційна частина.

Режими (на всіх шарах ≥ K):
   corr    R істинна, діагональ НАША
   corrN   R оцінена з N семплів (вішартів шум), діагональ НАША
   full    усе істинне                       (контроль зверху)
   diag    лише діагональ істинна            (контроль знизу, ×5.9 шкоди)

🔮 C1 (45%) `corr` допомагає (>1.1) ⟹ змішування несиметричне
🔮 C2 (40%) `corr` ШКОДИТЬ ⟹ гейт узгодженості діє в обидва боки, гілка мертва
🔮 C3 (60%) якщо `corr` живий, при N=6144 лишається >70% виграшу
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

def wishart_noise(cp, N, rs):
    """вибіркова коваріація гаусових даних: Var(E_ij) = (c_ii c_jj + c_ij²)/N"""
    dg = np.diag(cp)
    v = (np.outer(dg, dg) + cp*cp)/float(N)
    E = rs.standard_normal((NW, NW))*np.sqrt(v)
    E = 0.5*(E + E.T)*np.sqrt(2.0)      # симетризація зі збереженням дисперсії
    return cp + E

def closure(W, K=None, mode="", cpT=None, N=None, rs=None):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        if K is not None and i >= K and mode:
            so = np.sqrt(np.maximum(np.diag(cp_), 1e-20))
            ct = cpT[i].astype(np.float64)
            if mode == "full": cp_ = ct
            elif mode == "diag":
                st = np.sqrt(np.maximum(np.diag(ct), 1e-20))
                g = st/so; cp_ = cp_*np.outer(g, g)
            elif mode in ("corr", "corrN"):
                src = ct if mode == "corr" else wishart_noise(ct, N, rs)
                ss = np.sqrt(np.maximum(np.diag(src), 1e-20))
                R = src/np.outer(ss, ss)
                cp_ = R*np.outer(so, so)          # НАША діагональ
                cp_ = cp_ + np.diag(so*so - np.diag(cp_))
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wbcov/cov_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, T=d["relu_%d" % (D-1)],
                  cp=[d["cp_%d" % L] for L in range(D)]))
lg("мереж %d" % len(B))
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
lg("база %.6e · adj %.4e" % (BASE, sc(BASE)))

rs = np.random.RandomState(31771)
print("\n── на всіх шарах ≥ K, ×база у MSE ──")
print("%4s | %8s %8s %8s | %9s %9s %9s"
      % ("K", "full", "diag", "corr", "corr6144", "corr2048", "corr1024"))
for K in [0, 8, 12, 14]:
    row = []
    for md, N in (("full", None), ("diag", None), ("corr", None),
                  ("corrN", 6144), ("corrN", 2048), ("corrN", 1024)):
        v = float(np.mean([cerr(closure(b["W"], K, md, b["cp"], N, rs), b["T"]) for b in B]))
        row.append(BASE/v)
    print("%4d | %8.4f %8.4f %8.4f | %9.4f %9.4f %9.4f" % (K, *row))
    sys.stdout.flush()
lg("ГОТОВО")
print("DONE_CORR117")
