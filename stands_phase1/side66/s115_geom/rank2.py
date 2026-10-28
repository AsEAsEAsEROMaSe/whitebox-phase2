"""С115-п: СКІЛЬКИ РАНГУ ПОТРІБНО ЗАМИКАННЮ У ФАЗІ 2 (питання ЮЗЕРА «привести до 3D»).

ДВА ЗАКРИТІ ПРОЧИТАННЯ (не переміряю):
  3D як квадратура  — [С115-и] K=3 знімає 9% похибки, зняття йде за дисперсією
  3D як ранг стану  — [С115-и/сьогодні] PR(Cov h) 60.2, PR(J) 32.0 у фазі 2, не 2.8

ЦЕ ПРО ТРЕТЄ: скільки рангу треба, щоб замикання НЕ зіпсувалось. Питання ЦІНИ:
бал = a·p/B, і здешевлене замикання конвертується в семпли МК напряму.

СХЕМА: у рекурсії `C` обрізаємо до рангу k (eigh, топ-k), далі як завжди.
  ranklow  — чистий ранг k
  rankdiag — ранг k + ТОЧНА діагональ (реальна схема так і робила б, це дешево)
🔒 k=1024 мусить відтворити базу БІТ У БІТ.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 k=3 руйнує (похибка >1%, тобто ×10 гірше)                    80%
  P2 k=32 ще помітно гірше за базу (×1.5+)                        60%
  P3 k=256 ≈ база (в межах 5%)                                    70%
  P4 діагональ рятує СИЛЬНО: rankdiag(32) кращий за ranklow(256)  55%
  🔒 P5 k=1024 = база бітово                                      95%
"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np, glob
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
RANKS = [1, 3, 8, 32, 64, 128, 256, 512, 1024]

def lowrank(C, k, keepdiag):
    if k >= NW: return C
    dg = np.diag(C).copy()
    ev, U = np.linalg.eigh(C)
    U = U[:, -k:]; ev = np.maximum(ev[-k:], 0.0)
    R = (U*ev) @ U.T
    if keepdiag: R = R + np.diag(dg - np.diag(R))
    return R

def closure(W, k=None, keepdiag=False):
    m = np.zeros(NW); C = np.eye(NW); MS = []
    for i in range(D):
        w = W[i]; q = w.T @ C @ w
        if i > KSIG: q = q*KAPPA
        s = np.sqrt(np.maximum(np.diag(q), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t); m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(q, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy())
        if i == D-1: break
        C = q*np.outer(Ph, Ph) + 0.5*(q*q)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
        if k is not None: C = lowrank(C, k, keepdiag)
    return MS

def cerr(x, tr):
    a = (x@tr)/max((x*x).sum(),1e-300); e = a*x - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

NETS = []
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:8]:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    c = "/home/trim/wblogs/w2_%s.npy" % nm
    if os.path.exists(c):
        NETS.append((nm, np.load(c).astype(np.float64), np.asarray(d["m_half"])[:,1,:]))
log("мереж: %d" % len(NETS))
base = [cerr(closure(W)[15], mB[15]) for _, W, mB in NETS]
B0 = float(np.median(base)); log("база %.4f%%" % (100*B0))

# спектр C по глибині (довідка: скільки рангу там взагалі є)
_, W0, _ = NETS[0]
m = np.zeros(NW); C = np.eye(NW); prs = []
for i in range(D):
    w = W0[i]; q = w.T @ C @ w
    if i > KSIG: q = q*KAPPA
    s = np.sqrt(np.maximum(np.diag(q),1e-20)); mp = m@w; t = mp/s; ph = phi(t); Ph = Phi(t)
    m = mp*Ph + s*ph
    if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(q,s,t,Ph,ph,m,i/float(D))
    dd = (mp*mp+s*s)*Ph + (mp*s)*ph - m*m
    if i == D-1: break
    C = q*np.outer(Ph,Ph) + 0.5*(q*q)*np.outer(ph/s,ph/s); C = C + np.diag(dd-np.diag(C))
    e = np.maximum(np.linalg.eigvalsh(C), 0.0); prs.append((e.sum()**2)/(e*e).sum())
print("\nPR(C) нашого замикання по шарах: " + " ".join("%.0f" % v for v in prs))

print("\nранг   ranklow            rank+діагональ")
for k in RANKS:
    lo = [cerr(closure(W, k, False)[15], mB[15]) for _, W, mB in NETS]
    di = [cerr(closure(W, k, True)[15],  mB[15]) for _, W, mB in NETS]
    l, dv = float(np.median(lo)), float(np.median(di))
    print("%5d  %.4f%% ×%.3f     %.4f%% ×%.3f" % (k, 100*l, B0/l, 100*dv, B0/dv), flush=True)
log("ГОТОВО")
