"""ФУНКЦІЯ ПЕРЕДАЧІ КАНАЛУ `s` — БЕЗ ІСТИНИ, самими скінченними різницями.
Збурюємо `s` на шарі l на +ε узгоджено (cp ← Dg·cp·Dg) і міряємо відносний
зсув `mp₁₅` та `m₁₅`. Це дає, СКІЛЬКИ важить похибка `s` на кожній глибині,
ще до того, як банк скаже, ЯКА вона там.
"""
import sys, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
def psi(t): return phi(t) + t*Phi(t)

def closure(W, eps=0.0, lay=-1):
    m = np.zeros(NW); C = np.eye(NW); OMP = []; TT = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20))
        if i == lay and eps != 0.0:
            g = 1.0 + eps; cp_ = cp_*(g*g); s = s*g
        mp = m @ w; OMP.append(mp.copy()); TT.append(mp/s)
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m, np.stack(OMP), np.stack(TT)

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NN = 2; EPS = 0.01
G = np.zeros((NN, D, 2)); TM = np.zeros((NN, D, 2))
for k in range(NN):
    W = np.asarray(tb["weights"][k].as_py(), np.float64).reshape(D, NW, NW)
    m0, mp0, tt0 = closure(W)
    for L in range(D):
        m1, mp1, _ = closure(W, EPS, L)
        G[k, L, 0] = (np.linalg.norm(mp1[15]-mp0[15])/np.linalg.norm(mp0[15]))/EPS
        G[k, L, 1] = (np.linalg.norm(m1-m0)/np.linalg.norm(m0))/EPS
        t = tt0[L]
        TM[k, L] = [np.mean(np.abs(t)), np.mean(phi(t)/np.maximum(psi(t), 1e-12))]
g = G.mean(0); tm = TM.mean(0)
print("ФУНКЦІЯ ПЕРЕДАЧІ `s` (%d мереж, ε=%.0e, підсилення = δ_out/δ_s)" % (NN, EPS))
print("%4s %8s %9s %14s %14s" % ("шар","сер|t|","лок φ/ψ","→ mp₁₅","→ m₁₅ (вихід)"))
for L in range(D):
    print("%4d %8.3f %9.4f %13.4f %13.4f" % (L, tm[L,0], tm[L,1], g[L,0], g[L,1]))
print("\nсума підсилень по мілких 1..6 = %.3f · по глибоких 9..15 = %.3f"
      % (g[1:7,1].sum(), g[9:16,1].sum()))
