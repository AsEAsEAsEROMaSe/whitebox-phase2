"""С118 — ЧИ ТРИКУТНИЙ ДОДАНОК ПОРОДЖЕННЯ ЗАКРИВАЄ 12% κ₃ НА ШАРІ 1?

gen118: ланцюгові доданки (хаб + дві руки, amax=3) дають R² 0.88 і масштаб 1.22 на l=1, де вхід
p_0 гаусів ТОЧНО ⟹ решта — доданки з усіма трьома ребрами (a,b,c ≥ 1). Найнижчий: (2,2,2)/(1,1,1):
   T3_ijk = 8 c2_i c2_j c2_k ρ_ij ρ_ik ρ_jk  ⟹  κ_aab += 8 Σ_k c2_k W_kb · diag(B_kᵀ ρ B_k)_a,
   B_k = (c2 ⊙ ρ[:,k])[:,None] ⊙ W.   O(n⁴) — лише стенд (GPU).
Порівняння з P3f_1 (sub118): R² ланцюг · R² ланцюг+трикутник · масштаб.
🔮 P1 (55%): трикутник піднімає R² з 0.88 до ≥ 0.95 і масштаб до ≤ 1.08
🔮 P2 (45%): ні (≤ 0.91) ⟹ решта у вищих ρ-степенях/кластерах, хабова форма їх не несе
"""
import sys, os, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq
from math import erf, factorial
D, NW = 16, 1024
NET = os.environ.get("NET", "regina-johnston"); L = int(os.environ.get("LAYER", "1"))
OUT = "/home/trim/wblogs"
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def phi(t): return np.exp(-0.5*t*t)/np.sqrt(2*np.pi)
_erf = np.vectorize(erf)
def Phi(t): return 0.5*(1.0 + _erf(t/np.sqrt(2)))
def He(n, t):
    a, b = np.ones_like(t), t
    if n == 0: return a
    for k in range(1, n):
        a, b = b, t*b - k*a
    return b
def hcoef(s, t, N):
    ph = phi(t); c = [s*(ph + t*Phi(t)), s*Phi(t)]; f = 1.0
    for n in range(2, N+1):
        f *= n; c.append(s*((-1)**n)*He(n-2, t)*ph/f)
    return c
def gen_chain(mp, Sp, W, amax):
    s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s)
    c = hcoef(s, t, 2*amax); rp = {1: rho}
    for a in range(2, amax+1): rp[a] = rp[a-1]*rho
    A = {a: rp[a] @ (c[a][:, None]*W) for a in range(1, amax+1)}
    K = np.zeros((NW, NW))
    for a in range(1, amax+1):
        for b in range(1, amax+1):
            f = factorial(a+b); ch = c[a+b]
            K += 2*f*((ch[:, None]*W*A[a]).T @ A[b]) + f*((A[a]*A[b]).T @ (ch[:, None]*W))
    return K

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
idx = NAMES.index(NET)
W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
bank = np.load("%s/sub118_%s.npz" % (OUT, NET))
if L == 1:
    mp = np.zeros(NW); Sp = W[0].T @ W[0]
else:
    mu = bank["mu_%d" % (L-1)]; S = bank["S_%d" % (L-1)]; mp = mu @ W[L-1]; Sp = W[L-1].T @ S @ W[L-1]
Kt = bank["P3f_%d" % L]
Kc = gen_chain(mp, Sp, W[L], 3)
s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s); c = hcoef(s, t, 4)
def R2(p, tr): return 1 - ((tr - p)**2).sum()/(tr**2).sum()
def sc(p, tr): return float((p*tr).sum()/(p*p).sum())
lg("l=%d ланцюг amax=3: R² %.4f · масштаб %.3f" % (L, R2(Kc, Kt), sc(Kc, Kt)))
# трикутник (1,1,1) на GPU
rg = cp.asarray(rho, cp.float32); Wg = cp.asarray(W[L], cp.float32); c2 = cp.asarray(c[2], cp.float32)
T3 = cp.zeros((NW, NW), cp.float64)
for k in range(NW):
    Bk = (c2*rg[:, k])[:, None]*Wg                  # (n × n): B_k[i,a]
    d = ((rg @ Bk)*Bk).sum(0)                        # diag(B_kᵀ ρ B_k)_a
    T3 += cp.outer(d, c2[k]*Wg[k, :]).astype(cp.float64)
    if k % 256 == 255: lg("  k=%d" % (k+1))
T3 = 8*cp.asnumpy(T3)
lg("трикутник: ‖T3‖/‖Kt‖ %.3f · R² ланцюг+трикутник %.4f · масштаб %.3f · R² лише трикутник проти залишку %.4f"
   % (np.linalg.norm(T3)/np.linalg.norm(Kt), R2(Kc + T3, Kt), sc(Kc + T3, Kt), R2(T3, Kt - Kc)))
np.savez("%s/tri118_%s_L%d.npz" % (OUT, NET, L), T3=T3, Kc=Kc, Kt=Kt)
lg("ГОТОВО"); print("DONE_TRI")
