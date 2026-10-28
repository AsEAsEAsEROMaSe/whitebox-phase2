"""С118 — ЧИ ПОРОДЖУЄТЬСЯ κ₃ ПЕРЕДАКТИВАЦІЙ «МЕЛЕРОМ ТРЕТЬОГО ПОРЯДКУ» З (μ, Σ) ПОПЕРЕДНЬОГО ШАРУ?

Для гаусової трійки з кореляціями: E[He_p(x)He_q(y)He_r(z)] = Σ_{a+b=p, a+c=q, b+c=r}
p!q!r!/(a!b!c!) ρ_xy^a ρ_xz^b ρ_yz^c. Кумулянт κ₃(f_i,f_j,f_k) для f = ReLU(p) — зв'язні
доданки. «Ланцюгові» (одна вершина-хаб i, c=0): Σ_{a,b≥1} (a+b)! c_{a+b}(i) c_a(j) c_b(k) ρ_ij^a ρ_ik^b
— контрактуються з w за O(n³) (матмули); «трикутні» (a,b,c≥1) — O(n⁴), тут ВІДКИНУТО.
Прогноз мішаних кумулянтів наступної передактивації p_l = h_l W_l, h_l = ReLU(p_{l−1}):
   κ_aab(p_l) = Σ_ijk κ₃(h_l)_ijk w_ia w_ja w_kb, γ₃(p_l)_a = κ_aaa
з ГАУСОВОГО p_{l−1} (mp, Σ_p з банку sub118). Порівняння з істинними P3f_l (банк).
🔒 l=1: p_0 = zW_0 гаусів ТОЧНО ⟹ R² = 1 − (обрізання трикутних + ρ^{>A}) — пломба формули
l≥2: залишок = ПЕРЕНОС κ₃(p_{l−1}) через ReLU (+ вищі кумулянти) — те, що треба нести станом.
🔮 P1 (60%): R²(l=1) ≥ 0.95 для γ₃ і κ_aab   🔮 P2 (50%): R²(l=15) ≤ 0.5 (перенос домінує на глибині)
"""
import sys, os, time, glob
import numpy as np
from math import erf
D, NW = 16, 1024
OUT = "/home/trim/wblogs"
AMAX = int(os.environ.get("AMAX", "3"))     # макс. степінь ρ на ребро
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
SQ2PI = np.sqrt(2*np.pi)
def phi(t): return np.exp(-0.5*t*t)/SQ2PI
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
from math import factorial

def gen_k3(mp, Sp, W, amax=AMAX):
    """κ_aab(p_l) [n×n] з гаусового p_{l−1} ~ N(mp, Sp): ланцюгові доданки до ρ^amax на ребро"""
    s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s)
    c = hcoef(s, t, 2*amax)
    rp = {1: rho}
    for a in range(2, amax+1): rp[a] = rp[a-1]*rho
    # A[a] = ρ^{∘a} (c_a ⊙ W)   (n × n_a)
    A = {a: rp[a] @ (c[a][:, None]*W) for a in range(1, amax+1)}
    K = np.zeros((NW, NW))
    for a in range(1, amax+1):
        for b in range(1, amax+1):
            f = factorial(a+b); ch = c[a+b]
            # хаб i (c_{a+b}), i→j (ρ^a, c_a), i→k (ρ^b, c_b); контракція w_ia w_ja w_kb:
            #   Σ_i ch_i w_ia A[a]_ia A[b]_ib  → (ch⊙W⊙A[a])ᵀ A[b]
            K += f*((ch[:, None]*W*A[a]).T @ A[b])
            # хаб j: Σ_j ch_j w_ja A[a]_ja A[b]_jb  (i→j ребро ρ^a з c_a на i; j→k ρ^b)  — те саме за симетрією a↔a
            K += f*((ch[:, None]*W*A[a]).T @ A[b])
            # хаб k: Σ_k ch_k w_kb A[a]_ka A[b]_ka → (A[a]⊙A[b])ᵀ (ch⊙W)
            K += f*((A[a]*A[b]).T @ (ch[:, None]*W))
    return K

import pyarrow.parquet as pq
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
files = sorted(glob.glob("%s/sub118_*.npz" % OUT))
lg("банків %d · AMAX=%d" % (len(files), AMAX))
ROWS = []
for fn in files:
    nm = os.path.basename(fn)[len("sub118_"):-4]
    idx = NAMES.index(nm)
    W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
    d = np.load(fn)
    mu = [np.zeros(NW)] + [d["mu_%d" % l] for l in range(1, D)]
    S = [np.eye(NW)] + [d["S_%d" % l] for l in range(1, D)]
    row = []
    for l in range(1, D):
        mp = mu[l-1] @ W[l-1]; Sp = W[l-1].T @ S[l-1] @ W[l-1]; Sp = 0.5*(Sp + Sp.T)
        Kp = gen_k3(mp, Sp, W[l])
        Kt = d["P3f_%d" % l]
        sl = np.sqrt(np.diag(W[l].T @ S[l] @ W[l]))
        g3t = np.diag(Kt)/sl**3; g3p = np.diag(Kp)/sl**3
        off = ~np.eye(NW, dtype=bool)
        kt = Kt/np.outer(sl*sl, sl); kp = Kp/np.outer(sl*sl, sl)
        R2d = 1 - ((g3t - g3p)**2).sum()/(g3t**2).sum()
        R2o = 1 - ((kt - kp)[off]**2).sum()/(kt[off]**2).sum()
        # найкращий масштаб (чи це недооцінка амплітуди)
        ad = float(g3p @ g3t)/float(g3p @ g3p); ao = float((kp[off]*kt[off]).sum()/(kp[off]**2).sum())
        row.append((l, R2d, R2o, ad, ao, g3t.std(), np.corrcoef(g3t, g3p)[0, 1]))
    ROWS.append(row)
    lg("%-20s l=1: R²γ₃ %.3f R²κ_aab %.3f · l=8: %.3f %.3f · l=15: %.3f %.3f" % (
        nm, row[0][1], row[0][2], row[7][1], row[7][2], row[14][1], row[14][2]))
print("\n── прогноз κ₃(p_l) з ГАУСОВОГО p_{l−1} (ланцюгові доданки до ρ^%d), середнє по %d мережах ──" % (AMAX, len(ROWS)))
print("  l | R² γ₃   corr    масшт | R² κ_aab  масшт | sd γ₃ істин")
for i in range(D-1):
    v = np.array([r[i] for r in ROWS]).mean(0)
    print(" %2d | %6.3f  %6.3f  %6.3f | %6.3f    %6.3f | %.4f" % (v[0], v[1], v[6], v[3], v[2], v[4], v[5]))
lg("ГОТОВО"); print("DONE_GEN")
