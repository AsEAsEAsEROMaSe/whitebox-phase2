"""С119 — ПЕРЕХРЕСНІ ЧЛЕНИ κ₃×ρ У ПЕРЕНОСІ κ₃ ЧЕРЕЗ ReLU (перший порядок за κ₃, виведено з Віка):
h = Σ c_n He_n(p̃).  cum(He2_i,He2_j,He1_k) = 4ρ_ij κ̃_ijk + 2ρ_ik κ̃_ijj + 2ρ_jk κ̃_iij ;  cum(He3_i,He1_j,He1_k) = 3ρ_ij κ̃_iik + 3ρ_ik κ̃_iij ;
cum(He2_i,He1_j,He1_k) не має κ₃.  Разом із лінійним c1c1c1 κ̃ це ВСІ члени першого порядку за κ₃ (без κ₄ і κ₃²).
Стенд: регіна, T = κ₃(p) (k3full118 L14, p = (h_14−μ)W_14), Kt = κ_aab(p_next) банку, Klin+Kgen = закон переносу (R² 0.992).
Питання: чи пояснює Kcross залишок Δ = Kt − Klin − Kgen.
🔒 S0: швидкий шлях (n³ + слаби) == груба сила einsum на n=8 (1e-10)
🔮 P1 (50%): R²(Kcross, Δ) ≥ 0.5 і R²(Klin+Kgen+Kcross, Kt) ≥ 0.996
🔮 P2 (35%): ні — залишок у κ₄ або в породженні (трикутник/кластери)
"""
import sys, os, time, numpy as np
from math import erf
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def phi(t): return np.exp(-0.5*t*t)/np.sqrt(2*np.pi)
_erf = np.vectorize(erf)
def Phi(t): return 0.5*(1.0 + _erf(t/np.sqrt(2)))
def He(n, t):
    a, b = np.ones_like(t), t
    if n == 0: return a
    for k in range(1, n): a, b = b, t*b - k*a
    return b
def hcoef(s, t, N):
    ph = phi(t); c = [s*(ph + t*Phi(t)), s*Phi(t)]; f = 1.0
    for n in range(2, N+1): f *= n; c.append(s*((-1)**n)*He(n-2, t)*ph/f)
    return c
def pairterm(al, be, ga, Mij, Mik, Mjk, W):
    """K[a,b] = Σ_ijk al_i be_j ga_k Mij Mik Mjk W_ia W_ja W_kb, рівно одна з M = None (n³)"""
    if Mjk is None:
        G = Mij @ (be[:, None]*W); H = Mik @ (ga[:, None]*W); return (al[:, None]*W*G).T @ H
    if Mik is None:
        G = Mij.T @ (al[:, None]*W); H = Mjk @ (ga[:, None]*W); return (be[:, None]*W*G).T @ H
    G = Mik.T @ (al[:, None]*W); H = Mjk.T @ (be[:, None]*W); return (G*H).T @ (ga[:, None]*W)
def diag_terms(a, b, d, rho, Dg, W):
    """усі члени з діагональними зрізами κ̃_xxy = Dg[x,y]; повертає (D-частина, B-частина)"""
    DgT = Dg.T
    D = (pairterm(2*b, b, a, DgT, rho, None, W)      # 2 b_i b_j a_k ρ_ik κ̃_ijj   (κ̃_ijj = Dg[j,i])
       + pairterm(2*b, b, a, Dg, None, rho, W)       # 2 b_i b_j a_k ρ_jk κ̃_iij
       + pairterm(2*b, a, b, rho, DgT, None, W)      # 2 b_i a_j b_k ρ_ij κ̃_ikk   (κ̃_ikk = Dg[k,i])
       + pairterm(2*b, a, b, None, Dg, rho, W)       # 2 b_i a_j b_k ρ_kj κ̃_iik
       + pairterm(2*a, b, b, rho, None, DgT, W)      # 2 a_i b_j b_k ρ_ji κ̃_jkk   (κ̃_jkk = Dg[k,j])
       + pairterm(2*a, b, b, None, rho, Dg, W))      # 2 a_i b_j b_k ρ_ki κ̃_jjk
    B = (pairterm(3*d, a, a, rho, Dg, None, W)       # 3 d_i a_j a_k ρ_ij κ̃_iik
       + pairterm(3*d, a, a, Dg, rho, None, W)       # 3 d_i a_j a_k ρ_ik κ̃_iij
       + pairterm(3*a, d, a, rho, None, Dg, W)       # 3 d_j a_i a_k ρ_ji κ̃_jjk  (al=a_i, be=3d_j)
       + pairterm(3*a, d, a, DgT, None, rho, W)      # 3 d_j a_i a_k ρ_jk κ̃_jji  (κ̃_jji = Dg[j,i])
       + pairterm(3*a, a, d, None, rho, DgT, W)      # 3 d_k a_i a_j ρ_ki κ̃_kkj  (пари (i,k)=ρ, (j,k)=Dgᵀ)
       + pairterm(3*a, a, d, None, rho, DgT, W))     # 3 d_k a_i a_j ρ_kj κ̃_kki  (κ̃_kki = Dg[k,i])
    return D, B
def full_terms(a, b, rho, s, W, slab, xp):
    """F1+F2+F3 через слаби T_i (сирий T, W̃ = W/s): 4ρκ̃ члени"""
    Wt = xp.asarray(W/s[:, None]); a_ = xp.asarray(a); b_ = xp.asarray(b); rg = xp.asarray(rho)
    V = a_[:, None]*Wt; Q = b_[:, None]*Wt; n = W.shape[0]
    K = xp.zeros((n, n), xp.float64)
    for i in range(n):
        Ti = xp.asarray(slab(i))
        Ui = (rg[i]*b_)[:, None]*Wt
        F1 = Ui.T @ (Ti @ V)                       # F1[a,b]; F2 = F1ᵀ
        F3 = Q.T @ ((rg*Ti) @ Q)
        K += 4*b_[i]*Wt[i][:, None]*(F1 + F1.T) + 4*a_[i]*Wt[i][:, None]*F3
        if (i+1) % 256 == 0: lg("  слаб %d/%d" % (i+1, n))
    return K if xp is np else xp.asnumpy(K)
def R2(p, t): return 1 - ((t-p)**2).sum()/(t**2).sum()
def sc(p, t): return float((p*t).sum()/(p*p).sum())
# ── S0: пломба на n=8 проти грубої сили ──
rng = np.random.default_rng(3); n = 8
T = rng.standard_normal((n, n, n)); T = T + T.transpose(1, 0, 2) + T.transpose(2, 1, 0) + T.transpose(0, 2, 1) + T.transpose(1, 2, 0) + T.transpose(2, 0, 1)
s = rng.uniform(0.5, 2, n); A = rng.standard_normal((n, n)); rho = A @ A.T; dd = np.sqrt(np.diag(rho)); rho = rho/np.outer(dd, dd)
a, b, d = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n); W = rng.standard_normal((n, n))
kt = T/(s[:, None, None]*s[None, :, None]*s[None, None, :]); Dg = np.einsum("iij->ij", kt)
X = (4*np.einsum("i,j,k,ij,ijk->ijk", b, b, a, rho, kt) + 4*np.einsum("i,j,k,ik,ijk->ijk", b, a, b, rho, kt) + 4*np.einsum("i,j,k,jk,ijk->ijk", a, b, b, rho, kt)
   + 2*np.einsum("i,j,k,ik,ji->ijk", b, b, a, rho, Dg) + 2*np.einsum("i,j,k,jk,ij->ijk", b, b, a, rho, Dg)
   + 2*np.einsum("i,j,k,ij,ki->ijk", b, a, b, rho, Dg) + 2*np.einsum("i,j,k,kj,ik->ijk", b, a, b, rho, Dg)
   + 2*np.einsum("i,j,k,ji,kj->ijk", a, b, b, rho, Dg) + 2*np.einsum("i,j,k,ki,jk->ijk", a, b, b, rho, Dg)
   + 3*np.einsum("i,j,k,ij,ik->ijk", d, a, a, rho, Dg) + 3*np.einsum("i,j,k,ik,ij->ijk", d, a, a, rho, Dg)
   + 3*np.einsum("i,j,k,ji,jk->ijk", a, d, a, rho, Dg) + 3*np.einsum("i,j,k,jk,ji->ijk", a, d, a, rho, Dg)
   + 3*np.einsum("i,j,k,ki,kj->ijk", a, a, d, rho, Dg) + 3*np.einsum("i,j,k,kj,ki->ijk", a, a, d, rho, Dg))
Kb = np.einsum("ijk,ia,ja,kb->ab", X, W, W, W)
Dp, Bp = diag_terms(a, b, d, rho, Dg, W); Fp = full_terms(a, b, rho, s, W, lambda i: T[i], np)
Kf = Dp + Bp + Fp
lg("🔒 S0 швидкий шлях проти грубої сили (n=8): max|Δ|/max|K| %.2e" % (np.abs(Kf - Kb).max()/np.abs(Kb).max()))
assert np.abs(Kf - Kb).max()/np.abs(Kb).max() < 1e-9
# ── регіна L14 ──
sys.path.insert(0, "/home/trim/gpupkg"); import cupy as cp, pyarrow.parquet as pq
OUT = "/home/trim/wblogs"; NET = "regina-johnston"; L = 14; NW = 1024
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]; idx = NAMES.index(NET)
W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(16, NW, NW)
bank = np.load("%s/sub118_%s.npz" % (OUT, NET)); mp = bank["mu_%d" % L] @ W[L]; Sp = W[L].T @ bank["S_%d" % L] @ W[L]
s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s); c = hcoef(s, t, 4); a, b, d = c[1], c[2], c[3]
Tm = np.load("%s/k3full118_%s_L%d_T.npy" % (OUT, NET, L), mmap_mode="r")
Dg = np.empty((NW, NW)); 
for i in range(NW): Dg[i] = Tm[i, i, :]
Dg = Dg/(s[:, None]**2*s[None, :])
lg("Dg готово · 🔒 діагональ T_aaa/s³ = γ₃ банку: R² %.3f" % R2(np.diag(Dg), np.diag(bank["P3f_%d" % L])/s**3))
Wn = W[L+1]
Dp, Bp = diag_terms(a, b, d, rho, Dg, Wn); lg("D,B готово")
Fp = full_terms(a, b, rho, s, Wn, lambda i: np.ascontiguousarray(Tm[i]).astype(np.float32).astype(np.float64), cp); lg("F готово")
Kc = Dp + Bp + Fp
z = np.load("%s/k3full118_%s_L%d.npz" % (OUT, NET, L)); Klin, Kgen, Kt = z["Klin"], z["Kgen"], z["Kt"]
base = Klin + Kgen; Dl = Kt - base
off = ~np.eye(NW, dtype=bool)
lg("база перенос+породження: R² %.4f · залишок ‖Δ‖/‖Kt‖ %.3f" % (R2(base, Kt), np.linalg.norm(Dl)/np.linalg.norm(Kt)))
for nm, K in [("F (4ρκ̃)", Fp), ("D (He2He2He1 діаг)", Dp), ("B (He3)", Bp), ("F+D+B", Kc)]:
    lg("%-22s ‖K‖/‖Δ‖ %.3f · R²(K,Δ) %.3f · масштаб %.3f · R²(база+K, Kt) %.4f · позадіаг %.4f · γ₃ %.4f" % (
        nm, np.linalg.norm(K)/np.linalg.norm(Dl), R2(K, Dl), sc(K, Dl), R2(base + K, Kt),
        1 - ((Kt - base - K)[off]**2).sum()/(Kt[off]**2).sum(), R2(np.diag(base + K), np.diag(Kt))))
A3 = np.c_[Klin.ravel(), Kgen.ravel(), Kc.ravel()]; cf = np.linalg.lstsq(A3, Kt.ravel(), rcond=None)[0]
lg("МНК (перенос, породження, перехресні): %.3f %.3f %.3f · R² %.4f" % (cf[0], cf[1], cf[2], R2(A3 @ cf, Kt.ravel())))
np.savez("%s/cross118_%s_L%d.npz" % (OUT, NET, L), F=Fp, D=Dp, B=Bp)
lg("ГОТОВО"); print("DONE_CROSS")
