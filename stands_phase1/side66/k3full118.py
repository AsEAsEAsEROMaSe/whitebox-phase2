"""С118 — ПОВНИЙ ТЕНЗОР κ₃(p_14) ОДНІЄЇ МЕРЕЖІ: ЙОГО ВЛАСНИЙ РАНГ І ЗАКОН ПЕРЕНОСУ

gen118: породження κ₃ з гаусового p_{l−1} дає R² 0.88 на l=1 і лише 0.15 на l=15 ⟹ κ₃
на глибині є ПЕРЕНЕСЕНИМ. inj118b: підпростір власних Σ(h) рангу 256 тримає 93% γ₃, але
дає лише ×2.0 проти ×3.9 (повний 3-й порядок). Питання: (1) який ВЛАСНИЙ ранг тензора
(HOSVD) і чи його власний базис кращий за власні Σ; (2) чи є перенос через ReLU лінійним
відгуком κ₃(h_15) ≈ c1⊗c1⊗c1 ⊙ κ₃(p_14) + породження(гаус p_14).

T = E[p̃⊗p̃⊗p̃], p̃ = (h_14−μ)W_14 (центрована передактивація шару 14), fp32, 2²⁰ семплів.
Контракції: K(X) = T(X,X,X)_{ab} = Σ_ijk T_ijk X_ia X_ja X_kb  (κ_aab(p_15) при X = c1⊙W_15)
   S1: X = I ⟹ K = T_{aab} ⟹ 🔒 діагональ = P3f_14 банку (sub118) з точністю шуму
   перенос: K_pred = T(c1⊙W_15) + gen_k3(mp_14, Σ_p14, W_15) проти P3f_15 → R²
   ранг: G = T₍₁₎T₍₁₎ᵀ → спектр; E_r(власний) = 1 − ‖K(X) − K(P_rX)‖²/‖K(X)‖²; те саме з P_r = топ-r Σ_p14
🔮 P1 (55%): перенос «c1c1c1 + породження» дає R² ≥ 0.8 для κ_aab(p_15)
🔮 P2 (50%): власний базис тензора при r=256 тримає ≥ 0.97 енергії K (проти ~0.93 у базисі Σ)
🔮 P3 (60%): ефективний ранг (PR спектру G) < 100
"""
import sys, os, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq
from math import erf, factorial
D, NW = 16, 1024
L = int(os.environ.get("LAYER", "14"))
NTRUTH = 1 << 20
BATCH = 1 << 13
CH = 32
NET = os.environ.get("NET", "regina-johnston")
OUT = "/home/trim/wblogs"
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
def gen_k3(mp, Sp, W, amax=3):
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
Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
seed = 930000

def fwd(x, k0, k1):
    for k in range(k0, k1): x = cp.maximum(x @ Wg[k], 0.0)
    return x
# прохід 1: μ_L
mu = cp.zeros(NW, cp.float64); done, b = 0, 0
while done < NTRUTH:
    n = min(BATCH, NTRUTH - done)
    z = cp.random.RandomState(seed=seed + b).standard_normal((n, NW), dtype=cp.float32)
    mu += fwd(z, 0, L).astype(cp.float64).sum(0); done += n; b += 1
mu /= NTRUTH
lg("μ_%d готово · 🔒 проти банку: %.2e" % (L, float(cp.abs(mu - cp.asarray(bank["mu_%d" % L])).max())))
# прохід 2: T = E[p̃⊗p̃⊗p̃], fp32  (або з диска, якщо вже рахували)
TN = "%s/k3full118_%s_L%d_T.npy" % (OUT, NET, L)
T = cp.zeros((NW, NW, NW), cp.float32)
WL = Wg[L]; muf = mu.astype(cp.float32)
done, b = 0, 0
if os.path.exists(TN):
    Tm = np.load(TN, mmap_mode="r")
    for c0 in range(0, NW, CH): T[c0:c0+CH] = cp.asarray(np.ascontiguousarray(Tm[c0:c0+CH]))
    done = NTRUTH; lg("тензор ПРОЧИТАНО з диска")
while done < NTRUTH:
    n = min(BATCH, NTRUTH - done)
    z = cp.random.RandomState(seed=seed + b).standard_normal((n, NW), dtype=cp.float32)
    p = (fwd(z, 0, L) - muf) @ WL                       # (n, NW) fp32 центрована
    for c0 in range(0, NW, CH):
        Y = (p[:, c0:c0+CH, None]*p[:, None, :]).reshape(n, CH*NW)      # (n, CH·NW)
        T[c0:c0+CH] += (Y.T @ p).reshape(CH, NW, NW)
        del Y
    del z, p; done += n; b += 1
    if b % 16 == 0: lg("  тензор: %d/%d" % (done, NTRUTH))
if not os.path.exists(TN):
    T /= np.float32(NTRUTH)
    Tm = np.lib.format.open_memmap(TN + ".tmp.npy", mode="w+", dtype=np.float32, shape=(NW, NW, NW))
    for c0 in range(0, NW, CH): Tm[c0:c0+CH] = cp.asnumpy(T[c0:c0+CH])
    Tm.flush(); del Tm; os.replace(TN + ".tmp.npy", TN)
nrm = 0.0
for c0 in range(0, NW, CH): nrm += float((T[c0:c0+CH].astype(cp.float64)**2).sum())
lg("тензор готово · ‖T‖ %.4e · на диску %s" % (np.sqrt(nrm), TN))
# 🔒 S1: діагональ T_{aab} проти P3f_L банку
Kd = cp.zeros((NW, NW), cp.float64)
for c0 in range(0, NW, CH):
    Kd[c0:c0+CH] = T[c0:c0+CH, cp.arange(c0, c0+CH), :].astype(cp.float64) if False else cp.stack([T[i, i, :] for i in range(c0, c0+CH)]).astype(cp.float64)
P3b = bank["P3f_%d" % L]; Kdn = cp.asnumpy(Kd)
lg("🔒 S1: T_aab проти банку P3f_%d: R² %.4f · corr діаг γ₃ %.4f" % (
    L, 1 - ((Kdn - P3b)**2).sum()/(P3b**2).sum(), np.corrcoef(np.diag(Kdn), np.diag(P3b))[0, 1]))

def contract(X):
    """K_ab = Σ_ijk T_ijk X_ia X_ja X_kb"""
    Xg = cp.asarray(X, cp.float32); K = cp.zeros((NW, NW), cp.float64)
    for c0 in range(0, NW, CH):
        Z = cp.matmul(cp.matmul(Xg.T[None], T[c0:c0+CH]), Xg[None])      # (CH, NW, NW): X^T T_i X
        K += cp.einsum("ia,iab->ab", Xg[c0:c0+CH], Z).astype(cp.float64)
        del Z
    return cp.asnumpy(K)

# перенос до шару L+1
S_L = bank["S_%d" % L]; mu_L = bank["mu_%d" % L]
mp = mu_L @ W[L]; Sp = W[L].T @ S_L @ W[L]; Sp = 0.5*(Sp + Sp.T)
s = np.sqrt(np.diag(Sp)); t = mp/s; c = hcoef(s, t, 2)
# 🚨 T у СИРИХ одиницях p ⟹ лінійний відгук ReLU = Φ(t) (без s!); c1 = sΦ дало масштаб 13.9 = 1/s³
X = Phi(t)[:, None]*W[L+1]                                 # Φ ⊙ W_{L+1}
Kt = bank["P3f_%d" % (L+1)]
Klin = contract(X); Kgen = gen_k3(mp, Sp, W[L+1])
def R2(a, b): return 1 - ((b - a)**2).sum()/(b**2).sum()
off = ~np.eye(NW, dtype=bool)
for nm_, Kp in [("лише перенос c1c1c1", Klin), ("лише породження", Kgen), ("перенос + породження", Klin + Kgen)]:
    lg("перенос → κ₃(p_%d): %-22s R² повний %.3f · позадіаг %.3f · діаг(γ₃) %.3f · масштаб %.3f" % (
        L+1, nm_, R2(Kp, Kt), 1 - ((Kt - Kp)[off]**2).sum()/(Kt[off]**2).sum(),
        R2(np.diag(Kp), np.diag(Kt)), float((Kp*Kt).sum()/(Kp*Kp).sum())))
# HOSVD: G = T(1) T(1)^T
G = cp.zeros((NW, NW), cp.float64)
for j in range(NW):
    Tj = cp.ascontiguousarray(T[:, j, :]); G += (Tj @ Tj.T).astype(cp.float64); del Tj
ev, V = cp.linalg.eigh(G); ev = cp.asnumpy(ev)[::-1]; V = cp.asnumpy(V)[:, ::-1]
ev = np.maximum(ev, 0); cum = np.cumsum(ev)/ev.sum()
lg("HOSVD спектр: PR %.1f · енергія G у топ-r: 32 %.3f · 74 %.3f · 128 %.3f · 256 %.3f · 512 %.3f" % (
    ev.sum()**2/(ev**2).sum(), cum[31], cum[73], cum[127], cum[255], cum[511]))
lamS, VS = np.linalg.eigh(Sp); VS = VS[:, ::-1]
Sh = np.load("%s/covbank118_%s.npz" % (OUT, NET))["S_%d" % (L+1)]      # Σ(h_{L+1}) = Cov(ReLU p_L)
lamH, VH = np.linalg.eigh(Sh); VH = VH[:, ::-1]
Kfull = Klin
print("\n── енергія перенесеного K = T(P_r X) проти T(X) (X = Φ⊙W_%d) ──" % (L+1))
print("   r | власний базис T | базис власних Σ_p | базис власних Σ(h_%d)" % (L+1))
for r in (32, 74, 128, 256, 512):
    Ko = contract(V[:, :r] @ (V[:, :r].T @ X)); Ks = contract(VS[:, :r] @ (VS[:, :r].T @ X)); Kh = contract(VH[:, :r] @ (VH[:, :r].T @ X))
    print(" %4d |   %.4f          |   %.4f          |   %.4f" % (r, R2(Ko, Kfull), R2(Ks, Kfull), R2(Kh, Kfull)))
# найкращий спільний масштаб переносу і породження (МНК 2×2) — чи є ще щось окрім цих двох
A2 = np.c_[Klin.ravel(), Kgen.ravel()]; coef = np.linalg.lstsq(A2, Kt.ravel(), rcond=None)[0]
lg("МНК (перенос, породження) → κ₃(p_%d): коефіцієнти %.3f %.3f · R² %.4f" % (L+1, coef[0], coef[1], R2(A2 @ coef, Kt.ravel())))
np.savez("%s/k3full118_%s_L%d.npz" % (OUT, NET, L), ev=ev, V=V, Klin=Klin, Kgen=Kgen, Kt=Kt, Kd=Kdn)
lg("ГОТОВО"); print("DONE_K3FULL")
