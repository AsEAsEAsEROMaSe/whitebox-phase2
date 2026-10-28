"""С118 — ЯДРО κ₄(p_14) У ТОП-r p-БАЗИСІ: ЧИ ДОСЯЖНА ПОНЕЙРОННА γ₄ НИЗЬКОРАНГОВО?

inj118c: 3-й порядок обох каналів ×3.9, + понейронна γ₄ ×14.7 ⟹ γ₄ — це ×3.7 і саме вона
відділяє «#28» від «топ-10». γ₄ не виводиться з γ₃ (R² 0.2–0.3), з семплів SNR 0.26.
Єдиний детермінований шлях — рекурсія 4-го порядку: ядро r⁴ у підпросторі, перенос 4r⁵.
r=256 неможливий (r⁴ = 4.3e9), r=128 — 2.7e8 (перенос 1.4e11/шар ≈ 1 B — дорого), тож
потрібно ЩЕ нижчий власний ранг. Тут: (1) енергія δ₄ = (γ₄/24)s(t²−1)φ, яку відтворює
ядро r=128 у базисі власних Σ_p; (2) HOSVD-спектр ядра κ₄ — його власний ранг.

q = p̃ V_r (N × r), M4 = E[q⊗q⊗q⊗q] як матриця (r² × r²) = Q2ᵀQ2/N, Q2 = vec(q⊗q)
κ₄ = M4 − (Λ⊗Λ + 2 перестановки), Λ = VᵀΣ_pV. γ₄_a ≈ g2ₐᵀ κ₄ g2ₐ / s_a⁴, g_a = Vᵀw_a, g2 = vec(gg ᵀ)
🔒 r=1024 неможливо; пломба — γ₃ з тих самих q проти банку (E3 при r=128 ≈ 0.95 у p-базисі, k3full)
🔒 гаусів контроль: q_g ~ N(0, Λ) тими ж акумуляторами ⟹ κ₄ ≈ шум, E4 ≈ 0
🔮 P1 (45%): E4(δ₄, r=128) ≥ 0.80   🔮 P2 (50%): PR(κ₄-ядро) ≤ 20   🔮 P3 (60%): E4 < E3 при тому ж r
"""
import sys, os, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq
from math import erf
D, NW = 16, 1024
L = int(os.environ.get("LAYER", "14"))
R = int(os.environ.get("R", "128"))
NTRUTH = 1 << 20
BATCH = 1 << 13
NET = os.environ.get("NET", "regina-johnston")
OUT = "/home/trim/wblogs"
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def phi(t): return np.exp(-0.5*t*t)/np.sqrt(2*np.pi)

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
idx = NAMES.index(NET)
W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
bank = np.load("%s/sub118_%s.npz" % (OUT, NET)); ga = np.load("%s/gamall_%s.npz" % (OUT, NET))
mu_L = bank["mu_%d" % L]; S_L = bank["S_%d" % L]
mp = mu_L @ W[L]; Sp = W[L].T @ S_L @ W[L]; Sp = 0.5*(Sp + Sp.T)
s = np.sqrt(np.diag(Sp)); t = mp/s
lam, V = np.linalg.eigh(Sp); V = V[:, ::-1][:, :R]; lam = lam[::-1][:R]
lg("базис Σ_p топ-%d: частка дисперсії %.4f" % (R, lam.sum()/np.trace(Sp)))
Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
Vg = cp.asarray(V.astype(np.float32)); WL = Wg[L]; muf = cp.asarray(mu_L.astype(np.float32))
Lam = cp.asarray(np.diag(np.sqrt(lam)).astype(np.float32))
def fwd(x, k0, k1):
    for k in range(k0, k1): x = cp.maximum(x @ Wg[k], 0.0)
    return x
M4 = cp.zeros((R*R, R*R), cp.float32); M4g = cp.zeros((R*R, R*R), cp.float32)   # fp32: 2×1.07 ГБ, інакше OOM
M3 = cp.zeros((R, R*R), cp.float64); M2 = cp.zeros((R, R), cp.float64)
done, b = 0, 0; seed = 940000
while done < NTRUTH:
    n = min(BATCH, NTRUTH - done)
    rs = cp.random.RandomState(seed=seed + b)
    z = rs.standard_normal((n, NW), dtype=cp.float32)
    q = ((fwd(z, 0, L) - muf) @ WL) @ Vg                       # (n, R)
    Q2 = (q[:, :, None]*q[:, None, :]).reshape(n, R*R)
    M4 += Q2.T @ Q2; M3 += (q.T @ Q2).astype(cp.float64); M2 += (q.T @ q).astype(cp.float64)
    qg = rs.standard_normal((n, R), dtype=cp.float32) @ Lam
    Q2g = (qg[:, :, None]*qg[:, None, :]).reshape(n, R*R)
    M4g += Q2g.T @ Q2g
    del z, q, Q2, qg, Q2g; done += n; b += 1
    if b % 32 == 0: lg("  моменти: %d/%d" % (done, NTRUTH))
M4 /= np.float32(NTRUTH); M4g /= np.float32(NTRUTH); M4 = cp.asnumpy(M4); M4g = cp.asnumpy(M4g)
M3 = cp.asnumpy(M3)/NTRUTH; M2 = cp.asnumpy(M2)/NTRUTH
cp.get_default_memory_pool().free_all_blocks()
lg("моменти готово · 🔒 Λ проти M2: %.2e" % np.abs(M2 - np.diag(lam)).max())
def kappa4(M4, C):
    """κ₄ як (r²×r²): M4[(ij),(kl)] − C_ij C_kl − C_ik C_jl − C_il C_jk"""
    C = C.astype(np.float32)
    K = M4; K -= np.outer(C.ravel(), C.ravel())
    K -= np.einsum("ik,jl->ijkl", C, C).reshape(R*R, R*R); K -= np.einsum("il,jk->ijkl", C, C).reshape(R*R, R*R)
    return K
K4g = kappa4(M4g, np.diag(lam)); n4g = np.linalg.norm(K4g)
Gm = V.astype(np.float32); G2 = np.einsum("ai,aj->aij", Gm, Gm).reshape(NW, R*R)
k4g = np.einsum("ak,ak->a", G2 @ K4g, G2).astype(np.float64); del K4g, M4g
K4 = kappa4(M4, M2)
lg("‖κ₄‖ %.4e · 🔒 гаусів контроль ‖κ₄_g‖ %.4e (частка %.4f)" % (np.linalg.norm(K4), n4g, n4g/np.linalg.norm(K4)))
# понейронні γ₃, γ₄ з ядра
# g_a = Vᵀ w_a, де p_a = Σ_i h_i W_L[i,a]  ⟹ у p-просторі g_a = Vᵀ e_a = V[a, :]
k4 = np.einsum("ak,ak->a", G2 @ K4, G2).astype(np.float64)
# простіше: κ₃_aaa = Σ_ijk M3[i,(jk)] g_i g_j g_k
k3 = np.einsum("ai,ak,ik->a", Gm.astype(np.float64), G2.astype(np.float64), M3)
g3t = ga["g3_%d" % L]; g4t = ga["g4_%d" % L]
g3p = k3/s**3; g4p = k4/s**4; g4pg = k4g/s**4
d3t = -(g3t/6)*s*t*phi(t); d3p = -(g3p/6)*s*t*phi(t)
d4t = (g4t/24)*s*(t*t-1)*phi(t); d4p = (g4p/24)*s*(t*t-1)*phi(t); d4g = (g4pg/24)*s*(t*t-1)*phi(t)
def E(p_, t_): return 1 - ((t_ - p_)**2).sum()/(t_**2).sum()
lg("🔒 γ₃ з ядра r=%d: E3(δ₃) %.4f · corr %.4f" % (R, E(d3p, d3t), np.corrcoef(g3p, g3t)[0, 1]))
lg("γ₄ з ядра r=%d: E4(δ₄) %.4f · corr %.4f · масштаб %.3f · 🔒 гаусів контроль E4 %.4f" % (
    R, E(d4p, d4t), np.corrcoef(g4p, g4t)[0, 1], float(d4p @ d4t/(d4p @ d4p)), E(d4g, d4t)))
# HOSVD ядра κ₄: Ґрам моди 1 (r × r)
K4t = K4.reshape(R, R*R*R); Gr = K4t @ K4t.T
ev = np.linalg.eigvalsh(Gr)[::-1]; ev = np.maximum(ev, 0); cum = np.cumsum(ev)/ev.sum()
lg("HOSVD κ₄-ядра: PR %.1f · енергія у топ-k: 8 %.3f · 16 %.3f · 32 %.3f · 64 %.3f" % (ev.sum()**2/(ev**2).sum(), cum[7], cum[15], cum[31], cum[63]))
# те саме для κ₃ (M3 = центровані треті моменти = κ₃ у базисі q)
Gr3 = M3 @ M3.T; ev3 = np.maximum(np.linalg.eigvalsh(Gr3)[::-1], 0); cum3 = np.cumsum(ev3)/ev3.sum()
lg("HOSVD κ₃-ядра (r=%d): PR %.1f · топ-k: 8 %.3f · 16 %.3f · 32 %.3f · 64 %.3f" % (R, ev3.sum()**2/(ev3**2).sum(), cum3[7], cum3[15], cum3[31], cum3[63]))
# γ₄ з ядра, обрізаного до топ-k ВЛАСНОГО базису (усі 4 моди)
evk, Uk = np.linalg.eigh(Gr); Uk = Uk[:, ::-1]
for k in (16, 32, 64):
    P = Uk[:, :k] @ Uk[:, :k].T
    Gp = (Gm @ P).astype(np.float32); G2p = np.einsum("ai,aj->aij", Gp, Gp).reshape(NW, R*R)
    k4k = np.einsum("ak,ak->a", G2p @ K4, G2p).astype(np.float64); d4k = (k4k/s**4/24)*s*(t*t-1)*phi(t)
    lg("   γ₄ з власного базису k=%d: E4(δ₄) %.4f (проти повного r=%d ядра: %.4f)" % (k, E(d4k, d4t), R, E(d4k, d4p)))
np.savez("%s/k4core118_%s_L%d_r%d.npz" % (OUT, NET, L, R), K4=K4.astype(np.float32), M3=M3, lam=lam, V=V, g4p=g4p, g3p=g3p)
lg("ГОТОВО"); print("DONE_K4CORE")
