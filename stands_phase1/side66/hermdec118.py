"""С119 — ЕРМІТОВИЙ РОЗКЛАД κ_aab(p_15) ЗА БЛОКАМИ (n,m,l) НА ІСТИННИХ І НА ГАУСОВИХ СЕМПЛАХ (регіна, L=14).
h_i = Σ_n c_n(i) He_n(x_i), x = (p−mp)/s ⟹ (hW)_a = Σ_n u_n[a], u_n = (c_n⊙He_n(x))W ⟹ κ_aab = Σ_{nml} cum(ũ_n[a],ũ_m[a],ũ_l[b]).
Той самий розклад на x_g ~ N(0,ρ) = ТОЧНЕ породження (усі кластери). Різниця true−gauss за блоками = негаусовий внесок:
(1,1,1)=перенос κ₃ · (2,2,1)=4ρκ₃+… · (2,1,1)=κ₄ · (3,1,1)=κ₃×ρ (He3) · вищі.
🔒 S1: Σ блоків(істина, до 4-го порядку) проти прямого cum3(hW) і проти банку P3f_15 · S2: gauss (2,1,1) проти Kgen хабової форми (amax=1)
🔮 P1 (45%): дефіцит Δ = Kt − Klin − Kgen пояснюється блоком (2,1,1) (κ₄) на R² ≥ 0.5
🔮 P2 (30%): дефіцит у ТОЧНОМУ породженні (gauss усі блоки − Kgen), тобто в кластерах, яких хабова форма не несе
🔮 P3 (25%): решта — вищі блоки (3,·,·),(4,·,·) або (2,2,·)
"""
import sys, os, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg"); import cupy as cp, pyarrow.parquet as pq
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
OUT = "/home/trim/wblogs"; NET = os.environ.get("NET", "regina-johnston"); L = int(os.environ.get("LAYER", "14")); NW = 1024
NORD = 4; NS = 1 << 20; BATCH = 1 << 13
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]; idx = NAMES.index(NET)
W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(16, NW, NW)
bank = np.load("%s/sub118_%s.npz" % (OUT, NET)); mu = bank["mu_%d" % L]; mp = mu @ W[L]; Sp = W[L].T @ bank["S_%d" % L] @ W[L]
s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s); c = hcoef(s, t, NORD)
Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(16)]
mug = cp.asarray(mu.astype(np.float32)); mpg = cp.asarray(mp.astype(np.float32)); sg = cp.asarray(s.astype(np.float32))
cg = [cp.asarray(np.asarray(ck, np.float32)) for ck in c]; Wn = Wg[L+1]
Lch = cp.asarray(np.linalg.cholesky(rho + 1e-10*np.eye(NW)).astype(np.float32))
def fwd(x, k0, k1):
    for k in range(k0, k1): x = cp.maximum(x @ Wg[k], 0.0)
    return x
def he_feats(x):
    """список ũ_n = (c_n ⊙ He_n(x)) W, n=1..NORD  (N×n fp32) — БЕЗ центрування; центруємо після (середні)"""
    Hs = [cp.ones_like(x), x]
    for k in range(1, NORD): Hs.append(x*Hs[-1] - k*Hs[-2])
    return [(cg[n]*Hs[n]) @ Wn for n in range(1, NORD+1)]
def run(kind, seed):
    """накопичує E[u_n u_m u_l], E[u_n u_m], E[u_n] і прямий cum3(hW) для істини"""
    E1 = [cp.zeros(NW, cp.float64) for _ in range(NORD)]; E2 = {}; E3 = {}
    D1 = cp.zeros(NW, cp.float64); D2 = cp.zeros((NW, NW), cp.float64); D3 = cp.zeros((NW, NW), cp.float64)
    done, b = 0, 0
    while done < NS:
        n = min(BATCH, NS - done)
        z = cp.random.RandomState(seed=seed + b).standard_normal((n, NW), dtype=cp.float32)
        if kind == "true":
            h = fwd(z, 0, L); x = ((h - mug) @ Wg[L])/sg
            hw = cp.maximum(x*sg + mpg, 0.0) @ Wn; D1 += hw.sum(0); D2 += (hw.T @ hw); D3 += ((hw*hw).T @ hw)
        else:
            x = z @ Lch.T
        U = he_feats(x)
        for i in range(NORD):
            E1[i] += U[i].sum(0)
            for j in range(i, NORD):
                E2[(i, j)] = E2.get((i, j), 0) + (U[i].T @ U[j])
                P = U[i]*U[j]
                for k in range(NORD): E3[(i, j, k)] = E3.get((i, j, k), 0) + (P.T @ U[k])
        done += n; b += 1
    E1 = [e/NS for e in E1]; E2 = {k: v/NS for k, v in E2.items()}; E3 = {k: v/NS for k, v in E3.items()}
    blocks = {}
    for (i, j, k), M in E3.items():
        # cum(u_i[a], u_j[a], u_k[b]) = E[xyz] − E[x]E[yz] − E[y]E[xz] − E[z]E[xy] + 2E[x]E[y]E[z]
        e2ij = E2[(i, j)]; e2ik = E2[(min(i, k), max(i, k))]; e2jk = E2[(min(j, k), max(j, k))]
        diag_ij = cp.diagonal(e2ij) if True else None
        # E[y_a z_b] = e2jk[a,b] (або транспонована якщо j>k)
        ejk = e2jk if j <= k else e2jk.T; eik = e2ik if i <= k else e2ik.T
        cum = M - E1[i][:, None]*ejk - E1[j][:, None]*eik - E1[k][None, :]*diag_ij[:, None] + 2*E1[i][:, None]*E1[j][:, None]*E1[k][None, :]
        blocks[(i+1, j+1, k+1)] = cp.asnumpy(cum)
    out = {"blocks": blocks}
    if kind == "true":
        m1 = D1/NS; m2 = D2/NS; m3 = D3/NS
        # cum3 центрованих: E[(x_a−μ_a)²(x_b−μ_b)] = E[x_a²x_b] − 2μ_a E[x_a x_b] − μ_b E[x_a²] + 2μ_a²μ_b
        ea2 = cp.diagonal(m2)
        out["Kdirect"] = cp.asnumpy(m3 - 2*m1[:, None]*m2 - m1[None, :]*ea2[:, None] + 2*(m1**2)[:, None]*m1[None, :])
    return out
def R2(p, q): return 1 - ((q-p)**2).sum()/(q**2).sum()
lg("старт %s L=%d" % (NET, L))
tr = run("true", 9300000); lg("істина готова"); ga = run("gauss", 9400000); lg("гаус готовий")
z = np.load("%s/k3full118_%s_L%d.npz" % (OUT, NET, L)); Klin, Kgen, Kt = z["Klin"], z["Kgen"], z["Kt"]
Dl = Kt - Klin - Kgen; nK = np.linalg.norm(Kt)
sum_true = sum(tr["blocks"].values()); sum_g = sum(ga["blocks"].values())
lg("🔒 S1: Σблоків(істина,≤4) проти банку P3f_%d: R² %.4f · проти прямого cum3: R² %.4f · прямий cum3 проти банку: R² %.4f" % (
    L+1, R2(sum_true, Kt), R2(sum_true, tr["Kdirect"]), R2(tr["Kdirect"], Kt)))
lg("🔒 S2: gauss-блок (2,1,1)+(1,2,1) проти Kgen (хаб amax=1, (2,1,1)-чейн): R² %.4f · масштаб %.3f" % (
    R2(ga["blocks"][(1, 2, 1)] + ga["blocks"][(1, 1, 2)]*0 + ga["blocks"][(2, 2, 1)]*0, Kgen) if False else -1, -1))
# таблиця блоків
print("\nблок (n,m,l)   ‖true‖/‖Kt‖  ‖gauss‖/‖Kt‖  ‖true−gauss‖/‖Kt‖   R²(true−gauss, Δ)   R²(gauss, Kgen-хаб)")
keys = sorted(tr["blocks"].keys())
NG = {}
for k in keys:
    bt, bg = tr["blocks"][k], ga["blocks"][k]; d = bt - bg; NG[k] = d
    print("  %s      %6.3f       %6.3f        %6.3f            %+.3f" % (k, np.linalg.norm(bt)/nK, np.linalg.norm(bg)/nK, np.linalg.norm(d)/nK, R2(d, Dl) if np.linalg.norm(d) > 1e-12*nK else 0))
lg("Σ gauss (усі блоки) = ТОЧНЕ породження: R²(·, Kgen) %.4f · масштаб(Kgen→Σg) %.3f · ‖Σg−Kgen‖/‖Kt‖ %.3f" % (
    R2(Kgen, sum_g), float((Kgen*sum_g).sum()/(Kgen*Kgen).sum()), np.linalg.norm(sum_g - Kgen)/nK))
lg("(1,1,1) true−gauss проти Klin (перенос κ₃): R² %.4f · масштаб %.3f" % (R2(Klin, NG[(1, 1, 1)]), float((Klin*NG[(1,1,1)]).sum()/(Klin*Klin).sum())))
# де дефіцит: Δ' = Kt − Klin − Σgauss  (точне породження замість хабового)
Dg_ = Kt - Klin - sum_g
lg("залишок при ТОЧНОМУ породженні: ‖Kt−Klin−Σg‖/‖Kt‖ %.3f (було з Kgen: %.3f)" % (np.linalg.norm(Dg_)/nK, np.linalg.norm(Dl)/nK))
rest = sum(v for k, v in NG.items() if k != (1, 1, 1))
lg("негаусові блоки крім (1,1,1) разом: ‖·‖/‖Kt‖ %.3f · R² проти Kt−Klin−Σg: %.4f" % (np.linalg.norm(rest)/nK, R2(rest, Dg_)))
for k in [(2, 1, 1), (1, 1, 2), (2, 2, 1), (3, 1, 1), (1, 1, 3), (2, 1, 2), (1, 2, 2)]:
    if k in NG: lg("   %s проти Kt−Klin−Σg: R² %+.3f · ‖·‖ %.3f" % (k, R2(NG[k], Dg_), np.linalg.norm(NG[k])/nK))
np.savez("%s/hermdec118_%s_L%d.npz" % (OUT, NET, L), **{"t_%d%d%d" % k: v for k, v in tr["blocks"].items()}, **{"g_%d%d%d" % k: v for k, v in ga["blocks"].items()}, Kdirect=tr["Kdirect"])
lg("ГОТОВО"); print("DONE_HERM")
