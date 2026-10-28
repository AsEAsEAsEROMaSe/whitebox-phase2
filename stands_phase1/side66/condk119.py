"""С119 — СТЕЛЯ ГІБРИДА «ТОЧНО В ТОП-k ПІДПРОСТОРІ + УМОВНО-ГАУСОВЕ ЗАМИКАННЯ ДЛЯ РЕШТИ» НА ОСТАННЬОМУ КРОЦІ.
Ідея ЮЗЕРА (05.09): цілитись у точність через точну обробку домінантного підпростору.
Оракул: істинні семпли p_16 (вихідна передактивація), y = U_kᵀ(p−m) (топ-k власних Σ_p), p_i | y ≈ N(m_i + B_i·y, s⊥_i²)
(лінійна регресія на y, спільна s⊥). Тоді E[ReLU(p_i)] ≈ E_y[ f(m_i + B_i y, s⊥_i) ] — усереднення по істинних y (2²⁰ семплів = точна
k-вимірна інтеграція). k=0 — звичайний гаусів крок при ТОЧНИХ (m,s) (🔒 має дати ≈0.046%, S116). Похибка проти істини банку (2²¹).
Це ВЕРХНЯ МЕЖА класу: якою була б похибка, якби розподіл y ми знали точно на кожному шарі і решта була умовно гаусовою.
🔮 P1 (35%): k=8 знімає ≥ ×3 (0.046% → ≤0.015%) ⟹ клас живий, далі — як нести розподіл y
🔮 P2 (40%): треба k ≥ 64 для ×3 ⟹ квадратура неможлива, клас мертвий
🔮 P3 (25%): між (k≈16–32) ⟹ живий лише з високопорядковим замиканням у k вимірах
"""
import sys, os, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg"); import cupy as cp, pyarrow.parquet as pq
from math import erf
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
OUT = "/home/trim/wblogs"; NW = 1024; D = 16; NS = 1 << 20; BATCH = 1 << 13
NETS = os.environ.get("NETS", "regina-johnston").split(","); KS = [int(x) for x in os.environ.get("KS", "0,1,2,4,8,16,32,64,128").split(",")]
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
def Phi(t): return 0.5*(1.0 + cp.asarray(np.vectorize(erf)(cp.asnumpy(t)/np.sqrt(2)))) if False else 0.5*(1.0 + cupy_erf(t/np.sqrt(2)))
import cupyx.scipy.special as css
def cupy_erf(x): return css.erf(x)
def f_relu(m, s):
    t = m/s; ph = cp.exp(-0.5*t*t)/np.sqrt(2*np.pi); Ph = 0.5*(1.0 + cupy_erf(t/np.sqrt(2)))
    return m*Ph + s*ph
RES = {k: [] for k in KS}
for nm in NETS:
    idx = NAMES.index(nm)
    W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
    Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
    T = cp.asarray(np.load("%s/covbank118_%s.npz" % (OUT, nm))["mu_16"])
    # прохід 1: моменти p_16 (fp64 накопичення)
    m1 = cp.zeros(NW, cp.float64); m2 = cp.zeros((NW, NW), cp.float64); done, b = 0, 0; seed = 777000
    def fwd(z):
        x = z
        for k in range(D-1): x = cp.maximum(x @ Wg[k], 0.0)
        return x @ Wg[D-1]
    while done < NS:
        n = min(BATCH, NS - done); z = cp.random.RandomState(seed=seed + b).standard_normal((n, NW), dtype=cp.float32)
        p = fwd(z).astype(cp.float64); m1 += p.sum(0); m2 += p.T @ p; done += n; b += 1
    m1 /= NS; S = m2/NS - cp.outer(m1, m1); s = cp.sqrt(cp.diagonal(S))
    KB = os.environ.get("KBASIS", "eig")
    if KB == "eig":
        ev, U = cp.linalg.eigh(S); U = U[:, ::-1]
    elif KB == "k3":
        Kb = cp.asarray(np.load("%s/sub118_%s.npz" % (OUT, nm))["P3f_15"]); _, U = cp.linalg.eigh(Kb @ Kb.T + Kb.T @ Kb); U = U[:, ::-1]
    elif KB == "rand":
        U, _ = cp.linalg.qr(cp.asarray(np.random.default_rng(5).standard_normal((NW, NW))))
    kmax = max(KS); Uk = U[:, :kmax]
    # регресія p на y = U_kᵀ(p−m) для довільного ортонормованого U_k: B_k = S U_k (U_kᵀ S U_k)⁻¹ ; s⊥² = s² − diag(B_k U_kᵀ S)
    SU = S @ Uk; G = Uk.T @ SU
    # прохід 2: для кожного k — накопичити E_y[f(m + B_k y, s⊥_k)] ; B_k = S U_k (U_kᵀ S U_k)⁻¹ = U_k (власний базис ⟹ B_k = U_k), s⊥² = s² − Σ_{j<k} ev_j U_ij²
    acc = {k: cp.zeros(NW, cp.float64) for k in KS}
    Bk = {}; sperp = {}
    for k in KS:
        if k == 0: Bk[k] = None; sperp[k] = s; continue
        Bk[k] = SU[:, :k] @ cp.linalg.inv(G[:k, :k]); sperp[k] = cp.sqrt(cp.maximum(s**2 - (Bk[k]*SU[:, :k]).sum(1), 1e-12))
    done, b = 0, 0; mcdir = cp.zeros(NW, cp.float64)
    while done < NS:
        n = min(BATCH, NS - done); z = cp.random.RandomState(seed=seed + b).standard_normal((n, NW), dtype=cp.float32)
        p = fwd(z).astype(cp.float64); y = (p - m1[None, :]) @ Uk                # (n×kmax)
        mcdir += cp.maximum(p, 0.0).sum(0)
        for k in KS:
            mc = m1[None, :] + (y[:, :k] @ Bk[k].T if k > 0 else 0.0)        # умовне середнє (n×NW)
            fk = f_relu(mc, sperp[k][None, :]); acc[k] += fk.sum(0)*(n if fk.shape[0] == 1 else 1)
        done += n; b += 1
    line = []
    for k in KS:
        est = acc[k]/NS; e = 100*float(cp.linalg.norm(est - T)/cp.linalg.norm(T)); RES[k].append(e); line.append("k=%d %.4f%%" % (k, e))
    # 🔒 пломба: пряма МК-оцінка E[ReLU(p)] тими ж семплами проти банку (підлога шуму 2²⁰ проти 2²¹)
    lg("%-18s %s · 🔒МК 2²⁰ проти банку %.4f%%" % (nm, " · ".join(line), 100*float(cp.linalg.norm(mcdir/NS - T)/cp.linalg.norm(T))))
print("СЕРЕДНЄ (RMS по мережах): " + " · ".join("k=%d %.4f%%" % (k, np.sqrt(np.mean(np.square(RES[k])))) for k in KS))
print("DONE_CONDK")
