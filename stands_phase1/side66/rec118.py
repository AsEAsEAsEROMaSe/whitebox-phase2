"""С118 — РЕКУРСІЯ ТРЕТЬОГО ПОРЯДКУ (mp, Σ_p, ядро κ₃ r³) — ПЕРША ЗБІРКА ВСІХ ЗМІРЯНИХ ЛАНОК

Ланки (усі зміряні С117–С118): крок m = Гаус + γ₃ (res117 R² 0.88) + γ₄c пошарова стала
(inj118c: ×14.05) · крок C = Мелер n≤NM + парний Еджворт κ_aab (cstep118 R² 0.996) + 1-D діагональ ·
перенос κ₃ через ReLU = T(Φ,Φ,Φ) (k3full118b R² 0.947, +породження 0.992) · породження — ланцюгові
доданки Мелера 3-го порядку (gen118 R² 0.88 на l=1) · представлення — ядро r³ у базисі V (n×r).
Стан на p_l: mp, Sp, V, C  (T_p ≈ Σ_abc C_abc V_a⊗V_b⊗V_c, сирі одиниці p).
Базис наступного шару: BASIS=qr → V' = orth(W_{l+1}ᵀ(Φ⊙V)) (перенос ТОЧНИЙ, породження проєктується);
BASIS=eig → топ-r власних Σ_p' (обидва — обрізання).
🔒 S1 (SEAL=1, n=r=64): κ_aab із ядра породження ≡ gen_k3 (щільна формула gen118) до 1e-10
🔒 S2: R=0 (без κ₃, γ₄c=0) ≡ base inj118 (Гаус NM=8): 0.1365% / 0.0990% на двох мережах
🔮 P1 (50%): r=128, qr, amax=2 дає ≥ ×4 у E² проти base (оракул both_f4c ×14)
🔮 P2 (60%): r=256 ≥ r=128 на ≥ ×1.3 · 🔮 P3 (55%): eig ≈ qr у межах 10%
"""
import sys, os, time, glob
import numpy as np
from math import erf, factorial
D = 16
NW = int(os.environ.get("NW", "1024"))
R = int(os.environ.get("R", "128"))
NM = int(os.environ.get("NM", "8"))
AMAX = int(os.environ.get("AMAX", "2"))
BASIS = os.environ.get("BASIS", "qr")
G4C = float(os.environ.get("G4C", "1.0"))      # множник на криву γ₄ (0 = вимкнути)
NOGEN = int(os.environ.get("NOGEN", "0")); NOTRANS = int(os.environ.get("NOTRANS", "0"))
SEAL = int(os.environ.get("SEAL", "0")); DIAG = int(os.environ.get("DIAG", "0")); HYB = int(os.environ.get("HYB", "1"))
NETS = os.environ.get("NETS", "regina-johnston,matthew-johnson").split(",")
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
G4CURVE = np.load("%s/g4_layer_curve.npy" % OUT)      # γ₄ передактивації p_l, l=0..15 (24 мережі)

def gen_k3_dense(mp, Sp, W, amax):
    """щільна перевірка (gen118): κ_aab(p') з гаусового p"""
    s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s)
    c = hcoef(s, t, 2*amax); rp = {1: rho}
    for a in range(2, amax+1): rp[a] = rp[a-1]*rho
    A = {a: rp[a] @ (c[a][:, None]*W) for a in range(1, amax+1)}
    K = np.zeros((W.shape[1], W.shape[1]))
    for a in range(1, amax+1):
        for b in range(1, amax+1):
            f = factorial(a+b); ch = c[a+b]
            K += 2*f*((ch[:, None]*W*A[a]).T @ A[b]) + f*((A[a]*A[b]).T @ (ch[:, None]*W))
    return K

def gen_core(mp, Sp, P, amax):
    """ядро породження у базисі V' (P = V'ᵀ W_{l+1}ᵀ, r×n): Σ_ab f·sym3(Σ_i ch_i P_i ⊗ (P Raᵀ)_i ⊗ (P Rbᵀ)_i)"""
    s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s)
    c = hcoef(s, t, 2*amax); rp = {1: rho}
    for a in range(2, amax+1): rp[a] = rp[a-1]*rho
    r = P.shape[0]
    Y = {a: P @ (rp[a]*c[a][None, :]).T for a in range(1, amax+1)}     # (r×n): (P Raᵀ), Ra = ρ^a diag(c_a)
    C = np.zeros((r, r, r))
    for a in range(1, amax+1):
        for b in range(1, amax+1):
            f = factorial(a+b); X = P*c[a+b][None, :]
            KR = (Y[a][:, None, :]*Y[b][None, :, :]).reshape(r*r, -1)        # (r², n)
            H = (X @ KR.T).reshape(r, r, r)                                   # хаб у першій моді
            C += f*(H + np.transpose(H, (1, 0, 2)) + np.transpose(H, (1, 2, 0)))     # хаб у 1-й / 2-й / 3-й моді
    return C

def core_contract(C, V):
    """κ_aab (n×n) і κ_aaa з ядра: M_{ic} = Σ_ab C_abc V_ia V_ib ; κ_aab = M Vᵀ"""
    r = V.shape[1]
    G2 = (V[:, :, None]*V[:, None, :]).reshape(V.shape[0], r*r)
    M = G2 @ C.reshape(r*r, r)
    return M @ V.T

LAST = {}
def step(mp, Sp, V, C, Wn, l, prev=None):
    """один шар: p_l → h_{l+1} → p_{l+1}; повертає новий стан і (m, Cov) для h_{l+1}.
    HYB=1: свіже породження κ₃(p_l) з гаусового p_{l−1} (prev = (mp_{l−1}, Sp_{l−1}, W_l)) рахується
    ЩІЛЬНО O(n³) для кроку, а в ядро додається лише для ПЕРЕНОСУ далі."""
    n = len(mp)
    s = np.sqrt(np.maximum(np.diag(Sp), 1e-300)); t = mp/s; rho = Sp/np.outer(s, s)
    ph = phi(t); Ph = Phi(t); c = hcoef(s, t, max(NM, 4))
    Kab = np.zeros((n, n))
    if C is not None: Kab = Kab + core_contract(C, V)
    if HYB and prev is not None: Kab = Kab + gen_k3_dense(prev[0], prev[1], prev[2], AMAX)
    LAST["Kab"] = Kab
    if C is not None or (HYB and prev is not None):
        k_aab = Kab/np.outer(s*s, s); g3 = np.diag(k_aab)
    else:
        k_aab = np.zeros((n, n)); g3 = np.zeros(n)
    g4 = G4C*G4CURVE[l]
    # крок m
    m = mp*Ph + s*ph - (g3/6)*s*t*ph + (g4/24)*s*(t*t-1)*ph
    # крок C
    Cov = np.zeros((n, n)); f = 1.0; rp = np.ones_like(rho)
    for k in range(1, NM+1):
        f *= k; rp = rp*rho; Cov += f*np.outer(c[k], c[k])*rp
    Cov += np.outer(c[2], c[1])*k_aab + np.outer(c[1], c[2])*k_aab.T \
         + rho*(6*np.outer(c[3], c[2])*k_aab + 6*np.outer(c[2], c[3])*k_aab.T
                + 4*np.outer(c[4]*g3, c[1]) + 4*np.outer(c[1], c[4]*g3))
    e2 = (mp*mp + s*s)*Ph + mp*s*ph + s*s*ph*(g3/3 - g4*t/12)
    np.fill_diagonal(Cov, e2 - m*m)
    # наступна передактивація
    mp2 = m @ Wn; Sp2 = Wn.T @ Cov @ Wn; Sp2 = 0.5*(Sp2 + Sp2.T)
    if R == 0: return mp2, Sp2, None, None, m, Cov
    Vh = Ph[:, None]*V if V is not None else None                         # перенос: T_h = T_p(Φ,Φ,Φ)
    Vt = Wn.T @ Vh if Vh is not None else None                            # (n×r) образ базису
    if BASIS == "eig" or Vt is None:
        ev, U = np.linalg.eigh(Sp2); V2 = U[:, ::-1][:, :R]
    else:
        V2, _ = np.linalg.qr(Vt)
    C2 = np.zeros((R, R, R))
    Cf = C
    if HYB and prev is not None and not NOGEN and V is not None:
        Cf = (C if C is not None else 0) + gen_core(prev[0], prev[1], V.T @ prev[2].T, AMAX)   # свіже → у пам'ять
    if Cf is not None and not NOTRANS and Vt is not None:
        Mx = V2.T @ Vt                                                    # (r×r)
        C2 = np.einsum("abc,da->dbc", Cf, Mx); C2 = np.einsum("dbc,eb->dec", C2, Mx); C2 = np.einsum("dec,fc->def", C2, Mx)   # Mx[d,a] = Σ_i V2_id Vt_ia
    if not HYB and not NOGEN:
        C2 = C2 + gen_core(mp, Sp, V2.T @ Wn.T, AMAX)
    return mp2, Sp2, V2, C2, m, Cov

def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = x*a - T
    return 100*np.sqrt(float(r @ r)/float(T @ T))

import pyarrow.parquet as pq
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]

if SEAL:
    # 🔒 S1 на міні-мережі n=r: ядро породження ≡ щільна формула; перенос ядра ≡ пряма контракція
    n = NW; idx = NAMES.index(NETS[0])
    Wf = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, 1024, 1024)[:, :n, :n]*np.sqrt(1024/n)
    mp = np.zeros(n); Sp = Wf[0].T @ Wf[0]
    P = np.eye(n)                                                          # V' = I, r = n
    Cg = gen_core(mp, Sp, P @ Wf[1].T, AMAX)
    Kc = core_contract(Cg, np.eye(n)); Kd = gen_k3_dense(mp, Sp, Wf[1], AMAX)
    lg("🔒 S1 породження: max|Δ| %.2e при max|K| %.2e" % (np.abs(Kc - Kd).max(), np.abs(Kd).max()))
    # перенос: випадкове ядро в базисі V, Vt = WᵀΦV; контракція через M vs через щільний тензор
    rs = np.random.RandomState(0); r = 8; V = np.linalg.qr(rs.randn(n, r))[0]; C = rs.randn(r, r, r)
    Ph = rs.rand(n); Vt = Wf[1].T @ (Ph[:, None]*V); V2, _ = np.linalg.qr(Vt); Mx = V2.T @ Vt
    C2 = np.einsum("abc,da,eb,fc->def", C, Mx, Mx, Mx)
    Td = np.einsum("abc,ia,jb,kc->ijk", C, Vt, Vt, Vt)                     # щільний перенесений тензор
    Kd2 = np.einsum("iik->ik", Td); Kc2 = core_contract(C2, V2)
    lg("🔒 S1 перенос: max|Δ| %.2e при max|K| %.2e" % (np.abs(Kc2 - Kd2).max(), np.abs(Kd2).max()))
    sys.exit(0)

lg("R=%d NM=%d AMAX=%d BASIS=%s G4C=%.2f NOGEN=%d NOTRANS=%d HYB=%d" % (R, NM, AMAX, BASIS, G4C, NOGEN, NOTRANS, HYB))
RES = []
for nm in NETS:
    idx = NAMES.index(nm)
    W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
    cb = np.load("%s/covbank118_%s.npz" % (OUT, nm)); T = cb["mu_16"]
    sb = "%s/sub118_%s.npz" % (OUT, nm); sbk = np.load(sb) if os.path.exists(sb) else None
    mp = np.zeros(NW); Sp = W[0].T @ W[0]; V = None; C = None
    if R > 0:
        ev, U = np.linalg.eigh(Sp); V = U[:, ::-1][:, :R]; C = np.zeros((R, R, R))
    diag = []; k3diag = []; prev = None
    for l in range(D):
        Wn = W[l+1] if l+1 < D else np.eye(NW)
        mp0, Sp0 = mp, Sp
        mp, Sp, V, C, m, Cov = step(mp, Sp, V, C, Wn, l, prev)
        if DIAG and sbk is not None and 1 <= l <= 15:
            Kt = sbk["P3f_%d" % l]; Kr = LAST["Kab"]; off = ~np.eye(NW, dtype=bool)
            r2o = 1 - ((Kt - Kr)[off]**2).sum()/(Kt[off]**2).sum(); r2d = 1 - ((np.diag(Kt) - np.diag(Kr))**2).sum()/(np.diag(Kt)**2).sum()
            sc = float((Kr*Kt).sum()/(Kr*Kr).sum())
            k3diag.append("l%d R²off %.2f R²γ₃ %.2f масшт %.2f" % (l, r2o, r2d, sc))
        prev = (mp0, Sp0, Wn) if l+1 < D else None
        if sbk is not None and l+1 < D and (l+1) in (4, 8, 12, 15):
            mt = sbk["mu_%d" % (l+1)]; St = sbk["S_%d" % (l+1)]
            mpt = mt @ W[l+1]; spt = np.sqrt(np.diag(W[l+1].T @ St @ W[l+1])); sp_ = np.sqrt(np.diag(Sp))
            diag.append("p%d: mp %.3f%% sd %.3f%%" % (l+1, cerr(mp, mpt), 100*np.sqrt(np.mean((sp_/spt - 1)**2))))
    e = cerr(m, T); RES.append(e)
    lg("%-20s похибка %.4f%% · %s" % (nm, e, " · ".join(diag)))
    if k3diag: lg("   κ₃ рекурсії проти банку: " + " · ".join(k3diag))
print("СЕРЕДНЄ %.4f%%  (E² середнє %.4e)" % (np.sqrt(np.mean(np.square(RES))), np.mean(np.square(RES))))
lg("ГОТОВО"); print("DONE_REC")
