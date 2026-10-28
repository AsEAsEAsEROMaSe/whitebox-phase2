"""С118 — РЕКУРСІЯ ТРЕТЬОГО ПОРЯДКУ У ХАБОВІЙ ФОРМІ: ПЕРЕНОС κ₃ ТОЧНИЙ, БЕЗ ПІДПРОСТОРУ

rec118 (ядро r=128): κ₃ рекурсії проти банку R² 0.65–0.80, масштаб 1.4–1.5 ⟹ підпростір губить
перенесену пам'ять на мілких шарах (спектр Σ_p плоский). Тут пам'ять несеться ТОЧНО:
джерело, народжене на шарі k (породження з гаусового p_k, ланцюгові доданки (a,b)):
   H^{ab}(i;j,k) = c_{a+b,i} · Ra_ij · Rb_ik,  Ra = ρ_k^{∘a} diag(c_a)      (хаб i, руки j,k)
   κ₃(h_{k+1}) = Σ_ab (a+b)! · sym3(H^{ab})
Перенос h→p: усі три моди на W; p→h: множення на Φ(t). Для хабової форми це матмули на
хаб-матриці (diag(c_{a+b})·X) і руки (Ra X, Rb X), X — накопичений добуток W·Φ·W·…
Контракція κ_aab(p_l) = Σ_ab (a+b)! [2 (Hm⊙Am)ᵀ Bm + (Am⊙Bm)ᵀ Hm].
Ціна: ~13 матмулів n³ на джерело на шар ⟹ ~3e12 флопів на мережу (стенд, не сабміт).
🔒 S1: κ₃ на l=1 ≡ gen_k3_dense (R² 0.87 проти банку, як rec118 HYB) · 🔒 S2: R=0 ≡ base
Виводить: похибка виходу, κ₃ проти банку по шарах, і ВНЕСОК кожного джерела у ‖κ_aab(p_15)‖
(чи гаснуть старі джерела — від цього залежить ціна у сабміті).
🔮 P1 (55%): κ₃ R² ≥ 0.85 на всіх шарах (точний перенос знімає масштаб 1.4)
🔮 P2 (50%): похибка виходу ≤ 0.07% на regina (×4 проти base 0.1365)
"""
import sys, os, time
import numpy as np
from math import erf, factorial
D = 16; NW = int(os.environ.get("NW", "1024"))
NM = int(os.environ.get("NM", "8")); AMAX = int(os.environ.get("AMAX", "2"))
G4C = float(os.environ.get("G4C", "1.0")); DIAG = int(os.environ.get("DIAG", "1"))
MAXAGE = int(os.environ.get("MAXAGE", "99"))          # скільки шарів жити джерелу (для оцінки ціни)
KSCALE = float(os.environ.get("KSCALE", "1.0"))
DAMP = float(os.environ.get("DAMP", "1.0"))       # EDQNM-гасіння: κ₃ джерела множиться на DAMP за кожен шар переносу
PSD = int(os.environ.get("PSD", "0"))
KTRUE = float(os.environ.get("KTRUE", "0"))
KMODE = os.environ.get("KMODE", "all")
DUMP = int(os.environ.get("DUMP", "0"))
KRANK = int(os.environ.get("KRANK", "0"))
GENS = float(os.environ.get("GENS", "1.0"))        # масштаб ПОРОДЖЕННЯ (новонароджене джерело), сталий
GENS4 = float(os.environ.get("GENS4", "0.0"))
KSD = float(os.environ.get("KSD", "1.0"))          # масштаб ДІАГОНАЛІ Kab на шарах l>=KSL
KSO = float(os.environ.get("KSO", "1.0"))          # масштаб ПОЗАДІАГОНАЛІ Kab на шарах l>=KSL
KSL = int(os.environ.get("KSL", "8"))
KDIR = os.environ.get("KDIR", "")                   # лінійний відгук: Kab += KEPS·X_l, X = delta | ktop8 | ktop64 | krest64 | self | rand (норма = ‖Δ‖), для l>=KSL
KEPS = float(os.environ.get("KEPS", "1.0"))
STATE = int(os.environ.get("STATE", "0"))
FITW = int(os.environ.get("FITW", "0"))
KP = int(os.environ.get("KP", "0"))        # офіційні mini-100 + моменти keenanpepper N=1e8 (/home/trim/wb2data/kp_p2/mlp_%05d.npz); NETS = індекси
G4T = float(os.environ.get("G4T", "0"))    # KP: понейронна істинна γ₄ у кроках m/e2 (замість G4CURVE), вага
G3T = float(os.environ.get("G3T", "0"))    # KP: понейронна істинна γ₃ (діагональ) у кроках m/e2/Cov, вага (змішування з нашою)
K4P = float(os.environ.get("K4P", "0"))
K4B = os.environ.get("K4B", "")        # KP: банк-МОДЕЛЬ для k22/k31/g4 (файл kp119d) замість істинних; шаблон із %d    # KP: парні блоки κ₂₂, κ₃₁ істинні у кроці Cov (1-й порядок), вага             # на l=15: МНК-підбір ваг хабів живих джерел до дефіциту Δ = банк − Kab (LSQR через оператор і спряжений)           # друкувати похибку стану (mp, s) на l=15 проти банку (mu_15, S_15)      # масштаб породження 1 + GENS4·γ₄(l)/γ₄(15) (κ₄ робить блок He2He1He1 більшим за гаусів)           # оракул: Kab += топ-KRANK SVD-частина (банк − Kab)             # зберегти Kab по шарах у /home/trim/wblogs/rec118h_kab_<net>.npz             # all | diag | off | shallow(l<=7) | deep(l>=8)       # оракул: Kab ← (1−KTRUE)·наша + KTRUE·банк P3f_l (внутрішній оракул цього ж стенду)
TRI = int(os.environ.get("TRI", "0"))               # трикутний доданок породження (1,1,1) точно на GPU: у Knext і в ядро
if TRI:
    sys.path.insert(0, "/home/trim/gpupkg"); import cupy as cp
def tri_kab(c2, rho, W):
    """Ktri[a,b] = 8 Σ_k c2_k W_kb diag(B_kᵀ ρ B_k)_a, B_k = (c2⊙ρ[:,k])[:,None]⊙W  (O(n⁴), GPU fp32)"""
    n = W.shape[0]; rg = cp.asarray(rho, cp.float32); Wg = cp.asarray(W, cp.float32); cg = cp.asarray(c2, cp.float32)
    K = cp.zeros((n, n), cp.float64)
    for k in range(n):
        Bk = (cg*rg[:, k])[:, None]*Wg
        d = ((rg @ Bk)*Bk).sum(0)
        K += cp.outer(d, cg[k]*Wg[k, :]).astype(cp.float64)
    return 8.0*cp.asnumpy(K)
def tri_core(c2, rho, W, V):
    """C[αβγ] = 8 Σ_i U_iα (Uᵀ (ρ⊙ρ_i⊗ρ_i) U)_βγ, U = c2[:,None]⊙(W V)   (O(n³ r), GPU fp32)"""
    rg = cp.asarray(rho, cp.float32); U = cp.asarray(c2, cp.float32)[:, None]*(cp.asarray(W, cp.float32) @ cp.asarray(V, cp.float32))
    r = U.shape[1]; C = cp.zeros((r, r, r), cp.float64)
    for i in range(rg.shape[0]):
        M = rg*rg[:, i][:, None]*rg[:, i][None, :]
        C += U[i][:, None, None]*(U.T @ M @ U)[None, :, :].astype(cp.float64)
    return 8.0*cp.asnumpy(C)               # друкувати min eig(Cov) після Еджворта     # множник на κ₃ (компенсація обрізаних трикутних доданків)
FOLD = int(os.environ.get("FOLD", "99"))    # джерела віку ≥ FOLD згортаються у ядро r³ (базис = простір рядків κ_aab)
R = int(os.environ.get("R", "128")); SEAL = int(os.environ.get("SEAL", "0"))
FOLDL = int(os.environ.get("FOLDL", "0"))   # згортати лише з шару ≥ FOLDL (мілкі — хабово точно)
PRH = float(os.environ.get("PRH", "0"))     # обрізання ХАБІВ: лишати i з φ(t_i) > PRH·max φ
PRC = float(os.environ.get("PRC", "0"))     # обрізання СТОВПЦІВ при переносі: лишати j з Φ(t_j) > PRC
FL = {"n": 0.0}
def mm(A, B):
    FL["n"] += A.shape[0]*B.shape[1]*(2*A.shape[1]-1); return A @ B
NETS = os.environ.get("NETS", "regina-johnston").split(",")
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
G4CURVE = np.load("%s/g4_layer_curve.npy" % OUT)

class Source:
    """κ₃(h_{k+1}) з гаусового p_k у хабовій формі; матриці живуть у поточному просторі (після X)"""
    def __init__(self, mp, Sp, amax):
        s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s)
        c = hcoef(s, t, 2*amax); rp = {1: rho}
        for a in range(2, amax+1): rp[a] = rp[a-1]*rho
        self.amax = amax
        ph = phi(t); I = np.where(ph > PRH*ph.max())[0] if PRH > 0 else np.arange(len(t))   # хаби
        self.arm = {a: (rp[a]*c[a][None, :])[I] for a in range(1, amax+1)}     # Ra = ρ^a diag(c_a)  (|I|×n)
        self.hub = {q: np.eye(len(t))[I]*c[q][None, :] for q in range(2, 2*amax+1)}   # diag(c_{a+b})[I]
        self.age = 0; self.act = None
    def push(self, X):
        if self.act is not None: X = X[self.act]
        for a in self.arm: self.arm[a] = mm(self.arm[a], X)
        for q in self.hub: self.hub[q] = mm(self.hub[q], X)*DAMP
        self.age += 1; self.act = None
    def scale_cols(self, v):
        act = np.where(v > PRC)[0] if PRC > 0 else None
        for a in self.arm: self.arm[a] = (self.arm[a]*v[None, :])[:, act] if act is not None else self.arm[a]*v[None, :]
        for q in self.hub: self.hub[q] = (self.hub[q]*v[None, :])[:, act] if act is not None else self.hub[q]*v[None, :]
        self.act = act
    def kab(self):
        K = 0
        for a in range(1, self.amax+1):
            for b in range(1, self.amax+1):
                f = factorial(a+b); Hm = self.hub[a+b]; Am = self.arm[a]; Bm = self.arm[b]
                K = K + f*(2*mm((Hm*Am).T, Bm) + mm((Am*Bm).T, Hm))
        return K

    def fold(self, V):
        """ядро (r×r×r) цього джерела у базисі V (n×r): Σ_ab f·sym3(Σ_i (HmV)_i ⊗ (AmV)_i ⊗ (BmV)_i)"""
        r = V.shape[1]; C = np.zeros((r, r, r))
        Y = {a: mm(self.arm[a], V).T for a in self.arm}                       # (r×n)
        for a in range(1, self.amax+1):
            for b in range(1, self.amax+1):
                f = factorial(a+b); X = mm(self.hub[a+b], V).T
                KR = (Y[a][:, None, :]*Y[b][None, :, :]).reshape(r*r, -1)
                H = mm(X, KR.T).reshape(r, r, r)
                C += f*(H + np.transpose(H, (1, 0, 2)) + np.transpose(H, (1, 2, 0)))
        return C

class SourceF21:
    """негаусів зворотний зв'язок κ₂₁(p_k) у породження κ₃(h_{k+1}): хаб H=diag(w)[I], руки A≠B (6 розміщень).
    f21a: E[He₂He₂He₁]∋2κ₂₁_ij ρ_jk — хаб j, A=c₂∘κ₂₁[:,j], B=c₁∘ρ_j, w=2c₂·F21A · f21b: E[He₃He₁He₁]∋3κ₂₁_ij ρ_ik — хаб i, A=c₁∘κ₂₁[i,:], B=c₁∘ρ_i, w=3c₃·F21B"""
    def __init__(self, mp, Sp, k_aab, kind):
        s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s); c = hcoef(s, t, 4)
        ph = phi(t)
        if kind == "a":   w = 2*c[2]*F21A
        elif kind == "c": w = 4*c[4]*F31B          # f31b: E[He₄(x_i)He₁He₁] ∋ 4κ₃₁_ij ρ_ik — хаб i, A=c₁∘κ₃₁[i,:], B=c₁∘ρ_i (k_aab тут = κ₃₁ нормована)
        else:             w = 3*c[3]*F21B
        I = np.where(np.abs(w) > F21PRH*np.abs(w).max())[0] if F21PRH > 0 else (np.where(ph > PRH*ph.max())[0] if PRH > 0 else np.arange(len(t)))
        self.B = (rho*c[1][None, :])[I]
        if kind == "a": self.A = (k_aab.T*c[2][None, :])[I]
        else:           self.A = (k_aab*c[1][None, :])[I]     # "b" і "c": рядок i = c₁(j)·κ[i,j]
        self.H = np.eye(len(t))[I]*w[None, :]
        self.age = 0; self.act = None; self.amax = 0
    def push(self, X):
        if self.act is not None: X = X[self.act]
        self.H = mm(self.H, X)*DAMP; self.A = mm(self.A, X); self.B = mm(self.B, X); self.age += 1; self.act = None
    def scale_cols(self, v):
        act = np.where(v > PRC)[0] if PRC > 0 else None
        for k in ("H", "A", "B"):
            M = getattr(self, k)*v[None, :]; setattr(self, k, M[:, act] if act is not None else M)
        self.act = act
    def kab(self):
        H, A, B = self.H, self.A, self.B
        return 2*(mm((H*A).T, B) + mm((H*B).T, A) + mm((A*B).T, H))
    def fold(self, V):
        r = V.shape[1]; X = mm(self.H, V).T; Ya = mm(self.A, V).T; Yb = mm(self.B, V).T
        KR = (Ya[:, None, :]*Yb[None, :, :]).reshape(r*r, -1); T3 = mm(X, KR.T).reshape(r, r, r)   # H⊗A⊗B
        C = np.zeros((r, r, r))
        for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)): C += np.transpose(T3, perm)
        return C
F21 = int(os.environ.get("F21", "0")); F21A = float(os.environ.get("F21A", "1.0")); F21B = float(os.environ.get("F21B", "1.0"))
F21L = int(os.environ.get("F21L", "1")); F21PRH = float(os.environ.get("F21PRH", "0")); F21AGE = int(os.environ.get("F21AGE", "99")); FOLDG = int(os.environ.get("FOLDG", "0")); FOLDB = int(os.environ.get("FOLDB", "0")); F31 = int(os.environ.get("F31", "0")); F31B = float(os.environ.get("F31B", "1.0"))   # 1: згортати лише Source, F21 лишати   # обрізання хабів F21 за |w| (0 ⟹ як PRH за φ)    # з якого шару породження вмикати зв'язок (κ₂₁(p_0)=0 і так)

def core_contract(C, V):
    r = V.shape[1]
    G2 = (V[:, :, None]*V[:, None, :]).reshape(V.shape[0], r*r)
    return mm(mm(G2, C.reshape(r*r, r)), V.T)

def core_transport(C, Mx):
    FL["n"] += 3*2*Mx.shape[0]**4
    C2 = np.einsum("abc,da->dbc", C, Mx); C2 = np.einsum("dbc,eb->dec", C2, Mx)
    return np.einsum("dec,fc->def", C2, Mx)

def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = x*a - T
    return 100*np.sqrt(float(r @ r)/float(T @ T))

import pyarrow.parquet as pq
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
if SEAL:
    n = NW; idx = NAMES.index(NETS[0])
    Wf = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, 1024, 1024)[:, :n, :n]*np.sqrt(1024/n)
    mp = np.zeros(n); Sp = Wf[0].T @ Wf[0]; src = Source(mp, Sp, AMAX); src.push(Wf[1])
    Kd = src.kab(); V = np.eye(n); Kc = core_contract(src.fold(V), V)
    lg("🔒 S1 згортка (r=n): max|Δ| %.2e при max|K| %.2e" % (np.abs(Kc - Kd).max(), np.abs(Kd).max()))
    rs = np.random.RandomState(1); Ph = rs.rand(n); C1 = src.fold(V); src.scale_cols(Ph); src.push(Wf[2]); Kd2 = src.kab()
    Vt = Wf[2].T @ (Ph[:, None]*V); V2, _ = np.linalg.qr(Vt); Kc2 = core_contract(core_transport(C1, V2.T @ Vt), V2)
    lg("🔒 S2 перенос ядра ≡ перенос джерела: max|Δ| %.2e при max|K| %.2e" % (np.abs(Kc2 - Kd2).max(), np.abs(Kd2).max()))
    sys.exit(0)
lg("NM=%d AMAX=%d G4C=%.2f MAXAGE=%d KSCALE=%.2f FOLD=%d R=%d FOLDL=%d PRH=%.3f PRC=%.3f" % (NM, AMAX, G4C, MAXAGE, KSCALE, FOLD, R, FOLDL, PRH, PRC))
RES = []
def _kp_load(i):
    """офіційна мережа i (індекс у mini-100) + моменти keenanpepper → W, T, банк P3f (центральні κ21), κ22/κ31 (нормовані), γ₃/γ₄"""
    import glob
    kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i); nm = str(kp["mlp_name"])
    W = None
    for f in sorted(glob.glob("/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a/data/mini-*.parquet")):
        t = pq.read_table(f, columns=["mlp_name"]); nms = [x.as_py() for x in t["mlp_name"]]
        if nm in nms:
            W = np.asarray(pq.read_table(f, columns=["weights"])["weights"][nms.index(nm)].as_py(), np.float64).reshape(D, NW, NW); break
    assert W is not None, nm
    mu = kp["pre_mean"]; m2 = kp["pre_m2"]; m3 = kp["pre_m3"]; m4 = kp["pre_m4"]
    sd = np.sqrt(m2 - mu*mu)
    c3 = m3 - 3*mu*m2 + 2*mu**3; c4 = m4 - 4*mu*m3 + 6*mu*mu*m2 - 3*mu**4
    g3 = c3/sd**3; g4 = c4/sd**4 - 3.0
    bank = {}
    for l in range(1, D):
        a = mu[l]; M11 = kp["pre_M11"][l].astype(np.float64); M21 = kp["pre_M21"][l].astype(np.float64)
        M31 = kp["pre_M31"][l].astype(np.float64); M22 = kp["pre_M22"][l].astype(np.float64); E2 = np.diag(M11)
        k21 = M21 - a[None, :]*E2[:, None] - 2*a[:, None]*M11 + 2*(a*a)[:, None]*a[None, :]          # E[z̃_i² z̃_j]
        C11 = M11 - np.outer(a, a)
        # E[(zi−a)³(zj−b)] = M31 − b·m3_i − 3a·M21 + 3ab·E2_i + 3a²·M11 − 3a³b
        mu31 = M31 - a[None, :]*m3[l][:, None] - 3*a[:, None]*M21 + 3*(a[:, None]*a[None, :])*E2[:, None] \
               + 3*(a*a)[:, None]*M11 - 3*(a**3)[:, None]*a[None, :]
        M12 = M21.T
        mu22 = M22 - 2*a[None, :]*M21 - 2*a[:, None]*M12 + (a*a)[None, :]*E2[:, None] + (a*a)[:, None]*E2[None, :] \
               + 4*(a[:, None]*a[None, :])*M11 - 3*(a*a)[:, None]*(a*a)[None, :]
        k22 = mu22 - np.outer(E2 - a*a, E2 - a*a) - 2*C11*C11
        k31 = mu31 - 3*(E2 - a*a)[:, None]*C11
        bank["P3f_%d" % l] = k21
        bank["k22_%d" % l] = k22/np.outer(sd[l]**2, sd[l]**2); bank["k31_%d" % l] = k31/np.outer(sd[l]**3, sd[l])
        if l == 1 or l == 15:
            lg("   🔒KP l=%d: діаг κ21 vs c3 %.2e · діаг κ22 vs c4 %.2e · діаг κ31 vs c4 %.2e · діаг k22 vs γ₄ %.2e" % (l,
               np.abs(np.diag(k21) - c3[l]).max()/np.abs(c3[l]).max(), np.abs(np.diag(k22) - c4[l]).max()/np.abs(c4[l]).max(),
               np.abs(np.diag(k31) - c4[l]).max()/np.abs(c4[l]).max(), np.abs(np.diag(bank["k22_%d" % l]) - g4[l]).max()))
    bank["mu"] = mu; bank["sd"] = sd; bank["g3"] = g3; bank["g4"] = g4
    return nm, W, kp["gt_mean"][15].astype(np.float64), bank

for nm in NETS:
    if KP:
        _kpi = int(nm); nm, W, T, kpb = _kp_load(_kpi)
    else:
        idx = NAMES.index(nm)
        W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
        T = np.load("%s/covbank118_%s.npz" % (OUT, nm))["mu_16"]
    KBANK = os.environ.get("KBANK", "")   # банк κ_aab для KTRUE (шаблон з %s); за замовчуванням істина sub118
    sb = (KBANK % nm) if (KBANK and not KP) else "%s/sub118_%s.npz" % (OUT, nm); sbk = np.load(sb) if os.path.exists(sb) else None
    _sbt = "%s/sub118_%s.npz" % (OUT, nm); sbt = np.load(_sbt) if os.path.exists(_sbt) else sbk   # істина для STATE
    if KP: sbk = kpb; sbt = kpb
    if KP and K4B:
        _kb = np.load(K4B % _kpi if "%d" in K4B else K4B)
        for l in range(1, D):
            kpb["k22_%d" % l] = _kb["k22_%d" % l]; kpb["k31_%d" % l] = _kb["k31_%d" % l]; kpb["g4"][l] = _kb["g4_%d" % l]
    if KP and KBANK: sbk = dict(np.load(KBANK % _kpi))
    mp = np.zeros(NW); Sp = W[0].T @ W[0]; sources = []; Cc = None; Vc = None; Knext = None
    k3diag = []; contrib = None
    _dump = {}
    for l in range(D):
        n = NW
        s = np.sqrt(np.maximum(np.diag(Sp), 1e-300)); t = mp/s; rho = Sp/np.outer(s, s)
        ph = phi(t); Ph = Phi(t); c = hcoef(s, t, max(NM, 4))
        # κ₃(p_l): сума джерел (усі вже перенесені у простір p_l) + ядро (згорнуті старі)
        Kab = np.zeros((n, n)); parts = []
        if Knext is not None:
            Kab = Knext
        else:
            for src in sources:
                Kk = src.kab(); Kab += Kk; parts.append(np.linalg.norm(Kk))
            if Cc is not None: Kab = Kab + core_contract(Cc, Vc)
        if l == 15 and not parts: parts = [np.linalg.norm(src.kab()) for src in sources]
        if KTRUE > 0 and sbk is not None and 1 <= l <= 15:
            Kt_ = sbk["P3f_%d" % l]; Kmix = (1-KTRUE)*Kab + KTRUE*Kt_
            if KMODE == "all": Kab = Kmix
            elif KMODE == "diag": Kab = Kab - np.diag(np.diag(Kab)) + np.diag(np.diag(Kmix))
            elif KMODE == "off": Kab = Kmix - np.diag(np.diag(Kmix)) + np.diag(np.diag(Kab))
            elif KMODE == "shallow" and l <= 7: Kab = Kmix
            elif KMODE == "deep" and l >= 8: Kab = Kmix
        if KDIR and sbk is not None and KSL <= l <= 15:
            _D = sbk["P3f_%d" % l] - Kab; _nD = np.linalg.norm(_D)
            if KDIR == "delta": _X = _D
            elif KDIR in ("ktop8", "ktop64", "krest64"):
                _r = 8 if KDIR == "ktop8" else 64; _U, _S, _Vt = np.linalg.svd(sbk["P3f_%d" % l], full_matrices=False)
                _P = _U[:, :_r] @ (_U[:, :_r].T @ _D @ _Vt[:_r].T) @ _Vt[:_r]; _X = _P if KDIR != "krest64" else _D - _P
            elif KDIR == "self": _X = Kab*(_nD/np.linalg.norm(Kab))
            elif KDIR == "rand": _R = np.random.default_rng(l).standard_normal(Kab.shape); _X = _R*(_nD/np.linalg.norm(_R))
            Kab = Kab + KEPS*_X
        if KRANK > 0 and sbk is not None and 1 <= l <= 15:
            _D = sbk["P3f_%d" % l] - Kab; _U, _S, _Vt = np.linalg.svd(_D, full_matrices=False)
            Kab = Kab + (_U[:, :KRANK]*_S[:KRANK]) @ _Vt[:KRANK]
        if l == 15: contrib = parts
        if DUMP and 1 <= l <= 15: _dump["K_%d" % l] = Kab.copy()
        if DIAG and sbk is not None and 1 <= l <= 15:
            Kt = sbk["P3f_%d" % l]; off = ~np.eye(n, dtype=bool)
            r2o = 1 - ((Kt - Kab)[off]**2).sum()/(Kt[off]**2).sum(); r2d = 1 - ((np.diag(Kt) - np.diag(Kab))**2).sum()/(np.diag(Kt)**2).sum()
            k3diag.append("l%d R²off %.2f R²γ₃ %.2f масшт %.2f" % (l, r2o, r2d, float((Kab*Kt).sum()/(Kab*Kab).sum())))
        if (KSD != 1.0 or KSO != 1.0) and l >= KSL:
            _d = np.diag(np.diag(Kab)); Kab = KSO*(Kab - _d) + KSD*_d
        if FITW and l == 15 and sbk is not None:
            _Dl = sbk["P3f_15"] - Kab; _nD = np.linalg.norm(_Dl)
            _srcs = [sr for sr in sources if sr.hub is not None]
            def _apply(wv):
                K_ = np.zeros((n, n)); o = 0
                for sr in _srcs:
                    Hm, Am = sr.hub[2], sr.arm[1]; wi = wv[o:o+Hm.shape[0]][:, None]; o += Hm.shape[0]
                    K_ += 2*((Hm*Am*wi).T @ Am) + ((Am*Am*wi).T @ Hm)
                return K_
            def _adj(Mx):
                out = []
                for sr in _srcs:
                    Hm, Am = sr.hub[2], sr.arm[1]
                    out.append(2*np.einsum("ia,ia->i", Hm*Am, Am @ Mx.T) + np.einsum("ia,ia->i", Am*Am, Hm @ Mx.T))
                return np.concatenate(out)
            from scipy.sparse.linalg import LinearOperator, lsqr
            _nw = sum(sr.hub[2].shape[0] for sr in _srcs)
            _op = LinearOperator((n*n, _nw), matvec=lambda v: _apply(v).ravel(), rmatvec=lambda r: _adj(r.reshape(n, n)))
            _w = lsqr(_op, _Dl.ravel(), iter_lim=60, atol=1e-10, btol=1e-10)[0]
            _Kf = _apply(_w); _off = ~np.eye(n, dtype=bool)
            _U, _S, _Vt = np.linalg.svd(sbk["P3f_15"], full_matrices=False); _P = lambda M: M - _U[:, :64] @ (_U[:, :64].T @ M @ _Vt[:64].T) @ _Vt[:64]
            lg("   FITW l=15: %d ваг · R²(K(w),Δ) %.3f · R² поза топ-64 %.3f · R² діаг %.3f · ‖w‖/‖c2-еквів‖ %.2f" % (
                _nw, 1 - np.linalg.norm(_Kf - _Dl)**2/_nD**2, 1 - np.linalg.norm(_P(_Kf) - _P(_Dl))**2/np.linalg.norm(_P(_Dl))**2,
                1 - ((np.diag(_Kf) - np.diag(_Dl))**2).sum()/(np.diag(_Dl)**2).sum(), np.linalg.norm(_w)/np.sqrt(_nw)))
        if STATE and l == 15:
            if KP: _mt = kpb["mu"][15]; _st = kpb["sd"][15]
            else: _cb = np.load("%s/covbank118_%s.npz" % (OUT, nm)); _mt = _cb["mu_15"] @ W[15]; _st = np.sqrt(np.diag(W[15].T @ _cb["S_15"] @ W[15]))
            lg("   СТАН l=15: похибка mp %.4f%% · s %.4f%% · γ₃ (діаг k_aab) R² %.3f" % (100*np.linalg.norm(mp - _mt)/np.linalg.norm(_mt), 100*np.linalg.norm(s - _st)/np.linalg.norm(_st), 1 - ((np.diag(Kab/np.outer(s*s, s)) - np.diag(sbt["P3f_15"])/_st**3)**2).sum()/((np.diag(sbt["P3f_15"])/_st**3)**2).sum() if sbt is not None else float("nan")))
        k_aab = KSCALE*Kab/np.outer(s*s, s); g3 = np.diag(k_aab); g4 = G4C*G4CURVE[l]
        if KP and G4T > 0 and l >= 1: g4 = (1-G4T)*g4 + G4T*kpb["g4"][l]
        if KP and G3T > 0 and l >= 1:
            g3 = (1-G3T)*g3 + G3T*kpb["g3"][l]; np.fill_diagonal(k_aab, g3)
        m = mp*Ph + s*ph - (g3/6)*s*t*ph + (g4/24)*s*(t*t-1)*ph
        Cov = np.zeros((n, n)); f = 1.0; rp = np.ones_like(rho)
        for k in range(1, NM+1):
            f *= k; rp = rp*rho; Cov += f*np.outer(c[k], c[k])*rp
        Cov += np.outer(c[2], c[1])*k_aab + np.outer(c[1], c[2])*k_aab.T \
             + rho*(6*np.outer(c[3], c[2])*k_aab + 6*np.outer(c[2], c[3])*k_aab.T
                    + 4*np.outer(c[4]*g3, c[1]) + 4*np.outer(c[1], c[4]*g3))
        if KP and K4P > 0 and l >= 1:
            _k22 = kpb["k22_%d" % l]; _k31 = kpb["k31_%d" % l]
            Cov += K4P*(np.outer(c[2], c[2])*_k22 + np.outer(c[3], c[1])*_k31 + np.outer(c[1], c[3])*_k31.T
                        + rho*(8*np.outer(c[4], c[2])*_k31 + 8*np.outer(c[2], c[4])*_k31.T))
        e2 = (mp*mp + s*s)*Ph + mp*s*ph + s*s*ph*(g3/3 - g4*t/12)
        np.fill_diagonal(Cov, e2 - m*m)
        if PSD and l < D-1:
            _ev = np.linalg.eigvalsh(Cov); _neg = _ev[_ev < 0]
            lg('   PSD l=%d min eig %.3e · max eig %.3e · від\'ємних %d · Σ|neg|/Σ|pos| %.2e' % (l, _ev.min(), _ev.max(), len(_neg), -_neg.sum()/_ev[_ev>0].sum()))
        if l + 1 >= D: break
        Wn = W[l+1]
        # перенос джерел p_l → h_{l+1} (Φ) → p_{l+1} (W); нове джерело народжується у h_{l+1}
        for src in sources: src.scale_cols(Ph)
        sources = [src for src in sources if src.age < MAXAGE and not (isinstance(src, SourceF21) and src.age >= F21AGE)]
        sources.append(Source(mp, Sp, AMAX))
        if F21 and l >= F21L:
            sources.append(SourceF21(mp, Sp, k_aab, "a")); sources.append(SourceF21(mp, Sp, k_aab, "b"))
        if F31 and KP and l >= 1:
            sources.append(SourceF21(mp, Sp, kpb["k31_%d" % l], "c"))      # ОРАКУЛ: істинна κ₃₁(p_l) датасету
        _g = GENS*(1.0 + GENS4*G4CURVE[l]/G4CURVE[15])
        if _g != 1.0:
            _gs = [x for x in sources if isinstance(x, Source)][-1]
            for q in _gs.hub: _gs.hub[q] = _gs.hub[q]*_g
        for src in sources: src.push(Wn)
        mp = m @ Wn; Sp = mm(mm(Wn.T, Cov), Wn); Sp = 0.5*(Sp + Sp.T)
        if FOLD < 99 and l + 1 >= FOLDL:
            # точна κ_aab(p_{l+1}) усіх джерел + перенесеного ядра (базис Vt неортонормований — контракція чинна)
            Vt = mm(Wn.T, Ph[:, None]*Vc) if Vc is not None else None
            Knext = sum(src.kab() for src in sources) + (core_contract(Cc, Vt) if Cc is not None else 0)
            if TRI:
                Ktri = tri_kab(c[2], rho, Wn); Knext = Knext + Ktri
            # новий базис = простір рядків κ_aab (= span базису тензора)
            Kb = (sum(src.kab() for src in sources if isinstance(src, Source)) + (core_contract(Cc, Vt) if Cc is not None else 0)) if FOLDB else Knext   # FOLDB: базис лише з g-джерел
            ev, U = np.linalg.eigh(mm(Kb.T, Kb)); V2 = U[:, ::-1][:, :R]; FL["n"] += 9.0*n**3
            C2 = core_transport(Cc, V2.T @ Vt)*DAMP if Cc is not None else np.zeros((R, R, R))
            if TRI: C2 = C2 + tri_core(c[2], rho, Wn, V2)
            keep = []
            for src in sources:
                if src.age >= FOLD and not (FOLDG and isinstance(src, SourceF21)): C2 = C2 + src.fold(V2)
                else: keep.append(src)
            sources, Cc, Vc = keep, C2, V2
    e = cerr(m, T); RES.append(e)
    if DUMP: np.savez("%s/rec118h_kab_%s.npz" % (OUT, nm), **_dump)
    _a = float(m @ T)/float(m @ m); _raw = 100*np.sqrt(float((m-T)@(m-T))/float(T@T))
    lg("   СИРА %.4f%% · косинусна %.4f%% · масштаб a* %.6f" % (_raw, e, _a))
    np.savez("%s/rec118h_m_%s.npz" % (OUT, nm), m=m, T=T)
    lg("%-20s похибка %.4f%% · флопи (матмули κ₃/ядра/Sp/eigh) %.3e = %.3f B" % (nm, e, FL["n"], FL["n"]/2.2e12)); FL["n"] = 0.0
    if k3diag: lg("   κ₃ проти банку: " + " · ".join(k3diag))
    if contrib:
        tot = np.linalg.norm(sum(src.kab() for src in sources)) if False else sum(contrib)
        lg("   внесок джерел у ‖κ_aab(p_15)‖ за віком (найстарше → наймолодше): " + " ".join("%.3f" % (x/sum(contrib)) for x in contrib))
print("СЕРЕДНЄ %.4f%%" % np.sqrt(np.mean(np.square(RES))))
lg("ГОТОВО"); print("DONE_RECH")
