"""С118 — ГЕЙТ БУДІВНИЦТВА: ЧИ ВЕДЕ ТОЧНА ПОПРАВКА ОБОХ КАНАЛІВ КРОКУ РЕКУРСІЮ ДО ІСТИНИ?

С117 вкидала точний залишок кроку СЕРЕДНЬОГО у продакшен-рекурсію: ×1.23 у MSE (5..15
×1.26). Пояснення: sd хибна на 2.9% і через φ(t)·δs годує mp; коваріаційний канал не
чіпали. cstep118 показав, що залишок кроку C — теж третій порядок (κ_aab, R² 0.996).
Питання: якщо додати ТОЧНІ Δm_l і ΔC_l (від істинного стану, банк covbank118) на КОЖНОМУ
шарі до ПРОСТОЇ гаусової рекурсії (μ, Σ повна) — чи сходить стан до істини?

Крок (той самий для означення Δ і для рекурсії):  mp = μW, Cp = WᵀΣW, s, t, ρ
   m' = mpΦ + sφ ;  Cov' = Σ_{n≤NM} n!·c_n c_nᵀ ρⁿ, c_n = s(−1)ⁿHe_{n−2}(t)φ/n!, c1 = sΦ ;
   діагональ точна: E[ReLU²] − m'².
Δm_l = μ_{l+1} − m'(μ_l,Σ_l) ; ΔC_l = Σ_{l+1} − Cov'(μ_l,Σ_l) ;  μ_0 = 0, Σ_0 = I.
Рекурсія: (m,C) ← step(m,C) + c·(Δm_l | ΔC_l за режимом).  Похибка: косинусна по μ_16.

🔒 S1: Δ_0 ≈ підлога банку (шар 0 гаусів ТОЧНО): ‖Δm_0‖/‖μ_1‖ ≲ 0.1%, ‖ΔC_0‖/‖Σ_1‖ ≲ 1%
🔒 S2: reset@15 (істинний стан на вході останнього кроку) ≈ 0.046% [С116: 0.0459%]
🔒 S3: both, c=1, усі шари ⟹ ЯКЩО стан слідує істині, похибка → S2
🔮 P1 (50%): both c=1 дає похибку ≤ 0.5·base (×4 у MSE)
🔮 P2 (35%): both c=1 ≥ 0.8·base ⟹ розходження внутрішнє, детермінований шлях капнутий ~×1.3
🔮 P3 (65%): covdiag-only ГІРШЕ за базу, covoff-only краще (гейт узгодженості)
🔮 P4 (60%): mean-only ≈ ×1.2–1.3 у MSE (репліка С117 на іншій рекурсії)
"""
import sys, os, time, glob
import numpy as np
from math import erf
D, NW = 16, 1024
OUT = "/home/trim/wblogs"
NM = int(os.environ.get("NM", "8"))
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

def step(mu, S, W):
    mp = mu @ W; Cp = W.T @ S @ W; Cp = 0.5*(Cp + Cp.T)
    s = np.sqrt(np.maximum(np.diag(Cp), 1e-300)); t = mp/s; rho = Cp/np.outer(s, s)
    ph = phi(t); Ph = Phi(t)
    m = mp*Ph + s*ph
    c = [None, s*Ph]; f = 1.0
    for n in range(2, NM+1):
        f *= n; c.append(s*((-1)**n)*He(n-2, t)*ph/f)
    C = np.zeros_like(Cp); f = 1.0; rp = np.ones_like(rho)
    for n in range(1, NM+1):
        f *= n; rp = rp*rho; C += f*np.outer(c[n], c[n])*rp
    e2 = (mp*mp + s*s)*Ph + mp*s*ph
    np.fill_diagonal(C, e2 - m*m)
    return m, C

def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T
    return 100*np.sqrt(float(r @ r)/float(T @ T))

import pyarrow.parquet as pq
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
MODES = ["none", "mean", "cov", "covdiag", "covoff", "both", "both9", "reset15", "reset8"]
CS = [1.0, 0.5]
RES = {}
files = sorted(glob.glob("%s/covbank118_*.npz" % OUT))
lg("банків %d · NM=%d" % (len(files), NM))
for fn in files:
    nm = os.path.basename(fn)[len("covbank118_"):-4]
    idx = NAMES.index(nm)
    W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
    d = np.load(fn)
    mu = [np.zeros(NW)] + [d["mu_%d" % l] for l in range(1, D+1)]
    S = [np.eye(NW)] + [d["S_%d" % l] for l in range(1, D+1)]
    T = mu[D]
    dm, dC = [], []
    for l in range(D):
        m, C = step(mu[l], S[l], W[l])
        dm.append(mu[l+1] - m); dC.append(S[l+1] - C)
    seal = "🔒Δ_0: m %.3f%% C %.2f%% · Δ_8: m %.3f%% C %.2f%% · Δ_15: m %.3f%% C %.2f%%" % (
        100*np.linalg.norm(dm[0])/np.linalg.norm(mu[1]), 100*np.linalg.norm(dC[0])/np.linalg.norm(S[1]),
        100*np.linalg.norm(dm[8])/np.linalg.norm(mu[9]), 100*np.linalg.norm(dC[8])/np.linalg.norm(S[9]),
        100*np.linalg.norm(dm[15])/np.linalg.norm(mu[16]), 100*np.linalg.norm(dC[15])/np.linalg.norm(S[16]))
    row = {}
    for mode in MODES:
        for c in (CS if mode not in ("none", "reset15", "reset8") else [1.0]):
            m, C = np.zeros(NW), np.eye(NW)
            for l in range(D):
                if mode == "reset15" and l == 15: m, C = mu[15], S[15]
                if mode == "reset8" and l == 8: m, C = mu[8], S[8]
                m, C = step(m, C, W[l])
                if mode in ("mean", "both") or (mode == "both9" and l >= 9): m = m + c*dm[l]
                if mode in ("cov", "both") or (mode == "both9" and l >= 9): C = C + c*dC[l]
                if mode == "covdiag": C = C + c*np.diag(np.diag(dC[l]))
                if mode == "covoff": C = C + c*(dC[l] - np.diag(np.diag(dC[l])))
            row[(mode, c)] = cerr(m, T)
    RES[nm] = row
    lg("%-20s %s" % (nm, seal))
    lg("   base %.4f%% · mean %.4f · cov %.4f · both %.4f · both(0.5) %.4f · reset15 %.4f · reset8 %.4f · covdiag %.4f · covoff %.4f"
       % (row[("none", 1.0)], row[("mean", 1.0)], row[("cov", 1.0)], row[("both", 1.0)], row[("both", 0.5)],
          row[("reset15", 1.0)], row[("reset8", 1.0)], row[("covdiag", 1.0)], row[("covoff", 1.0)]))

print("\n── похибка виходу (%%, косинусна), середнє по %d мережах · ×MSE проти бази ──" % len(RES))
base = np.mean([r[("none", 1.0)]**2 for r in RES.values()])
for mode in MODES:
    for c in (CS if mode not in ("none", "reset15", "reset8") else [1.0]):
        v = [r[(mode, c)] for r in RES.values()]
        e2 = np.mean([x*x for x in v])
        print("%-9s c=%.1f | %.4f%%  ×MSE %7.3f  (мін %.4f макс %.4f)" % (mode, c, np.sqrt(e2), base/e2, min(v), max(v)))
lg("ГОТОВО")
print("DONE_INJ")
