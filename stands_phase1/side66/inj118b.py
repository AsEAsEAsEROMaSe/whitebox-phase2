"""С118 — СТЕЛЯ «3-Й ПОРЯДОК (ПОВНИЙ / У ПІДПРОСТОРІ r) ДЛЯ ОБОХ КАНАЛІВ» У РЕКУРСІЇ.

inj118: точні Δm, ΔC на всіх шарах ⟹ 0 (тавтологія), половинні ⟹ рівно 0.5·base,
кожен канал сам ⟹ ×1.18. Відгук ЛІНІЙНИЙ і канали ЗЧЕПЛЕНІ ⟹ похибка ≈ (1−q)·base,
де q — якість поправки ОБОХ каналів. Тут q задається 3-м порядком Еджворта:
   Δ̂m_l = −(γ₃/6)·s·t·φ(t)                                (res117: R² 0.879 сам γ₃)
   Δ̂C_l = парний Еджворт 3-го порядку з κ_aab (cstep118: R² 0.996), діагональ — 1-D
кумулянти — з ІСТИННОГО стану (банк sub118), повні або обмежені топ-r власних Σ(h_l).
Рекурсія проста (μ, Σ повна), NM=8. Похибка косинусна по μ_16 (covbank118).

🔒 S1: «exact» (Δ з того ж банку) ⟹ 0.000 · 🔒 S2: q_m(full) ≈ 0.88, q_C(full) ≈ 0.99
🔮 P1 (55%): edg3 full дає похибку ≤ 0.35·base (×8 MSE)
🔮 P2 (60%): edg3 r=256 ≤ 0.5·base · r=74 ≤ 0.7·base
🔮 P3 (70%): mean-only edg3 ≈ ×1.15 (реплікує inj118 mean)
"""
import sys, os, time, glob
import numpy as np
from math import erf
D, NW = 16, 1024
OUT = "/home/trim/wblogs"
NM = int(os.environ.get("NM", "8"))
RS = [74, 128, 256]
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

def step(mu, S, W):
    mp = mu @ W; Cp = W.T @ S @ W; Cp = 0.5*(Cp + Cp.T)
    s = np.sqrt(np.maximum(np.diag(Cp), 1e-300)); t = mp/s; rho = Cp/np.outer(s, s)
    ph = phi(t); Ph = Phi(t)
    m = mp*Ph + s*ph
    c = hcoef(s, t, NM)
    C = np.zeros_like(Cp); f = 1.0; rp = np.ones_like(rho)
    for n in range(1, NM+1):
        f *= n; rp = rp*rho; C += f*np.outer(c[n], c[n])*rp
    e2 = (mp*mp + s*s)*Ph + mp*s*ph
    np.fill_diagonal(C, e2 - m*m)
    return m, C

def edg3(mu, S, W, P3):
    """поправки 3-го порядку від ІСТИННОГО стану (mu,S) і третіх моментів P3 = E[p̃²⊗p̃]"""
    mp = mu @ W; Cp = W.T @ S @ W; Cp = 0.5*(Cp + Cp.T)
    s = np.sqrt(np.diag(Cp)); t = mp/s; rho = Cp/np.outer(s, s)
    ph = phi(t); c = hcoef(s, t, 4)
    k_aab = P3/np.outer(s*s, s); g3 = np.diag(k_aab)
    dm = -(g3/6)*s*t*ph
    dC = np.outer(c[2], c[1])*k_aab + np.outer(c[1], c[2])*k_aab.T \
       + rho*(6*np.outer(c[3], c[2])*k_aab + 6*np.outer(c[2], c[3])*k_aab.T
              + 4*np.outer(c[4]*g3, c[1]) + 4*np.outer(c[1], c[4]*g3))
    dE2 = s*s*ph*g3/3; dE1 = dm
    np.fill_diagonal(dC, dE2 - 2*c[0]*dE1)
    return dm, dC

def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T
    return 100*np.sqrt(float(r @ r)/float(T @ T))

import pyarrow.parquet as pq
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
RES = {}
files = sorted(glob.glob("%s/sub118_*.npz" % OUT))
lg("банків %d · NM=%d" % (len(files), NM))
for fn in files:
    nm = os.path.basename(fn)[len("sub118_"):-4]
    cb = "%s/covbank118_%s.npz" % (OUT, nm)
    if not os.path.exists(cb): lg("%-20s нема covbank (μ_16)" % nm); continue
    idx = NAMES.index(nm)
    W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
    d = np.load(fn); T = np.load(cb)["mu_16"]
    mu = [np.zeros(NW)] + [d["mu_%d" % l] for l in range(1, D)]
    S = [np.eye(NW)] + [d["S_%d" % l] for l in range(1, D)]
    # точні Δ (для шарів 0..14; шар 15 → ціль μ_16 з covbank — інший потік семплів)
    dm, dC = [None]*D, [None]*D
    for l in range(D-1):
        m, C = step(mu[l], S[l], W[l]); dm[l] = mu[l+1] - m; dC[l] = S[l+1] - C
    # оцінки 3-го порядку
    EST = {}
    for tag in ["f"] + ["r%d" % r for r in RS]:
        em, eC = [np.zeros(NW)]*1 + [None]*(D-1), [np.zeros((NW, NW))]*1 + [None]*(D-1)
        for l in range(1, D):
            em[l], eC[l] = edg3(mu[l], S[l], W[l], d["P3%s_%d" % (tag, l)])
        EST[tag] = (em, eC)
    # якість по шарах 1..14 (де є точна Δ)
    def q(est, ex):
        num = sum(((ex[l] - est[l])**2).sum() for l in range(1, D-1)); den = sum((ex[l]**2).sum() for l in range(1, D-1))
        return 1 - num/den
    qual = {tag: (q(EST[tag][0], dm), q(EST[tag][1], dC)) for tag in EST}
    row = {}
    def rec(mode, tag=None, c=1.0):
        m, C = np.zeros(NW), np.eye(NW)
        for l in range(D):
            m, C = step(m, C, W[l])
            if mode == "exact" and l < D-1: m = m + dm[l]; C = C + dC[l]
            if mode in ("both", "mean") and l >= 1: m = m + c*EST[tag][0][l]
            if mode in ("both", "cov") and l >= 1: C = C + c*EST[tag][1][l]
        return cerr(m, T)
    row["base"] = rec("none"); row["exact14"] = rec("exact")
    for tag in EST:
        row["both_" + tag] = rec("both", tag); row["mean_" + tag] = rec("mean", tag); row["cov_" + tag] = rec("cov", tag)
        row["both05_" + tag] = rec("both", tag, 0.5)
    RES[nm] = (row, qual)
    lg("%-20s base %.4f%% · 🔒exact(0..14) %.4f · q(full) m %.3f C %.3f · q(r74) m %.3f C %.3f · q(r256) m %.3f C %.3f"
       % (nm, row["base"], row["exact14"], qual["f"][0], qual["f"][1], qual["r74"][0], qual["r74"][1], qual["r256"][0], qual["r256"][1]))
    lg("   both: full %.4f · r74 %.4f · r128 %.4f · r256 %.4f | mean full %.4f · cov full %.4f · both(0.5) full %.4f"
       % (row["both_f"], row["both_r74"], row["both_r128"], row["both_r256"], row["mean_f"], row["cov_f"], row["both05_f"]))

print("\n── похибка виходу (%%, косинусна), середнє по %d мережах · ×MSE проти бази ──" % len(RES))
base = np.mean([r["base"]**2 for r, _ in RES.values()])
keys = ["base", "exact14"] + ["%s_%s" % (m, t) for t in ["f", "r74", "r128", "r256"] for m in ["both", "mean", "cov", "both05"]]
for k in keys:
    v = [r[k] for r, _ in RES.values()]; e2 = np.mean([x*x for x in v])
    print("%-12s | %.4f%%  ×MSE %8.3f  (мін %.4f макс %.4f)" % (k, np.sqrt(e2), base/e2, min(v), max(v)))
print("якість q (1 − ‖Δ−Δ̂‖²/‖Δ‖², шари 1..14), середнє:")
for tag in ["f", "r74", "r128", "r256"]:
    print("   %-5s q_m %.3f · q_C %.3f" % (tag, np.mean([q_[tag][0] for _, q_ in RES.values()]), np.mean([q_[tag][1] for _, q_ in RES.values()])))
lg("ГОТОВО")
print("DONE_INJB")
