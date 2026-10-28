"""С119 — СУМІШ У ПРОСТОРІ ВХОДУ: z = v·y + z⊥, y на вузлах Гаусса–Ерміта (q), z⊥~N(0,I−vvᵀ); на кожному вузлі — наша рекурсія
(AMAX=1 FOLD=4, κ₃) з початком mp₀ = y·vᵀW₀, Sp₀ = W₀ᵀ(I−vvᵀ)W₀; вихід = Σ w_i m(y_i). Умовність по ВХОДУ точна і діє на всій
траєкторії (на відміну від condk119, що умовлював стан на останньому кроці). q=1 ≡ база.
Напрямки: rand (контроль) · w0 (топ лівий сингулярний вектор W₀, макс. підсилення) · g (⟨∇f⟩: середній градієнт суми виходів, GPU).
🔮 P1 (30%): w0/g при q=9 дають ≥ ×1.10 у похибці, rand ≈ ×1.00 ⟹ клас живий, далі багатовимірні вузли
🔮 P2 (50%): усі ≈ ×1.00±0.03 ⟹ одновимірне розщеплення входу нічого не міняє (негаусовість ізотропна і тут)
🔮 P3 (20%): гірше (умовні компоненти з ненульовим середнім входу ламають калібровку G4CURVE/α)
"""
import sys, os, time
for k, v in {"NM": "6", "AMAX": "1", "FOLD": "4", "R": "64", "PRH": "0.05", "PRC": "0.03", "DIAG": "0", "SEAL": "0"}.items(): os.environ.setdefault(k, v)
Q = int(os.environ.get("Q", "9")); DIRS = os.environ.get("DIRS", "rand,w0").split(","); NETS_ = os.environ.get("NETS", "regina-johnston").split(",")
src = open("rec118h.py").read()
exec(src.split("import pyarrow")[0])                       # константи, phi/Phi/hcoef, Source, core_*, cerr, G4CURVE, lg
import numpy as np, pyarrow.parquet as pq
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
def recurse(W, mp, Sp):
    sources = []; Cc = None; Vc = None; Knext = None
    for l in range(D):
        n = NW
        s = np.sqrt(np.maximum(np.diag(Sp), 1e-300)); t = mp/s; rho = Sp/np.outer(s, s)
        ph = phi(t); Ph = Phi(t); c = hcoef(s, t, max(NM, 4))
        Kab = np.zeros((n, n))
        if Knext is not None: Kab = Knext
        else:
            for sr in sources: Kab += sr.kab()
            if Cc is not None: Kab = Kab + core_contract(Cc, Vc)
        k_aab = Kab/np.outer(s*s, s); g3 = np.diag(k_aab); g4 = G4C*G4CURVE[l]
        m = mp*Ph + s*ph - (g3/6)*s*t*ph + (g4/24)*s*(t*t-1)*ph
        Cov = np.zeros((n, n)); f = 1.0; rp = np.ones_like(rho)
        for k in range(1, NM+1):
            f *= k; rp = rp*rho; Cov += f*np.outer(c[k], c[k])*rp
        Cov += np.outer(c[2], c[1])*k_aab + np.outer(c[1], c[2])*k_aab.T \
             + rho*(6*np.outer(c[3], c[2])*k_aab + 6*np.outer(c[2], c[3])*k_aab.T + 4*np.outer(c[4]*g3, c[1]) + 4*np.outer(c[1], c[4]*g3))
        e2 = (mp*mp + s*s)*Ph + mp*s*ph + s*s*ph*(g3/3 - g4*t/12)
        np.fill_diagonal(Cov, e2 - m*m)
        if l + 1 >= D: break
        Wn = W[l+1]
        for sr in sources: sr.scale_cols(Ph)
        sources.append(Source(mp, Sp, AMAX))
        for sr in sources: sr.push(Wn)
        mp = m @ Wn; Sp = mm(mm(Wn.T, Cov), Wn); Sp = 0.5*(Sp + Sp.T)
        Vt = mm(Wn.T, Ph[:, None]*Vc) if Vc is not None else None
        Knext = sum(sr.kab() for sr in sources) + (core_contract(Cc, Vt) if Cc is not None else 0)
        ev, U = np.linalg.eigh(mm(Knext.T, Knext)); V2 = U[:, ::-1][:, :R]
        C2 = core_transport(Cc, V2.T @ Vt) if Cc is not None else np.zeros((R, R, R))
        keep = []
        for sr in sources:
            if sr.age >= FOLD: C2 = C2 + sr.fold(V2)
            else: keep.append(sr)
        sources, Cc, Vc = keep, C2, V2
    return m
def grad_dir(W):
    """⟨∇_z Σ_a out_a⟩ по 256 семплах (numpy, кусково-лінійно): напрямок середнього градієнта"""
    rng = np.random.default_rng(11); g = np.zeros(NW)
    for _ in range(4):
        z = rng.standard_normal((64, NW)); x = z; masks = []
        for k in range(D):
            p = x @ W[k]; masks.append(p > 0); x = np.maximum(p, 0.0)
        v = np.ones((64, NW))
        for k in range(D-1, -1, -1):
            v = (v*masks[k]) @ W[k].T
        g += v.sum(0)
    return g/np.linalg.norm(g)
xs, ws = np.polynomial.hermite_e.hermegauss(Q); ws = ws/ws.sum()
RES = {}
for nm in NETS_:
    idx = NAMES.index(nm); W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
    T = np.load("%s/covbank118_%s.npz" % (OUT, nm))["mu_16"]
    base = recurse(W, np.zeros(NW), W[0].T @ W[0]); eb = cerr(base, T)
    lg("%-18s база q=1: %.4f%%" % (nm, eb)); RES.setdefault("base", []).append(eb)
    for dn in DIRS:
        if dn == "rand": v = np.random.default_rng(3).standard_normal(NW)
        elif dn == "w0": v = np.linalg.svd(W[0])[0][:, 0]
        elif dn == "g": v = grad_dir(W)
        v = v/np.linalg.norm(v); a = v @ W[0]                      # a = vᵀW₀: зсув p₁ на одиницю y
        Sp0 = W[0].T @ W[0] - np.outer(a, a)
        m = sum(w*recurse(W, x*a, Sp0) for x, w in zip(xs, ws))
        e = cerr(m, T); RES.setdefault(dn, []).append(e)
        lg("   %-5s q=%d: %.4f%%  (×%.3f)" % (dn, Q, e, eb/e))
print("СЕРЕДНЄ: " + " · ".join("%s %.4f%%" % (k, np.sqrt(np.mean(np.square(v)))) for k, v in RES.items()))
print("DONE_MIX")
