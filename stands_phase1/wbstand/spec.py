"""С115-у: ЧИ ВКАЗУЄ ШУМ, ДЕ ПОХИБКА? (ідея ЮЗЕРА «зібрати шум між семплами і шарами»)

ТОЧНА ФОРМА: розкласти по власному базису істинної Cov(h_L) три величини —
   λ_i            = дисперсія напрямку  ⟹ де живе ШУМ МК (шум ∝ λ_i/N)
   ⟨b, u_i⟩²      = похибка замикання   ⟹ де живе ПОХИБКА
   ⟨m, u_i⟩²      = сам сигнал
Якщо шум і похибка в ОДНИХ напрямках — «шум вказує де шукати» працює
(і це рівно вінерів/спектральний бленд). Якщо в РІЗНИХ — мертво за побудовою.

🔒 b береться з ПОЛОВИНИ A банку, Cov — зі СВІЖИХ семплів ⟹ незалежні.
🔒 ізотропна опора = k/1024 (частка, якби величина була рівномірна).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 λ концентрується: топ-64 несуть >70% сліду                    80%
  P2 ‖b‖² НАБАГАТО плоскіша: топ-64 несуть 10..25%                 65%
  P3 ‖m‖² концентрується як λ (сигнал і шум в одному підпросторі)  70%
  P4 ⟹ «шум вказує на похибку» дає ×1.0x                           75%
"""
import os, sys, glob, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp
D, NW = 16, 1024
NS = 1 << 20
CH = 1 << 13
LAYERS = [8, 12, 14, 15]
KS = [1, 4, 16, 64, 256, 512]
T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def cl_np(W):
    m = np.zeros(NW); C = np.eye(NW); MS = []
    for i in range(D):
        w = W[i]; q = w.T @ C @ w
        if i > KSIG: q = q*KAPPA
        s = np.sqrt(np.maximum(np.diag(q), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t); m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(q, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy())
        if i == D-1: break
        C = q*np.outer(Ph, Ph) + 0.5*(q*q)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS

ACC = {L: {"lam": [], "b": [], "m": [], "ov": [], "ov1": [], "b1": [], "l1": [], "ovs": []} for L in LAYERS}
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:6]:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    wf = "/home/trim/wblogs/w2_%s.npy" % nm
    if not os.path.exists(wf): continue
    mA = np.asarray(d["m_half"])[:, 0, :]
    W = np.load(wf); Wg = [cp.asarray(W[i]) for i in range(D)]
    MS = cl_np(W.astype(np.float64))
    sm = {L: cp.zeros(NW, dtype=cp.float64) for L in LAYERS}
    S2 = {L: cp.zeros((NW, NW), dtype=cp.float64) for L in LAYERS}
    done = 0; c = 0
    while done < NS:
        b = min(CH, NS-done)
        z = cp.random.RandomState(seed=808+c).standard_normal((b, NW), dtype=cp.float32)
        x = z
        for i in range(D):
            x = cp.maximum(x @ Wg[i], 0.0)
            if i in LAYERS:
                sm[i] += x.sum(0, dtype=cp.float64); S2[i] += (x.T @ x).astype(cp.float64)
        done += b; c += 1
    inv = 1.0/float(NS)
    for L in LAYERS:
        mu = sm[L]*inv; Cv = S2[L]*inv - cp.outer(mu, mu)
        ev, U = cp.linalg.eigh(0.5*(Cv + Cv.T))
        ev = cp.asnumpy(ev)[::-1]; U = cp.asnumpy(U)[:, ::-1]      # спадання
        bb = MS[L] - mA[L]; mm = MS[L]
        pb = (U.T @ bb)**2; pm = (U.T @ mm)**2; pl = np.maximum(ev, 0.0)
        ACC[L]["lam"].append(np.cumsum(pl)/pl.sum())
        ACC[L]["b"].append(np.cumsum(pb)/pb.sum())
        ACC[L]["m"].append(np.cumsum(pm)/pm.sum())
        # перекриття профілів (косинус між нормованими спектрами)
        a1 = pl/pl.sum(); a2 = pb/pb.sum()
        ACC[L]["ov"].append(float((a1*a2).sum()/np.sqrt((a1*a1).sum()*(a2*a2).sum())))
        # 🔑 БЕЗ ТОП-1: топ-1 = масштаб, його `_ai` бере даром ⟹ вирок на залишку
        c1 = pl[1:]/pl[1:].sum(); c2 = pb[1:]/pb[1:].sum()
        ACC[L]["ov1"].append(float((c1*c2).sum()/np.sqrt((c1*c1).sum()*(c2*c2).sum())))
        ACC[L]["b1"].append(np.cumsum(pb[1:])/pb[1:].sum())
        ACC[L]["l1"].append(np.cumsum(pl[1:])/pl[1:].sum())
        # 🔒 контроль: перемішана похибка (ізотропна за побудовою)
        rs = np.random.default_rng(5).permutation(NW-1)
        c3 = pb[1:][rs]/pb[1:].sum()
        ACC[L]["ovs"].append(float((c1*c3).sum()/np.sqrt((c1*c1).sum()*(c3*c3).sum())))
    log("%-18s готово" % nm)
    del Wg, sm, S2; cp.get_default_memory_pool().free_all_blocks()

print("\nчастка величини в ТОП-k власних напрямках Cov(h_L)   [🔒ізотропно = k/1024]")
for L in LAYERS:
    lam = np.median(np.array(ACC[L]["lam"]), 0)
    bb  = np.median(np.array(ACC[L]["b"]), 0)
    mm  = np.median(np.array(ACC[L]["m"]), 0)
    print("\nшар %d" % L)
    print("  k      1      4     16     64    256    512")
    print("  ізотр  %s" % " ".join("%6.3f" % (k/1024.0) for k in KS))
    print("  λ ШУМ  %s" % " ".join("%6.3f" % lam[k-1] for k in KS))
    print("  m сигн %s" % " ".join("%6.3f" % mm[k-1] for k in KS))
    print("  b ПОХБ %s" % " ".join("%6.3f" % bb[k-1] for k in KS))
    b1 = np.median(np.array(ACC[L]["b1"]), 0); l1 = np.median(np.array(ACC[L]["l1"]), 0)
    print("  --- БЕЗ ТОП-1 (масштаб знято, його бере _ai) ---")
    print("  ізотр  %s" % " ".join("%6.3f" % (k/1023.0) for k in KS))
    print("  λ ШУМ  %s" % " ".join("%6.3f" % l1[k-1] for k in KS))
    print("  b ПОХБ %s" % " ".join("%6.3f" % b1[k-1] for k in KS))
    print("  перекриття λ↔b:  повне %.4f · БЕЗ ТОП-1 %.4f · 🔒переміш %.4f"
          % (np.median(ACC[L]["ov"]), np.median(ACC[L]["ov1"]), np.median(ACC[L]["ovs"])))
log("ГОТОВО")
