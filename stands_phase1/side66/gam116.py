"""С116: БАНК ПОНЕЙРОННИХ МОМЕНТІВ ПЕРЕДАКТИВАЦІЇ (до 4-го) НА ГЛИБОКИХ ШАРАХ.

МОТИВ [ceil116b]: вкидання ТОЧНИХ (mp, C) на шар 15 лишає 0.0481% похибки —
це ЧИСТА ціна гаусової гіпотези на ОДНОМУ кроці. Потрібно 0.0266% ⟹ ×1.81.
Питання: чи покривають цю щілину ПОНЕЙРОННІ γ₃, γ₄?

🚨 КОРПУСНЕ ЗАСТЕРЕЖЕННЯ С104: табличні «γ по шарах» — це γ ПРОЄКЦІЙ, інший
об'єкт. Тут рахуються САМЕ ПОНЕЙРОННІ моменти `p_j = w_j·h`.

Центруємо на замикальному `mp` ДО піднесення до степеня ⟹ центральні суми
добре обумовлені за один прохід.
"""
import os, sys, time, glob
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

D, NW = 16, 1024
NTOT  = 1 << 22          # 4 194 304
BATCH = 1 << 16
LAY   = [0, 8, 12, 13, 14, 15]
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def closure_mp(W):
    """замикальні mp по шарах — використовуємо ЛИШЕ як зсув для центрування"""
    m = np.zeros(NW); C = np.eye(NW); MP = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        MP.append(mp.copy())
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MP

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
WANT = [str(np.load(f, allow_pickle=True)["name"])
        for f in sorted(glob.glob("/home/trim/wblogs/truthC_*.npz"))]
lg("мереж до обробки: %d · N=%d · шари %s" % (len(WANT), NTOT, LAY))

for nm in WANT:
    out = "/home/trim/wblogs/gam116_%s.npz" % nm
    if os.path.exists(out): lg("%s вже є, пропускаю" % nm); continue
    i = NAMES.index(nm)
    W = np.asarray(tb["weights"][i].as_py(), dtype=np.float64).reshape(D, NW, NW)
    MP = closure_mp(W)
    Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
    SH = {L: cp.asarray(MP[L]) for L in LAY}                  # зсув
    S = {L: cp.zeros((4, NW), cp.float64) for L in LAY}       # Σ(p−sh)^k
    R = {L: cp.zeros(NW, cp.float64) for L in LAY}            # Σ relu(p)
    done = 0; b = 0
    while done < NTOT:
        n = min(BATCH, NTOT - done)
        rs = cp.random.RandomState(seed=770000 + i*1000 + b)
        x = rs.standard_normal((n, NW), dtype=cp.float32)
        for k in range(D):
            p = x @ Wg[k]
            if k in LAY:
                d1 = (p - SH[k]).astype(cp.float64)
                S[k][0] += d1.sum(0); d2 = d1*d1
                S[k][1] += d2.sum(0); S[k][2] += (d2*d1).sum(0); S[k][3] += (d2*d2).sum(0)
                R[k] += cp.maximum(p, 0).astype(cp.float64).sum(0)
                del d1, d2
            x = cp.maximum(p, 0.0)
        del x, p
        cp.get_default_memory_pool().free_all_blocks()
        done += n; b += 1
    res = {}
    for L in LAY:
        s1, s2, s3, s4 = [cp.asnumpy(S[L][k])/NTOT for k in range(4)]
        sh = MP[L]
        mu = sh + s1                                   # справжнє середнє p
        m2 = s2 - s1*s1
        m3 = s3 - 3*s1*s2 + 2*s1**3
        m4 = s4 - 4*s1*s3 + 6*s1*s1*s2 - 3*s1**4
        sd = np.sqrt(np.maximum(m2, 1e-30))
        res["mp_%d" % L]  = mu
        res["sd_%d" % L]  = sd
        res["g3_%d" % L]  = m3/sd**3
        res["g4_%d" % L]  = m4/sd**4 - 3.0
        res["relu_%d" % L] = cp.asnumpy(R[L])/NTOT
    np.savez_compressed(out, name=nm, n=NTOT, **res)
    lg("%-18s готово · γ₃(ш15) сер %+.4f sd %.4f · γ₄(ш15) сер %+.4f sd %.4f · "
       "🔒γ₃(ш0) %+.5f γ₄(ш0) %+.5f"
       % (nm, res["g3_15"].mean(), res["g3_15"].std(),
          res["g4_15"].mean(), res["g4_15"].std(),
          res["g3_0"].mean(), res["g4_0"].mean()))
    del Wg, S, R, SH
    cp.get_default_memory_pool().free_all_blocks()
lg("ГОТОВО")
