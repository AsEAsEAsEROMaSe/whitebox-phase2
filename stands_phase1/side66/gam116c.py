"""С116-б: БАНК ПОНЕЙРОННИХ МОМЕНТІВ НА ВСІХ 16 ШАРАХ (gam116 мав лише 6).

МОТИВ: `sdfix116` прикладав універсальну поправку log(s_true/s_our)=a+b·t²
ЛИШЕ НА ШАРІ 15, як фініш. Усередину рекурсії вона не заходила жодного разу.
А `m = mp·Φ(t) + s·φ(t)` ⟹ похибка `s` ВХОДИТЬ У `m`, отже у `mp` наступного
шару, з коефіцієнтом передачі `φ(t)/ψ(t)`:
    L1  t≈0.484  φ/ψ = 0.517      ← ВЕЛИКИЙ
    L8  t≈1.726  φ/ψ = 0.051
    L15 t≈2.694  φ/ψ = 0.004      ← мізерний
⟹ на глибині `s` не важить, а на МІЛКИХ шарах важить сильно — і саме там
корпус ніколи не міряв похибку `s` (профіль prop116 починався з шару 8).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 (60%) вкидання істинної `s` НА ВСІХ шарах ріже похибку mp₁₅ у ≥×1.5
  P2 (35%) ріже у ≥×3 (стільки треба, щоб відкрити ворота γ₃)
  P3 (75%) передача домінована МІЛКИМИ шарами 1..6, не глибокими
  P4 (50%) наша `s` на мілких шарах уже точна (<0.5%) ⟹ стеля мала, ідея вмирає
🔒 ПЛОМБА: шар 0 мусить дати похибку `s` = шум банку (C=I ⟹ cp=W₀ᵀW₀ ТОЧНЕ)
"""
import os, sys, time, glob
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

D, NW = 16, 1024
NTOT  = 1 << 23
BATCH = 1 << 16
LAY   = list(range(D))
NNET  = 12
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def closure_mp(W):
    m = np.zeros(NW); C = np.eye(NW); MP = []; SD = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        MP.append(mp.copy()); SD.append(s.copy())
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MP, SD

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
WANT = NAMES[:NNET]
lg("мереж %d · N=%d · шари ВСІ 16" % (len(WANT), NTOT))

for nm in WANT:
    out = "/home/trim/wblogs/gamhalf_%s.npz" % nm
    tmp = out + ".tmp"
    if os.path.exists(out): lg("%s вже є" % nm); continue
    i = NAMES.index(nm)
    W = np.asarray(tb["weights"][i].as_py(), dtype=np.float64).reshape(D, NW, NW)
    MP, SD = closure_mp(W)
    Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
    SH = {L: cp.asarray(MP[L]) for L in LAY}
    S = {(h,L): cp.zeros((4, NW), cp.float64) for h in (0,1) for L in LAY}
    R = {(h,L): cp.zeros(NW, cp.float64) for h in (0,1) for L in LAY}
    done = 0; b = 0
    while done < NTOT:
        n = min(BATCH, NTOT - done); hf = 0 if done < NTOT//2 else 1
        rs = cp.random.RandomState(seed=990000 + i*1000 + b)
        x = rs.standard_normal((n, NW), dtype=cp.float32)
        for k in range(D):
            p = x @ Wg[k]
            d1 = (p - SH[k]).astype(cp.float64)
            S[(hf,k)][0] += d1.sum(0); d2 = d1*d1
            S[(hf,k)][1] += d2.sum(0); S[(hf,k)][2] += (d2*d1).sum(0); S[(hf,k)][3] += (d2*d2).sum(0)
            R[(hf,k)] += cp.maximum(p, 0).astype(cp.float64).sum(0)
            del d1, d2
            x = cp.maximum(p, 0.0)
        del x, p
        cp.get_default_memory_pool().free_all_blocks()
        done += n; b += 1
    res = {}; NH = NTOT//2
    for h in (0,1):
        for L in LAY:
            s1, s2, s3, s4 = [cp.asnumpy(S[(h,L)][k])/NH for k in range(4)]
            mu = MP[L] + s1
            m2 = s2 - s1*s1; m3 = s3 - 3*s1*s2 + 2*s1**3
            m4 = s4 - 4*s1*s3 + 6*s1*s1*s2 - 3*s1**4
            sd = np.sqrt(np.maximum(m2, 1e-30))
            res["h%d_mp_%d" % (h,L)] = mu; res["h%d_sd_%d" % (h,L)] = sd
            res["h%d_g3_%d" % (h,L)] = m3/sd**3; res["h%d_g4_%d" % (h,L)] = m4/sd**4 - 3.0
            res["h%d_relu_%d" % (h,L)] = cp.asnumpy(R[(h,L)])/NH
    np.savez_compressed(tmp, name=nm, n=NTOT, **res); os.replace(tmp + ".npz", out)
    lg("%-18s готово · 🔒A-vs-B relu ш15 %.4f%% · γ₃ ш15 A %.4f B %.4f"
       % (nm, 100*np.linalg.norm(res["h0_relu_15"]-res["h1_relu_15"])/np.linalg.norm(res["h1_relu_15"]),
          res["h0_g3_15"].std(), res["h1_g3_15"].std()))
    del Wg, S, R, SH
    cp.get_default_memory_pool().free_all_blocks()
lg("ГОТОВО")
