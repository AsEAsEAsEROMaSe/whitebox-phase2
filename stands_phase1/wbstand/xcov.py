"""С115-о: МІЖШАРОВА КОВАРІАЦІЯ АКТИВАЦІЙ. Ідея ЮЗЕРА «кореляція всього до всього».

ОБ'ЄКТ: G_{l,l'} = Cov(h_l, h_l'). Замикання несе ЛИШЕ 16 діагональних блоків;
усі 120 позадіагональних не формуються ніколи.

НУЛЬ-МОДЕЛЬ — ЛЕМА СТАЙНА (точна для спільно гаусових):
    Cov(X, relu(Y)) = Cov(X, Y)·E[relu'(Y)] = Cov(X,Y)·Φ(t)
⟹ передбачення  pred_{l,l+1} = (C_ll @ W_{l+1}) * Φ̂_{l+1}
   Δ = C_{l,l+1} − pred   є РІВНО мірою негаусовості СПІЛЬНОГО розподілу.
Усе рахується з ОДНИХ І ТИХ САМИХ семплів ⟹ похибки нашого замикання не входять.

🔒 ПЛОМБА НА КІНЦІ ОСІ: пара (z, p_0) спільно гаусова ТОЧНО (Cov(z,z)=I) ⟹
   Cov(z, h_0) = W_0 * Φ̂_0 мусить справдитись до вибіркового шуму.
🔒 ПЛОМБА МАСШТАБУВАННЯ: N та N/4 ⟹ шум удвічі вниз, справжня Δ стоїть.

СТАДІЯ B: ранг якобіана J_{l→15} = Π (W_k ⊙ D_k). Запис «ранг 2.74 ⟹ розкопки
неможливі» — це ФАЗА 1 (256×32); у фазі 2 PR стану 60.2, не 2.8 ⟹ перевірити.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 пломба (z,h_0): Δ < 0.5%                                     85%
  P2 сусідні пари: Δ = 2..15%, росте з глибиною                   60%
  P3 далекі пари (l→15) мають Δ БІЛЬШУ за сусідні                 70%
  P4 PR(J_{l→15}) у фазі 2 НАБАГАТО більший за 2.74               80%
  P5 масштабування: Δ не змінюється при N×4                       80%
"""
import os, sys, time, glob
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp

NNET  = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LOGN  = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEED  = int(sys.argv[3]) if len(sys.argv) > 3 else 4242
CHUNK = 1 << 13
D, NW = 16, 1024
FAR   = [0, 2, 4, 8, 12, 14]
N     = 1 << LOGN
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
log("cupy %s · карт видно %d" % (cp.__version__, cp.cuda.runtime.getDeviceCount()))

def fro(a): return float(cp.sqrt((a*a).sum()))

def one_net(Wg, n, sd):
    sm = [cp.zeros(NW, dtype=cp.float64) for _ in range(D)]
    mk = [cp.zeros(NW, dtype=cp.float64) for _ in range(D)]
    Sd = [cp.zeros((NW, NW), dtype=cp.float64) for _ in range(D)]
    Sa = [cp.zeros((NW, NW), dtype=cp.float64) for _ in range(D-1)]
    Sf = {l: cp.zeros((NW, NW), dtype=cp.float64) for l in FAR}
    Sz = cp.zeros((NW, NW), dtype=cp.float64)
    done = 0; c = 0
    while done < n:
        b = min(CHUNK, n-done)
        z = cp.random.RandomState(seed=sd+c).standard_normal((b, NW), dtype=cp.float32)
        H = []
        x = z
        for i in range(D):
            p = x @ Wg[i]
            m = (p > 0.0)
            x = cp.maximum(p, 0.0)
            H.append(x); mk[i] += m.sum(0, dtype=cp.float64)
        for i in range(D):
            sm[i] += H[i].sum(0, dtype=cp.float64)
            Sd[i] += (H[i].T @ H[i]).astype(cp.float64)
        for i in range(D-1):
            Sa[i] += (H[i].T @ H[i+1]).astype(cp.float64)
        for l in FAR:
            Sf[l] += (H[l].T @ H[D-1]).astype(cp.float64)
        Sz += (z.T @ H[0]).astype(cp.float64)
        done += b; c += 1
        del H, z, x, p, m
    inv = 1.0/float(n)
    M  = [s*inv for s in sm]
    PH = [s*inv for s in mk]
    Cd = [Sd[i]*inv - cp.outer(M[i], M[i]) for i in range(D)]
    Ca = [Sa[i]*inv - cp.outer(M[i], M[i+1]) for i in range(D-1)]
    Cf = {l: Sf[l]*inv - cp.outer(M[l], M[D-1]) for l in FAR}
    Cz = Sz*inv                                   # E[z]=0 ⟹ Cov(z,h0)=E[z h0ᵀ]
    return M, PH, Cd, Ca, Cf, Cz

W_ALL = sorted(glob.glob("/home/trim/wblogs/w2_*.npy"))[:NNET]
log("мереж: %d · N=%d" % (len(W_ALL), N))
RES = {"seal": [], "adj": [], "far": [], "pr": [], "half": []}

for f in W_ALL:
    nm = os.path.basename(f)[3:-4]
    W = np.load(f)
    Wg = [cp.asarray(W[i]) for i in range(D)]

    M, PH, Cd, Ca, Cf, Cz = one_net(Wg, N, SEED)

    # 🔒 ПЛОМБА (z, h0): Cov(z,h0) = W0 * Φ̂0
    pz = Wg[0] * PH[0][None, :].astype(cp.float32)
    sealv = fro(Cz - pz)/fro(Cz)
    RES["seal"].append(sealv)

    # сусідні пари
    adj = []
    for i in range(D-1):
        pred = (Cd[i] @ Wg[i+1].astype(cp.float64)) * PH[i+1][None, :]
        adj.append(fro(Ca[i] - pred)/fro(Ca[i]))
    RES["adj"].append(adj)

    # далекі пари l→15: ланцюг Стайна
    far = []
    for l in FAR:
        P = Cd[l].copy()
        for k in range(l+1, D):
            P = (P @ Wg[k].astype(cp.float64)) * PH[k][None, :]
        far.append(fro(Cf[l] - P)/fro(Cf[l]))
    RES["far"].append(far)

    # СТАДІЯ B: ранг якобіана J_{l→15} на РЕАЛЬНИХ масках одного семпла
    zz = cp.random.RandomState(seed=SEED+9991).standard_normal((1, NW), dtype=cp.float32)
    Dm = []; x = zz
    for i in range(D):
        p = x @ Wg[i]; Dm.append((p > 0).astype(cp.float32)[0]); x = cp.maximum(p, 0.0)
    pr = []
    for l in FAR:
        J = cp.eye(NW, dtype=cp.float32)
        for k in range(l+1, D):
            J = (J @ Wg[k]) * Dm[k][None, :]
        sv = cp.linalg.svd(J, compute_uv=False).astype(cp.float64)
        e = sv*sv
        pr.append(float((e.sum()**2)/(e*e).sum()))
    RES["pr"].append(pr)

    # 🔒 масштабування: половинний N, окремий сід
    M2, PH2, Cd2, Ca2, _, _ = one_net(Wg, N//4, SEED+777)
    a2 = []
    for i in (1, 8, 14):
        pred = (Cd2[i] @ Wg[i+1].astype(cp.float64)) * PH2[i+1][None, :]
        a2.append(fro(Ca2[i] - pred)/fro(Ca2[i]))
    RES["half"].append(a2)

    log("%-18s 🔒пломба(z,h0) %.5f · Δсусід ш1 %.4f ш8 %.4f ш14 %.4f · PR(J 0→15) %.1f"
        % (nm, sealv, adj[1], adj[8], adj[14], pr[0]))
    del Wg, Cd, Ca, Cf, Cd2, Ca2
    cp.get_default_memory_pool().free_all_blocks()

A = np.array(RES["adj"]); F = np.array(RES["far"]); P = np.array(RES["pr"]); H2 = np.array(RES["half"])
print("\n🔒 ПЛОМБА (z,h0), мусить бути ~вибірковий шум: %s" % " ".join("%.5f" % v for v in RES["seal"]))
print("\nΔ СУСІДНІХ пар (‖Cov − Стайн‖/‖Cov‖), медіана по мережах:")
print("  шар:  " + " ".join("%6d" % i for i in range(D-1)))
print("  Δ:    " + " ".join("%6.3f" % v for v in np.median(A, 0)))
print("\nΔ ДАЛЕКИХ пар l→15:")
print("  шар:  " + " ".join("%6d" % l for l in FAR))
print("  Δ:    " + " ".join("%6.3f" % v for v in np.median(F, 0)))
print("\nPR якобіана J_{l→15}  [фаза 1 давала 2.74 для 0→31]:")
print("  шар:  " + " ".join("%6d" % l for l in FAR))
print("  PR:   " + " ".join("%6.1f" % v for v in np.median(P, 0)))
print("\n🔒 МАСШТАБУВАННЯ N/4 (шари 1,8,14): %s" % " ".join("%.4f" % v for v in np.median(H2, 0)))
print("   проти повного N:                  %s"
      % " ".join("%.4f" % v for v in [np.median(A, 0)[i] for i in (1, 8, 14)]))
log("ГОТОВО")
