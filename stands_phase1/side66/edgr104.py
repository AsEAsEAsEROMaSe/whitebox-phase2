"""edgr104 — ЧИ Є ПОХИБКА КРОКУ ЕДЖВОРТІВСЬКОЮ НАСПРАВДІ (пункт 5 аудиту).

ПІД ПИТАННЯМ ГОЛОВНИЙ ВИСНОВОК: `equil104b` показав cos(b, b_Еджворт) = −0.03…+0.06
і з цього виведено «похибка кроку НЕ є негаусовістю 3-4 порядку» — а це, своєю
чергою, пояснює, чому всі γ-ідеї повертались із ×1.0x.

🚨 АЛЕ ЕДЖВОРТ ТАМ БРАВСЯ З *ТАБЛИЧНИМИ* γ — по ОДНОМУ числу на шар, з корпусної
кривої. Якщо γ помітно варіює МІЖ НЕЙРОНАМИ, табличний варіант губить усю
структуру, і ортогональність була б артефактом таблиці, а не фактом про задачу.

ЩО РОБИМО: рахуємо γ₃, γ₄ ПОНЕЙРОННО з власних семплів і порівнюємо ТРИ версії
еджвортівського члена з реальною похибкою кроку `b`:
    (t) табличні γ (як в equil104b)      — відтворення
    (r) РЕАЛЬНІ понейронні γ             — головне питання
    (m) реальні γ, але гаусовий крок з gam=1 (без нашої ручки на аргумент)

МЕТРИКИ: cos по шарах · частка поясненої дисперсії R² · оптимальний коефіцієнт
(закон корпусу: поправку міряти з ФІТОВАНИМ коефіцієнтом, не з вагою 1).

🔒 ПЛОМБИ: (1) шар 0: `p` там гаусів ТОЧНО ⟹ і `b`, і `edg` мусять бути ≈0
           (2) розкид γ між нейронами друкуємо явно — якщо він малий, питання
               знімається саме собою і табличний варіант виправданий
           (3) знак: `b = наш_крок − істина`, а Еджворт передбачає поправку ДО
               гаусового ⟹ очікуємо cos(b, edg) ≈ −1, якщо теорія вірна

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
   - реальні γ дадуть |cos| > 0.5 хоч на глибоких шарах:        40%
   - розкид γ₄ між нейронами виявиться > 50% від середнього:    55%
   - висновок equil104b встоїть (|cos| < 0.2 і з реальними γ):  50%
"""
import glob
import math
import sys
import time

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NPAIRS = 131072
CHUNK = 4096
KSIG, KAP, GAM = 8, 0.9985, 1.048
GX = [0, 1, 4, 8, 16, 24, 31]
G3T = np.interp(np.arange(DEPTH), GX,
                [0.003, 0.211, 0.302, 0.331, 0.411, 0.412, 0.398])
G4T = np.interp(np.arange(DEPTH), GX,
                [0.005, 0.049, 0.133, 0.125, 0.231, 0.239, 0.220])

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)


def stream(nmax):
    got = 0
    for fp in sorted(glob.glob(SRC + "/data-*.arrow")):
        if got >= nmax:
            break
        for b in pa.ipc.open_stream(pa.memory_map(fp, "rb")):
            if got >= nmax:
                break
            xw = b.column("weights")
            while not pa.types.is_primitive(xw.type):
                xw = xw.flatten()
            nb = b.num_rows
            wv = np.asarray(xw).reshape(nb, DEPTH, W_, W_)
            for j in range(min(nb, nmax - got)):
                yield np.array(wv[j], dtype=np.float32)
            got += min(nb, nmax - got)
            del xw, wv


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


ACC = {k: np.zeros(DEPTH) for k in ("ct", "cr", "cm", "r2t", "r2r", "kt",
                                    "kr", "g3sd", "g4sd", "g3m", "g4m",
                                    "bn", "n")}
t0 = time.time()
for net, Wn in enumerate(stream(NNET)):
    Wg = [cp.asarray(Wn[L]) for L in range(DEPTH)]
    W64 = [w.astype(cp.float64) for w in Wg]
    N = 2 * NPAIRS
    rng = cp.random.default_rng(1104 + net)
    S1 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    S2 = cp.zeros((DEPTH, W_, W_), dtype=cp.float64)
    P1 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    P2 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    P3 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    P4 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    O1 = cp.zeros(W_, dtype=cp.float64)
    done = 0
    while done < NPAIRS:
        nb = min(CHUNK, NPAIRS - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(DEPTH):
            a64 = a.astype(cp.float64)
            S1[i] += a64.sum(0)
            S2[i] += a64.T @ a64
            p = (a @ Wg[i]).astype(cp.float64)
            P1[i] += p.sum(0)
            q = p * p
            P2[i] += q.sum(0)
            P3[i] += (q * p).sum(0)
            P4[i] += (q * q).sum(0)
            a = cp.maximum(p, 0.0).astype(cp.float32)
            if i == DEPTH - 1:
                O1 += a.astype(cp.float64).sum(0)
        done += nb
    del z, a, a64, p, q
    cp.get_default_memory_pool().free_all_blocks()
    mt, out_t = S1 / N, O1 / N
    Ct = S2 / N - mt[:, :, None] * mt[:, None, :]
    Ct = 0.5 * (Ct + cp.transpose(Ct, (0, 2, 1)))
    # 🔑 РЕАЛЬНІ ПОНЕЙРОННІ γ передактивацій
    mu_ = P1 / N
    v = cp.maximum(P2 / N - mu_ * mu_, 1e-30)
    sdv = cp.sqrt(v)
    m3 = P3 / N - 3.0 * mu_ * (P2 / N) + 2.0 * mu_ ** 3
    m4 = (P4 / N - 4.0 * mu_ * (P3 / N) + 6.0 * mu_ * mu_ * (P2 / N)
          - 3.0 * mu_ ** 4)
    G3R = m3 / sdv ** 3
    G4R = m4 / sdv ** 4 - 3.0
    del S1, S2, P1, P2, P3, P4
    cp.get_default_memory_pool().free_all_blocks()

    for i in range(DEPTH):
        w = W64[i]
        cpm = w.T @ Ct[i] @ w
        if i > KSIG:
            cpm = cpm * KAP
        s2 = cp.maximum(cp.diagonal(cpm), 1e-30)
        s = cp.sqrt(s2)
        mp = mt[i] @ w
        tgt = out_t if i == DEPTH - 1 else mt[i + 1]

        def edg(g3, g4, gam):
            u = gam * (mp / s)
            ph = phi(u)
            return s * (-(g3 / 6.0) * u * ph
                        + (g4 / 24.0) * (u * u - 1.0) * ph)

        u = GAM * (mp / s)
        mn = mp * ndtr(u) + s * phi(u)
        b = mn - tgt
        u1 = mp / s
        mn1 = mp * ndtr(u1) + s * phi(u1)
        b1 = mn1 - tgt
        nb_ = float(cp.linalg.norm(b))
        for key, e, bb in (("t", edg(G3T[i], G4T[i], GAM), b),
                           ("r", edg(G3R[i], G4R[i], GAM), b),
                           ("m", edg(G3R[i], G4R[i], 1.0), b1)):
            ne = float(cp.linalg.norm(e))
            c = float(bb @ e) / max(nb_ * ne, 1e-300)
            ACC["c" + key][i] += c
            if key in ("t", "r"):
                # оптимальний коефіцієнт і частка поясненої дисперсії
                k = float(bb @ e) / max(float(e @ e), 1e-300)
                ACC["k" + key][i] += k
                res = bb - k * e
                ACC["r2" + key][i] += 1.0 - float(res @ res) / max(
                    float(bb @ bb), 1e-300)
        ACC["g3m"][i] += float(cp.mean(G3R[i]))
        ACC["g4m"][i] += float(cp.mean(G4R[i]))
        ACC["g3sd"][i] += float(cp.std(G3R[i]))
        ACC["g4sd"][i] += float(cp.std(G4R[i]))
        ACC["bn"][i] += nb_ / float(cp.linalg.norm(tgt))
        ACC["n"][i] += 1.0
    del Wg, W64, Ct, mt
    cp.get_default_memory_pool().free_all_blocks()
    print(f"  ✅ мережа {net} ({time.time()-t0:.1f} с)", flush=True)

n = ACC["n"]
print(f"\n{'шар':>4}{'‖b‖/‖m‖':>10}{'γ₄ сер':>8}{'γ₄ sd':>8}{'γ₃ сер':>8}"
      f"{'γ₃ sd':>8}{'cos табл':>10}{'cos РЕАЛ':>10}{'cos gam=1':>10}"
      f"{'R² реал':>9}")
for i in (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 31):
    print(f"{i:>4}{100*ACC['bn'][i]/n[i]:>9.3f}%{ACC['g4m'][i]/n[i]:>8.3f}"
          f"{ACC['g4sd'][i]/n[i]:>8.3f}{ACC['g3m'][i]/n[i]:>8.3f}"
          f"{ACC['g3sd'][i]/n[i]:>8.3f}{ACC['ct'][i]/n[i]:>10.4f}"
          f"{ACC['cr'][i]/n[i]:>10.4f}{ACC['cm'][i]/n[i]:>10.4f}"
          f"{ACC['r2r'][i]/n[i]:>9.4f}")
print("\n🔒 ПЛОМБА (1): шар 0 — `p` гаусів ТОЧНО ⟹ і b, і edg мусять бути ≈0")
print("🔒 ПЛОМБА (3): теорія передбачає cos ≈ −1 (b = наше − істина, а Еджворт")
print("   дає поправку ДО гаусового) ⟹ великий ВІД'ЄМНИЙ cos = Еджворт працює")
dp = np.mean([ACC["cr"][i] / n[i] for i in range(12, 32)])
dt = np.mean([ACC["ct"][i] / n[i] for i in range(12, 32)])
r2 = np.mean([ACC["r2r"][i] / n[i] for i in range(12, 32)])
print(f"\nсереднє по шарах 12..31:  cos табл {dt:+.4f} · cos РЕАЛ {dp:+.4f} · "
      f"R² реал {r2:+.4f}")
if abs(dp) > 0.5:
    print("🚨 ВИСНОВОК equil104b СПРОСТОВАНО: похибка кроку Є еджвортівською,")
    print("   а ортогональність була артефактом ТАБЛИЧНИХ γ")
elif abs(dp) > 2 * abs(dt) + 0.1:
    print("⚠️ ЧАСТКОВО: реальні γ дають помітно більше за табличні, але не все")
else:
    print("✅ ВИСНОВОК equil104b ВСТОЯВ: похибка кроку НЕ є негаусовістю 3-4")
    print("   порядку — і це не артефакт таблиці")
