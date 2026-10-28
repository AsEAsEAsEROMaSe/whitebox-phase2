"""lamdeep92 — λ-РОДИНИ НА ВСІХ 31 ШАРАХ, А НЕ НА 8. ГЛИБОКЕ ЗАМИКАННЯ ВЖЕ ОПЛАЧЕНЕ.

════════════════ ЗВІДКИ ІДЕЯ ════════════════
Сабміт застосовує чотири λ-родини (δσ, δμ, δE[a²], δмаска) лише на шарах 1..8.
Причина в докстрінгу `estimator_sub87`: «замикання коштує 2 матмулі 256³ на шар
⟹ глибше не окупає». 🔑 **Ця причина померла в sub91**: там глибоке замикання
женеться крізь УСІ 32 шари заради бленду, тобто вже оплачене, а його пошарові
виходи на шарах 9..31 просто ВИКИДАЮТЬСЯ (`if 1 <= i <= ksig` у
`_closure_parts`).
```
ціна розширення = 4 редукції на шар (4·2N·256) проти матмулу (2·2N·256²)
                = 1/128 вартості шару ⟹ ~0.09% B флопами
                + ~160 викликів fnp × 3.0–4.7 мкс      ⟹ ~0.03% B
```

════════════════ ЧОМУ САМЕ ЦЕ, А НЕ ЗАМИКАННЯ ════════════════
Калібровка `MSE = V·E²/(V+E²)`, `V=2.050e-06`, `E²=2.118e-05` (відтворює рекорд
1.8691e-07 проти 1.86762e-07, 0.08%):
```
замикання ×k  ⟹ виграш = (k+10.33)/11.33   ⟹ d/dk = 0.088
дисперсія МК ÷j ⟹ виграш = (1+10.33j)/11.33 ⟹ d/dj = 0.912   ← у 10.33× важче
щоб побити лідера: j = 3.9   (або k = 31.2)
```
λ-родини б'ють саме по `V`.

════════════════ ДВА ВИПРАВЛЕННЯ, ПРО ЯКІ ПОПЕРЕДЖАЄ ПАМ'ЯТЬ ════════════════
1. 🚨 `sub91_port.py` несе `W_BLEND = 0.0651` (від sub91), а кандидати мають
   **0.0883**. С84 на цьому міряла не ту базу
   ([[feedback-new-stand-inherits-constants-not-just-truth]]). Перевизначаю.
2. 🚨 `closure91` зберігає `s/m/q/P` лише для `1..ksig` ⟹ кличу з
   `ksig = depth-1`, щоб дістати всі. `m_deep` від цього не міняється
   (ранній `break` спрацьовує лише при `i > ksig`) — пломба P2.

════════════════ ЩО ПОРІВНЮЄТЬСЯ ════════════════
```
BASE   λ зашиті в сабміті, шари 1..8            ← нинішній стан
FIT8   λ ПЕРЕФІТЕНІ на офіційних, шари 1..8     ← скільки дає сам новий прилад
FIT31  λ перефітені, шари 1..31                 ← ідея
DEEP   λ лише на шарах 9..31                    ← чи глибина несе НОВЕ
⛔ЧУЖА R від ІНШОЇ мережі                        ← контроль
```
Усе — гребенева регресія з **LOO ПО МЕРЕЖАХ** (одиниця незалежності — мережа).

════════════════ ПЛОМБИ ════════════════
P1 ГРАДУЮВАННЯ: BASE проти офіційної істини мусить дати відому відносну
   похибку порту. Розбіжність порту з грейдером (2.13e-06 проти 1.863e-06)
   ВІДОМА і відкрита — фіксую число, висновки роблю лише з ВІДНОШЕНЬ.
P2 `m_deep` при `ksig=depth-1` ≡ `m_deep` при `ksig=8` (машинно).
P3 FIT8 із гребенем→∞ мусить збігтися з «λ=0», а не з BASE.
P4 ⛔ЧУЖА мусить дати ×≤1.00.
P5 λ=0 (lam_off) — окремий рядок: показує, скільки дають родини взагалі.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) FIT31 поб'є FIT8 щонайменше в 1.15× по MSE          шанс 55%
 (б) більша частина виграшу прийде з шарів 25..31         шанс 60%
 (в) FIT8 сам по собі дасть ×1.05 над BASE (новий прилад) шанс 50%
 (г) підсумковий `j` не дійде до 3.9                      шанс 90%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

sys.path.insert(0, "/mnt/win_d/llm/White-Box/side66")
import sub91_port as P                                        # noqa: E402

P.W_BLEND = 0.0883                                            # 🚨 виправлення 1
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DEV = sys.argv[2] if len(sys.argv) > 2 else "cpu"
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "mini"
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/" + SPLIT)


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        fm = np.asarray(t.column("final_means").combine_chunks().flatten()
                        ).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i].astype(np.float64)
            got += 1


COL = list(range(31))
BASE, ZERO, DEEPM, TRUTH, RR = [], [], [], [], []
p2 = 0.0
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        wc, npair, rng = P.cut_budget91(W, DEV)
        depth = len(wc)
        cl_all = P.closure91(wc, depth - 1, DEV, kdeep=depth)
        if q == 0:                                           # ⛔P2
            cl_8 = P.closure91(wc, P.KSIG, DEV, kdeep=depth)
            p2 = float((cl_all[4] - cl_8[4]).abs().max())
            NP0 = npair
        st = rng.bit_generator.state
        pb, _ = P.predict91(wc, npair, DEV, rng=rng, cl=cl_all)      # λ зашиті
        rng.bit_generator.state = st
        pz, R = P.predict91(wc, npair, DEV, rng=rng, cl=cl_all,
                            collect=COL, lam_off=True)               # λ=0 + R
        BASE.append(pb.cpu().numpy())
        ZERO.append(pz.cpu().numpy())
        DEEPM.append(cl_all[4].cpu().numpy())
        RR.append(R.cpu().numpy())
        TRUTH.append(truth)
        if q % 25 == 0:
            print(f"  net{q:>4d} n_pairs={npair} ({time.time()-t0:.0f} с)",
                  flush=True)

BASE = np.array(BASE); ZERO = np.array(ZERO); DEEPM = np.array(DEEPM)
TRUTH = np.array(TRUTH); RR = np.array(RR)          # (Q, 124, 256)
Q = len(TRUTH)
print(f"\n  час {time.time()-t0:.0f} с · {Q} мереж · n_pairs={NP0} · {DEV}")
print(f"⛔P2 m_deep(ksig=31) ≡ m_deep(ksig=8): max|Δ| = {p2:.3e} "
      f"{'✅' if p2 < 1e-12 else '🚨'}")

W_B, AL = P.W_BLEND, P.ALPHA
# pr(λ) = (1−w)·(ours0 − Σλ_k R_k) + w(1+α)m_deep ;  ours0 = ZERO до бленду
ours0 = (ZERO - W_B * (1.0 + AL) * DEEPM) / (1.0 - W_B)
const = (1.0 - W_B) * ours0 + W_B * (1.0 + AL) * DEEPM
TGT = (const - TRUTH).reshape(Q, 256)               # ціль регресії
DES = -(1.0 - W_B) * RR                             # (Q,124,256)


def fit(cols, ridge, foreign=False):
    """LOO по мережах через ЗНЯТТЯ внеску мережі з глобальної матриці Грама.

    `G_(−q) = G − D_q D_qᵀ`, `b_(−q) = b − D_q t_q` ⟹ один розклад 124×124 на
    мережу замість перезбирання всієї системи. Інакше 600 мереж не порахувати.
    """
    D = DES[:, cols, :]
    if foreign:
        D = np.roll(D, 1, axis=0)
    p = len(cols)
    Gq = np.einsum('qik,qjk->qij', D, D)          # (Q,p,p)
    bq = np.einsum('qik,qk->qi', D, TGT)          # (Q,p)
    G, b = Gq.sum(0), bq.sum(0)
    tr = ridge * np.trace(G) / p
    err = np.zeros((Q, 256))
    for q in range(Q):
        coef = np.linalg.solve(G - Gq[q] + tr * np.eye(p), b - bq[q])
        err[q] = TGT[q] - D[q].T @ coef
    return float((err * err).mean())


mse_base = float(((BASE - TRUTH) ** 2).mean())
mse_zero = float(((ZERO - TRUTH) ** 2).mean())
z = float((TRUTH * TRUTH).mean())
print(f"\n🔒 P1 ГРАДУЮВАННЯ: BASE (λ зашиті, w={W_B}) MSE = {mse_base:.6e} · "
      f"відн {100*np.sqrt(mse_base/z):.4f}%")
print(f"   λ=0 (родини вимкнені)        MSE = {mse_zero:.6e} ⟹ родини дають "
      f"×{mse_zero/mse_base:.3f}")

C8 = [4 * k + f for k in range(8) for f in range(4)]
C31 = list(range(124))
CD = [4 * k + f for k in range(8, 31) for f in range(4)]
print("\n" + "=" * 96)
print(f"λ-РОДИНИ: 8 ШАРІВ ПРОТИ 31 · гребінь скановано · LOO по {Q} мережах")
print("=" * 96)
print(f"{'гребінь':>10s}{'FIT8':>14s}{'FIT31':>14s}{'DEEP(9..31)':>15s}"
      f"{'⛔ЧУЖА31':>14s}")
best = {}
for rg in (1e-4, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0):
    r8, r31 = fit(C8, rg), fit(C31, rg)
    rd, rf = fit(CD, rg), fit(C31, rg, foreign=True)
    for nm, v in (('FIT8', r8), ('FIT31', r31), ('DEEP', rd), ('ЧУЖА', rf)):
        if nm not in best or v < best[nm][0]:
            best[nm] = (v, rg)
    print(f"{rg:>10.0e}{r8:>14.5e}{r31:>14.5e}{rd:>15.5e}{rf:>14.5e}")

print("\n" + "=" * 96)
print("📣 ВИРОК — ПЕРЕКЛАД У БАЛ")
print("=" * 96)
# 🚨 `mse_*` — ПОВНІ MSE прогнозу ⟹ виграш на балі = їх ВІДНОШЕННЯ.
#    Формулу (1+10.33j)/11.33 сюди застосовувати НЕ МОЖНА — це подвійний
#    рахунок: вона переводить `j` (коефіцієнт при V) у повний MSE, а він тут
#    уже зміряний. `j` показую окремо, лише щоб бачити, наскільки це від V.
print(f"{'варіант':>12s}{'MSE':>14s}{'бал × над BASE':>17s}{'екв. j по V':>14s}"
      f"{'бал':>13s}")
V_, E2_ = 2.050e-06, 2.118e-05
for nm in ('FIT8', 'FIT31', 'DEEP', 'ЧУЖА'):
    v, rg = best[nm]
    k = mse_base / v
    j = (k * (V_ + E2_) - V_) / E2_
    print(f"{nm:>12s}{v:>14.5e}{k:>17.4f}{j:>14.3f}"
          f"{1.86762e-07/k:>13.4e}   гребінь {rg:.0e}")
print(f"\n   потрібно бал ×3.66 (j=3.9) щоб побити лідера 5.102e-08")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/lamdeep92.npz",
         BASE=BASE, ZERO=ZERO, DEEPM=DEEPM, TRUTH=TRUTH, RR=RR)
