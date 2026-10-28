"""mix102d — ВИРІШАЛЬНИЙ: чи виживе ×1.41 при фіті проти ВЛАСНОЇ МК?

ЛАНЦЮГ: mix102b дав оракульну стелю ×1.4149 для поправки до `m` вздовж
напрямків змішування (🔒чужі напрямки ×1.018 ⟹ контроль зламав).
mix102c показав: `a_k` ІНДИВІДУАЛЬНІ (|сер|/sd 0.06…0.61, corr із усіма
кандидатами ≤0.05, крос = 🔒rand) ⟹ спільної константи не буде.
АЛЕ `_ai` у сабміті теж індивідуальний — він фітується проти власної МК.

ПИТАННЯ: `a_k` фітуються проти `mmc` (МК з бюджетом сабміту), вимір — проти
ІСТИНИ. Скільки з ×1.41 виживе?
🚨 ПАСТКА, ЯКУ МИ ВЖЕ ЛОВИЛИ (С100): фіт проти `ours` дає зміщення, бо `ours`
містить control variate, ПОБУДОВАНУ ІЗ ЗАМИКАННЯ ⟹ фітуємо проти СИРОЇ МК.
🚨 ЗАКОН С99: оптимальна поправка СТИСНЕНА ⟹ скануємо shrink, не беремо 1.0.

ЦІНА РЕАЛІЗАЦІЇ (рахувати ДО, не після):
    eigh на Ĉ фінального шару — ОДИН раз, 1.51e8 = 0.55% підлоги
    проєкції K×256 + МНК (K+1)² — дрібниця
⟹ ≈0.6% бюджету, множник 0.1004 → 0.1010, тобто −0.6% балу
⟹ гейт: чиста поправка мусить дати > ×1.01 у похибці, щоб окупитись

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 K=0 відтворить базу (пломба)                                   95%
 (б) фіт проти МК збереже >50% оракульного виграшу при K=4             40%
 (в) оптимальний shrink буде < 1.0                                     80%
 (г) 🔒 фіт проти ПЕРЕМІШАНОЇ МК (чужа мережа) зламає                  85%
 (д) оптимальне K буде 2…4, не 8 (шум з'їдає хвіст)                    65%

ВИРОК: якщо найкраще (K, shrink) дає < ×1.01 проти істини ⟹ гілка мертва
за шумом МК, і це остаточно для всього класу «поправка до m у власному
базисі C».
"""
import glob
import math
import os
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
CACHE = "/home/trim/mix102c_cache.npz"
MCCACHE = "/home/trim/mix102d_mc.npz"
NPAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
KS = [0, 1, 2, 3, 4, 6, 8]
SHR = [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]

d = np.load(CACHE)
MH, MT, VV, LM = d["MH"], d["MT"], d["VV"], d["LM"]
NNET = MH.shape[0]

if not os.path.exists(MCCACHE):
    if cp.cuda.runtime.getDeviceCount() != 1:
        sys.exit(2)
    tb = pa.ipc.open_stream(pa.memory_map(
        sorted(glob.glob(SRC + "/data-*.arrow"))[0], "rb")).read_all()
    xx = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xx.type):
        xx = xx.flatten()
    Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
    print(f"mix102d · МК {NPAIR} пар × {NREP} реплік для {NNET} мереж",
          flush=True)

    def draw(seed, npair):
        rng = cp.random.default_rng(seed)
        z = rng.standard_normal((npair, W_), dtype=cp.float32)
        z = cp.concatenate([z, -z], axis=0).astype(cp.float64)
        cov = (z.T @ z) / z.shape[0]
        wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
        return (z @ wh).astype(cp.float32)

    MC = np.zeros((NNET, NREP, W_))
    for net in range(NNET):
        Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
              for i in range(DEPTH)]
        for r in range(NREP):
            h = draw(1200000 + net * 4523 + r * 91, NPAIR)
            for i in range(DEPTH):
                h = cp.maximum(h @ Wg[i], 0.0)
            MC[net, r] = cp.asnumpy(h.astype(cp.float64).mean(0))
        del Wg
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  мережа {net} готова", flush=True)
    np.savez(MCCACHE, MC=MC)

MC = np.load(MCCACHE)["MC"]
print(f"mix102d · {NNET} мереж · МК {MC.shape[1]} реплік", flush=True)
mce = np.mean([100.0 * math.sqrt(float((MC[n, r] - MT[n]) @ (MC[n, r] - MT[n]))
                                 / float(MT[n] @ MT[n]))
               for n in range(NNET) for r in range(MC.shape[1])])
print(f"🔒 похибка самої МК {NPAIR} пар: {mce:.4f}%  "
      f"(записане для сабміту 0.1454%)")


def basis(n, K, V):
    B = [MH[n]]
    for k in range(K):
        B.append(float(V[:, k] @ MH[n]) * V[:, k])
    return np.stack(B, 1)


def rel(v, n):
    return 100.0 * math.sqrt(float(v @ v) / float(MT[n] @ MT[n]))


def run(K, sh, foreign=False):
    e = []
    for n in range(NNET):
        V = VV[(n + 1) % NNET] if foreign else VV[n]
        B = basis(n, K, V)
        for r in range(MC.shape[1]):
            a, *_ = np.linalg.lstsq(B, MC[n, r], rcond=None)
            a0, *_ = np.linalg.lstsq(B[:, :1], MC[n, r], rcond=None)
            af = np.concatenate([a0, np.zeros(K)])
            v = B @ (af + sh * (a - af))
            e.append(rel(v - MT[n], n))
    return float(np.mean(e))


base = run(0, 0.0)
print("\n" + "=" * 76)
print(f"ФІТ ПРОТИ ВЛАСНОЇ МК, вимір проти ІСТИНИ   [база {base:.4f}%]")
print(f"{'K':>3}" + "".join(f"{f'sh={s}':>11}" for s in SHR))
print("-" * 76)
BEST = (1.0, 0, 0.0)
for K in KS:
    row = []
    for sh in SHR:
        e = run(K, sh)
        row.append(base / e)
        if base / e > BEST[0]:
            BEST = (base / e, K, sh)
    print(f"{K:>3}" + "".join(f"{x:>11.4f}" for x in row))
print("-" * 76)
g, K, sh = BEST
print(f"НАЙКРАЩЕ: K={K} shrink={sh} ⟹ ×{g:.4f}")
if K > 0:
    ef = run(K, sh, foreign=True)
    print(f"🔒 КОНТРОЛЬ «чужі напрямки» на тих самих (K,shrink): "
          f"{ef:.4f}% = ×{base/ef:.4f}  (мусить бути ≈1.00 або гірше)")
    per = [np.mean([rel(basis(n, K, VV[n]) @ (
        (lambda a, a0: np.concatenate([a0, np.zeros(K)])
         + sh * (a - np.concatenate([a0, np.zeros(K)])))(
            np.linalg.lstsq(basis(n, K, VV[n]), MC[n, r], rcond=None)[0],
            np.linalg.lstsq(basis(n, K, VV[n])[:, :1], MC[n, r],
                            rcond=None)[0])) - MT[n], n)
        for r in range(MC.shape[1])]) for n in range(NNET)]
    b0 = [np.mean([rel(basis(n, 0, VV[n]) @ np.linalg.lstsq(
        basis(n, 0, VV[n]), MC[n, r], rcond=None)[0] - MT[n], n)
        for r in range(MC.shape[1])]) for n in range(NNET)]
    win = sum(1 for n in range(NNET) if per[n] < b0[n])
    print(f"ЗНАКОВИЙ по мережах: {win}/{NNET}")
e2 = (base / g) ** 2 / base ** 2
V, E2 = 1.9227e-06, 2.3385e-05
new = V * (E2 / g ** 2) / (V + E2 / g ** 2)
print(f"\nУ БАЛІ: E² ×{g**2:.4f} ⟹ MSE {V*E2/(V+E2):.4e} → {new:.4e} "
      f"= ×{(V*E2/(V+E2))/new:.4f}  (мінус ціна eigh 0.6%)")
