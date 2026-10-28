"""mom4_102 — ЧЕТВЕРТІ МОМЕНТИ ВИБІРКИ: ОСТАННЯ ТОЧНО ВІДОМА ІСТИНА.

ІДЕЯ ЮЗЕРА («думка натовпу»): зробити вибірку РЕПРЕЗЕНТАТИВНОЮ в тому, що ми
знаємо напевно. Вибілювання вже робить це для ДРУГИХ моментів входу (32896
ступенів свободи, ×2.34). Але для гаусового `z` ми знаємо точно і ЧЕТВЕРТІ:

    E[(u·z)⁴] = 3‖u‖⁴        для БУДЬ-ЯКОГО напрямку u — ТОЧНА ТОТОЖНІСТЬ

Повний тензор 256⁴ = 4.3e9 неможливий, але вздовж ВИБРАНИХ напрямків — дешево.
Напрямки очевидні: стовпці `W₀`, бо саме вони визначають, що бачить шар 1.

🚦 ГЕЙТ, ЯКИЙ ЦЯ ІДЕЯ ПРОХОДИТЬ, А ПОПЕРЕДНІ ДВІ — НІ:
   цільова величина відома ТОЧНО (тотожність), а не взята із замикання.
   Саме на цьому стоїть λ-блок (E[h₁]=‖w‖/√2π) і саме тому він єдиний
   окупається ×10.9.

ЩО МІРЯЄМО, ПО КОЖНІЙ МЕРЕЖІ:
   для NREP незалежних вибірок бюджету сабміту (антитетичні + ВИБІЛЕНІ):
     e4_j = mean((u_j·z)⁴)/(3‖u_j‖⁴) − 1     надлишок 4-го моменту, 256 чисел
     e2_j = mean((u_j·z)²)/‖u_j‖² − 1        🔒 ПЛОМБА: після вибілювання ≈0
     X    = (out_MC − truth)·t̂               ЗНАКОВА проєкція похибки виходу
     X2   = ‖out_MC − truth‖²                модуль похибки
   і рахуємо corr по вибірках ВСЕРЕДИНІ мережі (мережа фіксована ⟹ чистий
   вплив вибірки, а не властивостей мережі).

🔒 ПЛОМБИ / КОНТРОЛІ:
   (1) e2 мусить бути ~1e-15 (вибілювання виправляє другі моменти ТОЧНО)
   (2) БЕЗ вибілювання e2 має бути ~1/√N ⟹ прилад бачить різницю
   (3) 🔒 ЧУЖА: e4 з ІНШОЇ вибірки проти нашої похибки ⟹ corr ≈ 0
   (4) істина — незалежна МК 4.19e6 ⟹ її шум ЗАНИЖУЄ corr, не завищує

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 e2 після вибілювання < 1e-12                                 90%
 (б) e4 після вибілювання лишиться помітним (~1/√N ≈ 1%)             80%
 (в) |corr(mean e4, X)| > 0.3                                        30%
 (г) 🔒 чужа дасть |corr| < 0.1                                      85%
 (д) багатовимірний R² (кілька агрегатів e4) > 0.25                  25%

ВИРОК: (в) < 0.3 І (д) < 0.25 ⟹ четверті моменти не несуть інформації про
похибку виходу ⟹ ОСТАННЯ точно відома істина вичерпана, і клас
«зробити натовп репрезентативнішим» закрито ПОВНІСТЮ.
"""
import glob
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 40
NPAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 3072      # пар = бюджет сабміту
NTRUE = int(sys.argv[4]) if len(sys.argv) > 4 else 2097152

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"mom4_102 · {NNET} мереж · {NREP} вибірок × {NPAIR} пар · істина {NTRUE}",
      flush=True)


def draw(rng, npair, whiten=True):
    """вибірка сабміту: антитетичні пари + вибілювання"""
    z = rng.standard_normal((npair, W_), dtype=cp.float32)
    z = cp.concatenate([z, -z], axis=0).astype(cp.float64)
    if whiten:
        cov = (z.T @ z) / z.shape[0]
        wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
        z = z @ wh
    return z


def fwd(Wg, z):
    h = z.astype(cp.float32)
    for i in range(DEPTH):
        h = cp.maximum(h @ Wg[i], 0.0)
    return h.astype(cp.float64).mean(0)


rows = []
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    U = cp.asarray(np.ascontiguousarray(Wall[net][0], dtype=np.float64))  # (256,256)
    un2 = (U * U).sum(0)                    # ‖u_j‖²
    # істина — незалежна велика МК
    rng0 = cp.random.default_rng(50000 + net)
    acc = cp.zeros(W_, dtype=cp.float64)
    n = 0
    while n < NTRUE // 2:
        nb = min(8192, NTRUE // 2 - n)
        acc += fwd(Wg, draw(rng0, nb)) * (2 * nb)
        n += nb
    truth = acc / NTRUE
    th = truth / cp.sqrt((truth * truth).sum())

    E4, E2, X, X2 = [], [], [], []
    for r in range(NREP):
        rng = cp.random.default_rng(900000 + 1000 * net + r)
        z = draw(rng, NPAIR)
        p = z @ U                                    # (N, 256) проєкції u_j·z
        m2 = (p * p).mean(0)
        m4 = (p ** 4).mean(0)
        E2.append(cp.asnumpy(m2 / un2 - 1.0))
        E4.append(cp.asnumpy(m4 / (3.0 * un2 * un2) - 1.0))
        o = fwd(Wg, z)
        d = o - truth
        X.append(float((d * th).sum()))
        X2.append(float((d * d).sum()))
        del z, p
    E4 = np.array(E4); E2 = np.array(E2)
    X = np.array(X); X2 = np.array(X2)
    y1 = E4.mean(1)                                   # загальний надлишок
    y2 = (E4 * np.sqrt(cp.asnumpy(un2))[None, :]).mean(1)   # зважений на ‖u‖
    y3 = np.abs(E4).mean(1)                           # модуль
    ych = np.roll(y1, 1)                              # 🔒 ЧУЖА вибірка
    rows.append(dict(net=net, e2=np.abs(E2).mean(), e4=np.abs(E4).mean(),
                     c1=np.corrcoef(y1, X)[0, 1], c2=np.corrcoef(y2, X)[0, 1],
                     c3=np.corrcoef(y3, X2)[0, 1], cch=np.corrcoef(ych, X)[0, 1],
                     r2=np.corrcoef(np.c_[y1, y2, y3] @ np.linalg.lstsq(
                         np.c_[y1, y2, y3, np.ones(NREP)],
                         X, rcond=None)[0][:3], X)[0, 1] ** 2,
                     sd=X.std(), rms=float(cp.sqrt((truth * truth).mean()))))
    print(f"  мережа {net}: |e2|={rows[-1]['e2']:.2e} |e4|={rows[-1]['e4']:.4f} "
          f"corr={rows[-1]['c1']:+.3f} 🔒чужа={rows[-1]['cch']:+.3f}", flush=True)
    del Wg, U
    cp.get_default_memory_pool().free_all_blocks()

print("\n" + "=" * 74)
g = lambda k: np.array([r[k] for r in rows])
print(f"🔒(а) |e2| після вибілювання  {g('e2').mean():.3e}   (мусить бути ~1e-15)")
print(f"  (б) |e4| після вибілювання  {g('e4').mean():.5f}   "
      f"(очікувано ~1/√N = {1/math.sqrt(2*NPAIR):.5f})")
print(f"🚨(в) corr(mean e4, ЗНАКОВА похибка)     {g('c1').mean():+.4f} "
      f"± {g('c1').std():.4f}")
print(f"      corr(зважений e4, похибка)         {g('c2').mean():+.4f}")
print(f"      corr(|e4|, КВАДРАТ похибки)        {g('c3').mean():+.4f}")
print(f"🔒(г) ЧУЖА вибірка                       {g('cch').mean():+.4f} "
      f"± {g('cch').std():.4f}")
print(f"  (д) R² трьох агрегатів                 {g('r2').mean():.4f}")
print("\nВИРОК: (в) < 0.3 і (д) < 0.25 ⟹ четверті моменти не несуть інформації")
print("       про похибку виходу ⟹ ОСТАННЯ точно відома істина вичерпана")
