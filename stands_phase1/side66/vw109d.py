"""vw109d — КУТ 9: чи породжує КРИТИЧНІСТЬ ранг ансамбля віртуальних ваг.

Замикає петлю з трьох окремих результатів корпусу в одну криву:

    критичність (50% затворів) → високий ранг ансамбля → замикання провалюється

Ланка 3 вже зміряна (`vw109b`: corr(ранг, косинусна похибка) = +0.80 на 50
мережах). Ланка 1 ніколи не мірялась — тут вона.

ДВІ ОСІ ВІДХОДУ ВІД КРИТИЧНОСТІ, обидві з корпусу:
  (а) частка від'ємних ваг: перевертаємо частку `f` від'ємних у додатні.
      f=1 ⟹ всі ваги додатні ⟹ мережа ЛІНІЙНА з шару 2 ⟹ ранг мусить впасти в 1.
  (б) біас `b`: зсуває частку вимкнених від 50%. Корпус: b=0.15 робить задачу
      вшестеро легшою, а екстраполяція з [0.15,1.5] назад недооцінює похибку ×7.1
      ⟹ b=0 є КАСПОМ. Тут перевіряється, чи видно касп у РАНЗІ.

🔒 ПЛОМБИ
 D1 f=0 мусить відтворити ранг із `vw109` (та сама мережа, той самий прилад)
 D2 f=1 мусить дати ранг ≈1 і частку відкритих затворів ≈1 з шару 2
 D3 частка відкритих при f=0 мусить дати 0.49…0.50 на КОЖНОМУ шарі

📣 ПЕРЕДБАЧЕННЯ: ранг падає МОНОТОННО по обох осях (85%); падіння по `b` між 0
   і 0.15 різкіше за наступні кроки, тобто касп видно і в ранзі (50%).
"""
import glob
import json
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
NPROBE = 16
FS = (0.0, 0.25, 0.5, 0.75, 1.0)
BS = (0.0, 0.05, 0.15, 0.5, 1.5)  # у одиницях σ шару
OUT = "/home/trim/wblogs/vw109d.json"

import cupy as cp                                             # noqa: E402

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 ВИДНО НЕ ОДНУ КАРТУ — вихід.", flush=True)
    sys.exit(2)

Wall = []
for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    xx = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xx.type):
        xx = xx.flatten()
    Wall.append(np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_))
Wall = np.concatenate(Wall, 0)
print(f"vw109d · {NNET} мереж · N={NSAMP} · проб={NPROBE}", flush=True)


def renorm(W, z):
    """🚨 Перевертання знаків лишає |w| тим самим, але робить середнє ваги
    НЕНУЛЬОВИМ ⟹ сума 256 додатних доданків вибухає і fp32 дає nan. Це та сама
    пастка, що з обрізанням ширини: без перенормування міняється МАСШТАБ, а не
    досліджувана вісь. Тут кожен шар ділиться на зміряний коефіцієнт підсилення
    норми, тож мережа лишається одиничного підсилення, як при He-ініціалізації."""
    a = z[:512]
    out = []
    for L in range(DEPTH):
        p = a @ W[L]
        g = float(cp.mean(cp.sqrt(cp.sum(p * p, 1)))
                  / cp.maximum(cp.mean(cp.sqrt(cp.sum(a * a, 1))), 1e-30))
        Wn = W[L] / max(g, 1e-20)
        out.append(Wn)
        a = cp.maximum(a @ Wn, 0.0)
    return out


def ens_rank(W, z, bias, U, rel=False):
    """ранг ансамбля віртуальних ваг + частка відкритих затворів по шарах."""
    masks, a, frac = [], z, []
    for i in range(DEPTH):
        p = a @ W[i]
        # rel=True ⟹ біас у одиницях ВЛАСНОГО масштабу шару: сталий біас на
        # глибині нічого не зсуває, бо масштаб передактивацій росте
        p = p + (bias * float(p.std()) if rel else bias)
        mk = p > 0
        masks.append(mk)
        frac.append(float(mk.mean()))
        a = p * mk
    v = cp.broadcast_to(U[None], (NSAMP, NPROBE, W_)).copy()
    for i in range(DEPTH):
        v = (v.reshape(-1, W_) @ W[i]).reshape(NSAMP, NPROBE, W_)
        if bias != 0.0:
            v = v * masks[i][:, None, :]
        else:
            v = v * masks[i][:, None, :]
    X = v.reshape(NSAMP, NPROBE * W_)
    X = X - X.mean(0)
    S = (X.T @ X) / NSAMP
    tr = float(cp.trace(S))
    pr = tr * tr / max(float(cp.sum(S * S)), 1e-300)
    del v, X, S
    return pr, frac


rows = {"f": {}, "b": {}}
for net in range(NNET):
    rng = cp.random.default_rng(9109 + net)
    z = rng.standard_normal((NSAMP, W_), dtype=cp.float32)
    U = rng.standard_normal((NPROBE, W_), dtype=cp.float32)
    U /= cp.sqrt(cp.sum(U * U, 1, keepdims=True))
    W0 = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float32))
          for L in range(DEPTH)]
    # вісь (а): перевертаємо частку f від'ємних ваг у додатні
    rs = cp.random.default_rng(555 + net)
    U01 = [rs.random((W_, W_), dtype=cp.float32) for _ in range(DEPTH)]
    for f in FS:
        W = renorm([cp.where((W0[L] < 0) & (U01[L] < f), -W0[L], W0[L])
                    for L in range(DEPTH)], z)
        pr, fr = ens_rank(W, z, 0.0, U)
        rows["f"].setdefault(str(f), []).append((pr, fr[1], fr[8], fr[31]))
        del W
    # вісь (б): біас
    for b in BS:
        pr, fr = ens_rank(W0, z, b, U, rel=True)
        rows["b"].setdefault(str(b), []).append((pr, fr[1], fr[8], fr[31]))
    print(f"  ✅ мережа {net}", flush=True)
    cp.get_default_memory_pool().free_all_blocks()

json.dump(rows, open(OUT, "w"))
print("\n🎯 ВІСЬ (а): ЧАСТКА ВІД'ЄМНИХ ВАГ, ПЕРЕВЕРНУТИХ У ДОДАТНІ")
print("   f      ранг ансамбля      відкритих затворів (шар 1 · 8 · 31)")
for f in FS:
    a = np.array(rows["f"][str(f)])
    print(f"  {f:.2f}      {a[:,0].mean():8.1f}         "
          f"{a[:,1].mean():.3f} · {a[:,2].mean():.3f} · {a[:,3].mean():.3f}")
print("\n🎯 ВІСЬ (б): БІАС У ОДИНИЦЯХ σ ШАРУ")
print("   b      ранг ансамбля      відкритих затворів (шар 1 · 8 · 31)")
for b in BS:
    a = np.array(rows["b"][str(b)])
    print(f"  {b:.2f}      {a[:,0].mean():8.1f}         "
          f"{a[:,1].mean():.3f} · {a[:,2].mean():.3f} · {a[:,3].mean():.3f}")
r0 = np.array(rows["f"]["0.0"])[:, 0].mean()
r1 = np.array(rows["f"]["1.0"])[:, 0].mean()
print(f"\n🔒 D1 f=0 ранг {r0:.1f}  (vw109 на тих самих мережах давав ~200)")
print(f"🔒 D2 f=1 ранг {r1:.2f}  (мусить бути ≈1: мережа лінійна)")
print(f"🔒 D3 частка відкритих при f=0: "
      f"{np.array(rows['f']['0.0'])[:,1:].mean():.4f} (мусить бути ≈0.49)")
