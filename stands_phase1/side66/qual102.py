"""qual102 — ЧИ Є В ВИБІРКИ УНІВЕРСАЛЬНА «ЯКІСТЬ»? (зауваження ЮЗЕРА)

ЮЗЕР: «а може ти не застосовуєш цей надлишок вірно?» — і він має рацію.
ТЕОРІЯ: після наших виправлень похибка МК розкладається як
    mean(f)−E[f] = f₁·mean(z)        = 0 ТОЧНО  (антитетичні пари)
                 + f₂·(mean(z²)−I)   = 0 ТОЧНО  (вибілювання, зміряно 1e-15)
                 + f₃·mean(z³)       = 0 ТОЧНО  (антитетичні пари)
                 + f₄·(mean(z⁴)−E)   ≠ 0        ⟸ ЄДИНЕ, ЩО ЛИШИЛОСЬ
⟹ похибка МУСИТЬ визначатись четвертими моментами. mom4_102 дав corr 0.017,
   тобто НУЛЬ — отже помилка в ПРОЄКЦІЇ: я брав лише діагональні (u·z)⁴ по 256
   напрямках із рівними вагами, а потрібна згортка з ТЕНЗОРОМ f_ijkl, де
   змішані члени (u·z)²(v·z)² (їх 256²/2) можуть нести основну вагу.

ЗАМІСТЬ ВГАДУВАТИ ВАГУ — ПЕРЕВІРЯЄМО, ЧИ Є ЩО ЗВАЖУВАТИ:
   якщо вибірка «погана», вона мусить бути поганою ДЛЯ ВСІХ МЕРЕЖ.
   Беремо ОДНІ Й ТІ САМІ NREP вибірок і проганяємо через NNET РІЗНИХ мереж.
   corr похибок між мережами по вибірках = чи існує універсальна якість.

🔑 ЦЕ СИЛЬНІШИЙ ТЕСТ, НІЖ БУДЬ-ЯКА КОНКРЕТНА ОЗНАКА: він не залежить від
   того, як ми виражаємо «якість» — лише від того, чи вона взагалі існує.
   Якщо corr ≈ 0, то жодна ознака вибірки не допоможе, і шукати проєкцію
   немає сенсу. Якщо corr > 0.3 — важіль існує, і його варто виражати.

ЩО МІРЯЄМО:
   d[r,net] = (out_MC(вибірка r, мережа net) − truth[net]) · t̂[net]   ЗНАКОВА
   D[r,net] = ‖out_MC − truth‖²                                       МОДУЛЬ
   corr між СТОВПЦЯМИ (мережами) по рядках (вибірках)
   🔒 КОНТРОЛЬ: те саме для НЕЗАЛЕЖНИХ вибірок на кожну мережу ⟹ має дати 0

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 контроль (незалежні вибірки) дасть |corr| < 0.15                85%
 (б) знакова похибка: сер |corr| між мережами > 0.3                    25%
 (в) МОДУЛЬ похибки: сер corr > 0.3  (якість може бути ненаправленою)  40%
 (г) перша головна компонента матриці похибок пояснить >40% дисперсії  30%

ВИРОК: (б) і (в) < 0.2 ⟹ універсальної якості вибірки НЕМАЄ ⟹ похибка МК
є специфічною для мережі, і клас «відбирати/виправляти вибірку» закрито
ОСТАТОЧНО, незалежно від вибору ознаки.
"""
import glob
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 60
NPAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 3072
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
print(f"qual102 · {NNET} мереж · {NREP} СПІЛЬНИХ вибірок × {NPAIR} пар", flush=True)


def draw(seed, npair):
    rng = cp.random.default_rng(seed)
    z = rng.standard_normal((npair, W_), dtype=cp.float32)
    z = cp.concatenate([z, -z], axis=0).astype(cp.float64)
    cov = (z.T @ z) / z.shape[0]
    wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
    return (z @ wh).astype(cp.float32)


def fwd(Wg, z):
    h = z
    for i in range(DEPTH):
        h = cp.maximum(h @ Wg[i], 0.0)
    return h.astype(cp.float64).mean(0)


# СПІЛЬНІ вибірки — ті самі для всіх мереж (у цьому вся суть тесту)
Z = [draw(400000 + r, NPAIR) for r in range(NREP)]
# 🔒 КОНТРОЛЬ — незалежні вибірки на кожну мережу
Zc = {}

D = np.zeros((NREP, NNET))
A = np.zeros((NREP, NNET))
Dc = np.zeros((NREP, NNET))
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    acc = cp.zeros(W_, dtype=cp.float64)
    n = 0
    while n < NTRUE // 2:
        nb = min(8192, NTRUE // 2 - n)
        acc += fwd(Wg, draw(770000 + net * 97 + n, nb)) * (2 * nb)
        n += nb
    truth = acc / NTRUE
    th = truth / cp.sqrt((truth * truth).sum())
    for r in range(NREP):
        d = fwd(Wg, Z[r]) - truth
        A[r, net] = float((d * th).sum())
        D[r, net] = float((d * d).sum())
        zc = draw(600000 + net * 1000 + r, NPAIR)     # 🔒 незалежна
        dc = fwd(Wg, zc) - truth
        Dc[r, net] = float((dc * th).sum())
        del zc
    print(f"  мережа {net} готова", flush=True)
    del Wg
    cp.get_default_memory_pool().free_all_blocks()


def pairmean(M):
    C = np.corrcoef(M.T)
    iu = np.triu_indices(M.shape[1], 1)
    return C[iu].mean(), C[iu].std(), np.abs(C[iu]).mean()


ma, sa, aa = pairmean(A)
md, sd, ad = pairmean(np.log(D))
mc, sc, ac = pairmean(Dc)
w, _ = np.linalg.eigh(np.corrcoef(A.T))
print("\n" + "=" * 70)
print(f"(б) ЗНАКОВА похибка: сер corr між мережами  {ma:+.4f} ± {sa:.4f} "
      f"| сер |corr| {aa:.4f}")
print(f"(в) МОДУЛЬ (log ‖d‖²):  сер corr           {md:+.4f} ± {sd:.4f} "
      f"| сер |corr| {ad:.4f}")
print(f"🔒(а) КОНТРОЛЬ (незалежні вибірки)          {mc:+.4f} ± {sc:.4f} "
      f"| сер |corr| {ac:.4f}")
print(f"(г) 1-ша головна компонента пояснює         {100*w[-1]/w.sum():.1f}% "
      f"(ізотропний рівень {100/NNET:.1f}%)")
print("\nВИРОК: (б) і (в) < 0.2 ⟹ універсальної якості вибірки НЕМАЄ ⟹")
print("       клас «відбирати/виправляти вибірку» закрито ОСТАТОЧНО")
