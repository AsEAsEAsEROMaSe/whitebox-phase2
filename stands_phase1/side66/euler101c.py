"""euler101c — ЧИ Є `Δout` КОРИСНОЮ CONTROL VARIATE ДЛЯ МК-ГІЛКИ?

`E[out] = E[Δout]` — ТОЧНА тотожність (перевірено euler101: L=1 аналітично,
L=32 чисельно). ⟹ `d = out − Δout` має РІВНО нульове середнє ⟹ це ідеальна
control variate, незалежно від того, наскільки `Δ`-оцінка сама по собі шумна.

    Ê(λ) = mean(out) − λ·mean(out − Δout)
    Var мінімальна при λ* = Cov(out, d)/Var(d)
    редукція = 1 − corr²(out, d)

🚨 ЧОМУ ЦЕ МОЖЕ ПРАЦЮВАТИ, ХОЧА `Δ` ШУМНІША: control variate не потребує, щоб
   допоміжна величина була точною — лише щоб вона КОРЕЛЮВАЛА з ціллю і мала
   відоме середнє. Тут середнє відоме ТОЧНО (= самій цілі).
🚨 ЧОМУ МОЖЕ НЕ ПРАЦЮВАТИ: якщо `corr(out, d) ≈ 0`, редукції нема; а якщо
   `Var(d) >> Var(out)`, то λ* ≈ 0 і виграш нульовий.

ВАГА ПИТАННЯ: МК-гілка = 91% балу (`V = 1.9227e-06`), і `MSE ≤ V` структурно.
Будь-яка реальна редукція `V` б'є прямо в бал — на відміну від замикання (8.8%).

🔒 ПЛОМБА: `mean(d)` мусить бути НУЛЕМ у межах шуму (це і є тотожність).
   Якщо ні — або формула `Δ`, або реалізація хибні, і корелювати нема сенсу.
🔒 КОНТРОЛЬ: та сама редукція для ПЕРЕМІШАНОЇ `d` (розрив зв'язку з `out`)
   мусить дати нуль.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) 🔒 mean(d)/sd(d) буде в межах ±3 (тобто нуль)                     75%
 (б) |corr(out, d)| > 0.3                                              40%
 (в) редукція дисперсії > ×1.2                                         25%
 (г) λ* виявиться далеко від 1.0 (тобто `Δ` НЕ є «другою копією» out)   70%
"""
import glob
import math
import sys

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402

W_ = 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
EPS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
BATCH = 64

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не одну карту", flush=True)
    sys.exit(2)

f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, 32, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"euler101c · {NNET} мереж × {NS} семплів · ε={EPS}", flush=True)


def one_net(net):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(32)]
    # масштаби s для ядра
    rng = cp.random.default_rng(555 + net)
    S1 = cp.zeros((32, W_), dtype=cp.float64)
    S2 = cp.zeros((32, W_), dtype=cp.float64)
    n = 0
    while n < 8192:
        nb = min(2048, 8192 - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = z
        for i in range(32):
            p = a @ Wg[i]
            p64 = p.astype(cp.float64)
            S1[i] += p64.sum(0)
            S2[i] += (p64 * p64).sum(0)
            if i < 31:
                a = cp.maximum(p, 0.0)
        n += nb
    mm = S1 / 8192
    s = cp.sqrt(cp.maximum(S2 / 8192 - mm * mm, 1e-30))
    wdt = cp.maximum(EPS * s, 1e-20).astype(cp.float32)

    rng = cp.random.default_rng(31337 + net)
    OU, DL = [], []
    n = 0
    while n < NS:
        nb = min(BATCH, NS - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        P, D = [], []
        a = z
        J = cp.tile(Wg[0], (nb, 1, 1))
        Js = []
        for i in range(32):
            p = a @ Wg[i]
            P.append(p)
            d = (p > 0).astype(cp.float32)
            D.append(d)
            Js.append(J)
            if i < 31:
                a = cp.maximum(p, 0.0)
                J = cp.matmul(J * d[:, None, :], Wg[i + 1])
        out = cp.maximum(P[31], 0.0)
        B = [None] * 32
        B[30] = Wg[31] * D[31][:, None, :]
        for i in range(29, -1, -1):
            B[i] = cp.matmul(Wg[i + 1] * D[i + 1][:, None, :], B[i + 1])
        tot = cp.zeros((nb, W_), dtype=cp.float32)
        for i in range(32):
            gn = cp.sum(Js[i] * Js[i], axis=1)
            ker = cp.exp(-0.5 * (P[i] / wdt[i]) ** 2) / (SQ2PI * wdt[i])
            g = gn * ker
            if i == 31:
                tot += g
            else:
                tot += cp.squeeze(cp.matmul(g[:, None, :], B[i]), 1)
        OU.append(out.mean(1).astype(cp.float64))
        DL.append(tot.mean(1).astype(cp.float64))
        n += nb
    return cp.concatenate(OU), cp.concatenate(DL)


def stats(o, dl):
    d = o - dl
    md, sd = float(cp.mean(d)), float(cp.std(d))
    n = len(o)
    z = md / (sd / math.sqrt(n)) if sd > 0 else 0.0
    oc = o - cp.mean(o)
    dc = d - cp.mean(d)
    den = float(cp.sqrt(cp.sum(oc * oc) * cp.sum(dc * dc)))
    r = float(cp.sum(oc * dc)) / den if den > 0 else 0.0
    lam = float(cp.sum(oc * dc) / cp.maximum(cp.sum(dc * dc), 1e-300))
    red = 1.0 / max(1.0 - r * r, 1e-12)
    return md, sd, z, r, lam, red


print(f"\n{'мережа':>7s}{'mean(d)/se':>12s}{'corr(out,d)':>13s}"
      f"{'λ*':>10s}{'×редукція V':>13s}")
print("=" * 56)
rs, reds = [], []
for net in range(NNET):
    o, dl = one_net(net)
    md, sd, z, r, lam, red = stats(o, dl)
    rs.append(r)
    reds.append(red)
    print(f"{net:>7d}{z:>12.2f}{r:>13.4f}{lam:>10.4f}{red:>13.4f}", flush=True)
print("=" * 56)
print(f"сер |corr| = {sum(abs(x) for x in rs)/len(rs):.4f} · "
      f"сер редукція = {sum(reds)/len(reds):.4f}")
print("🔒 mean(d)/se — це ПЛОМБА ТОТОЖНОСТІ: |z| ≲ 3 ⟹ середнє справді нуль")
