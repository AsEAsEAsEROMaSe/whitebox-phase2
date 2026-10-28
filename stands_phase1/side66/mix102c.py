"""mix102c — ЧИ Є СПІЛЬНА СТРУКТУРА В КОЕФІЦІЄНТАХ `a_k`?

ЩО ЗМУСИЛО (mix102b): поправка до `m` вздовж напрямків змішування має
ОРАКУЛЬНУ стелю ×1.3799 (K=8) при 🔒чужих напрямках ×1.0178 ⟹ контроль
зламав, сигнал реальний, і це у ×13 вище за стелю всього класу С101-г (×1.027).
АЛЕ спільні коефіцієнти на кросі дали ×0.78 / ×0.95 — гірше за базу.

ЄДИНЕ, ЩО ЛИШИЛОСЬ ПЕРЕВІРИТИ: чи `a_k` мають спільну ФОРМУ, яку можна
параметризувати 1–2 числами (як `κ`), чи вони просто різні.
🚨 ЗАКОН С100: правильний розмір сім'ї — 2–4 параметри; 17 параметрів,
   підібраних на 50 мережах, уже перефічені.

ЩО МІРЯЄМО:
 (1) самі `a_k/a₀` по мережах: медіана, розкид, ЧАСТКА ОДНАКОВОГО ЗНАКУ
 (2) кореляція `a_k` з кандидатами: λ_k/λ₁ · √λ_k · cos(v_k,m̂) · 1/k · const
 (3) однопараметрична сім'я `a_k = c·g_k` для кожного кандидата g,
     `c` фітується СПІЛЬНО, вирок — КРОС A→B (не in-sample)

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) знак `a_1` буде однаковим у >75% мереж                            55%
 (б) знайдеться кандидат g із |corr| > 0.5 до оракульних a_k           40%
 (в) однопараметрична сім'я дасть крос > ×1.02                         25%
 (г) 🔒 випадковий g дасть крос ≤ ×1.00                                85%

ВИРОК: якщо жоден кандидат не дає крос > ×1.02 ⟹ поправка існує, але
є ІНДИВІДУАЛЬНОЮ для мережі; тоді єдиний шлях — оцінювати `a_k` з ВЛАСНОЇ
МК кожної мережі (як `_ai`), і це вже інша гілка з іншою ціною.
"""
import glob
import math
import os
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
CACHE = "/home/trim/mix102c_cache.npz"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NTRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
KMAX = 8

if not os.path.exists(CACHE):
    if cp.cuda.runtime.getDeviceCount() != 1:
        print("ПІНУВАННЯ НЕ СПРАЦЮВАЛО", flush=True)
        sys.exit(2)
    tb = pa.ipc.open_stream(pa.memory_map(
        sorted(glob.glob(SRC + "/data-*.arrow"))[0], "rb")).read_all()
    xx = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xx.type):
        xx = xx.flatten()
    Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
    NNET = min(NNET, tb.num_rows)
    print(f"mix102c · збираю {NNET} мереж × {NTRUE}", flush=True)

    def draw(seed, npair):
        rng = cp.random.default_rng(seed)
        z = rng.standard_normal((npair, W_), dtype=cp.float32)
        z = cp.concatenate([z, -z], axis=0).astype(cp.float64)
        cov = (z.T @ z) / z.shape[0]
        wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
        return (z @ wh).astype(cp.float32)

    def step(m, C, W):
        mp = m @ W
        Cp = W.T @ C @ W
        s2 = cp.maximum(cp.diag(Cp), 1e-300)
        s = cp.sqrt(s2)
        t = mp / s
        Ph = ndtr(t)
        pf = cp.exp(-0.5 * t * t) / SQ2PI
        m_new = s * (pf + t * Ph)
        e2 = s2 * ((t * t + 1.0) * Ph + t * pf)
        var = cp.maximum(e2 - m_new * m_new, 1e-300)
        Cn = (Cp * cp.outer(Ph, Ph)
              + (Cp * Cp / (2.0 * cp.outer(s, s))) * cp.outer(pf, pf))
        idx = cp.arange(W_)
        Cn[idx, idx] = var
        return m_new, Cn

    MH, MT, VV, LM = [], [], [], []
    for net in range(NNET):
        Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
              for i in range(DEPTH)]
        Wg64 = [w.astype(cp.float64) for w in Wg]
        S = cp.zeros((DEPTH + 1, W_), dtype=cp.float64)
        Q = cp.zeros((W_, W_), dtype=cp.float64)
        n = 0
        while n < NTRUE // 2:
            nb = min(4096, NTRUE // 2 - n)
            h = draw(810000 + net * 7919 + n, nb)
            for i in range(DEPTH):
                h = cp.maximum(h @ Wg[i], 0.0)
                h64 = h.astype(cp.float64)
                S[i + 1] += h64.sum(0)
                if i == 0:
                    Q += h64.T @ h64
            n += nb
        mt = S / NTRUE
        m, C = mt[1].copy(), Q / NTRUE - cp.outer(mt[1], mt[1])
        for l in range(1, DEPTH):
            m, C = step(m, C, Wg64[l])
        w_, v_ = cp.linalg.eigh(C)
        MH.append(cp.asnumpy(m))
        MT.append(cp.asnumpy(mt[DEPTH]))
        VV.append(cp.asnumpy(v_[:, ::-1][:, :KMAX]))
        LM.append(cp.asnumpy(w_[::-1][:KMAX]))
        print(f"  мережа {net} готова", flush=True)
        del Wg, Wg64
        cp.get_default_memory_pool().free_all_blocks()
    np.savez(CACHE, MH=np.array(MH), MT=np.array(MT),
             VV=np.array(VV), LM=np.array(LM))

d = np.load(CACHE)
MH, MT, VV, LM = d["MH"], d["MT"], d["VV"], d["LM"]
NNET = MH.shape[0]
print(f"mix102c · {NNET} мереж із кешу", flush=True)


def basis(n, K):
    B = [MH[n]]
    for k in range(K):
        B.append(float(VV[n][:, k] @ MH[n]) * VV[n][:, k])
    return np.stack(B, 1)


def rel(v, n):
    return 100.0 * math.sqrt(float(v @ v) / float(MT[n] @ MT[n]))


A = np.zeros((NNET, KMAX + 1))
for n in range(NNET):
    B = basis(n, KMAX)
    A[n], *_ = np.linalg.lstsq(B, MT[n], rcond=None)
base = np.mean([rel(basis(n, 0) @ np.linalg.lstsq(basis(n, 0), MT[n],
                                                  rcond=None)[0] - MT[n], n)
                for n in range(NNET)])
ora = np.mean([rel(basis(n, KMAX) @ A[n] - MT[n], n) for n in range(NNET)])
R = A[:, 1:] / A[:, :1]

print("\n" + "=" * 74)
print(f"(1) ОРАКУЛЬНІ a_k/a₀   [база {base:.4f}%  стеля K={KMAX} {ora:.4f}% "
      f"= ×{base/ora:.4f}]")
print(f"{'k':>3} {'медіана':>10} {'сер':>10} {'sd':>10} {'знак+':>8} {'|сер|/sd':>9}")
print("-" * 74)
for k in range(KMAX):
    c = R[:, k]
    pos = int((c > 0).sum())
    print(f"{k+1:>3} {np.median(c):>10.4f} {c.mean():>10.4f} {c.std():>10.4f} "
          f"{max(pos, NNET-pos):>4}/{NNET} {abs(c.mean())/max(c.std(),1e-12):>9.3f}")

print("\n" + "=" * 74)
print("(2) З ЧИМ КОРЕЛЮЮТЬ a_k (усі мережі × усі k разом)")
cosm = np.array([[abs(float(VV[n][:, k] @ MH[n]))
                  / math.sqrt(float(MH[n] @ MH[n]))
                  for k in range(KMAX)] for n in range(NNET)])
lamn = LM / LM[:, :1]
CAND = {"const": np.ones_like(lamn), "lam/lam1": lamn,
        "sqrt(lam)": np.sqrt(np.maximum(lamn, 0)), "cos(v,m)": cosm,
        "1/k": np.tile(1.0 / np.arange(1, KMAX + 1), (NNET, 1)),
        "lam*cos": lamn * cosm}
flat = R.ravel()
for nm, g in CAND.items():
    gg = g.ravel()
    cc = float(np.corrcoef(flat, gg)[0, 1])
    print(f"   {nm:>10}  corr {cc:+.4f}")

print("\n" + "=" * 74)
print("(3) ОДНОПАРАМЕТРИЧНА СІМ'Я a_k = c·g_k  — ВИРОК НА КРОСІ A→B")
print(f"{'g':>10} {'in-sample':>11} {'×':>8} {'крос A→B':>11} {'×':>8} {'B→A':>10} {'×':>8}")
print("-" * 74)
IA, IB = list(range(0, NNET, 2)), list(range(1, NNET, 2))
rng = np.random.default_rng(7)
CAND["🔒rand"] = rng.standard_normal((NNET, KMAX))


def err_c(idx, g, c):
    e = []
    for n in idx:
        B = basis(n, KMAX)
        a = np.concatenate([[1.0], c * g[n]])
        v = B @ a
        sc = float(v @ MT[n]) / max(float(v @ v), 1e-300)
        e.append(rel(sc * v - MT[n], n))
    return float(np.mean(e))


for nm, g in CAND.items():
    grid = np.linspace(-0.5, 0.5, 201)
    ins = min(grid, key=lambda c: err_c(range(NNET), g, c))
    ca = min(grid, key=lambda c: err_c(IA, g, c))
    cb = min(grid, key=lambda c: err_c(IB, g, c))
    e_in = err_c(range(NNET), g, ins)
    e_ab = err_c(IB, g, ca)
    e_ba = err_c(IA, g, cb)
    print(f"{nm:>10} {e_in:>10.4f}% {base/e_in:>8.4f} {e_ab:>10.4f}% "
          f"{base/e_ab:>8.4f} {e_ba:>9.4f}% {base/e_ba:>8.4f}")
print("-" * 74)
print("🔒 rand мусить дати ≤×1.00 на кросі · ВИРОК: жоден g > ×1.02 ⟹ "
      "поправка ІНДИВІДУАЛЬНА для мережі")
