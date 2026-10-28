"""mix102b — ПОПРАВКА ДО `m` ВЗДОВЖ НАПРЯМКІВ ЗМІШУВАННЯ. СТЕЛЯ І РЕАЛІЗОВНЕ

ЗАУВАЖЕННЯ ЮЗЕРА: «чому нуль? може застосування як завжди не туди?» — і воно
влучне: mix102 крутив ФОРМУ СПЕКТРА `C` при сталому сліді, а `m` бачить `C`
майже тільки через рівень `s_a` (при сталому сліді E_w[wᵀCw] не змінюється)
⟹ та ручка фізично не доходила до цілі, симетричні крила були підписом
ефекту ДРУГОГО порядку.

КУДИ ТРЕБА: канал `m` (orac101: істинна C цілком = ×1.098, істинні m і C =
×30.9). Новий базис, якого НЕ БУЛО в стелі С101-г (там лише скаляри m,s,t,Φ,φ):
    m ← a₀·m̂ + Σ_k a_k·(v_k·m̂)·v_k
де v_k — головні напрямки ЗМІШУВАННЯ, тобто власні вектори НАШОЇ Ĉ.
🔑 ЇХ МИ ЗНАЄМО ТОЧНО: mix102 зміряв cos(v̂₁,v₁) = 0.991, топ-4 = 0.955.
🔑 a₀ у базисі ОБОВ'ЯЗКОВЕ — це і є `_ALPHA`; без нього вимір спіймає масштаб,
   який ми й так беремо даром (закон С101-в).

ТРИ РІВНІ, ВІД СТЕЛІ ДО РЕАЛІЗОВНОГО:
  ORA   a_k фітуються ПО МЕРЕЖІ проти ІСТИНИ            — абсолютна стеля
  SHR   ті самі a_k, але СПІЛЬНІ на всі мережі (крос A→B) — реалізовно як κ
  🔒CHU  v_k від ІНШОЇ мережі, фіт по мережі             — має зламати ORA

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 K=0 (лише a₀) відтворить косинусну похибку бази до 4 знаків     95%
 (б) ORA K=4 дасть > ×1.10                                             40%
 (в) SHR (спільні коефіцієнти, крос) дасть > ×1.02                     25%
 (г) 🔒 CHU буде помітно гіршою за ORA                                 75%
 (д) виграш ORA росте з K і насичується до K=8                         60%

ВИРОК: якщо ORA K=8 < ×1.05 ⟹ напрямки змішування як базис поправки до `m`
мертві, і канал `m` закрито з ЧЕТВЕРТОГО боку (після С101-г, С97, С74).
"""
import glob
import math
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

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NTRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
KS = [0, 1, 2, 4, 8, 16]

if cp.cuda.runtime.getDeviceCount() != 1:
    print("ПІНУВАННЯ НЕ СПРАЦЮВАЛО", flush=True)
    sys.exit(2)

tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"mix102b · {NNET} мереж · істина {NTRUE} · K {KS}", flush=True)


def draw(seed, npair):
    rng = cp.random.default_rng(seed)
    z = rng.standard_normal((npair, W_), dtype=cp.float32)
    z = cp.concatenate([z, -z], axis=0).astype(cp.float64)
    cov = (z.T @ z) / z.shape[0]
    wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
    return (z @ wh).astype(cp.float32)


def collect(Wg, seed0, ntot):
    S = cp.zeros((DEPTH + 1, W_), dtype=cp.float64)
    Q = cp.zeros((W_, W_), dtype=cp.float64)
    n = 0
    while n < ntot // 2:
        nb = min(4096, ntot // 2 - n)
        h = draw(seed0 + n, nb)
        for i in range(DEPTH):
            h = cp.maximum(h @ Wg[i], 0.0)
            h64 = h.astype(cp.float64)
            S[i + 1] += h64.sum(0)
            if i == 0:
                Q += h64.T @ h64
        n += nb
    m = S / ntot
    return m, Q / ntot - cp.outer(m[1], m[1])


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
    Cn = Cp * cp.outer(Ph, Ph) + (Cp * Cp / (2.0 * cp.outer(s, s))) * cp.outer(pf, pf)
    idx = cp.arange(W_)
    Cn[idx, idx] = var
    return m_new, Cn


def cerr(mh, mt):
    c2 = float((mh * mt).sum() ** 2 / ((mh * mh).sum() * (mt * mt).sum()))
    return 100.0 * math.sqrt(max(1.0 - c2, 0.0))


MH, MT, VV = [], [], []
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    Wg64 = [w.astype(cp.float64) for w in Wg]
    mt, C1 = collect(Wg, 810000 + net * 7919, NTRUE)
    m, C = mt[1].copy(), C1
    for l in range(1, DEPTH):
        m, C = step(m, C, Wg64[l])
    w_, v_ = cp.linalg.eigh(C)
    MH.append(cp.asnumpy(m))
    MT.append(cp.asnumpy(mt[DEPTH]))
    VV.append(cp.asnumpy(v_[:, ::-1][:, :max(KS)]))
    print(f"  мережа {net}: база {cerr(m, mt[DEPTH]):.4f}%", flush=True)
    del Wg, Wg64
    cp.get_default_memory_pool().free_all_blocks()


def basis(mh, V, K):
    B = [mh]
    for k in range(K):
        B.append(float(V[:, k] @ mh) * V[:, k])
    return np.stack(B, 1)


def fit_err(mh, mt, V, K):
    B = basis(mh, V, K)
    a, *_ = np.linalg.lstsq(B, mt, rcond=None)
    r = B @ a - mt
    return 100.0 * math.sqrt(float(r @ r) / float(mt @ mt)), a


print("\n" + "=" * 76)
print("СТЕЛЯ: поправка до m вздовж напрямків змішування (фіт ПО МЕРЕЖІ)")
print(f"{'K':>4} {'ORA':>10} {'×':>8} {'🔒CHU':>10} {'×':>8}")
print("-" * 76)
base = None
for K in KS:
    eo = np.mean([fit_err(MH[n], MT[n], VV[n], K)[0] for n in range(NNET)])
    ec = np.mean([fit_err(MH[n], MT[n], VV[(n + 1) % NNET], K)[0]
                  for n in range(NNET)])
    if base is None:
        base = eo
    print(f"{K:>4} {eo:>9.4f}% {base/eo:>8.4f} {ec:>9.4f}% {base/ec:>8.4f}")

print("\n" + "=" * 76)
print("РЕАЛІЗОВНЕ: СПІЛЬНІ коефіцієнти, крос A→B (мережі парні/непарні)")
print(f"{'K':>4} {'A→B':>10} {'×':>8} {'B→A':>10} {'×':>8}")
print("-" * 76)
IA = list(range(0, NNET, 2))
IB = list(range(1, NNET, 2))
for K in KS:
    if K == 0:
        print(f"{0:>4} {base:>9.4f}% {1.0:>8.4f} {base:>9.4f}% {1.0:>8.4f}")
        continue
    out = []
    for tr, te in ((IA, IB), (IB, IA)):
        G = np.zeros((K + 1, K + 1))
        rhs = np.zeros(K + 1)
        for n in tr:
            B = basis(MH[n], VV[n], K)
            sc = math.sqrt(float(MT[n] @ MT[n]))
            G += (B.T @ B) / (sc * sc)
            rhs += (B.T @ MT[n]) / (sc * sc)
        a = np.linalg.solve(G + 1e-12 * np.eye(K + 1), rhs)
        e = np.mean([100.0 * math.sqrt(
            float((basis(MH[n], VV[n], K) @ a - MT[n]) @
                  (basis(MH[n], VV[n], K) @ a - MT[n])) / float(MT[n] @ MT[n]))
            for n in te])
        out.append(e)
    print(f"{K:>4} {out[0]:>9.4f}% {base/out[0]:>8.4f} "
          f"{out[1]:>9.4f}% {base/out[1]:>8.4f}")
print("-" * 76)
print("🔒 K=0 — це РІВНО база після оптимального масштабу (тобто після _ALPHA)")
print("ВИРОК: ORA K=8 < ×1.05 ⟹ напрямки змішування як базис для `m` мертві")
