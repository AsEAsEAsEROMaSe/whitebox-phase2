"""indep101 — ЧОМУ participation ratio працює на шарі 1 і мовчить на глибині.

ГІПОТЕЗА (з neff101): формула κ₄(Σ_j W_ja h_j) = Σ_j W_ja⁴ κ₄(h_j) вимагає
НЕЗАЛЕЖНОСТІ компонент h_j. На шарі 1 вони майже незалежні, на глибині — ні.

ВИМІР — три незалежні міри залежності на кожному шарі:
  (1) ефективний ранг Cov(h)  = (Σλ)²/Σλ²   ⟹ скільки справді вільних напрямків
  (2) сер |корел| між парами h_j
  (3) 🔒 ПРЯМИЙ КОНТРОЛЬ: перемішати компоненти h незалежно між семплами
      (руйнує ВСІ міжнейронні зв'язки, лишає маргінали) і перерахувати γ₄(p).
      Якщо формула суми незалежних вірна — на перемішаних даних corr(γ₄,1/n_eff)
      МУСИТЬ ожити. Це не аргумент, а тест механізму.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) ефективний ранг Cov(h) впаде з ~200 (ш1) до <40 (ш31)          70%
 (б) на ПЕРЕМІШАНИХ даних corr(γ₄, 1/n_eff) підскочить до >0.3 на ВСІХ
     шарах ⟹ підтверджує, що вбиває саме залежність, а не глибина      60%
 (в) γ₄ на перемішаних буде МЕНШОЮ за реальну (залежність її роздуває)  55%
"""
import glob
import math
import sys

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
LS = [1, 4, 8, 16, 24, 31]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"indep101 · {NNET} мереж · {NS} семплів · шари {LS}", flush=True)


def g4_of(p):
    m1 = p.mean(0)
    d = p - m1
    v = (d * d).mean(0)
    k4 = (d ** 4).mean(0) - 3 * v * v
    return k4 / cp.maximum(v * v, 1e-30)


rows = []
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    rng = cp.random.default_rng(20260810 + net)
    for L in LS:
        z = rng.standard_normal((NS, W_), dtype=cp.float32)
        h = cp.concatenate([z, -z], axis=0)
        for i in range(L):
            h = cp.maximum(h @ Wg[i], 0.0)
        w = Wg[L].astype(cp.float64)
        w2 = w * w
        w4 = w2 * w2
        hd = h.astype(cp.float64)
        c = hd.var(0)
        neff = ((w2 * c[:, None]).sum(0) ** 2) / \
            cp.maximum((w4 * (c * c)[:, None]).sum(0), 1e-300)

        # (1) ефективний ранг Cov(h) та (2) середня |корел|
        hc = hd - hd.mean(0)
        cov = (hc.T @ hc) / hc.shape[0]
        dv = cp.sqrt(cp.maximum(cp.diagonal(cov), 1e-30))
        lam = cp.linalg.eigvalsh(cov)
        erank = float((lam.sum() ** 2) / cp.maximum((lam * lam).sum(), 1e-300))
        corrm = cov / cp.outer(dv, dv)
        off = corrm - cp.diag(cp.diagonal(corrm))
        mabs = float(cp.abs(off).sum() / (W_ * (W_ - 1)))

        g4_real = g4_of(hd @ w)

        # (3) 🔒 ПЕРЕМІШУВАННЯ: кожна компонента h перемішується СВОЄЮ
        #     перестановкою ⟹ маргінали ті самі, залежності знищені
        idx = cp.argsort(rng.random((hd.shape[0], W_), dtype=cp.float32), axis=0)
        hs = cp.take_along_axis(hd, idx, axis=0)
        g4_sh = g4_of(hs @ w)

        rows.append((net, L, cp.asnumpy(neff), cp.asnumpy(g4_real),
                     cp.asnumpy(g4_sh), erank, mabs))
    print(f"  мережа {net} готова", flush=True)

print(f"\n{'шар':>4s}{'eff.rank Cov(h)':>17s}{'сер|корел|':>12s}{'n_eff':>9s}"
      f"{'γ₄ реальна':>12s}{'γ₄ ПЕРЕМІШ':>12s}{'corr real':>11s}"
      f"{'corr ПЕРЕМІШ':>14s}")
print("=" * 92)
for L in LS:
    R = [r for r in rows if r[1] == L]
    cr = np.mean([np.corrcoef(1.0 / r[2], r[3])[0, 1] for r in R])
    cs = np.mean([np.corrcoef(1.0 / r[2], r[4])[0, 1] for r in R])
    print(f"{L:>4d}{np.mean([r[5] for r in R]):>17.1f}"
          f"{np.mean([r[6] for r in R]):>12.4f}"
          f"{np.concatenate([r[2] for r in R]).mean():>9.2f}"
          f"{np.concatenate([r[3] for r in R]).mean():>12.4f}"
          f"{np.concatenate([r[4] for r in R]).mean():>12.4f}"
          f"{cr:>11.4f}{cs:>14.4f}")
print("\n🔒 ЧИТАННЯ: якщо на ПЕРЕМІШАНИХ corr оживає, а на реальних ні —")
print("   формулу вбиває САМЕ залежність h_j, а не глибина і не шум.")
