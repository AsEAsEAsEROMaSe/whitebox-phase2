"""varq93 — ПРЯМИЙ ВИМІР `Var(Q̄_ℓ)` ПОВТОРНИМИ РОЗІГРАШАМИ. БЕЗ ПРИПУЩЕНЬ.

════════════════ СУПЕРЕЧКА, ЯКУ ЦЕ ВИРІШУЄ ════════════════
`bcross93` порівняв `b_ℓ` із НАЇВНИМ шумом `σ_Q,ℓ/√N` і дав перетин ℓ=14.
АРХІВ (RAUND 14) каже: `σ_Q` уже зменшена вибілюванням і антитетикою ⟹ треба
ДІЛИТИ на 1.692 ⟹ λ живий на всіх 31.
🚨 Я вважаю знак зворотним:
```
σ_Q — ПОСЕМПЛЕВИЙ розкид a², властивість розподілу; вибілювання й антитетика
його НЕ чіпають. Вони зменшують Var(Q̄) ⟹ Var(Q̄) = σ_Q²/(N·g), g > 1
⟹ поріг b < σ_Q/√(N·g) СУВОРІШИЙ ⟹ відношення треба МНОЖИТИ на √g
⟹ перетин зсувається РАНІШЕ за 14, а не пізніше
```
Крім того мій `bcross93` антитетику НЕ вмикав узагалі.

⟹ **Не сперечаюсь припущеннями. Міряю `Var(Q̄)` як розкид по НЕЗАЛЕЖНИХ
розіграшах повного конвеєра сабміту** (пілот+обрізання+вибілювання+антитетика,
`N = 2·n_pairs`). Це знімає всі припущення про `g` одразу.

════════════════ ПЛОМБИ ════════════════
P1 ⛔БЕЗ каналів (сирий гаусів вхід, без антитетики) виміряний `Var(Q̄)`
   мусить збігтися з наївним `σ_Q²/N` у межах ~10%. Якщо ні — розкид по
   розіграшах міряє не те.
P2 `g_ℓ = наївний/виміряний` мусить бути ≥ 1 (канали не можуть ШКОДИТИ
   дисперсії середнього). Якщо десь <1 — стенд бракований.
P3 `b_ℓ` беру з ВЕЛИКОЇ МК (окремий сід, N=65536), як у `bcross93`.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) `g_ℓ` вийде 1.5–4 на малих шарах і впаде до ~1.2 на глибині   шанс 55%
 (б) перетин зсунеться на ℓ = 4..9 (тобто `_KSIG=8` виправданий)   шанс 65%
 (в) АРХІВ помиляється у знаку                                     шанс 80%
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

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 16
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i]
            got += 1


def qbar(wc, npair, seed, chan=True):
    """Q̄_ℓ = сер. a² по семплах. `chan` — вибілювання+антитетика як у сабміті."""
    g = torch.Generator().manual_seed(seed)
    n = wc[0].shape[0]
    if chan:
        z = torch.randn(npair, n, generator=g)
        cov = (z.T @ z) / float(npair)
        ev, V = torch.linalg.eigh(cov.double())
        wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
        y = z @ (wh @ wc[0])
        x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    else:
        z = torch.randn(2 * npair, n, generator=g)
        x = torch.relu(z @ wc[0])
    out = [x.double().pow(2).mean(0)]
    for i in range(1, len(wc)):
        x = torch.relu(x @ wc[i])
        out.append(x.double().pow(2).mean(0))
    return out


def truth_q(wc, N, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(N, wc[0].shape[0], generator=g)
    out, sq = [], []
    x = z
    for i in range(len(wc)):
        x = torch.relu(x @ wc[i])
        a2 = x.double().pow(2)
        out.append(a2.mean(0)); sq.append(a2.std(0))
    return out, sq


RB, RN_ON, RN_OFF, RNAIVE = [], [], [], []
t0 = time.time()
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i])) for i in range(32)]
        wc, npair, _ = P.cut_budget91(W, "cpu")
        depth = len(wc)
        _, _, q_cl, _, _ = P.closure91(wc, depth - 1, "cpu", kdeep=depth)
        qt, sq = truth_q(wc, 65536, 900000 + 41 * q)
        # b_ℓ для шарів 1..depth-1 (q_cl[i-1] ↔ шар i)
        RB.append([float((q_cl[i - 1] - qt[i]).abs().mean())
                   for i in range(1, depth)])
        RNAIVE.append([float(sq[i].mean()) / np.sqrt(2 * npair)
                       for i in range(1, depth)])
        for chan, dst in ((True, RN_ON), (False, RN_OFF)):
            reps = [qbar(wc, npair, 30000 + 97 * q + 7 * r, chan)
                    for r in range(NREP)]
            # 🚨 ширини шарів РІЗНІ після обрізання ⟹ стакати по шару, не разом
            dst.append([float(torch.stack([r[i] for r in reps])
                              .std(0, unbiased=True).mean())
                        for i in range(1, depth)])
        print(f"  net{q} n_pairs={npair} ({time.time()-t0:.0f} с)", flush=True)

B = np.array(RB); NON = np.array(RN_ON); NOFF = np.array(RN_OFF)
NAI = np.array(RNAIVE)
print(f"\n  час {time.time()-t0:.0f} с · {NNET} мереж · {NREP} розіграшів\n")
print("=" * 96)
print("ПРЯМИЙ ВИМІР ШУМУ `Q̄` ПРОТИ НАЇВНОГО `σ_Q/√N`")
print("=" * 96)
p1 = float(np.median(NOFF.mean(0) / NAI.mean(0)))
print(f"⛔P1 БЕЗ каналів: виміряний/наївний = {p1:.4f} "
      f"{'✅' if 0.85 < p1 < 1.15 else '🚨 стенд бракований'}")
g = NAI.mean(0) / NON.mean(0)
print(f"⛔P2 g = наївний/виміряний(з каналами) ≥ 1 скрізь: "
      f"{'✅' if g.min() > 0.95 else '🚨 мін ' + f'{g.min():.3f}'}")
print(f"\n{'шар':>5s}{'b_ℓ':>12s}{'наївний':>12s}{'ВИМІРЯНИЙ':>12s}{'g':>8s}"
      f"{'b/вимір':>10s}{'b/наївн':>10s}")
for i in (0, 2, 4, 6, 7, 8, 11, 15, 19, 23, 29):
    if i >= B.shape[1]:
        continue
    print(f"{i+1:>5d}{B[:,i].mean():>12.4e}{NAI[:,i].mean():>12.4e}"
          f"{NON[:,i].mean():>12.4e}{g[i]:>8.3f}"
          f"{(B[:,i]/NON[:,i]).mean():>10.4f}{(B[:,i]/NAI[:,i]).mean():>10.4f}")
r_on = (B / NON).mean(0)
r_na = (B / NAI).mean(0)
c_on = next((i + 1 for i in range(len(r_on)) if r_on[i] >= 1.0), None)
c_na = next((i + 1 for i in range(len(r_na)) if r_na[i] >= 1.0), None)
print(f"\n📣 ПЕРЕТИН зі СПРАВЖНІМ шумом: ℓ = {c_on}   (з наївним: ℓ = {c_na})")
print(f"   сабміт має _KSIG = 8 · АРХІВ передбачав «λ живий на всіх 31»")
print(f"   ⟹ {'✅ Я МАВ РАЦІЮ: канали ЗСУВАЮТЬ перетин РАНІШЕ' if c_on and c_na and c_on < c_na else '🚨 АРХІВ МАВ РАЦІЮ'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/varq93.npz",
         B=B, NON=NON, NOFF=NOFF, NAI=NAI)
