"""mask100g — GPU-версія маскового тесту (cupy), СТІЙКА ДО ВІДВАЛУ КАРТИ.

Та сама математика, що `mask100c`, але:
  · рахує на GPU (пінування СТРОГО по UUID)
  · пише результат ПІСЛЯ КОЖНОЇ МЕРЕЖІ у власний json
  · при перезапуску ПІДХОПЛЮЄ зроблене і рахує лише те, чого бракує
  · будь-яка помилка карти НЕ вбиває прогін: мережа позначається як невдала,
    робиться пауза й повторна спроба; після 3 невдач поспіль — чистий вихід
    зі збереженням усього, що вже пораховано

🚨 ПІНУВАННЯ: карта задається ЛИШЕ через `CUDA_VISIBLE_DEVICES=GPU-<uuid>`.
   Індекси перенумеровуються, і `cuda:1` без `CUDA_DEVICE_ORDER=PCI_BUS_ID`
   може влучити у ФІЗИЧНУ карту 0 (ТАБУ). З UUID процес бачить рівно одну карту
   (перевірено: `getDeviceCount() == 1`).
   робочі: 1 = GPU-bff69481-0d35-74f0-8871-68eb429cdc34
           2 = GPU-eaa8132f-e46f-2090-d943-0481fbe03aef
           3 = GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26
   ⛔ index 0 = GPU-3cfa18b3-... ТАБУ · index 4 = 1080 Ti — монітор ЮЗЕРА

ВИКОРИСТАННЯ:
   CUDA_VISIBLE_DEVICES=GPU-<uuid> python3 mask100g.py <tag> <net0> <nnet> <npairs>

🔒 ПЛОМБА ПЕРЕНОСУ: `mask100` на CPU дав для пари (1,2) розбіжність 6.48%,
   (8,9) 3.38%, (24,25) 2.43%. GPU мусить відтворити це в межах шуму — інакше
   результати не використовуються (закон «спершу еталон, тоді прилад»).
"""
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
TAG = sys.argv[1] if len(sys.argv) > 1 else "a"
NET0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NNET = int(sys.argv[3]) if len(sys.argv) > 3 else 8
NPAIRS_TOT = int(sys.argv[4]) if len(sys.argv) > 4 else 262144
CHUNK = 16384
NPAIR = 3000
CUTS = [(1, 2), (8, 9), (16, 17), (24, 25), (8, 24)]
LAYERS = sorted({l for c in CUTS for l in c} | {16})
OUT = f"/mnt/win_d/llm/White-Box/relay/mask100n_{TAG}.json"
N = 2 * NPAIRS_TOT

# ── стан: підхоплюємо зроблене, якщо прогін уже падав ────────────────────────
state = {}
if os.path.exists(OUT):
    try:
        state = json.load(open(OUT))
        print(f"↻ підхоплено {len(state)} готових мереж із {OUT}", flush=True)
    except Exception as e:                                    # noqa: BLE001
        print(f"⚠️ не зміг прочитати чекпоінт ({e}), починаю з нуля", flush=True)


def save():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, OUT)          # атомарно: обрив під час запису не псує файл


import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

ndev = cp.cuda.runtime.getDeviceCount()
props = cp.cuda.runtime.getDeviceProperties(0)
name = props["name"].decode()
if ndev != 1:
    print(f"🚨 ВИДНО {ndev} КАРТ — пінування НЕ спрацювало. Вихід, щоб не "
          f"влучити в табу-карту.", flush=True)
    sys.exit(2)
print(f"mask100g[{TAG}] · карта {name} · мережі {NET0}..{NET0+NNET-1} · "
      f"{N} семплів · шум P̂ ≈ {100*0.5/math.sqrt(N)/0.25:.3f}%", flush=True)

_xs, _ws = np.polynomial.legendre.leggauss(48)
QU = 0.5 * 16.0 * _xs
QW = 0.5 * 16.0 * _ws


def bivar_orthant(t1, t2, rho):
    q = cp.sqrt(cp.maximum(1.0 - rho * rho, 1e-12))
    out = cp.zeros_like(t1)
    for uk, wk in zip(QU, QW):
        out += (wk * (math.exp(-0.5 * uk * uk) / SQ2PI)
                * ndtr((t2 + rho * uk) / q) * (uk > -t1))
    return out


f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)

rng0 = np.random.default_rng(20260809)
II = {c: cp.asarray(rng0.integers(0, W_, NPAIR)) for c in CUTS}
JJ = {c: cp.asarray(rng0.integers(0, W_, NPAIR)) for c in CUTS}


def one_net(net):
    W = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float32))
         for L in range(DEPTH)]
    rng = cp.random.default_rng(555 + net)
    S1 = {L: cp.zeros(W_, dtype=cp.float64) for L in LAYERS}
    S2 = {L: cp.zeros(W_, dtype=cp.float64) for L in LAYERS}
    CPa = {c: cp.zeros(NPAIR, dtype=cp.float64) for c in CUTS}
    JN = {c: cp.zeros(NPAIR, dtype=cp.float64) for c in CUTS}
    done = 0
    while done < NPAIRS_TOT:
        nb = min(CHUNK, NPAIRS_TOT - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        A = cp.concatenate([z, -z], axis=0)
        keep, a = {}, A
        for i in range(DEPTH):
            p = a @ W[i]
            if i in LAYERS:
                keep[i] = p
                p64 = p.astype(cp.float64)
                S1[i] += p64.sum(0)
                S2[i] += (p64 * p64).sum(0)
            if i < DEPTH - 1:
                a = cp.maximum(p, 0.0)
        for c in CUTS:
            l1, l2 = c
            ii, jj = II[c], JJ[c]
            CPa[c] += (keep[l1][:, ii].astype(cp.float64)
                       * keep[l2][:, jj].astype(cp.float64)).sum(0)
            JN[c] += ((keep[l1][:, ii] > 0)
                      & (keep[l2][:, jj] > 0)).sum(0, dtype=cp.float64)
        del keep, a, A, z
        done += nb
    devs, rhos, tmed = [], [], None
    for c in CUTS:
        l1, l2 = c
        ii, jj = II[c], JJ[c]
        m1, m2 = S1[l1] / N, S1[l2] / N
        s1 = cp.sqrt(cp.maximum(S2[l1] / N - m1 * m1, 1e-30))
        s2 = cp.sqrt(cp.maximum(S2[l2] / N - m2 * m2, 1e-30))
        rho = cp.clip((CPa[c] / N - m1[ii] * m2[jj])
                      / cp.maximum(s1[ii] * s2[jj], 1e-30), -0.999, 0.999)
        pred = bivar_orthant(m1[ii] / s1[ii], m2[jj] / s2[jj], rho)
        real = JN[c] / N
        devs.append(float(cp.abs(real - pred).mean()
                          / cp.maximum(cp.abs(real).mean(), 1e-30)))
        rhos.append(float(cp.abs(rho).mean()))
    m16 = S1[16] / N
    s16 = cp.sqrt(cp.maximum(S2[16] / N - m16 * m16, 1e-30))
    tmed = float(cp.median(cp.abs(m16 / s16)))
    return {"devs": devs, "rhos": rhos, "tmed": tmed}


NLIST = [32768, 131072, 300000, 1048576]
fails = 0
for net in range(NET0, NET0 + NNET):
  for _NP in NLIST:
    key = f"{net}_{_NP}"
    if key in state:
        continue
    NPAIRS_TOT = _NP
    N = 2 * _NP
    try:
        t0 = time.time()
        globals()["NPAIRS_TOT"] = _NP
        globals()["N"] = 2 * _NP
        state[key] = one_net(net)
        save()
        d = state[key]["devs"]
        fails = 0
        print(f"  ✅ мережа {net}, N={2*_NP:>8d}: розбіжність "
              f"{100*float(np.mean(d)):.3f}%  "
              f"[{' '.join(f'{100*x:.2f}' for x in d)}]  "
              f"шум {100*0.5/math.sqrt(2*_NP)/0.25:.3f}%  "
              f"{time.time()-t0:.1f} с", flush=True)
    except Exception as e:                                    # noqa: BLE001
        fails += 1
        print(f"  ⚠️ мережа {net} N={_NP} ВПАЛА ({type(e).__name__}: {e}) — "
              f"невдача {fails}/3", flush=True)
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:                                     # noqa: BLE001
            pass
        time.sleep(10)
        if fails >= 3:
            print(f"🚨 ТРИ НЕВДАЧІ ПОСПІЛЬ — карта, схоже, відвалилась. "
                  f"Збережено {len(state)} мереж у {OUT}. Перезапуск підхопить.",
                  flush=True)
            save()
            sys.exit(3)

save()
print(f"\n✅ ГОТОВО: {len(state)} мереж → {OUT}")
if state:
    print("\n🔒 ПЛОМБА МАСШТАБУВАННЯ: чи залежить ознака від N?")
    print(f"{'N семплів':>12s}{'розбіжність':>14s}{'шум':>9s}{'мереж':>7s}")
    for _NP in NLIST:
        vs = [np.mean(v["devs"]) for k, v in state.items()
              if k.endswith(f"_{_NP}")]
        if vs:
            print(f"{2*_NP:>12d}{100*np.mean(vs):>13.3f}%"
                  f"{100*0.5/math.sqrt(2*_NP)/0.25:>8.3f}%{len(vs):>7d}")
