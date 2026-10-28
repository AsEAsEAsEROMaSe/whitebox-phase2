"""blend91 — №2: ПОМЕРÉЖНА ВАГА БЛЕНДУ ЗАМІСТЬ ОДНІЄЇ КОНСТАНТИ 0.0883.

════════════════ МЕХАНІЗМ, А НЕ ЗДОГАДКА ════════════════
Оптимальна вага бленду двох незалежних оцінювачів:
    w* = Var(МК) / (Var(МК) + b²),   b² = квадрат похибки замикання
Ми ставимо ОДНУ константу `_W_BLEND=0.0883` на ВСІ мережі, тоді як `V/N`
розлітається між мережами у **×11.86** (зміряно `alloc91`).

🔑 І `b²` НЕ ТРЕБА ФІТИТИ. Ми спостерігаємо `d = (1+α)·m_deep − ours`, а
`E[d²] = b² + V/N` (похибки незалежні: одна від МК-шуму, друга від замикання)
⟹ `b² = ⟨d²⟩ − V/N`  і  **`w* = (V/N) / ⟨d²⟩`  — БЕЗ ЖОДНОГО ПАРАМЕТРА**.
`V/N` уже пораховано в сабміті (понейронна `Var(x̄_j)`), ціна = ДВІ редукції.

⚠️ НАШ ЗАКОН: «величину в знаменнику `Nβ²` не оцінювати з тих самих даних —
вона вироджується в шум». Тому знаменник беремо УСЕРЕДНЕНИЙ ПО 256 НЕЙРОНАХ
(померéжно), а понейронний варіант міряємо ОКРЕМО як заздалегідь підозрілий.

════════════════ ПЛОМБИ ════════════════
P1 ГЛОБАЛЬНА схема мусить відтворити сабмітну вагу: `w` LOO ≈ 0.0883.
P2 ТОТОЖНІСТЬ: `⟨d²⟩ − V/N` мусить бути ДОДАТНИМ на переважній більшості
   мереж (інакше модель незалежності похибок хибна) — рахуємо частку.
P3 ⛔КОНТРОЛЬ «ЧУЖА»: `w_i` — перестановка тих самих ваг між мережами.
   Мусить дати ×<1.00. Якщо чужа виграє — виграє розкид, а не інформація.
P4 усе LOO по мережах; α фітиться глобально LOO.

════════════════ ПЕРЕДБАЧЕННЯ (записано ДО запуску) ════════════════
 (а) P1 зійдеться на 0.088 ± 0.01                     — шанс 75%
 (б) померéжна `w*` дасть ×1.03…1.08                  — шанс 55%
 (в) понейронна виродиться (≤ померéжної)             — шанс 70%
 (г) чужа дасть <×1.00                                — шанс 85%
"""
import glob
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

sys.path.insert(0, "/mnt/win_d/llm/White-Box/side66")
import sub91_port as P                                        # noqa: E402

P.TARGET_UTIL = 0.09011
P.W_BLEND = 0.0883
P.ALPHA = -0.00751

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
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
        fm = np.asarray(t.column("final_means").combine_chunks().flatten()
                        ).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i].astype(np.float64)
            got += 1


OU, CL, TR, VV, VH = [], [], [], [], []
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth) in enumerate(rows(NNET)):
        Wt = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
              for i in range(32)]
        wc, npair, rng = P.cut_budget91(Wt, DEV)
        _, _, _, (ours, m_deep, vout, vvec) = P.predict91(
            wc, npair, DEV, rng=rng, raw=True)
        OU.append(ours.cpu().numpy())
        CL.append(m_deep.cpu().numpy())
        VV.append(vvec.cpu().numpy())
        TR.append(truth)
        if q % 100 == 0:
            print(f"  net{q:>4d} ({time.time()-t0:.0f} с)", flush=True)
OU, CL, TR, VV = map(np.array, (OU, CL, TR, VV))
N = len(OU)
print(f"\n  час {time.time()-t0:.0f} с · мереж {N}\n", flush=True)


def mse(pred):
    return float(((pred - TR) ** 2).mean())


def alpha_loo():
    """Спільний α: (1+α)·CL ≈ TR."""
    num = (CL * TR).sum(1)
    den = (CL * CL).sum(1)
    a = np.empty(N)
    for q in range(N):
        o = np.arange(N) != q
        a[q] = num[o].sum() / max(den[o].sum(), 1e-300)
    return a


AL = alpha_loo()
D = AL[:, None] * CL - OU                 # спостережуване d
d2 = (D * D).mean(1)                      # ⟨d²⟩ по нейронах
vn = VV.mean(1)                           # V/N по нейронах
b2 = d2 - vn                              # ⛔ P2
print(f"⛔ P2 `b² = ⟨d²⟩ − V/N` додатне на "
      f"{100*float((b2 > 0).mean()):.1f}% мереж  "
      f"(медіана b² {np.median(b2):.3e}, медіана V/N {np.median(vn):.3e})")
print(f"   розкид V/N між мережами ×{vn.max()/vn.min():.2f} · CV {vn.std()/vn.mean():.3f}")
print(f"   розкид w* = (V/N)/⟨d²⟩ : мін {np.min(vn/d2):.4f} · "
      f"медіана {np.median(vn/d2):.4f} · макс {np.max(vn/d2):.4f}")

rng = np.random.default_rng(0)


def run(wvec, name):
    """LOO по мережах: α беремо LOO, w — за схемою."""
    p = OU + wvec[:, None] * (AL[:, None] * CL - OU)
    return name, mse(p)


# --- схеми ---
res = []
# 1) ГЛОБАЛЬНА w, фітиться LOO (це сабміт)
wg = np.empty(N)
for q in range(N):
    o = np.arange(N) != q
    num = (D[o] * (TR[o] - OU[o])).sum()
    den = (D[o] * D[o]).sum()
    wg[q] = num / max(den, 1e-300)
res.append(run(wg, "ГЛОБАЛЬНА w (сабміт)"))
res.append(run(np.full(N, 0.0883), "константа 0.0883"))
# 2) ТЕОРЕТИЧНА померéжна, БЕЗ параметрів
wt = np.clip(vn / np.maximum(d2, 1e-300), 0.0, 1.0)
res.append(run(wt, "🎯 w* = (V/N)/⟨d²⟩"))
# 3) та сама, але з одним масштабним коефіцієнтом (LOO)
sc = np.empty(N)
for q in range(N):
    o = np.arange(N) != q
    num = (wt[o, None] * D[o] * (TR[o] - OU[o])).sum()
    den = ((wt[o, None] * D[o]) ** 2).sum()
    sc[q] = num / max(den, 1e-300)
res.append(run(np.clip(sc * wt, 0, 1), "w* × масштаб (LOO)"))
# 4) ⛔ ЧУЖА: перестановка тих самих ваг
res.append(run(wt[rng.permutation(N)], "⛔ЧУЖА (перестановка w*)"))
# 5) ПОНЕЙРОННА (заздалегідь підозріла)
wpn = np.clip(VV / np.maximum(D * D, 1e-300), 0.0, 1.0)
pn = OU + wpn * (AL[:, None] * CL - OU)
res.append(("ПОНЕЙРОННА w*_j", mse(pn)))
# 6) ПОНЕЙРОННА з бінуванням по t (усереднення знаменника)
ordr = np.argsort(VV, axis=1)
wb = np.empty_like(wpn)
NB = 8
for q in range(N):
    idx = ordr[q]
    for b in range(NB):
        sl = idx[b * 256 // NB:(b + 1) * 256 // NB]
        wb[q, sl] = np.clip(VV[q, sl].mean()
                            / max((D[q, sl] ** 2).mean(), 1e-300), 0, 1)
res.append(("понейронна, 8 бінів", mse(OU + wb * (AL[:, None] * CL - OU))))

base = dict(res)["ГЛОБАЛЬНА w (сабміт)"]
print("\n" + "=" * 96)
print(f"ПОМЕРÉЖНА ВАГА БЛЕНДУ · {N} мереж · офіційна істина 1e9 · LOO")
print("=" * 96)
print(f"{'схема':>28s} {'абс. MSE':>14s} {'бал ×':>10s} {'бал':>13s}")
for nm, v in res:
    print(f"{nm:>28s} {v:>14.6e} {base/v:>10.4f} {1.86762e-07*v/base:>13.4e}")
print(f"{'⛔ без бленду (чиста МК)':>28s} {mse(OU):>14.6e} "
      f"{base/mse(OU):>10.4f}")
print(f"\n⛔ P1 глобальна w LOO = {wg.mean():.4f}  (сабміт 0.0883) "
      f"{'✅' if abs(wg.mean()-0.0883) < 0.012 else '🚨'}")
g = base / dict(res)["🎯 w* = (V/N)/⟨d²⟩"]
gf = base / dict(res)["⛔ЧУЖА (перестановка w*)"]
print(f"\n📣 ВИРОК: w* без параметрів ×{g:.4f}  проти  ЧУЖОЇ ×{gf:.4f}")
print(f"   {'✅ ІНФОРМАЦІЯ' if g > gf * 1.01 else '⛔ ВИГРАЄ РОЗКИД, НЕ ІНФОРМАЦІЯ'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/blend91.npz",
         OU=OU, CL=CL, TR=TR, VV=VV, AL=AL, wt=wt, wg=wg)
