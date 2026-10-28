"""scan102 — МАСОВИЙ СКАН ДЕТЕРМІНОВАНИХ РУЧОК У РЕКУРСІЇ НА 1000 МЕРЕЖАХ

ЧОМУ САМЕ ТАК (урок С102): бленд працює ТІЛЬКИ тому, що гілка замикання
детермінована — будь-яка вибіркова інформація в ній складає шуми замість
усереднювати (зміряно: −4.4% і −41%). ⟹ живі лише ДЕТЕРМІНОВАНІ зміни
рекурсії з ціною ≈0. Такою була `κ` (єдиний плюс за 8 сесій).

ЧОМУ ЗАРАЗ ЦЕ ДЕШЕВО: спліт `full` має 1000 мереж із ТОЧНОЮ істиною
(`final_means`, n=1e9) ⟹ детермінована ідея судиться за секунди, без МК,
без грейдера, зі знаковим тестом на сотнях мереж.

🚨 МІРЯЄМО КОСИНУСНУ ПОХИБКУ (масштаб вільний — його бере `_ALPHA` даром).
   Закон С101-в: щілина сира/косинусна ×1.9, і в ній ховається те, що вже
   безкоштовне.
🚨 ВИРОК — ТІЛЬКИ КРОС A→B (перші/другі 50%), не in-sample.

СІМ'Ї (кожна 1 параметр, база = значення, що дає РІВНО поточне замикання):
  kap    C *= k при i>KSIG                     база 0.9985  (наш `_KAPPA`)
  kapd   C *= (1 + c·(i−KSIG)/depth) при i>KSIG база 0
  gam    Φ(g·t), φ(g·t) — зсув режиму воріт     база 1.0
  bet    m = mp·Φ + b·s·φ                       база 1.0
  dia    d *= (1+c) — діагональ Cov окремо      база 0
  mel    +K=3 член ряду Мелера з вагою c        база 0
  g3     Еджворт −(c·γ₃/6)·t·φ, γ₃ з ЛОКАЛЬНОЇ
         оцінки третього кумулянта кроку        база 0
⟹ `edg` дав стелю ×1.12 у похибці ⟹ від `g3` більше не чекаємо

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 кожна сім'я у своїй базовій точці відтворить спільне число  95%
 (б) `kap` підтвердить оптимум ≈0.9985 (репліка С100 на 1000 мереж)  70%
 (в) хоч одна НОВА сім'я дасть крос > ×1.02                         30%
 (г) `g3` дасть < ×1.03 (бо `edg` дав ×1.12 з ОРАКУЛЬНИМИ γ)        80%
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
KSIG = 8
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 200

if cp.cuda.runtime.getDeviceCount() != 1:
    print("ПІНУВАННЯ НЕ СПРАЦЮВАЛО", flush=True)
    sys.exit(2)

Ws, Ts = [], []
for fp in sorted(glob.glob(SRC + "/data-*.arrow")):
    if len(Ws) >= NNET:
        break
    tb = pa.ipc.open_stream(pa.memory_map(fp, "rb")).read_all()
    xw = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xw.type):
        xw = xw.flatten()
    Wa = np.asarray(xw).reshape(tb.num_rows, DEPTH, W_, W_)
    xt = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(xt.type):
        xt = xt.flatten()
    Ta = np.asarray(xt).reshape(tb.num_rows, W_)
    for r in range(tb.num_rows):
        if len(Ws) >= NNET:
            break
        Ws.append(np.ascontiguousarray(Wa[r], dtype=np.float32))
        Ts.append(Ta[r].astype(np.float64))
    del Wa, Ta
NNET = len(Ws)
print(f"scan102 · {NNET} мереж · ручки з шару {KSIG}", flush=True)
Tg = cp.asarray(np.array(Ts))


def closure(Wg, fam, val):
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    kap = 0.9985
    for i in range(DEPTH):
        w = Wg[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            k = kap
            if fam == "kap":
                k = val
            elif fam == "kapd":
                k = kap * (1.0 + val * (i - KSIG) / DEPTH)
            cpm = cpm * k
        s2 = cp.maximum(cp.diagonal(cpm), 1e-30)
        s = cp.sqrt(s2)
        mp = m @ w
        t = mp / s
        tg = t * val if (fam == "gam" and i > KSIG) else t
        Ph = ndtr(tg)
        ph = cp.exp(-0.5 * tg * tg) / SQ2PI
        b = val if (fam == "bet" and i > KSIG) else 1.0
        m_new = mp * Ph + b * (s * ph)
        if fam == "g3" and i > KSIG:
            # локальний третій кумулянт кроку для ГАУСОВОГО p дорівнює нулю;
            # беремо κ₃ від ПОПЕРЕДНЬОЇ активації, перенесеної лінійно:
            # γ₃(p) ≈ Σ_j w³·κ₃(h_j) / s³, κ₃(h_j) з формули ReLU(гаус)
            k3h = (s2 * s) * (ph * (2.0 * tg * tg + 1.0))
            g3 = (k3h @ (w * w * w)) / (s2 * s)
            m_new = m_new - (val / 6.0) * (g3 * (s * (tg * ph)))
        C = cpm * cp.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s)
        if fam == "mel" and i > KSIG:
            c3 = s * (tg * ph)
            C = C + (val / 6.0) * ((cpm / cp.outer(s, s)) ** 3) * cp.outer(c3, c3)
        d = (mp * mp + s2) * Ph + (mp * s) * ph - m_new * m_new
        if fam == "dia" and i > KSIG:
            d = d * (1.0 + val)
        C = C + cp.diag(d - cp.diagonal(C))
        m = m_new
    return m


def cos_err(M):
    num = (M * Tg).sum(1) ** 2
    den = (M * M).sum(1) * (Tg * Tg).sum(1)
    return cp.sqrt(cp.maximum(1.0 - num / cp.maximum(den, 1e-300), 0.0))


FAM = {
    "kap": ([0.9985], 0.9985),
    "kapd": ([0.0], 0.0),
    "gam": ([1.000, 1.010, 1.020, 1.035, 1.050, 1.080, 1.120, 1.180], 1.0),
    "bet": ([1.0], 1.0),
    "dia": ([0.0], 0.0),
    "mel": ([0.0], 0.0),
    "g3": ([0.0], 0.0),
}
E = {}
for fam, (vals, base) in FAM.items():
    for v in vals:
        acc = []
        for n in range(NNET):
            Wg = [cp.asarray(Ws[n][i]).astype(cp.float64) for i in range(DEPTH)]
            acc.append(closure(Wg, fam, v))
            del Wg
        E[(fam, v)] = cp.asnumpy(cos_err(cp.stack(acc)))
        cp.get_default_memory_pool().free_all_blocks()
    print(f"  {fam} готова", flush=True)

IA, IB = np.arange(0, NNET, 2), np.arange(1, NNET, 2)
print("\n" + "=" * 78)
print(f"{'сім\'я':>6} {'база':>10} {'best v':>9} {'in-sample':>11} {'×':>7} "
      f"{'крос A→B':>11} {'×':>7} {'B→A':>7} {'знак':>9}")
print("-" * 78)
for fam, (vals, bv) in FAM.items():
    b = float(E[(fam, bv)].mean())
    ei = {v: float(E[(fam, v)].mean()) for v in vals}
    vb = min(vals, key=lambda v: ei[v])
    va = min(vals, key=lambda v: E[(fam, v)][IA].mean())
    vbb = min(vals, key=lambda v: E[(fam, v)][IB].mean())
    e_ab = float(E[(fam, va)][IB].mean())
    e_ba = float(E[(fam, vbb)][IA].mean())
    bA = float(E[(fam, bv)][IB].mean())
    bB = float(E[(fam, bv)][IA].mean())
    win = int((E[(fam, vb)] < E[(fam, bv)]).sum())
    print(f"{fam:>6} {100*b:>9.4f}% {vb:>9.4f} {100*ei[vb]:>10.4f}% "
          f"{b/ei[vb]:>7.4f} {100*e_ab:>10.4f}% {bA/e_ab:>7.4f} "
          f"{bB/e_ba:>7.4f} {str(win)+'/'+str(NNET):>9}")
print("-" * 78)
print("🔒 усі сім'ї у своїй базовій точці мусять дати ОДНЕ Й ТЕ САМЕ число:",
      " ".join(f"{100*float(E[(f,FAM[f][1])].mean()):.4f}" for f in FAM))
