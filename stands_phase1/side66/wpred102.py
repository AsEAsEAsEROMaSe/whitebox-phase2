"""wpred102 — `eff_rank(C)` ЯК ПРОКСІ ДЛЯ `W*` ПО МЕРЕЖІ. Ціна 0.008% бюджету.

ЗВІДКИ: spec102 закрив «звести ранг до 2.8» (ручка P досягла рангу 1.38 і
зруйнувала відповідь у 123×), АЛЕ в його стовпчиках видно інше:
    corr( eff_rank(C₃₁) , похибка замикання ) = +0.67  на 16 мережах
    ранг 1.88·2.19·2.44 ⟹ похибки 0.326%·0.326%·0.366%
    ранг 6.37·7.02·7.68 ⟹ похибки 1.026%·0.575%·0.966%

ЧОМУ ЦЕ НЕ ПОВТОРЕННЯ `blend101` (де per-network W дав ×0.9667, ГІРШЕ константи):
    там проксі був `avg_variance`, і механізм смерті ЗМІРЯНИЙ — вона тягне
    ОБИДВІ компоненти однаково (нахили log V 0.5514, log E² 0.5015) ⟹ у
    ВІДНОШЕННІ скорочується, corr(W*, avg_var) = −0.034.
    `eff_rank` — властивість `C` ЗАМИКАННЯ ⟹ вона про `E²`, а не про обидві.

📏 ЗАКОН С98, ЯКИЙ ТУТ ВИРІШУЄ: «позитивна кореляція проксі не є виграшем —
   розкид проксі мусить збігатися з розкидом ЦІЛІ». Ціль тут — `W*_i`, а воно
   є ВІДНОШЕННЯМ ⟹ міряємо ОБИДВІ кореляції, і головна — з `log(V/E²)`.

ЩО МІРЯЄМО, ПО КОЖНІЙ МЕРЕЖІ:
   E²_i   похибка ЧИСТОГО замикання проти МК-істини 4.19e6      (косинусна)
   V_i    дисперсія МК-гілки при бюджеті сабміту — з ДВОХ незалежних
          реплік по 6144 пар: V̂ = (m₁ − m₂)²/2, усереднено по нейронах
   ER_i   eff_rank(C₃₁) — детерміновано, з нашої власної C
   W*_i   V_i/(V_i + E²_i) — оракульна вага бленду

ВИРОКИ (оголошені ДО прогону):
 (а) corr(log ER, log E²) > +0.5                                   65%
 (б) 🚨 ГОЛОВНИЙ: |corr(log ER, log(V/E²))| > 0.35                 35%
     ← якщо ні, ER скорочується у відношенні так само, як avg_var,
       і гілка мертва З ТИМ САМИМ механізмом
 (в) реалізовна W(ER) поб'є найкращу ФІКСОВАНУ W на кросі A→B      30%
 (г) 🔒 перемішаний ER дасть ×<1.00 (контроль)                     85%

ЦІЛЬ ЗВЕРХУ: оракульна W*_i дає ×1.0293 (blend101, 50 мереж).
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
KSIG, KAPPA = 8, 0.9985
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NTRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 4194304
NMC = int(sys.argv[3]) if len(sys.argv) > 3 else 6144

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"wpred102 · {NNET} мереж · істина {NTRUE} · МК {NMC}", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def closure(Wg):
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    er = None
    for i in range(DEPTH):
        w = Wg[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * KAPPA
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        mn = mp * Ph + s * ph
        if i == DEPTH - 1:
            return mn, er
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        Co = cpm * cp.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s)
        C = Co + cp.diag(d - cp.diagonal(Co))
        m = mn
        if i == DEPTH - 2:
            tr = float(cp.trace(C))
            er = tr * tr / max(float((C * C).sum()), 1e-300)


def mc_out(Wg, ns, seed):
    rng = cp.random.default_rng(seed)
    acc = cp.zeros(W_, dtype=cp.float64)
    n = 0
    while n < ns:
        nb = min(8192, ns - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        z = cp.concatenate([z, -z], axis=0)
        zz = z.astype(cp.float64)
        cov = (zz.T @ zz) / zz.shape[0]
        wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
        h = (zz @ wh).astype(cp.float32)
        for i in range(DEPTH):
            h = cp.maximum(h @ Wg[i], 0.0)
        acc += h.astype(cp.float64).sum(0)
        n += nb
    return acc / (2 * ns)


ER, E2, VV, TR = [], [], [], []
MD, M1, M2 = [], [], []
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float64))
          for i in range(DEPTH)]
    Wf = [w.astype(cp.float32) for w in Wg]
    truth = mc_out(Wf, NTRUE // 2, 70001 + net)
    md, er = closure(Wg)
    # масштаб знімаємо оптимально (закон: міряти на косинусній)
    a = float((md * truth).sum() / cp.maximum((md * md).sum(), 1e-30))
    e2 = float(((a * md - truth) ** 2).mean())
    # V: дві НЕЗАЛЕЖНІ репліки МК бюджету сабміту
    m1 = mc_out(Wf, NMC // 2, 313370 + net)
    m2 = mc_out(Wf, NMC // 2, 818181 + net)
    v = float((((m1 - m2) ** 2) / 2.0).mean())
    ER.append(er)
    E2.append(e2)
    VV.append(v)
    TR.append(cp.asnumpy(truth))
    MD.append(cp.asnumpy(a * md))
    M1.append(cp.asnumpy(m1))
    M2.append(cp.asnumpy(m2))
    print(f"  мережа {net}: ER={er:.2f}  E²={e2:.3e}  V={v:.3e}  "
          f"W*={v/(v+e2):.4f}", flush=True)
    del Wg, Wf
    cp.get_default_memory_pool().free_all_blocks()

ER = np.array(ER); E2 = np.array(E2); VV = np.array(VV)
WS = VV / (VV + E2)
lr, le, lv = np.log(ER), np.log(E2), np.log(VV)
print("\n" + "=" * 66)
print(f"(а) corr(log ER, log E²)        = {np.corrcoef(lr, le)[0,1]:+.4f}")
print(f"    corr(log ER, log V)         = {np.corrcoef(lr, lv)[0,1]:+.4f}")
print(f"🚨(б) corr(log ER, log(V/E²))   = {np.corrcoef(lr, lv-le)[0,1]:+.4f}"
      f"   ← ГОЛОВНИЙ")
print(f"    corr(log ER, W*)            = {np.corrcoef(lr, WS)[0,1]:+.4f}")
print(f"    нахил log E² на log ER      = {np.polyfit(lr, le, 1)[0]:+.4f}")
print(f"    нахил log V  на log ER      = {np.polyfit(lr, lv, 1)[0]:+.4f}")
print(f"    W* : сер {WS.mean():.4f} sd {WS.std():.4f} "
      f"діапазон [{WS.min():.4f}, {WS.max():.4f}]")

# (в) реалізовна W(ER) проти найкращої фіксованої — крос A/B
TRa = np.array(TR); MDa = np.array(MD)
MCa = (np.array(M1) + np.array(M2)) / 2.0


def mse_at(wvec):
    p = wvec[:, None] * MDa + (1.0 - wvec[:, None]) * MCa
    return ((p - TRa) ** 2).mean()


grid = np.arange(0.0, 0.401, 0.005)
h = NNET // 2
wf_a = grid[np.argmin([mse_at(np.full(h, g)) for g in grid])]
best_fixed_b = mse_at(np.full(NNET - h, wf_a))
# модель: W_i = 1/(1 + k0/ER_i^b) — підбір (k0,b) на A, вирок на B
bestp, bestv = None, 1e9
for b in np.arange(-3.0, 3.01, 0.25):
    for k0 in np.exp(np.arange(-4, 4.01, 0.25)):
        wv = 1.0 / (1.0 + k0 / np.maximum(ER[:h] ** b, 1e-12))
        vloss = ((wv[:, None] * MDa[:h] + (1 - wv[:, None]) * MCa[:h]
                  - TRa[:h]) ** 2).mean()
        if vloss < bestv:
            bestv, bestp = vloss, (k0, b)
k0, b = bestp
wvB = 1.0 / (1.0 + k0 / np.maximum(ER[h:] ** b, 1e-12))
mdl = ((wvB[:, None] * MDa[h:] + (1 - wvB[:, None]) * MCa[h:]
        - TRa[h:]) ** 2).mean()
orac = min(mse_at(np.where(np.arange(NNET) >= 0, WS, WS)), mse_at(WS))
print(f"\n(в) КРОС A→B  (підбір на {h}, вирок на {NNET-h})")
print(f"    найкраща ФІКСОВАНА W={wf_a:.3f}   {best_fixed_b:.6e}   ×1.0000")
print(f"    W(ER), k0={k0:.3f} b={b:+.2f}      {mdl:.6e}   "
      f"×{best_fixed_b/mdl:.4f}")
rng = np.random.default_rng(4242)
sh = [((lambda w: ((w[:, None]*MDa[h:] + (1-w[:, None])*MCa[h:] - TRa[h:])**2).mean())
       (1.0/(1.0 + k0/np.maximum(rng.permutation(ER[h:])**b, 1e-12))))
      for _ in range(200)]
print(f"🔒(г) ПЕРЕМІШАНИЙ ER ×200: сер ×{best_fixed_b/np.mean(sh):.4f}  "
      f"макс ×{best_fixed_b/np.min(sh):.4f}")
print(f"    🔒 ОРАКУЛЬНА W*_i (стеля)   {orac:.6e}   "
      f"×{best_fixed_b/orac:.4f}   [blend101: ×1.0293]")
print("\nВИРОК: (б) < 0.35 ⟹ ER скорочується у відношенні ⟹ мертво "
      "тим самим механізмом, що avg_variance")
