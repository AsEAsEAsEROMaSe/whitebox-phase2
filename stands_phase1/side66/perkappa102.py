"""perkappa102 — ІДЕЯ ЮЗЕРА «МЕТОД ПОШУКУ, ЩО ПІДЛАШТОВУЄТЬСЯ ПІД КОЖНУ МЕРЕЖУ».

ЩО ВЖЕ ЗМІРЯНО І ЧОМУ ЦЕ ІНШЕ:
  split101  фіт ВИХОДУ по мережі в базисі (m,s,t,Φ,φ)  стеля ×1.0269  ⛔
  blend101  W_BLEND по мережі                          реалізовно ×0.9667 ⛔
  ⟹ обидва підлаштовують РЕЗУЛЬТАТ. Тут підлаштовуємо ПАРАМЕТР РЕКУРСІЇ,
    який змінює всю траєкторію 23 шарів і чіпає `m` та `C` РАЗОМ.
С100 міряв κ як ГЛОБАЛЬНУ константу (LOO обрав 0.9985 усіма 50 фолдами) — це
доводить стійкість глобального значення, а НЕ відсутність кращого κ_i.

ПИТАННЯ, І ВОНО ПРО СТЕЛЮ:
  1. чи існує κ_i (по мережі), помітно кращий за глобальний κ = 0.9985?
  2. якщо так — чи знаходиться він БЕЗ істини, з власної МК?
  3. 🔒 чи не є весь виграш МАСШТАБОМ, який α бере даром?
     ⟹ МІРЯЄМО НА КОСИНУСНІЙ ПОХИБЦІ (закон С101-в: LOO/крос/чужа масштаб
        НЕ ЛОВЛЯТЬ; сира похибка бреше у ×1.9)

ЦІЛЬ: істина з банку МК (шумна, але незміщена). Для кожної мережі скануємо κ
і дивимось на КОСИНУСНУ похибку `1 − cos²(m_deep(κ), truth)`, у якій масштаб
уже поділений — саме той функціонал, що входить у бал після `_ai`.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) оптимуми κ_i розкидані ширше за 0.002 (крок сітки)              60%
 (б) оракульний κ_i дасть >×1.10 проти глобального на косинусній     25%
 (в) 🔒 κ_i, знайдений по МК, корелює з оракульним, r > 0.5          35%
 (г) 🔒 медіана κ_i буде ≈0.9985 (відтворить глобальний оптимум)     75%

ВИРОК: (б) < ×1.10 ⟹ гілка мертва НЕЗАЛЕЖНО від ціни прогонів — бо навіть
безкоштовний оракульний per-network κ не окупить 6% за прогін.
"""
import glob
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")   # ПОСТІЙНИЙ шлях: /tmp чиститься

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KSIG = 8
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NTRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 4194304   # "істина"
NMC = int(sys.argv[3]) if len(sys.argv) > 3 else 6144        # бюджет сабміту
KAPPAS = np.arange(0.988, 1.0081, 0.0010)

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"perkappa102 · {NNET} мереж · істина {NTRUE} · МК {NMC} · "
      f"{len(KAPPAS)} значень κ", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def closure(Wg, kap):
    """наше замикання з множником `kap` на cp при i > KSIG; віддає m шару 31"""
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    for i in range(DEPTH - 1):
        w = Wg[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * kap
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        mn = mp * Ph + s * ph
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        Co = cpm * cp.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s)
        C = Co + cp.diag(d - cp.diagonal(Co))
        m = mn
    w = Wg[DEPTH - 1]
    cpm = w.T @ C @ w * kap
    s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
    mp = m @ w
    t = mp / s
    return mp * ndtr(t) + s * phi(t)


def mc_out(Wg, ns, seed, whiten=True):
    """пряма МК з вибілюванням — та сама гілка, що в сабміті"""
    rng = cp.random.default_rng(seed)
    acc = cp.zeros(W_, dtype=cp.float64)
    n = 0
    while n < ns:
        nb = min(8192, ns - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        z = cp.concatenate([z, -z], axis=0)
        if whiten:
            zz = z.astype(cp.float64)
            cov = (zz.T @ zz) / zz.shape[0]
            # той самий вибілювач, що в сабміті: inv(cholesky(cov)).T
            wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
            z = (zz @ wh).astype(cp.float32)
        h = z
        for i in range(DEPTH):
            h = cp.maximum(h @ Wg[i], 0.0)
        acc += h.astype(cp.float64).sum(0)
        n += nb
    return acc / (2 * ns)


def cos_err(pred, truth):
    """1 − cos² ⟹ РІВНО MSE після оптимального масштабу (закон С100)"""
    pn = float((pred * pred).sum())
    tn = float((truth * truth).sum())
    ip = float((pred * truth).sum())
    return math.sqrt(max(1.0 - ip * ip / max(pn * tn, 1e-300), 0.0))


rows = []
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float64))
          for i in range(DEPTH)]
    Wf = [w.astype(cp.float32) for w in Wg]
    truth = mc_out(Wf, NTRUE // 2, 70001 + net)          # "істина"
    mcs = mc_out(Wf, NMC // 2, 313370 + net)             # бюджет сабміту
    eo, em = [], []
    for k in KAPPAS:
        md = closure(Wg, float(k))
        eo.append(cos_err(md, truth))
        em.append(cos_err(md, mcs))
    rows.append((net, np.array(eo), np.array(em)))
    io, im = int(np.argmin(eo)), int(np.argmin(em))
    print(f"  мережа {net}: κ*(істина)={KAPPAS[io]:.4f} "
          f"κ*(МК)={KAPPAS[im]:.4f} "
          f"похибка глоб={eo[np.argmin(np.abs(KAPPAS-0.9985))]*100:.4f}% "
          f"оракул={eo[io]*100:.4f}%", flush=True)
    del Wg, Wf
    cp.get_default_memory_pool().free_all_blocks()

EO = np.array([r[1] for r in rows])
EM = np.array([r[2] for r in rows])
ig = int(np.argmin(np.abs(KAPPAS - 0.9985)))
kbest_o = KAPPAS[EO.argmin(1)]
kbest_m = KAPPAS[EM.argmin(1)]
glob = EO[:, ig].mean()
best_glob = EO.mean(0).min()
orac = EO[np.arange(len(rows)), EO.argmin(1)].mean()
real = EO[np.arange(len(rows)), EM.argmin(1)].mean()

print("\n" + "=" * 78)
print(f"глобальний κ=0.9985            {100*glob:.4f}%   ×1.000")
print(f"найкращий ЄДИНИЙ κ={KAPPAS[EO.mean(0).argmin()]:.4f}       "
      f"{100*best_glob:.4f}%   ×{glob/best_glob:.4f}")
print(f"ОРАКУЛЬНИЙ κ_i (СТЕЛЯ)         {100*orac:.4f}%   ×{glob/orac:.4f}")
print(f"κ_i, знайдений по власній МК   {100*real:.4f}%   ×{glob/real:.4f}")
print(f"\nκ*(істина): медіана {np.median(kbest_o):.4f} sd {kbest_o.std():.4f} "
      f"діапазон [{kbest_o.min():.4f}, {kbest_o.max():.4f}]")
print(f"κ*(МК)    : медіана {np.median(kbest_m):.4f} sd {kbest_m.std():.4f}")
print(f"🔒 corr(κ*істина, κ*МК) = {np.corrcoef(kbest_o, kbest_m)[0,1]:.4f}")
print("\nВИРОК: оракульна стеля < ×1.10 ⟹ мертво НЕЗАЛЕЖНО від ціни прогонів")
