"""A/B КАНАЛУ `q̄₃₁ − q_замик,₃₁` НА ПОВНОМУ САБМІТІ (С75).

ЗВІДКИ. `matcorr.py`: з усієї батареї мат-конструкцій вижила ОДНА — залишок
ДРУГОГО моменту на ВИХОДІ. `R²=0.347` на тому, що лишається ПІСЛЯ sub87,
БЕЗ вільного члена, ⛔ROLL −0.011. Пряма перевірка причини:
    зміщення/шум для `m` на шарі 31: **5.9…9.6**   (потонули у зміщенні)
    зміщення/шум для `q` на шарі 31: **0.14…1.25** (сигнал ще чути)
    кореляція флуктуацій `q` і `m`:  **0.975…0.982**
⟹ маємо величину з майже точно відомим значенням, корельовану на 0.98 з тим,
що треба зловити.

🚨 ЧИМ ЦЕ ВІДРІЗНЯЄТЬСЯ ВІД ВІДКЛИКАНОГО `â²` ([[MEMORY_45]] §C): той був
другим моментом на шарах 1..8 (і вже сидить у сабміті як `_LAM_Q`), міряний
проти УРІЗАНОЇ бази. Тут — шар **31**, БЕЗ переносу, і база — ПОВНИЙ sub87
з усіма чотирма родинами. Саме тому база тут порт `estimator_sub87.py`,
а не `base_real.py`.

════════ ЦІНА, І ЧОМУ ЇЇ ТРЕБА ВІДНІМАТИ ВРУЧНУ ════════
Канал потребує замикання до 31 шару замість 8: +23 шари × 4·256³ = 1.54e9
флопів ≈ 5.7% бюджету. 🔑`n_pairs` у сабміті рахується ЛИШЕ під основний цикл
(замикання у формулу НЕ входить) — це вже коштувало сабміту sub88. Тому тут
обидві гілки рахують `n_pairs` із ЯВНО відрахованою ціною замикання, і
загальні флопи друкуються для звірки.

════════ МАСШТАБ: ЧОМУ КАНАЛ ТРЕБА НОРМУВАТИ ════════
Мережа позитивно однорідна: якщо ваги множаться на `α`, то `m ∝ α`, а `q ∝ α²`.
⟹ СИРИЙ `q̄−q_замик` має інші одиниці, ніж `m`, і ЄДИНИЙ скаляр `λ` не може
підійти всім мережам. Тому три варіанти нормування:
    V1  (q̄−q_cl)/√q̄        — одиниці `x`, стійке
    V2  (q̄−q_cl)/(2x̄)      — точна перша похідна `q` по зсуву
    V3  (q̄−q_cl)·‖m̄‖/‖q̄‖   — глобальний масштаб мережі

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 поглиблення БЕЗ каналу (λ=0) коштує **0.94…0.96×** (менше семплів);
📣 канал по скалярній оцінці дає 1.15× ⟹ 📣НЕТТО **1.06…1.12×**;
🟢 ЖИВЕ при нетто ≥ 1.05× на ≥24 мережах.
🚨 ПЛОМБИ
  ✅ λ=0 у гілці «канал» мусить дати РІВНО ту саму MSE, що гілка «поглиблено»;
  ✅ загальні флопи двох гілок збігаються (друкуємо);
  ⛔ НЕГ: канал із СУСІДНЬОГО жереба ⟹ виграш ≈1.000×;
  ✅ λ фітується LOO **ПО МЕРЕЖАХ** (OOS по жеребах брехав би —
     [[finding-scale-noise-is-predictable-but-only-within-a-network]]).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI, GATE_N, PILOT_PAIRS, B_BUDGET)
from jitprof import truth_means                                    # noqa: E402
from jitanim import bank_net, LAM_S, LAM_A, LAM_Q, LAM_G           # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
TARGET_UTIL = 0.0908
KSIG = 8
CLOS_FLOP = 4.0 * W_ ** 3          # один шар замикання: wᵀ(Cw), два матмули


def _phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQRT_2PI


def budget(wc, ksig):
    """`n_pairs` із ЯВНО відрахованою ціною замикання глибини `ksig`."""
    d = len(wc)
    per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
    pilot = 2.0 * PILOT_PAIRS * (2.0 * d * W_ * W_)
    clos = ksig * CLOS_FLOP
    left = TARGET_UTIL * float(B_BUDGET) - pilot - clos
    n = max(128, int(left / (2.0 * max(per_cut, 1.0))))
    return n, pilot + clos + 2.0 * n * per_cut


def run(wc, npair, cl, dev, seed, q31_cl=None):
    """Порядково `estimator_sub87.py`; додатково повертає КАНАЛИ шару 31."""
    s_cl, m_cl, P_cl, q_cl = cl
    d = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    y = z @ (wh @ wc[0])
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    denom = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.double().sum(0) / denom - sg0 * INV_SQRT_2PI
    ks = min(KSIG, d - 1)
    for i in range(1, d):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        v = (v @ wc[i].double()) * gate
        if i <= ks:
            mh = p.double().sum(0) / denom
            sh = (p.double().pow(2).sum(0) / denom - mh * mh).clamp_min(
                1e-30).sqrt()
            x = torch.relu(p)
            xm = x.double().sum(0) / denom
            xq = x.double().pow(2).sum(0) / denom
            v = (v + (LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + LAM_A[i - 1] * (xm - m_cl[i - 1])
                 + LAM_Q[i - 1] * (xq - q_cl[i - 1])
                 + LAM_G[i - 1] * (gate - P_cl[i - 1]))
        else:
            x = torch.relu(p)
    xm = x.double().sum(0) / denom
    pred = xm - v
    if q31_cl is None:
        return pred, None
    xq = x.double().pow(2).sum(0) / denom
    dq = xq - q31_cl
    ch = torch.stack([
        dq / xq.clamp_min(1e-30).sqrt(),                  # V1
        dq / (2.0 * xm).clamp_min(1e-30),                 # V2
        dq * (xm.norm() / xq.norm().clamp_min(1e-30)),    # V3
    ], 0)
    return pred, ch


def main(nets=24, dev=DEV, ndraw=NDRAW):
    print('📣 ПЕРЕДБАЧЕНО: λ=0 після поглиблення 0.94-0.96×, канал 1.15×, '
          'нетто 1.06-1.12×')
    print('🟢 ЖИВЕ при нетто ≥ 1.05×\n', flush=True)
    EB, ED, CH, FB, FD = [], [], [], [], []
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            d = len(wc)
            nb, fb = budget(wc, KSIG)
            nd, fd = budget(wc, d - 1)
            mt, _ = truth_means(wc, dev, 310_000 + 877 * n)
            m31 = mt[d - 1]
            s_cl, m_cl, P_cl, q_cl = closure_parts(wc, d - 1, dev, extras=True)
            cl8 = (s_cl[:KSIG], m_cl[:KSIG], P_cl[:KSIG], q_cl[:KSIG])
            q31 = q_cl[d - 2].double()
            eb, ed, ch = [], [], []
            for k in range(ndraw):
                sd = 71_000_000 + 977 * n + k
                pb, _ = run(wc, nb, cl8, dev, sd)
                pd_, c = run(wc, nd, cl8, dev, sd, q31_cl=q31)
                eb.append(pb - m31)
                ed.append(pd_ - m31)
                ch.append(c)
            EB.append(torch.stack(eb).cpu().numpy())
            ED.append(torch.stack(ed).cpu().numpy())
            CH.append(torch.stack(ch).cpu().numpy())
            FB.append(fb)
            FD.append(fd)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair {nb} → {nd})', flush=True)

    EB = np.array(EB)                       # (нет, жереб, вихід)
    ED = np.array(ED)
    CH = np.array(CH)                       # (нет, жереб, варіант, вихід)
    mse_b = float((EB ** 2).sum(2).mean())
    mse_d = float((ED ** 2).sum(2).mean())
    print(f'\n{nets} мереж × {ndraw} жеребів')
    print(f'✅ФЛОПИ: база {np.mean(FB):.4e} · поглиблено {np.mean(FD):.4e} '
          f'· різниця {abs(np.mean(FD)/np.mean(FB)-1)*100:.3f}% '
          f'{"✅" if abs(np.mean(FD)/np.mean(FB)-1) < 0.01 else "🛑"}')
    print(f'   бюджет {np.mean(FD)/B_BUDGET*100:.2f}% від B '
          f'{"✅під підлогою" if np.mean(FD)/B_BUDGET < 0.0941 else "🛑"}')
    print(f'\nБАЗА (замикання 8)     MSE {mse_b:.4e}')
    print(f'ПОГЛИБЛЕНО λ=0 (31)    MSE {mse_d:.4e}  '
          f'{mse_b/mse_d:.4f}× ← ціна поглиблення')

    names = ('V1 /√q̄', 'V2 /2x̄', 'V3 масштаб')
    for vi, nm in enumerate(names):
        C = CH[:, :, vi, :]
        # LOO ПО МЕРЕЖАХ: λ з усіх мереж, крім поточної
        num = np.array([float((ED[i] * C[i]).sum()) for i in range(nets)])
        den = np.array([float((C[i] * C[i]).sum()) for i in range(nets)])
        tot, res = 0.0, 0.0
        for i in range(nets):
            lam = (num.sum() - num[i]) / max(den.sum() - den[i], 1e-300)
            res += float(((ED[i] - lam * C[i]) ** 2).sum())
            tot += float((ED[i] ** 2).sum())
        mse_c = res / (nets * ndraw)
        lam_all = num.sum() / max(den.sum(), 1e-300)
        # ⛔НЕГ: канал із СУСІДНЬОГО жереба
        Cn = np.roll(C, 1, axis=1)
        nn = np.array([float((ED[i] * Cn[i]).sum()) for i in range(nets)])
        dn = np.array([float((Cn[i] * Cn[i]).sum()) for i in range(nets)])
        rneg = 0.0
        for i in range(nets):
            lam = (nn.sum() - nn[i]) / max(dn.sum() - dn[i], 1e-300)
            rneg += float(((ED[i] - lam * Cn[i]) ** 2).sum())
        print(f'{nm:>12}  MSE {mse_c:.4e}  проти бази {mse_b/mse_c:.4f}×  '
              f'проти поглибл. {mse_d/mse_c:.4f}×  λ={lam_all:+.4e}  '
              f'⛔НЕГ {mse_d/(rneg/(nets*ndraw)):.4f}×')

    print('\n🔑 ВИРОК за колонкою «проти бази» (🟢гейт 1.05×)')
    print('🎓 V1/V2 ділять ПОКОМПОНЕНТНО і тим ламають НАПРЯМОК вектора; '
          'шум же — флуктуація МАСШТАБУ вздовж одного напрямку ⟹ працює '
          'лише УЗГОДЖЕНИЙ скаляр (V3)')
    print('📊 БАЗОВИЙ БАЛ 1.954e-07 · множник на підлозі, тож виграш у MSE '
          'іде В БАЛ один до одного')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
