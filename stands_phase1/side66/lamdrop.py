"""ЧИ Є РОДИНИ `λ` БАЛАСТОМ? A/B НА ПОВНОМУ САБМІТІ. (швидкий, під дедлайн)

ЗВІДКИ (два незалежні рядки пам'яті):
  * «вибілювання 1.98×, якір+перенос 1.03×, `Q` 1.196×, `S`/`A`/`G` МЕРТВІ
    ⟹ 24 з 32 `λ` — баласт»;
  * розклад `‖b‖²`: обрізання 36% · **`λ` 55%** · вибілювання 5% · `v` ≈0
    ⟹ більшість ЗМІЩЕННЯ вносять самі канали.
Разом це означає: мертвий канал не просто не допомагає — він ДОДАЄ зміщення
замикання у відповідь. Викидання = один рядок, флопів менше (а ми під
підлогою ⟹ множник не зміниться).

🚨 БАЗА — ПОРТ `estimator_sub87.py` з усіма чотирма родинами.
🚨 `n_pairs` НЕ чіпаємо: варіанти відрізняються ЛИШЕ коефіцієнтами ⟹ флопи
   практично ті самі, порівняння чисте.

📣 ПЕРЕДБАЧЕННЯ: «лише Q» дасть 1.02-1.05×; викидання `Q` — помітно гірше.
🟢 ЖИВЕ при ≥1.02× (при підлозі множника це йде в бал один до одного).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI, GATE_N)
from jitprof import truth_means                                    # noqa: E402
from jitanim import bank_net, LAM_S, LAM_A, LAM_Q, LAM_G           # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 24
KSIG = 8
Z8 = (0.0,) * 8


def _phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQRT_2PI


def run(wc, npair, cl, dev, seed, ls, la, lq, lg):
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
            v = (v + (ls[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + la[i - 1] * (xm - m_cl[i - 1])
                 + lq[i - 1] * (xq - q_cl[i - 1])
                 + lg[i - 1] * (gate - P_cl[i - 1]))
        else:
            x = torch.relu(p)
    return x.double().sum(0) / denom - v


VAR = {
    'ПОВНИЙ sub87': (LAM_S, LAM_A, LAM_Q, LAM_G),
    'без S': (Z8, LAM_A, LAM_Q, LAM_G),
    'без A': (LAM_S, Z8, LAM_Q, LAM_G),
    'без G': (LAM_S, LAM_A, LAM_Q, Z8),
    'без S і A': (Z8, Z8, LAM_Q, LAM_G),
    'ЛИШЕ Q': (Z8, Z8, LAM_Q, Z8),
    'без Q (контроль)': (LAM_S, LAM_A, Z8, LAM_G),
    'БЕЗ УСІХ (лише v)': (Z8, Z8, Z8, Z8),
}


def main(nets=24, dev=DEV, ndraw=NDRAW):
    print('📣 ПЕРЕДБАЧЕНО: «лише Q» 1.02-1.05×; без Q — помітно гірше')
    print('🟢 ЖИВЕ при ≥1.02×\n', flush=True)
    acc = {k: [] for k in VAR}
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            mt, _ = truth_means(wc, dev, 310_000 + 877 * n)
            m31 = mt[d - 1]
            cl = closure_parts(wc, min(KSIG, d - 1), dev, extras=True)
            for k in range(ndraw):
                sd = 95_000_000 + 977 * n + k
                for nm, (a, b, c, e) in VAR.items():
                    pr = run(wc, npair, cl, dev, sd, a, b, c, e)
                    acc[nm].append(float((pr - m31).pow(2).sum()))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    base = float(np.mean(acc['ПОВНИЙ sub87']))
    print(f'\n{nets} мереж × {ndraw} жеребів')
    print(f'{"варіант":>22} {"MSE":>12} {"× проти sub87":>15}')
    for nm in VAR:
        v = float(np.mean(acc[nm]))
        print(f'{nm:>22} {v:>12.5e} {base/v:>15.4f}')
    best = max((base / float(np.mean(acc[nm])), nm) for nm in VAR
               if nm != 'ПОВНИЙ sub87')
    print(f'\n🔑 НАЙКРАЩИЙ: {best[1]} = {best[0]:.4f}×  (🟢гейт 1.02)')
    print('📊 БАЗОВИЙ БАЛ 1.954e-07 · множник на підлозі ⟹ виграш у MSE '
          'іде в бал один до одного')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
