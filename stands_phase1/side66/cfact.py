"""ЧИ ВИЖИВАЄ `q₃₁` ПРИ ФАКТОРИЗОВАНОМУ `C`? (вимір ТОЧНОСТІ, не виграшу)

ЗВІДКИ. Канал `q̄₃₁−q_замик` дає 1.0635× над своєю базою, але замикання до 31
шару коштує 6.3% семплів (0.9519×) ⟹ нетто 1.012×. Єдиний важіль — здешевити
САМЕ глибоке замикання.

ІДЕЯ. Дорогі тут ЛИШЕ два матмули `cp = wᵀ(Cw)` = 4·256³ на шар. Якщо брати
`C ≈ LLᵀ` рангу `r`, то `cp = (Lᵀw)ᵀ(Lᵀw)` коштує `4r·256²`, тобто у `256/r`
разів дешевше; плюс перефакторизація ~`4(r+p)·256²`. Разом ≈ `r/128` від повної.
🔑 ДІАГОНАЛЬ лишається ТОЧНОЮ (її вписує та сама формула `dd`), а саме вона
дає `s`, від якого залежить усе інше.
🔑 Мілкі шари (1..8) рахуємо ПОВНІСТЮ — там `C` майже одинична, ранг великий.
   Факторизуємо лише 9..31, де ранг активацій уже колапсує (PR 161.6 → 2.5).

════════ ЩО САМЕ МІРЯЄМО ════════
Не виграш, а ДОПУСК. Канал живий, поки зміщення замикання на `q` не переросло
шум вибірки на `q`. Зміряно раніше: `зміщ/шум|q = 0.14…1.25`.
⟹ факторизація має додати похибку `q`, МАЛУ проти того самого шуму.
Тому друкуємо: `‖q_ранг − q_повне‖ / ‖шум q̄‖` — прямо в одиницях допуску.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 r=64 — додаток ≲0.1 шуму (непомітно) · r=32 — 0.2-0.5 · r=16 — ~1 (на межі)
📣 r=8 — понад 2 шуми, тобто вже гірше за сам сигнал ⟹ мертво.
🟢 ЖИВЕ, якщо є `r` з додатком ≤0.35 шуму І економією ≥3× (тоді нетто ≈1.05×).
🚨 ПЛОМБИ
  ✅ r=256 (без урізання) мусить відтворити повне замикання БІТ-У-БІТ;
  ✅ флопи рахуються формулою і друкуються поруч із виміряним прискоренням.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_, INV_SQRT_2PI             # noqa: E402
from jitanim import bank_net                                       # noqa: E402
from q31ab import budget, KSIG                                     # noqa: E402
from q31dir import _phi, _Phi                                      # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
RANKS = (256, 128, 64, 32, 16, 8)
DEEP_FROM = KSIG          # мілкі шари 1..8 — завжди повне замикання


def closure_rank(wc, dev, r):
    """Замикання з `C ≈ LLᵀ` рангу `r` на шарах глибше `DEEP_FROM`."""
    d = len(wc)
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    q = None
    for i in range(d):
        w = wc[i].double()
        if i >= DEEP_FROM and r < C.shape[0]:
            # top-r власний підпростір; діагональ усе одно буде вписана точно
            ev, V = torch.linalg.eigh(C)
            L = V[:, -r:] * ev[-r:].clamp_min(0).sqrt()
            M = L.T @ w
            cp = M.T @ M
        else:
            cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(
            ph / s, ph / s)
        dd = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(dd - C.diagonal())
        q = dd + m * m
    return m, q


def flops(d, r):
    """Флопи замикання: повне на 1..DEEP_FROM, рангове далі."""
    full = DEEP_FROM * 4.0 * W_ ** 3
    deep = (d - DEEP_FROM) * (4.0 * min(r, W_) * W_ ** 2
                              + (4.0 * (min(r, W_) + 8) * W_ ** 2
                                 if r < W_ else 0.0))
    return full + deep


def main(nets=8, dev=DEV):
    print('📣 ПЕРЕДБАЧЕНО: r=64 ≲0.1 шуму · r=32 0.2-0.5 · r=16 ~1 · r=8 >2')
    print('🟢 ЖИВЕ при додатку ≤0.35 шуму І економії ≥3×\n', flush=True)
    acc = {r: [] for r in RANKS}
    spd = {}
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            d = len(wc)
            nd, _ = budget(wc, d - 1)
            _, q_full = closure_rank(wc, dev, W_)
            # шум вибірки на `q̄` — це і є одиниця допуску
            dq = []
            for k in range(24):
                g = torch.Generator(device=dev).manual_seed(5_500_000 + 77 * n
                                                            + k)
                z = torch.randn(nd, W_, device=dev, generator=g)
                x = torch.cat([z, -z], 0)
                cv = (x.T @ x) / float(2 * nd)
                ev, V = torch.linalg.eigh(cv.double())
                wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
                a = x @ wh
                for i in range(d):
                    a = torch.relu(a @ wc[i])
                dq.append(a.double().pow(2).mean(0))
            Q = torch.stack(dq)
            noise = float((Q - Q.mean(0)).pow(2).sum(1).mean() ** 0.5)
            for r in RANKS:
                _, q_r = closure_rank(wc, dev, r)
                acc[r].append(float((q_r - q_full).norm()) / noise)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    d0 = 32
    f_full = flops(d0, W_)
    print(f'\n{nets} мереж · допуск = шум вибірки на `q̄`\n')
    print(f'{"ранг":>6} {"додаток / шум":>15} {"флопи замик.":>14} '
          f'{"економія":>10} {"% бюджету":>11}')
    B = 272e9
    for r in RANKS:
        f = flops(d0, r)
        a = np.mean(acc[r])
        spd[r] = f_full / f
        print(f'{r:>6} {a:>15.4f} {f:>14.3e} {f_full/f:>9.2f}× '
              f'{f/B*100:>10.3f}%')
    print(f'\n✅ПЛОМБА r=256 (без урізання): {np.mean(acc[256]):.2e} '
          f'{"✅" if np.mean(acc[256]) < 1e-9 else "🛑 НЕ ТОТОЖНО"}')
    ok = [r for r in RANKS if r < W_ and np.mean(acc[r]) <= 0.35
          and spd[r] >= 3.0]
    print(f'🔑 ПРИДАТНІ РАНГИ: {ok if ok else "НЕМА"}')
    if ok:
        r = max(ok)
        # нетто = канал 1.0635 × (ціна нової глибини)
        base_cost = flops(d0, W_) / B
        new_cost = flops(d0, r) / B
        # 6.3% семплів коштували 0.9519× ⟹ лінійно за часткою бюджету
        price = 1.0 - (new_cost - DEEP_FROM * 4.0 * W_ ** 3 / B) / 0.0908 * (
            1.0 - 0.9519) / ((base_cost - DEEP_FROM * 4.0 * W_ ** 3 / B)
                             / 0.0908)
        print(f'   при r={r}: ціна ≈{price:.4f}× ⟹ нетто ≈'
              f'{1.0635 * price:.4f}×')
    print('📊 БАЗОВИЙ БАЛ 1.954e-07 · лідер 8e-9 ⟹ відставання 24×')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
