"""ВЕРХНЯ МЕЖА: ЕДЖВОРТ ЛИШЕ НА ШАРАХ 1..KSIG, ПРЯМО В БАЛІ (С84)

## ЗВІДКИ ЦЕ
`side66/selfcon.py` знайшов те, чого не бачили чотири попередні вироки:
поправка `κ₃+κ₄` у ланцюзі дає **×1.88 на шарі 2**, ×1.89 на шарі 5,
×1.51 на шарі 8, перетинає 1.0 на шарі 24 і на виході дає ×0.944.
🎓 Усі попередні виміри дивились ТІЛЬКИ на вихід ⟹ бачили ×0.94 і ховали ×1.9.
Механізм: локальна похибка форми `A` НЕ РОСТЕ вглиб (1.6e-3), а успадковане
збурення росте до 1.0e-2 ⟹ на шарі 32 форма — 2.2% похибки за квадратом,
на шарі 2 — майже вся.

## 🎯 ЧОМУ ЦЕ МОЖЕ ВПЛИНУТИ НА БАЛ
Ознака сабміту `δ = x̄ − m_cl` на шарах 1..8 = ШУМ ВИБІРКИ + ЗМІЩЕННЯ ЗАМИКАННЯ.
Зміряно (С75, `snr.py`): `r=‖зміщ‖/‖шум‖` для родини `A` = 2.42 на шарі 8.
Залишок після оптимальної усадки = `S²·r²/(1+r²)` ⟹ падіння `r` прямо збільшує
знятий шум. Це б'є в ДИСПЕРСІЙНІ 82% бюджету, а не в зміщення 7.6%.

## ЩО МІРЯЄМО
Замикання рахується ДВІЧІ на тих самих обрізаних вагах:
  ⛔ БАЗА   — `closure91` як у сабміті;
  🎯 ЕДЖ   — та сама рекурсія + `κ₃,κ₄` ІСТИННІ (велика вибірка крізь `wc`),
             поправка ТІЛЬКИ на шарах < KSIG_E (глибше вона шкодить).
Опори `s,m,q,P` з кожного варіанту підставляються у `predict91`, MSE — на БАНКУ,
крос-ціль `(p−T1)(p−T2)`.

🚨 `λ` САБМІТА ПІДІГНАНІ ПІД НИНІШНЄ ЗАМИКАННЯ ⟹ порівняння без перепідбору
   НЕЧЕСНЕ (та сама пастка, що `_ALPHA` у С83). Тому свіп ГЛОБАЛЬНОГО множника
   `c` на всі чотири родини `λ → c·λ`, окремо для кожного варіанту, і
   порівнюються ОПТИМУМИ. `c` — одне число на варіант, тож фора мінімальна й
   ОДНАКОВА для обох.

⛔ ПЛОМБИ:
 (а) ЕДЖ при `γ=0` мусить дати РІВНО базові опори (max|Δ| = 0);
 (б) БАЗА при `c=1` мусить відтворити відомий MSE `sub91_port`;
 (в) істинні `κ` рахуються крізь `wc` (обрізані!), а не `W` — інакше опори
     й кумулянти належать різним об'єктам;
 (г) шум істини знято крос-ціллю (T1·T2), як завжди на банку.

📣 ПЕРЕДБАЧЕНО: оптимум `c` для ЕДЖ зсунеться ВГОРУ (зміщення менше ⟹ можна
   брати більший `λ`), і виграш у MSE буде 1.01–1.05×. Якщо <1.005× —
   верхня межа мала, і дешеві `γ` шукати не варто. Якщо >1.02× — напрям живий
   і питання переходить у «звідки взяти `γ` в рантаймі».

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/edgeksig.py [cuda:N] [NETS] [log2N]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import (cut_budget91, predict91, closure91, _phi, _Phi,  # noqa: E402
                        KSIG, W_)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LG = int(sys.argv[3]) if len(sys.argv) > 3 else 18
KSIG_E = 12          # поправка діє на шарах < KSIG_E (за selfcon перетин ~24)
NS, BS, L_ = 1 << LG, 1 << 13, 32
NDRAW = 8
CS = np.array([0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж × {NDRAW} жеребів × {NS} семплів для κ, {DEV}\n',
      flush=True)


def closure_edge(wc, dev, g1=None, g2=None, kedge=0):
    """`closure91` + Еджворт на шарах i < kedge. g1,g2 — списки по шарах."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    s_o, m_o, q_o, p_o = [], [], [], []
    for i in range(len(wc)):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        if 1 <= i <= KSIG:
            s_o.append(s); p_o.append(Ph)
        m = mp * Ph + s * ph
        d2 = (mp * mp + s * s) * Ph + mp * s * ph
        if i < kedge and g1 is not None:
            m = m - (g1[i] / 6.0) * mp * ph
            d2 = d2 + (g1[i] / 3.0) * s * s * ph
        if i < kedge and g2 is not None:
            m = m + (g2[i] / 24.0) * s * (t * t - 1.0) * ph
            d2 = d2 - (g2[i] / 12.0) * s * s * t * ph
        if i == len(wc) - 1:
            break
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s,
                                                                    ph / s)
        C = C + torch.diag((d2 - m * m) - C.diagonal())
        if 1 <= i <= KSIG:
            m_o.append(m); q_o.append(C.diagonal() + m * m)
    return s_o, m_o, q_o, p_o, m


def true_cumulants(wc, dev, ns, bs):
    """ІСТИННІ γ₁,γ₂ маргіналів `h` на КОЖНОМУ шарі, крізь ОБРІЗАНІ ваги."""
    d = len(wc)
    S1 = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in wc]
    S2 = [torch.zeros_like(s) for s in S1]
    S3 = [torch.zeros_like(s) for s in S1]
    S4 = [torch.zeros_like(s) for s in S1]
    gen = torch.Generator(device=dev); gen.manual_seed(4242)
    done = 0
    while done < ns:
        b = min(bs, ns - done)
        a = torch.randn(b, W_, device=dev, generator=gen)
        for i in range(d):
            h = a @ wc[i]
            hd = h.double()
            S1[i] += hd.sum(0); S2[i] += (hd * hd).sum(0)
            S3[i] += (hd ** 3).sum(0); S4[i] += (hd ** 4).sum(0)
            a = torch.relu(h)
        done += b
    g1, g2 = [], []
    for i in range(d):
        mh = S1[i] / ns
        v = (S2[i] / ns - mh * mh).clamp_min(1e-30)
        k3 = S3[i] / ns - 3 * mh * (S2[i] / ns) + 2 * mh ** 3
        k4 = (S4[i] / ns - 4 * mh * (S3[i] / ns) + 6 * mh ** 2 * (S2[i] / ns)
              - 3 * mh ** 4 - 3 * v * v)
        g1.append(k3 / v ** 1.5); g2.append(k4 / v ** 2)
    return g1, g2


import sub91_port as SP                                            # noqa: E402
LAM0 = (SP.LAM_S, SP.LAM_A, SP.LAM_Q, SP.LAM_G)
MSE = np.zeros((len(FS), 3, len(CS)))
SEAL_G0 = 0.0
SEAL_BASE = 0.0

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        g1, g2 = true_cumulants(wc, DEV, NS, BS)
        cl_b = closure91(wc, KSIG, DEV, kdeep=len(wc))
        # 🔑 РОЗДІЛЕННЯ: опори 1..8 з поправкою, але `m_deep` — З БАЗИ.
        #    Інакше ефект на канали змішаний з ефектом на бленд (×0.944).
        _ce = closure_edge(wc, DEV, g1, g2, KSIG_E)
        cl_e = (_ce[0], _ce[1], _ce[2], _ce[3], cl_b[4])
        # 🔑 ЗНАК: та сама поправка з ОБЕРНЕНИМ знаком. Якщо вона краща —
        #    у мене помилка знаку, а не відкриття про канали.
        _cn = closure_edge(wc, DEV, [-x for x in g1], [-x for x in g2], KSIG_E)
        cl_n = (_cn[0], _cn[1], _cn[2], _cn[3], cl_b[4])
        if n == 0:                       # ⛔ПЛОМБА (а): γ=0 ⟹ база
            z0 = [torch.zeros_like(x) for x in g1]
            cl_z = closure_edge(wc, DEV, z0, z0, KSIG_E)
            SEAL_G0 = max(float((a - b).abs().max())
                          for u in range(4) for a, b in zip(cl_z[u], cl_b[u]))
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        for v, cl in enumerate((cl_b, cl_e, cl_n)):
            for k, c in enumerate(CS):
                SP.LAM_S = tuple(c * x for x in LAM0[0])
                SP.LAM_A = tuple(c * x for x in LAM0[1])
                SP.LAM_Q = tuple(c * x for x in LAM0[2])
                SP.LAM_G = tuple(c * x for x in LAM0[3])
                acc = 0.0
                for d in range(NDRAW):
                    pr, _ = predict91(wc, npair, DEV, seed=91_000_000 + 733*n + d,
                                      cl=cl)
                    acc += float(((pr - T1) * (pr - T2)).sum()) / NDRAW
                MSE[n, v, k] = acc
                if v == 0 and abs(c - 1.0) < 1e-9 and n == 0:
                    SEAL_BASE = acc
    SP.LAM_S, SP.LAM_A, SP.LAM_Q, SP.LAM_G = LAM0
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/edgeksig.npz', MSE=MSE, CS=CS)

print(f'\n⛔ПЛОМБА (а) ЕДЖ(γ=0) ≡ БАЗА: max|Δ| = {SEAL_G0:.2e} '
      f'{"✅" if SEAL_G0 == 0 else "🚨"}')
print(f'⛔ПЛОМБА (б) БАЗА при c=1, мережа 0: MSE = {SEAL_BASE:.4e}\n')

mb = MSE[:, 0, :].mean(0); me = MSE[:, 1, :].mean(0); mn = MSE[:, 2, :].mean(0)
print('🔑 m_deep У ВСІХ ТРЬОХ — З БАЗИ ⟹ при c=0 усі три мусять збігтися '
      f'(ефект чисто на КАНАЛИ): {mb[0]:.5e} {me[0]:.5e} {mn[0]:.5e}')
print(f'\n{"c (×λ)":>8s} {"⛔ БАЗА":>13s} {"🎯 ЕДЖ":>13s} {"ЕДЖ/БАЗА":>10s} '
      f'{"🔄 −ЕДЖ":>13s} {"−ЕДЖ/БАЗА":>10s}')
print('─' * 76)
for k, c in enumerate(CS):
    print(f'{c:8.2f} {mb[k]:13.5e} {me[k]:13.5e} {mb[k]/me[k]:10.4f} '
          f'{mn[k]:13.5e} {mb[k]/mn[k]:10.4f}')
kn = int(np.argmin(mn))
print(f'\n🔄 ОБЕРНЕНИЙ ЗНАК: оптимум c = {CS[kn]:.2f}, MSE = {mn[kn]:.5e}, '
      f'×{mb[int(np.argmin(mb))]/mn[kn]:.4f} до бази '
      f'{"🚨 ЗНАК ПЕРЕПЛУТАНО" if mn[kn] < mb[int(np.argmin(mb))] * 0.995 else "✅ знак правильний"}')
kb, ke = int(np.argmin(mb)), int(np.argmin(me))
print(f'\n🎯 ВИРОК')
print(f'   ⛔ БАЗА  оптимум c = {CS[kb]:.2f}, MSE = {mb[kb]:.5e}')
print(f'   🎯 ЕДЖ   оптимум c = {CS[ke]:.2f}, MSE = {me[ke]:.5e}')
g = mb[kb] / me[ke]
print(f'   ВИГРАШ НА ОПТИМУМАХ: ×{g:.4f}  ⟹ бал 1.954e-07 → {1.954e-07/g:.4e}')
print(f'   при c=1 (без перепідбору): ×{mb[3]/me[3]:.4f}')
if g < 1.005:
    print('   ⟹ ⛔ верхня межа МАЛА навіть з ІСТИННИМИ γ — дешеві γ шукати НЕ ВАРТО')
elif g < 1.02:
    print('   ⟹ ⚠️ межа помірна; питання, чи переживе вона дешеві γ і 48 мереж')
else:
    print('   ⟹ 🟢 напрям ЖИВИЙ; далі — звідки взяти γ у рантаймі')
print(f'\n📊 БАЛ 1.954e-07 · це ВЕРХНЯ МЕЖА (γ істинні, у сабміті їх нема)')
