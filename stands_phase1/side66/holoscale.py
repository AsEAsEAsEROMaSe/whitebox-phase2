"""ПЕРСОНАЛІЗОВАНИЙ МАСШТАБ ЧЕРЕЗ `⟨p,m_deep⟩` — ВИГРАШ У БАЛІ (С84)

## ЗВІДКИ
`side66/holo.py`: спостережне `ĉ = ⟨p,m_deep⟩/‖p‖²` корелює з ОРАКУЛЬНИМ
`c* = ⟨p,T⟩/‖p‖²` МІЖ мережами на **−0.5954** (⛔RND −0.1251), `R²=0.3545` —
вдвічі сильніше за найкращий предиктор С83 (+0.355, `R²`=0.126, ×1.0097).
За тією ж калібровкою (виграш ≈ `ρ²·(стеля−1)`, стеля ×1.0911) чекаємо ~×1.03.
🚨 Перша версія `holo.py` міряла ТАВТОЛОГІЮ (корел 1.0000 і в ⛔контролю).

## ЩО МІРЯЄМО
`pr' = ĉ_pred·pr`, `ĉ_pred = a + b·ĉ` — регресія `c*` на `ĉ`, **LOO ПО МЕРЕЖАХ**
(2 параметри; закон 11.76%·k/N ⟹ штраф 0.49% проти сигналу ~3%).
`MSE(c) = c²‖p‖² − c⟨p,T1+T2⟩ + ⟨T1,T2⟩` — аналітично для будь-якого `c`.
⛔ КОНТРОЛІ: RND-опора · одна корпусна константа · ЧУЖА мережа · оракул (стеля).

📣 ПЕРЕДБАЧЕНО: ×1.02–1.03. Ризик: `ĉ` варіює ВЧЕТВЕРО сильніше за `c*`
   (0.00189 проти 0.00049) ⟹ він переважно відображає `⟨p,b⟩`, і регресія
   мусить це стиснути нахилом ≪1. Якщо LOO ≤1.005 — не переживає перенесення.

## СТАРА ШАПКА
ГОЛОГРАФІЧНИЙ ТРЮК: ЧИ ВИДНО `⟨T,e⟩` ЧЕРЕЗ ІНТЕРФЕРЕНЦІЮ З `m_deep` (С84)

## ЗВІДКИ — АНАЛОГІЯ ЮЗЕРА ПРО ГОЛОГРАМИ
Голограма записує ФАЗУ, якої детектор не бачить (він міряє інтенсивність),
через ІНТЕРФЕРЕНЦІЮ з опорним променем.
🚨 У нас РІВНО ця проблема, і вона записана як фундаментальний блокер
([[finding-per-network-personalization-needs-an-unobservable-projection]]):
персоналізація по мережі закрита, бо оптимум залежить від **`⟨T,e⟩`** —
ЛІНІЙНОЇ проєкції похибки на істину, «а всі статистики КВАДРАТИЧНІ».

## 🔑 ІДЕЯ: `m_deep` ЯК ОПОРНИЙ ПРОМІНЬ
`p = T + e`, `m_deep` — ДЕТЕРМІНОВАНА оцінка істини із замикання. Тоді
    `⟨m_deep, p⟩ − ‖m_deep‖² = ⟨m_deep, e⟩ + ⟨m_deep, T − m_deep⟩`
Ліва частина ПОВНІСТЮ СПОСТЕРЕЖНА (обидві величини є в рантаймі).
Праворуч перший доданок **ЛІНІЙНИЙ ПО `e`** — це і є «фаза»; другий —
систематичний зсув, ОДНАКОВИЙ для всіх жеребів мережі.
⟹ 🎯 якщо `m_deep ≈ T`, то `⟨m_deep,e⟩ ≈ ⟨T,e⟩` і теза «всі статистики
квадратичні» ХИБНА: лінійна по `e` інформація доступна.

## ЩО МІРЯЄМО (48 мереж × 24 жереби, банк, T1/T2)
На кожному жеребі:
  ОРАКУЛ   `y = ⟨T, e⟩`, де `e = p − T` (T з банку; крос-ціль тут не потрібна,
           бо величина ЛІНІЙНА по T ⟹ беремо середнє T=(T1+T2)/2, а шум
           оцінюємо різницею T1,T2);
  СПОСТЕРЕЖНЕ `x = ⟨m_deep, p⟩ − ‖m_deep‖²`;
  ⛔контроль `x_rnd` з випадковим опорним вектором тієї ж норми;
  ⛔контроль `x_own` = `‖p‖² − ⟨m_deep,p⟩` (квадратична величина — має НЕ
           корелювати з фазою краще, ніж `x`).
🔑 ГОЛОВНЕ ЧИСЛО: кореляція `x` з `y` **ВСЕРЕДИНІ мережі** (по жеребах) — саме
вона потрібна для персоналізації. Кореляція МІЖ мережами не годиться: С83
показала, що там +0.71 при −0.83 всередині
([[finding-per-network-personalization-needs-an-unobservable-projection]]).

⛔ ПЛОМБИ:
 (а) `⟨T,e⟩` мусить мати НУЛЬОВЕ середнє по жеребах (бо `E[e]≈b` мале, але
     перевіряємо явно — якщо ні, ловимо зміщення, а не фазу);
 (б) шум оракула: та сама величина з `T1` і з `T2` окремо ⟹ їхня розбіжність;
 (в) ⛔RND-опора мусить дати корел ≈0.

📣 ПЕРЕДБАЧЕНО: `⟨m_deep,e⟩` і `⟨T,e⟩` відрізняються на `⟨b,e⟩`, де
   `b = m_deep − T` має ‖b‖/‖T‖ ≈ 0.6% ⟹ якщо `e` не корелює з `b` особливо
   сильно, кореляція має бути ВИСОКОЮ (≥0.9). Але є пастка: `x` містить
   `⟨m_deep, e⟩`, а `e` — це похибка ВСЬОГО прогнозу, тоді як для
   персоналізації потрібна проєкція на істину саме тієї частини, що
   налаштовується. Тож навіть при високій кореляції треба буде окремо
   перевірити, чи дає це виграш у балі. Якщо корел ВСЕРЕДИНІ <0.3 — трюк не
   працює, і теза «всі статистики квадратичні» встоїть.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/holo.py [cuda:N] [NETS] [NDRAW]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, GATE_N, INV_SQRT_2PI, PILOT_PAIRS)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 24
L_ = 32
# 🚨 КОНСТАНТИ ЯВНО — урок `feedback-new-stand-inherits-constants-not-just-truth`
W_BLEND, ALPHA = 0.0883, -0.00751
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
print(f'⛔КОНСТАНТИ СТЕНДУ: w={W_BLEND}, α={ALPHA} (sub92/sub93, НЕ порт!)\n',
      flush=True)

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
print(f'{len(FS)} мереж × {NDRAW} жеребів, {DEV}\n', flush=True)


def aux_parts(wc, dev):
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    AA, BB = [], []
    for i in range(len(wc)):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        if 1 <= i <= KSIG:
            s2ph = (s * s) * ph
            AA.append(s2ph); BB.append(s2ph * t)
        m = mp * Ph + s * ph
        if i < len(wc) - 1:
            d2 = (mp * mp + s * s) * Ph + mp * s * ph
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            C = C + torch.diag((d2 - m * m) - C.diagonal())
    return AA, BB


def run93(wc, npair, dev, seed, cl, aux):
    depth = len(wc)
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                     device=dev)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    y = z @ (wh @ wc[0])
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    den = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.double().sum(0) / den - sg0 * INV_SQRT_2PI
    s_cl, m_cl, q_cl, P_cl, m_deep = cl
    AA, BB = aux
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        pd = p.double(); p2 = pd * pd
        mh = pd.sum(0) / den
        e2 = p2.sum(0) / den
        sh2 = (e2 - mh * mh).clamp_min(1e-30)
        sh = sh2.sqrt()
        x = torch.relu(p)
        if i <= KSIG:
            e3 = (p2 * pd).sum(0) / den
            e4 = (p2 * p2).sum(0) / den
            mh2 = mh * mh
            k3 = e3 - 3.0 * mh * e2 + 2.0 * mh2 * mh
            k4 = (e4 - 4.0*mh*e3 + 6.0*mh2*e2 - 3.0*mh2*mh2 - 3.0*sh2*sh2)
            dq = ((k3 / (sh2 * sh)) * (AA[i-1] / 3.0)
                  - (k4 / (sh2 * sh2)) * (BB[i-1] / 12.0))
            xm = x.double().sum(0) / den
            xq = x.double().pow(2).sum(0) / den
            v = (v + (SP.LAM_S[i-1] * _phi(mh/sh)) * (sh - s_cl[i-1])
                 + SP.LAM_A[i-1] * (xm - m_cl[i-1])
                 + SP.LAM_Q[i-1] * ((xq - q_cl[i-1]) - dq)
                 + SP.LAM_G[i-1] * (gate - P_cl[i-1]))
        if i == depth - 1:
            last = x.double().sum(0) / den
    ours = last - v
    return ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)


PP = np.zeros(len(FS)); PT = np.zeros(len(FS)); TT = np.zeros(len(FS))
PM = np.zeros(len(FS)); PR_ = np.zeros(len(FS))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KSIG, DEV, kdeep=len(wc))
        aux = aux_parts(wc, DEV)
        md = cl[4]
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        Tm = 0.5 * (T1 + T2)
        gen = torch.Generator(device=DEV); gen.manual_seed(2024 + n)
        r = torch.randn(md.shape, dtype=torch.float64, device=DEV,
                        generator=gen)
        r = r * (md.norm() / r.norm())
        md2 = float((md * md).sum())
        for d in range(NDRAW):
            pr = run93(wc, npair, DEV, 107_000_000 + 733*n + d, cl, aux)
            PP[n] += float((pr * pr).sum()) / NDRAW
            PT[n] += float((pr * (T1 + T2)).sum()) / NDRAW
            TT[n] += float((T1 * T2).sum()) / NDRAW
            PM[n] += float((md * pr).sum()) / NDRAW
            PR_[n] += float((r * pr).sum()) / NDRAW
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/holoscale.npz',
         PP=PP, PT=PT, TT=TT, PM=PM, PR_=PR_)
N = len(FS)
cstar = PT / (2.0 * PP); chat = PM / PP; crnd = PR_ / PP
mse = lambda c: float((c**2 * PP - c * PT + TT).mean())
base, top = mse(np.ones(N)), mse(cstar)
print(f'\n⛔ПЛОМБА (а) c=1 ⟹ база sub93 = {base:.6e}')
print(f'⛔ПЛОМБА (б) c=c* ОРАКУЛ ⟹ СТЕЛЯ = {top:.6e} ⟹ ×{base/top:.4f}')
print(f'🔑 c*: сер {cstar.mean():.6f} розкид {cstar.std():.6f} · '
      f'ĉ: сер {chat.mean():.6f} розкид {chat.std():.6f}')
print(f'   корел ĉ↔c*: {np.corrcoef(chat, cstar)[0,1]:+.4f} · '
      f'⛔RND {np.corrcoef(crnd, cstar)[0,1]:+.4f}\n')
def loo_pred(x):
    o = np.zeros(N)
    for n in range(N):
        k = np.array([j for j in range(N) if j != n])
        o[n] = np.polyval(np.polyfit(x[k], cstar[k], 1), x[n])
    return o
def loo_const():
    return np.array([cstar[[j for j in range(N) if j != n]].mean()
                     for n in range(N)])
VAR = [('⛔ c=1 (sub93 як є)', np.ones(N)),
       ('🎯 ĉ=⟨p,m_deep⟩/‖p‖², LOO', loo_pred(chat)),
       ('⛔ RND-опора, LOO', loo_pred(crnd)),
       ('⛔ одна корпусна константа', loo_const()),
       ('⛔ ЧУЖА мережа', np.roll(loo_pred(chat), 1)),
       ('🟢 c* ОРАКУЛ (стеля)', cstar)]
print(f'{"варіант":>32s} {"MSE":>13s} {"×":>8s}')
print('─' * 56)
for nm, c in VAR:
    print(f'{nm:>32s} {mse(c):13.6e} {base/mse(c):8.4f}')
g = base / mse(VAR[1][1]); gr = base / mse(VAR[2][1])
gc = base / mse(VAR[3][1]); gx = base / mse(VAR[4][1])
print('\n🎯 ВИРОК')
print(f'   персоналізація через m_deep: ×{g:.4f} · ⛔RND ×{gr:.4f} · '
      f'⛔константа ×{gc:.4f} · ⛔чужа ×{gx:.4f} · 🟢стеля ×{base/top:.4f}')
print('   📌 С83 найкращий предиктор давав ×1.0097')
if g > 1.005 and g > max(gr, gc, gx) * 1.004:
    print(f'   ⟹ 🟢 ЖИВЕ ⟹ бал 1.928e-07 → {1.928e-07/g:.4e}')
else:
    print('   ⟹ ⛔ не переживає LOO/контролі')
print(f'\n📊 БАЛ: sub87 1.954e-07 · sub92 1.946e-07 · sub93 1.928e-07')
