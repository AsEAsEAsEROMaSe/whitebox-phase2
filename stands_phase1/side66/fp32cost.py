"""ЧИ ОБМЕЖУЄ НАС fp32 (питання ЮЗЕРА, С83)

## ДВА ОКРЕМІ ПИТАННЯ
1. **ТОЧНІСТЬ**: скільки MSE додає fp32 проти fp64 — чи є взагалі що вигравати;
2. **ЦІНА**: `C = F + 1e11·R`, множник `max(0.1, C/B)`, `B=2.72e11`.
   Наш `F = 0.0817·B = 2.22e10`, підлога тримається доки `C < 2.72e10`
   ⟹ запас **5.0e9 одиниць = 0.050 с** wall-time. Зараз `R ≈ 15 мс` = 1.5e9
   ⟹ вільно ще **3.5e9 = 0.035 с**, тобто fp64 безкоштовний, якщо сповільнює
   менш ніж у **2.3×**. 🚨 На P104 fp64 у 32× повільніше, але грейдер — CPU
   (сендбокс дозволяє лише numpy/scipy), де fp64 зазвичай ~2× повільніше.

## ЩО МІРЯЄМО (12 мереж, банкова істина, крос-ціль, ТІ САМІ семпли)
Прогноз рахується ДВІЧІ на ОДНОМУ жеребі, різниця лише в типі:
  A ⛔ усе fp32 (як у сабміті)
  B 🎯 усе fp64
  C 🎯 замикання fp64, вибірка fp32 (найдешевший гібрид: замикання не залежить
       від розміру вибірки, тож його fp64 коштує ~нуль)
  D 🎯 акумулятори сум fp64, решта fp32 (`Σx`, `Σx²`, `Σp³` — те, де
       накопичується 7400 доданків)
Плюс ПРЯМИЙ вимір часу numpy на CPU для fp32/fp64 матмулу розміру сабміта.

⛔ ПЛОМБИ: (а) A мусить відтворити `ab_files` MSE до шуму жеребу;
   (б) різниця A−B мусить бути ≪ самої похибки, інакше десь катастрофічне
       скорочення; (в) ⛔контроль: різниця A−A на двох викликах = РІВНО 0.

📣 ПЕРЕДБАЧЕНО: різниця fp32↔fp64 буде ~1e-6 відносно проти нашої похибки
   ~3e-2 ⟹ **fp32 не обмежує, запас ≥1000×**. Ризик лише в `sh²=Σp²/n − mh²`
   і `xq − xm²` (різниця близьких), але там mh мале проти s. Якщо ж різниця
   виявиться ≳1e-3 — це прихована втрата, і тоді гібрид C (замикання fp64)
   безкоштовно її прибирає.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/fp32cost.py [cuda:N] [NETS]
"""
import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import (cut_budget91, _phi, _Phi, W_, INV_SQRT_2PI,  # noqa: E402
                        GATE_N, KSIG, ALPHA, W_BLEND,
                        LAM_S, LAM_A, LAM_Q, LAM_G, PILOT_PAIRS)
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
L_ = 32
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def closure(wc, dev, dt, kdeep):
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=dt, device=dev)
    C = torch.eye(n0, dtype=dt, device=dev)
    s_o, m_o, q_o, p_o = [], [], [], []
    for i in range(kdeep):
        w = wc[i].to(dt)
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        if 1 <= i <= KSIG:
            s_o.append(s); p_o.append(Ph)
        m = mp * Ph + s * ph
        if i == kdeep - 1:
            break
        C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
        d = (mp*mp + s*s)*Ph + mp*s*ph - m*m
        C = C + torch.diag(d - C.diagonal())
        if 1 <= i <= KSIG:
            m_o.append(m); q_o.append(C.diagonal() + m*m)
    return s_o, m_o, q_o, p_o, m


def predict(wc, npair, dev, seed, dt_cl, dt_acc, dt_mm):
    """dt_cl — тип замикання, dt_acc — тип акумуляторів сум, dt_mm — матмулів."""
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                     device=dev).to(dt_mm)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).to(dt_mm)
    y = z @ (wh @ wc[0].to(dt_mm))
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    den = float(2*npair)
    sg0 = wc[0].to(dt_acc).pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.to(dt_acc).sum(0)/den - sg0*INV_SQRT_2PI
    s_cl, m_cl, q_cl, P_cl, m_deep = closure(wc, dev, dt_cl, len(wc))
    for i in range(1, len(wc)):
        p = x @ wc[i].to(dt_mm)
        gate = (p[:GATE_N] > 0).to(dt_acc).sum(0)/GATE_N
        v = (v @ wc[i].to(dt_acc)) * gate
        pd = p.to(dt_acc)
        mh = pd.sum(0)/den
        sh = (pd.pow(2).sum(0)/den - mh*mh).clamp_min(1e-30).sqrt()
        x = torch.relu(p)
        xm = x.to(dt_acc).sum(0)/den
        xq = x.to(dt_acc).pow(2).sum(0)/den
        if i - 1 < len(s_cl) and i <= KSIG:
            d = (_phi(mh/sh)*(sh - s_cl[i-1].to(dt_acc)),
                 xm - m_cl[i-1].to(dt_acc), xq - q_cl[i-1].to(dt_acc),
                 gate - P_cl[i-1].to(dt_acc))
            v = (v + LAM_S[i-1]*d[0] + LAM_A[i-1]*d[1]
                 + LAM_Q[i-1]*d[2] + LAM_G[i-1]*d[3])
        if i == len(wc)-1:
            last = x.to(dt_acc).sum(0)/den
    ours = last - v
    return (ours + W_BLEND*((1.0+ALPHA)*m_deep.to(dt_acc) - ours)).double()


F32, F64 = torch.float32, torch.float64
VAR = [('A ⛔ усе fp32 (сабміт)', F32, F32, F32),
       ('B 🎯 усе fp64', F64, F64, F64),
       ('C 🎯 замикання fp64, решта fp32', F64, F32, F32),
       ('D 🎯 акумулятори fp64, решта fp32', F32, F64, F32)]
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
NDRAW = 8
MSE = np.zeros((len(FS), len(VAR)))
DIFF = np.zeros((len(FS), len(VAR)))
SEAL = 0.0

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        for d in range(NDRAW):
            sd = 88_000_000 + 733*n + d
            prs = [predict(wc, npair, DEV, sd, *v[1:]) for v in VAR]
            if n == 0 and d == 0:      # ⛔контроль (в): A двічі = РІВНО 0
                SEAL = float((prs[0] - predict(wc, npair, DEV, sd,
                                               *VAR[0][1:])).abs().max())
            for k, pr in enumerate(prs):
                MSE[n, k] += float(((pr-T1)*(pr-T2)).sum())/NDRAW
                DIFF[n, k] += float((pr - prs[1]).norm()
                                    / prs[1].norm())/NDRAW
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}', flush=True)

print(f'\n⛔КОНТРОЛЬ (в) A двічі: max|Δ| = {SEAL:.2e} '
      f'{"✅ детерміновано" if SEAL == 0 else "🚨 недетермінізм"}')
print(f'\n{len(FS)} мереж × {NDRAW} жеребів, крос-ціль, банк\n')
print(f'{"варіант":36s} {"MSE":>12s} {"× до fp32":>10s} {"‖Δ‖/‖pr‖ до fp64":>18s}')
print('─'*82)
m0 = MSE[:, 0].mean()
for k, v in enumerate(VAR):
    print(f'{v[0]:36s} {MSE[:,k].mean():12.5e} {m0/MSE[:,k].mean():10.4f} '
          f'{DIFF[:,k].mean():18.3e}')
rel = np.sqrt(MSE[:, 0].mean()/np.mean([float(np.sum(TRUTH2[int(np.load(f)['seed'])][0]**2))
                                        for f in FS]))
print(f'\n🔑 НАША ВІДНОСНА ПОХИБКА: √(MSE/‖T‖²) = **{rel:.4f}** ({rel*100:.2f}%)')
print(f'🔑 ВІДНОСНА РІЗНИЦЯ fp32↔fp64: **{DIFF[:,0].mean():.3e}**')
print(f'   ⟹ запас у **{rel/max(DIFF[:,0].mean(),1e-30):.0f}×** '
      f'{"✅ fp32 НЕ ОБМЕЖУЄ" if rel/max(DIFF[:,0].mean(),1e-30) > 100 else "🚨 fp32 З’ЇДАЄ ТОЧНІСТЬ"}')

# ── ЦІНА: numpy на CPU, матмул розміру сабміта
print(f'\n💰 ЦІНА fp64 НА CPU (numpy, розмір сабміта {2*npair}×256×256)')
A32 = np.random.randn(2*npair, 256).astype(np.float32)
B32 = np.random.randn(256, 256).astype(np.float32)
A64, B64 = A32.astype(np.float64), B32.astype(np.float64)
for nm, a, b in (('fp32', A32, B32), ('fp64', A64, B64)):
    a @ b
    t0 = time.perf_counter()
    for _ in range(20):
        a @ b
    dt = (time.perf_counter()-t0)/20
    print(f'   {nm}: {dt*1e3:7.3f} мс/матмул ⟹ {dt*1e3*L_:7.2f} мс на 32 шари')
print(f'\n📊 БЮДЖЕТ: F=2.22e10, підлога тримається доки C<2.72e10 ⟹ запас '
      f'0.050 с; зараз R≈15 мс ⟹ fp64 дозволений, якщо сповільнить <2.3×')
