"""ЗМІЩЕННЯ ОБРІЗАННЯ — АНАЛІТИЧНО, ДВОМА ЗАМИКАННЯМИ (С84)

## ЗВІДКИ (метафора ЮЗЕРА: «мікробугорок на початку зміщує всю траєкторію»)
У прямому прочитанні метафора СПРОСТОВАНА: коефіцієнт передачі збурення через
шар `G = 0.22` (`side66/selfcon.py`) — крок ГАСИТЬ бугорок у 4.5×, а якір із
шару 0 приходить до виходу у 8.1× меншим (С75).
🔑 АЛЕ вона працює буквально для РІШЕННЯ, а не для величини: `cut_budget91`
дивиться 128 пілотних пар і НАЗАВЖДИ викидає нейрони з `max h < 0`. Це не
доданок до траєкторії — це ЗМІНА САМОЇ МЕРЕЖІ, крізь яку йде решта вибірки,
і воно не гасне.

## 📊 РОЗМІР УЖЕ ЗМІРЯНИЙ, ПОПРАВКА — НІ
С82 ([[reference-bank-truth-is-the-right-target]]): зсув «обрізана проти повної»
= **1.5e-04…6.1e-04** при RMS робочого 1.4e-03 ⟹ **до 20% MSE**, і він
БІЛЬШИЙ за обидва шуми, тобто систематичний.
С81: «нейрон із `p_on≈1%` на 256 пілотних семплах виглядає мертвим, пропуск
7.6%» — і там же дослівно: **«як внесок У БАЛ НЕ МІРЯНО»**.
📌 Обрізання САМЕ ПО СОБІ — чистий плюс (без нього −30.25% MSE), бо дає більше
пар за той самий бюджет. Питання не «чи обрізати», а «чи компенсувати зсув».

## 🔑 ІДЕЯ: ЗСУВ ОБЧИСЛЮВАНИЙ БЕЗ ВИБІРКИ
Замикання вже рахується на ОБРІЗАНИХ вагах (`m_deep`). Порахувати його ще раз
на ПОВНИХ — і різниця
    `D = m_deep(ПОВНІ W) − m_deep(ОБРІЗАНІ wc)`
є АНАЛІТИЧНОЮ оцінкою зсуву обрізання: детермінована, без краплі шуму вибірки.
💰 ЦІНА: 32 матмули 256×256 = 6.7e7 флоп ≈ **0.8% бюджету** (вибірка — 7400×32
таких), і йде ПОВЗ `n_pairs`, бо `per_sample_cut` не змінюється.
🚨 Вихід останнього шару НЕ обрізається по стовпцях (`if i < depth-1`), тож
обидва `m_deep` — 256-вимірні й віднімаються прямо.

## ЩО МІРЯЄМО
`pr' = pr + β·D`, `β` — один скаляр, ОПТИМУМ АНАЛІТИЧНИЙ (прогноз лінійний):
    `MSE(β) = ⟨e₁e₂⟩ + β⟨D(e₁+e₂)⟩ + β²⟨D²⟩`
⛔ ПЛОМБИ І КОНТРОЛІ:
 (а) `β=0` ⟹ РІВНО базовий MSE;
 (б) ⛔ЧУЖА МЕРЕЖА: `D` від сусідньої мережі (зсув обрізання — ВЛАСТИВІСТЬ
     МЕРЕЖІ, тож жеребний контроль тут не є нулем — урок С75);
 (в) ⛔ШУМ тієї ж норми;
 (г) ЧЕСНИЙ LOO по мережах — `β` фітується без поточної мережі;
 (д) істина — БАНК (ПОВНА мережа!), крос-ціль.
📌 Міряємо на sub93 як базі (найкращий готовий кандидат).

📣 ПЕРЕДБАЧЕНО: знак `β` має вийти ДОДАТНИМ (обрізання ВІДКИДАЄ внесок ⟹ наш
   прогноз занижений, `D>0` його повертає) і СТАБІЛЬНИМ по мережах — це
   головна ознака, що ловимо систематику, а не шум. Розмір: зсув ~20% MSE, але
   замикання знає його неточно (власна похибка форми 0.19% для `m`), тож чекаю
   ×1.01–1.04. Якщо `β` вийде близьким до 1.0 — різниця замикань відтворює
   зсув майже точно, і це сильний результат. Якщо LOO ≈ ⛔чужа мережа —
   різниця несе лише спільну корпусну складову, і треба буде дивитись, чи
   виживає вона як ОДНА КОНСТАНТА для всіх мереж.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/trimbias.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, GATE_N, INV_SQRT_2PI, PILOT_PAIRS,
                        ALPHA, W_BLEND)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
L_, NDRAW = 32, 8
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
    """sub93: sub92 + локальна чистка δ_Q вибірковими κ₃,κ₄."""
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
        pd = p.double()
        p2 = pd * pd
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
            k4 = (e4 - 4.0 * mh * e3 + 6.0 * mh2 * e2
                  - 3.0 * mh2 * mh2 - 3.0 * sh2 * sh2)
            dq = ((k3 / (sh2 * sh)) * (AA[i - 1] / 3.0)
                  - (k4 / (sh2 * sh2)) * (BB[i - 1] / 12.0))
            xm = x.double().sum(0) / den
            xq = x.double().pow(2).sum(0) / den
            v = (v + (SP.LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + SP.LAM_A[i - 1] * (xm - m_cl[i - 1])
                 + SP.LAM_Q[i - 1] * ((xq - q_cl[i - 1]) - dq)
                 + SP.LAM_G[i - 1] * (gate - P_cl[i - 1]))
        if i == depth - 1:
            last = x.double().sum(0) / den
    ours = last - v
    return ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)


NV = 3                       # 0: справжня D · 1: ⛔чужа мережа · 2: ⛔шум
E12 = np.zeros(len(FS))
LIN = np.zeros((len(FS), NV))
QUA = np.zeros((len(FS), NV))
DN = np.zeros(len(FS))       # ‖D‖ / ‖pr‖ — розмір зсуву
SEAL0 = 0.0
DB = []

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KSIG, DEV, kdeep=len(wc))
        aux = aux_parts(wc, DEV)
        # 🔑 ЗАМИКАННЯ НА ПОВНИХ ВАГАХ — те, чого сабміт не робить
        _, _, _, _, m_full = closure91(W, KSIG, DEV, kdeep=L_)
        D = m_full - cl[4]
        # ⛔ПЛОМБА: обидва `m_deep` мусять бути однакової довжини (256)
        SEAL0 = max(SEAL0, float(abs(len(m_full) - len(cl[4]))))
        DB.append(D.clone())
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        gen = torch.Generator(device=DEV); gen.manual_seed(555 + n)
        prev = DB[n - 1] if n > 0 else DB[0]
        rnd = torch.randn(D.shape, dtype=torch.float64, device=DEV,
                          generator=gen)
        VS = [D, prev, rnd * (D.norm() / rnd.norm())]
        for d in range(NDRAW):
            sd = 99_000_000 + 733 * n + d
            pr = run93(wc, npair, DEV, sd, cl, aux)
            e1, e2 = pr - T1, pr - T2
            es = e1 + e2
            E12[n] += float((e1 * e2).sum()) / NDRAW
            for k in range(NV):
                LIN[n, k] += float((VS[k] * es).sum()) / NDRAW
                QUA[n, k] += float((VS[k] * VS[k]).sum()) / NDRAW
            if d == 0:
                DN[n] = float(D.norm() / pr.norm())
    del W, wc
    if n > 1:
        DB[n - 2] = None
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}  ‖D‖/‖pr‖ = {DN[n]:.4f}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/trimbias.npz',
         E12=E12, LIN=LIN, QUA=QUA, DN=DN)

print(f'\n⛔ПЛОМБА розмірність m_deep: {SEAL0:.0f} '
      f'{"✅" if SEAL0 == 0 else "🚨 РІЗНІ ДОВЖИНИ"}')
base = E12.mean()
print(f'⛔ПЛОМБА (а) β=0 ⟹ базовий MSE sub93 = {base:.6e}')
print(f'🔑 РОЗМІР ЗСУВУ ‖D‖/‖pr‖: сер {DN.mean():.4f} '
      f'[{DN.min():.4f}…{DN.max():.4f}]\n')

NM = ['🎯 справжня D', '⛔ чужа мережа', '⛔ шум тієї ж норми']
print(f'{"варіант":>22s} {"β* (усі)":>10s} {"MSE(усі)":>13s} {"×":>8s} '
      f'{"LOO MSE":>13s} {"× LOO":>8s} {"β: ст.в":>9s}')
print('─' * 92)
res = []
for k in range(NV):
    bg = -LIN[:, k].sum() / (2.0 * max(QUA[:, k].sum(), 1e-300))
    va = (E12 + bg * LIN[:, k] + bg * bg * QUA[:, k]).mean()
    lo = 0.0
    bs = []
    for n in range(len(FS)):
        o = [j for j in range(len(FS)) if j != n]
        b = -LIN[o, k].sum() / (2.0 * max(QUA[o, k].sum(), 1e-300))
        bs.append(b)
        lo += E12[n] + b * LIN[n, k] + b * b * QUA[n, k]
    lo /= len(FS)
    res.append(base / lo)
    print(f'{NM[k]:>22s} {bg:10.4f} {va:13.6e} {base/va:8.4f} '
          f'{lo:13.6e} {base/lo:8.4f} {np.std(bs):9.5f}')

print('\n🎯 ВИРОК')
net = res[0] / max(res[1], res[2])
print(f'   справжня D: ×{res[0]:.4f} · найкращий контроль ×{max(res[1],res[2]):.4f}')
print(f'   ЧИСТИЙ понад контроль: ×{net:.4f}')
if res[0] > 1.002 and net > 1.002:
    print(f'   ⟹ 🟢 ЖИВЕ ⟹ бал 1.928e-07 → {1.928e-07/res[0]:.4e}')
    print('      наступне: зашити в естиматор (ціна ~0.8% бюджету, під підлогою)')
else:
    print('   ⟹ ⛔ різниця замикань НЕ відтворює зсув обрізання корисно')
print(f'\n📊 БАЛ: sub87 1.954e-07 · sub92 1.946e-07 · sub93 1.928e-07')
