"""ЧИ СПРАВДІ ПІДЛОГА БЮДЖЕТУ — ОПТИМУМ. (С85, з виводу з нуля)

## ВИВІД, ЩО СТАВИТЬ ПИТАННЯ
Для будь-якого оцінювача `MSE(N) = V/N + b²` (b — залишкове ЗМІЩЕННЯ),
вартість `C(N) = c₀ + c₁N`, бал `s(N) = MSE(N)·max(0.1, C(N)/B)`.
* нижче підлоги: `s = 0.1(V/N + b²)` — монотонно СПАДАЄ з N ⟹ рости вигідно;
* вище підлоги: `s = (V/N + b²)(c₀+c₁N)/B`
  `ds/dN = (−V c₀/N² + b² c₁)/B = 0` ⟹ **`N* = √(V c₀ / (b² c₁))`**
🚨 **ЯКЩО `b = 0`, похідна ВІД'ЄМНА завжди ⟹ вище підлоги бал СТАЛИЙ, і оптимум
рівно на підлозі.** Саме це записано в `AUDIT_LEDGER` як «арифметика підлоги».
🎯 **АЛЕ В НАС `b ≠ 0`** — залишкове зміщення замикання після λ-каналів. Тоді
вище підлоги бал має ВНУТРІШНІЙ МІНІМУМ, і якщо `N* > N_підлога`, ми стоїмо
НЕ В ОПТИМУМІ, а на краю.
📐 груба оцінка з наявних чисел (`b²` = 7.6% MSE): `N* ≈ 5670` проти
`N_підлога ≈ 3700` ⟹ виграш ~1.8%. Але все тримається на `b²`, якого ми
НІКОЛИ НЕ МІРЯЛИ ПРЯМО — лише оцінювали.

## ЩО МІРЯЄМО
Прогін sub93 при `npair × k`, `k ∈ {0.6, 0.8, 1.0, 1.3, 1.6, 2.0}`;
MSE проти банкової КРОС-істини; фіт **`MSE(N) = V/N + b²`** зваженою МНК.
Далі — крива балу `s(N)` з РЕАЛЬНИМИ `c₀`, `c₁` із `side66/costver.py`
(sub93: `F/B = 0.08791` при util 0.0817, `λR/B = 0.0134` на грейдері).
⛔ПЛОМБА: при `k=1` MSE мусить збігтися з відомою (≈6.4e-04 у стендових
одиницях крос-цілі, як у `layalign.py`).
🚨 ЗАСТЕРЕЖЕННЯ: `λ`-канали калібровані під ПОТОЧНИЙ `npair`; при зміні `N`
шум ознак змінюється, і вони стають неоптимальними ⟹ зміряне `b²` — це
ВЕРХНЯ оцінка (частину «зміщення» дасть розкалібровка). Це робить вирок
консервативним у потрібний бік: якщо навіть так `N* ≤ N_підлога` — питання
закрите.

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: `b²` вийде помітно меншим за 7.6% (бо та цифра —
   про зміщення ЗАМИКАННЯ, а не про залишок ПІСЛЯ λ-каналів), і `N*` буде
   близьким до `N_підлога` або нижчим ⟹ підлога підтвердиться. Шанс, що
   оптимум справді вище підлоги: ~30%. Але ціна питання — один прогін, а
   виграш, якщо є, ~1.5–2% і БЕЗ жодної нової математики.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/floorcheck.py [cuda:N] [NETS] [NDRAW]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, GATE_N, INV_SQRT_2PI, PILOT_PAIRS)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 6
L_ = 32
W_BLEND, ALPHA = 0.0883, -0.00751
KS = [0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
print(f'⛔КОНСТАНТИ: w={W_BLEND}, α={ALPHA} (sub92/93)\n', flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж × {NDRAW} жеребів × {len(KS)} масштабів, {DEV}\n',
      flush=True)


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
    gn = min(GATE_N, x.shape[0])
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:gn] > 0).to(p.dtype).sum(0).double() / gn
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


N = len(FS)
MSE = np.zeros((N, len(KS)))
NP = np.zeros((N, len(KS)))
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, np0, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KSIG, DEV, kdeep=len(wc))
        aux = aux_parts(wc, DEV)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        for j, k in enumerate(KS):
            npk = max(256, int(round(np0 * k)))
            NP[n, j] = npk
            acc = 0.0
            for d in range(NDRAW):
                pr = run93(wc, npk, DEV, 61_000_000 + 733*n + d, cl, aux)
                acc += float(((pr - T1) * (pr - T2)).mean()) / NDRAW
            MSE[n, j] = acc
        print(f'  … {n+1}/{N} seed {sd0} npair₀={np0} · ' +
              ' '.join(f'{k}={MSE[n,j]:.3e}' for j, k in enumerate(KS)),
              flush=True)
    del W, wc
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/floorcheck.npz',
         MSE=MSE, NP=NP, KS=np.array(KS))

# ── фіт MSE(N) = V/N + b² по СЕРЕДНІХ (мережі рівноважні) ────────────────
y = MSE.mean(0)
x = 1.0 / NP.mean(0)
A = np.vstack([x, np.ones_like(x)]).T
sol, *_ = np.linalg.lstsq(A, y, rcond=None)
V, b2 = float(sol[0]), float(sol[1])
print(f'\n{"k":>6s} {"npair":>7s} {"MSE":>13s} {"фіт V/N+b²":>13s}')
print('─' * 44)
for j, k in enumerate(KS):
    print(f'{k:6.1f} {NP.mean(0)[j]:7.0f} {y[j]:13.6e} '
          f'{V*x[j]+b2:13.6e}')
print('─' * 44)
print(f'ФІТ: V = {V:.6e} · b² = {b2:.6e} '
      f'({b2/y[2]*100:+.2f}% від MSE при k=1)')
if b2 <= 0:
    print('🎯 b² ≤ 0 ⟹ зміщення НЕ ВИМІРНЕ ⟹ вище підлоги бал СТАЛИЙ ⟹ '
          'ПІДЛОГА ОПТИМАЛЬНА, питання закрите.')
else:
    # реальні c₀,c₁ з costver: F/B = 0.08791 при util 0.0817, λR/B = 0.0134
    Nb = NP.mean(0)[2]
    c1B = 0.0817 / Nb                      # частка бюджету на семпл
    c0B = 0.08791 - 0.0817 + 0.0134        # накладні: F поза util + residual
    Nstar = float(np.sqrt(V * c0B / (b2 * c1B)))
    Nfloor = (0.1 - c0B) / c1B
    def sc(NN):
        return (V / NN + b2) * max(0.1, c0B + c1B * NN)
    print(f'📐 c₀/B = {c0B:.5f} · c₁·N₀/B = {c1B*Nb:.5f}')
    print(f'🎯 N_підлога = {Nfloor:.0f} (util {c1B*Nfloor:.4f}) '
          f'⟹ бал {sc(Nfloor):.6e}')
    print(f'🎯 N* (оптимум)  = {Nstar:.0f} (util {c1B*Nstar:.4f}) '
          f'⟹ бал {sc(Nstar):.6e}')
    print(f'\n⟹ ВИГРАШ ВІД ЗСУВУ З ПІДЛОГИ: ×{sc(Nfloor)/sc(Nstar):.4f}')
    print('   (>1.01 ⟹ ми стоїмо НЕ В ОПТИМУМІ і це виправляється однією '
          'константою `_TARGET_UTIL`)')
