"""КАНАЛИ НА ХВОСТІ (шари L−k…L), А НЕ НА ГОЛОВІ. (С85, з виводу з нуля)

## ЗВІДКИ — ТОЧНА ТОТОЖНІСТЬ І ПРОФІЛЬ
`ReLU(x)=(x+|x|)/2` ⟹ **`E[a_L] = Σ_ℓ 2^{−(L−ℓ+1)}(W_Lᵀ…W_{ℓ+1}ᵀ)E[|h_ℓ|]`**
✅ПЛОМБА тотожності: `|Σвнесків − m_deep|/|m_deep| = **7.233e-16**`.
📊 ЗМІРЯНИЙ ПРОФІЛЬ (`side66/depthprof.py`, 48 мереж), множник на шар
**1.4110** (теорія `√2/2` у зворотний бік = 1.4142 ⟹ збіг до 3-ї цифри):
| останніх шарів | 1 | 2 | 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|---|---|
| частка значення | 28.8% | 49.4% | 63.9% | 81.7% | **90.8%** | 96.7% | 99.4% |
🚨 **ПЕРШІ 19 ШАРІВ ДАЮТЬ 0.34%. А НАШІ КАНАЛИ СТОЯТЬ НА ШАРАХ 1..8**, де
сумарний прямий внесок **<0.01%**.

## 🎓 ЧОМУ КАНАЛИ НА ГОЛОВІ ВСЕ Ж ПРАЦЮЮТЬ (+24.9%) — І ЩО З ЦЬОГО ВИПЛИВАЄ
Похибки бувають ДВОХ типів із РІЗНОЮ динамікою:
* **АДИТИВНІ** (в `E[a]`) — згасають як `0.707^k`, тобто живуть лише у хвості;
* **МУЛЬТИПЛІКАТИВНІ** (в `σ`, масштабі) — `σ²_{ℓ+1} = diag(WᵀCov(a_ℓ)W)`
  залежить від `σ_ℓ` ⟹ відносна похибка множиться і **НЕ ЗГАСАЄ**.
Наші канали `S`,`Q` — про другі моменти, тобто про МАСШТАБ ⟹ вони й мали
працювати з голови. Узгоджено з «40.9% дисперсії = МАСШТАБ».
⟹ 🎯 **АДИТИВНА КОМПОНЕНТА СКОНЦЕНТРОВАНА В ОСТАННІХ 7 ШАРАХ, І ТАМ У НАС
НЕМАЄ НІЧОГО.** Це рівно та частина, яку жоден головний канал не бачить.

## 💰 ЧОМУ ЦЕ МОЖЕ БУТИ ДЕШЕВО (гейт ціни ДО реалізації)
Замикання вже йде до шару 32 (`kdeep=len(wc)` для `m_deep`) — опори
`s,m,q,P` на хвості просто **ВИКИДАЮТЬСЯ**. Зберегти їх коштує НУЛЬ флопів.
Доплата — лише вибіркові редукції на 7 шарах: ~4 редукції × 7 = 28 операцій,
разом із λ-арифметикою ≈ +200 викликів `fnp` = +2.5 мс = **+0.0009 до C/B**
(запас sub93 під підлогою — 0.0055) ⟹ вміщається.
📌 ⛔це НЕ «KSIG 8→24» (закрито раніше): там додавали шари 9–24 з внеском ~2%
   і платили ГЛИБШИМ замиканням; тут — хвіст із внеском 87% і опори безкоштовні.
📌 sub89 («канал `q31` + замикання на 31») дав MSE **−3.0%** — тобто хвостовий
   канал уже показував сигнал і помер ЛИШЕ ціною глибокого замикання.

## ЩО МІРЯЄМО (24 мережі × 8 жеребів, банк, крос-ціль)
`TAIL ∈ {0,1,2,3,5,7}` — скільки останніх шарів отримують канали.
Коефіцієнти: ОДИН спільний `λ` на родину (4 параметри), **LOO ПО МЕРЕЖАХ**
(закон ємності: 4 параметри при N=24 ⟹ штраф 1.96%).
⛔КОНТРОЛІ: ⚠️оракул (λ на всіх мережах — стеля) · ⛔ЧУЖА мережа · TAIL=0 (база).

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: сигнал БУДЕ (профіль і sub89 обидва вказують), але
   λ-канали на хвості мають гіршу опору (замикання там найменш точне) ⟹
   зміщення більше. Оцінка: оракул ×1.05–1.15, LOO ×1.02–1.06. Шанс, що
   переживе LOO і ціну: **45%** — найвища за сесію.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/tailchan.py [cuda:N] [NETS] [NDRAW]
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

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
L_ = 32
W_BLEND, ALPHA = 0.0883, -0.00751
TAILS = [0, 1, 2, 3, 5, 7]
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
print(f'{len(FS)} мереж × {NDRAW} жеребів, {DEV}\n', flush=True)


def closure_tail(wc, dev, tail):
    """Замикання: опори для шарів 1..KSIG І для ХВОСТА (безкоштовно)."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    L = len(wc)
    TS = {}
    for i in range(L):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        keep = (tail > 0 and i >= L - tail)
        if i < L - 1:
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            d = (mp*mp + s*s) * Ph + mp*s*ph - m*m
            C = C + torch.diag(d - C.diagonal())
            if keep:
                TS[i] = (s, m.clone(), C.diagonal() + m*m, Ph)
        elif keep:
            TS[i] = (s, m.clone(), None, Ph)
    return TS, m


def run(wc, npair, dev, seed, cl, TS, lam):
    """sub93 + хвостові канали з вагами `lam` = (S,A,Q,G)."""
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
    gn = min(GATE_N, x.shape[0])
    D = []                                   # внески хвостових каналів
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:gn] > 0).to(p.dtype).sum(0).double() / gn
        wd = wc[i].double()
        v = (v @ wd) * gate
        pd = p.double()
        mh = pd.sum(0) / den
        e2 = (pd * pd).sum(0) / den
        sh = (e2 - mh * mh).clamp_min(1e-30).sqrt()
        x = torch.relu(p)
        xm = x.double().sum(0) / den
        xq = x.double().pow(2).sum(0) / den
        if i <= KSIG:
            v = (v + (SP.LAM_S[i-1] * _phi(mh/sh)) * (sh - s_cl[i-1])
                 + SP.LAM_A[i-1] * (xm - m_cl[i-1])
                 + SP.LAM_Q[i-1] * (xq - q_cl[i-1])
                 + SP.LAM_G[i-1] * (gate - P_cl[i-1]))
        if i in TS:                          # 🎯 ХВІСТ
            s_t, m_t, q_t, P_t = TS[i]
            d = (lam[0] * (_phi(mh/sh) * (sh - s_t))
                 + lam[1] * (xm - m_t)
                 + lam[3] * (gate - P_t))
            if q_t is not None:
                d = d + lam[2] * (xq - q_t)
            v = v + d
        if i == depth - 1:
            last = xm
    ours = last - v
    return ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)


N = len(FS)
RES = {}
for tail in TAILS:
    # для кожного tail: збираємо базові й «одиничні» прогони, далі —
    # лінійність по λ (виграш є квадратичною формою) ⟹ оптимум аналітично
    P0 = np.zeros((N, NDRAW, W_))
    G = np.zeros((N, NDRAW, 4, W_))
    T1a = np.zeros((N, W_)); T2a = np.zeros((N, W_))
    for n, f in enumerate(FS):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair, _ = cut_budget91(W, DEV)
            cl = closure91(wc, KSIG, DEV, kdeep=len(wc))
            TS, _ = closure_tail(wc, DEV, tail)
            b1, b2 = TRUTH2[sd0]
            T1a[n] = b1; T2a[n] = b2
            for d in range(NDRAW):
                sd = 81_000_000 + 733*n + d
                base = run(wc, npair, DEV, sd, cl, TS, [0.0]*4)
                P0[n, d] = base.cpu().numpy()
                for j in range(4):
                    e = [0.0]*4; e[j] = 1.0
                    r = run(wc, npair, DEV, sd, cl, TS, e)
                    G[n, d, j] = (r - base).cpu().numpy()
        del W, wc
        torch.cuda.empty_cache()
    RES[tail] = (P0.copy(), G.copy())
    print(f'  … tail={tail} готовий', flush=True)

T1a = np.asarray(T1a); T2a = np.asarray(T2a)


def mse_of(lam, P0, G, idx):
    """MSE(λ) — квадратична форма; крос-ціль по T1,T2."""
    tot = 0.0
    for n in idx:
        for d in range(P0.shape[1]):
            p = P0[n, d] + sum(lam[j] * G[n, d, j] for j in range(4))
            tot += float(((p - T1a[n]) * (p - T2a[n])).mean())
    return tot / (len(idx) * P0.shape[1])


def fit(P0, G, idx):
    """Оптимальний λ (4 параметри) — розв'язок нормальних рівнянь."""
    A = np.zeros((4, 4)); b = np.zeros(4)
    for n in idx:
        for d in range(P0.shape[1]):
            e = P0[n, d] - 0.5 * (T1a[n] + T2a[n])
            for i in range(4):
                b[i] -= float((G[n, d, i] * e).mean())
                for j in range(4):
                    A[i, j] += float((G[n, d, i] * G[n, d, j]).mean())
    return np.linalg.solve(A + 1e-12 * np.eye(4), b)


print(f'\n{"TAIL":>5s} {"база":>12s} {"⚠️оракул":>12s} {"×":>8s} '
      f'{"LOO":>12s} {"×":>8s} {"⛔ЧУЖА ×":>9s}')
print('─' * 74)
allidx = list(range(N))
for tail in TAILS:
    P0, G = RES[tail]
    base = mse_of([0.0]*4, P0, G, allidx)
    if tail == 0:
        base0 = base
        print(f'{tail:5d} {base:12.5e} {"—":>12s} {"—":>8s} '
              f'{"—":>12s} {"—":>8s} {"—":>9s}')
        continue
    lo = fit(P0, G, allidx)
    orc = mse_of(lo, P0, G, allidx)
    tot = 0.0
    for n in range(N):
        li = fit(P0, G, [j for j in range(N) if j != n])
        tot += mse_of(li, P0, G, [n])
    loo = tot / N
    lsh = fit(P0, G, allidx[1:] + allidx[:1])
    alien = mse_of(np.roll(lo, 1), P0, G, allidx)
    print(f'{tail:5d} {base:12.5e} {orc:12.5e} {base0/orc:8.4f} '
          f'{loo:12.5e} {base0/loo:8.4f} {base0/alien:9.4f}')
print('─' * 74)
print('🎯 ГЕЙТ: LOO ≥ 1.02 І ⛔ЧУЖА ≈ 1.00 ⟹ канал на хвості ЖИВИЙ.')
