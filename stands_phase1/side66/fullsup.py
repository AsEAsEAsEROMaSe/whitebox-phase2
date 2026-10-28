"""ЧИ МОЖНА ВЗЯТИ ОПОРИ З ПОВНОГО ЗАМИКАННЯ, ЗВУЗИВШИ МАСКОЮ (С84)

## ЗВІДКИ
⛔`estimator_sub94.py`: компенсація зсуву обрізання `β·(m_deep(ПОВНІ) −
m_deep(обрізані))` дає ЧИСТИЙ сигнал (LOO ×1.0061, обидва контролі РІВНО 0,
`β*`=0.365 стабільний), але помирає ЦІНОЮ: другий прохід замикання = +65%
операцій ⟹ множник 0.10000 → 0.10529, повний бал **0.9626×**
([[finding-trimming-bias-is-computable-but-too-expensive]]).
🎯 ДЕШЕВША ІДЕЯ: рахувати повне замикання **ЗАМІСТЬ** обрізаного, а не
ДОДАТКОВО. Обрізані нейрони майже завжди неактивні ⟹ їхній внесок у `C` малий
⟹ опори `s,m,q,P` для λ можна брати з ПОВНОГО, звузивши маскою `alive`.
Доплата тоді — різниця ширин (на шарі 31 лишається 70.4%), а не другий прохід.

## 🚨 ПОРЯДОК: СПЕРШУ ПЛОМБА, ПОТІМ ЦІНА
Урок sub94 дослівно: **ГЕЙТ ЦІНИ РАХУВАТИ ДО РЕАЛІЗАЦІЇ**. Але тут ще дешевше
спершу перевірити ПРИДАТНІСТЬ: якщо опори розходяться, ціна вже не важить.

## ЩО МІРЯЄМО
1. ⛔ПЛОМБА ПРИДАТНОСТІ: `‖опора_повна[alive] − опора_обрізана‖` у масштабі
   САМОЇ ОЗНАКИ `δ` (не в масштабі опори!) — бо канал бачить саме `δ`.
   Гейт: якщо розбіжність ≳1% від `‖δ‖` — підміна псує λ-канали, СТОП.
2. Якщо пройшла — ПРЯМИЙ вимір MSE трьох варіантів на банку:
   ⛔ sub93 · 🎯 sub93 з опорами з ПОВНОГО замикання · 🎯 те саме + `β·D`.
3. ЦІНА: скільки матмулів коштує повне замикання проти обрізаного —
   рахується з РЕАЛЬНИХ ширин `wc`, а не з припущень.

⛔ ПЛОМБИ: (а) `alive`-звуження ТОТОЖНЕ тому, що робить `cut_budget91`
   (перевірка: `W[i][alive[i-1]][:,alive[i]] ≡ wc[i]` побітово);
   (б) варіант «опори обрізані, β=0» ≡ sub93 РІВНО;
   (в) істина — БАНК, крос-ціль.

📣 ПЕРЕДБАЧЕНО: розбіжність опор буде МАЛОЮ для `m`,`q` (обрізані нейрони
   майже неактивні, їхній внесок у середні мізерний), але може бути помітною
   для `P` (маска) — бо `Φ(t)` залежить від `t=m/s`, а `s` збирає внески з
   УСІХ вхідних нейронів, включно з викинутими. Якщо провалиться саме `P` —
   можна брати з повного лише `s,m,q`, а `P` лишити обрізаним; тоді
   доплата ще менша. Ціна: чекаю ×1.3 до нинішнього замикання, тобто ~+0.0016
   множника — вкладається під підлогу, якщо запас sub93 більший за це.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/fullsup.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (closure91, _phi, _Phi, KSIG, W_, GATE_N,   # noqa: E402
                        INV_SQRT_2PI, PILOT_PAIRS, SEED, MIN_PAIRS,
                        TARGET_UTIL, B_BUDGET, ALPHA, W_BLEND)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
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


def cut_with_mask(W, dev):
    """`cut_budget91`, але ще й повертає маски `alive` — вони тут потрібні."""
    depth = len(W)
    rng = np.random.default_rng(SEED)
    zp = torch.tensor(rng.standard_normal((PILOT_PAIRS, W_)).astype(np.float32),
                      device=dev)
    a = torch.cat([zp, -zp], 0)
    alive = []
    for i in range(depth):
        h = a @ W[i]
        if i < depth - 1:
            alive.append(h.max(0).values >= 0.0)
        a = torch.relu(h)
    wc = []
    for i in range(depth):
        wi = W[i]
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < depth - 1:
            wi = wi[:, alive[i]]
        wc.append(wi.contiguous())
    per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * depth * W_ * W_)
    left = TARGET_UTIL * float(B_BUDGET) - used
    return (wc, max(MIN_PAIRS, int(left / (2.0 * max(per_cut, 1.0)))),
            alive, per_cut)


def run(wc, npair, dev, seed, cl, beta, D):
    """sub93 з опорами `cl`, плюс `beta·D` до виходу."""
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
    s_cl, m_cl, q_cl, P_cl, m_deep, AA, BB = cl
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
    pr = ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)
    return pr if beta == 0.0 else pr + beta * D


def closure_aux(ws, dev):
    """`closure91` + опори `s²φ`, `s²tφ` (як у sub93)."""
    m = torch.zeros(ws[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(ws[0].shape[0], dtype=torch.float64, device=dev)
    s_o, m_o, q_o, p_o, AA, BB = [], [], [], [], [], []
    for i in range(len(ws)):
        w = ws[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        if 1 <= i <= KSIG:
            s_o.append(s); p_o.append(Ph)
            s2ph = (s * s) * ph
            AA.append(s2ph); BB.append(s2ph * t)
        m = mp * Ph + s * ph
        if i < len(ws) - 1:
            d2 = (mp * mp + s * s) * Ph + mp * s * ph
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            C = C + torch.diag((d2 - m * m) - C.diagonal())
            if 1 <= i <= KSIG:
                m_o.append(m); q_o.append(C.diagonal() + m * m)
    return s_o, m_o, q_o, p_o, m, AA, BB


NAMES = ['s (δs)', 'm (δmean)', 'q (δE[a²])', 'P (δмаска)']
REL = np.zeros((len(FS), KSIG, 4))     # ‖повна[alive] − обрізана‖ / ‖δ‖
E12 = np.zeros((len(FS), 3))
COST = np.zeros((len(FS), 2))          # матмули: обрізане, повне
SEALW = 0.0
SEAL93 = 0.0

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, alive, per_cut = cut_with_mask(W, DEV)
        # ⛔ПЛОМБА (а): звуження ТОТОЖНЕ тому, що робить cut_budget91
        for i in range(1, L_ - 1):
            ref = W[i][alive[i - 1], :][:, alive[i]]
            SEALW = max(SEALW, float((ref - wc[i]).abs().max()))
        cl_c = closure_aux(wc, DEV)
        cl_f = closure_aux(W, DEV)
        D = cl_f[4] - cl_c[4]
        # ── звужені опори з ПОВНОГО замикання
        cl_n = ([cl_f[0][k][alive[k + 1]] for k in range(len(cl_f[0]))],
                [cl_f[1][k][alive[k + 1]] for k in range(len(cl_f[1]))],
                [cl_f[2][k][alive[k + 1]] for k in range(len(cl_f[2]))],
                [cl_f[3][k][alive[k + 1]] for k in range(len(cl_f[3]))],
                cl_c[4],
                [cl_f[5][k][alive[k + 1]] for k in range(len(cl_f[5]))],
                [cl_f[6][k][alive[k + 1]] for k in range(len(cl_f[6]))])
        # ── ЦІНА: матмули замикання ∝ Σ (in·out + out·out) на шар
        COST[n, 0] = sum(float(w.shape[0]) * float(w.shape[1])
                         + float(w.shape[1]) ** 2 for w in wc)
        COST[n, 1] = sum(float(w.shape[0]) * float(w.shape[1])
                         + float(w.shape[1]) ** 2 for w in W)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        for d in range(NDRAW):
            sd = 101_000_000 + 733 * n + d
            prs = [run(wc, npair, DEV, sd, cl_c, 0.0, None),
                   run(wc, npair, DEV, sd, cl_n, 0.0, None),
                   run(wc, npair, DEV, sd, cl_n, 0.3651, D)]
            if d == 0:
                # ⛔ПЛОМБА (б): варіант 0 ≡ sub93 через незалежний шлях
                ref = closure91(wc, KSIG, DEV, kdeep=len(wc))
                SEAL93 = max(SEAL93, float((cl_c[4] - ref[4]).abs().max()))
                # розбіжність опор У МАСШТАБІ ОЗНАКИ δ
                pp = torch.cat([torch.relu(prs[0][:1])])   # заглушка форми
            for k in range(3):
                E12[n, k] += float(((prs[k] - T1) * (prs[k] - T2)).sum()) / NDRAW
        # розбіжність опор у масштабі δ — окремим коротким прогоном
        sd = 101_000_000 + 733 * n
        rng = np.random.default_rng(int(sd))
        rng.standard_normal((PILOT_PAIRS, W_))
        z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                         device=DEV)
        cov = (z.T @ z) / float(npair)
        ev, V = torch.linalg.eigh(cov.double())
        wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
        y = z @ (wh @ wc[0])
        x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
        den = float(2 * npair)
        for i in range(1, min(KSIG + 1, L_)):
            p = x @ wc[i]
            gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
            pd = p.double()
            mh = pd.sum(0) / den
            sh = (pd.pow(2).sum(0) / den - mh * mh).clamp_min(1e-30).sqrt()
            x = torch.relu(p)
            xm = x.double().sum(0) / den
            xq = x.double().pow(2).sum(0) / den
            dl = (_phi(mh / sh) * (sh - cl_c[0][i - 1]), xm - cl_c[1][i - 1],
                  xq - cl_c[2][i - 1], gate - cl_c[3][i - 1])
            for k in range(4):
                diff = (cl_n[k][i - 1] - cl_c[k][i - 1])
                if k == 0:
                    diff = _phi(mh / sh) * diff
                REL[n, i - 1, k] = float(diff.norm()
                                         / dl[k].norm().clamp_min(1e-300))
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/fullsup.npz',
         REL=REL, E12=E12, COST=COST)

print(f'\n⛔ПЛОМБА (а) звуження ≡ cut_budget91: max|Δ| = {SEALW:.2e} '
      f'{"✅" if SEALW == 0 else "🚨"}')
print(f'⛔ПЛОМБА (б) замикання стенду ≡ closure91: {SEAL93:.2e} '
      f'{"✅" if SEAL93 < 1e-12 else "🚨"}\n')

print('⛔ГОЛОВНА ПЛОМБА ПРИДАТНОСТІ: ‖опора_повна[alive] − обрізана‖ / ‖δ‖')
print(f'{"шар":>4s} ' + ' '.join(f'{x:>13s}' for x in NAMES))
print('─' * 60)
for i in range(KSIG):
    print(f'{i+1:4d} ' + ' '.join(f'{REL[:, i, k].mean():13.4f}'
                                 for k in range(4)))
mx = REL.mean(0).max()
print(f'\n   МАКСИМУМ по шарах і родинах: {mx:.4f} '
      f'({"✅ <1% — підміна безпечна" if mx < 0.01 else "🚨 ПІДМІНА ПСУЄ ОЗНАКУ"})')

r = COST[:, 1].mean() / COST[:, 0].mean()
print(f'\n💰 ЦІНА: повне/обрізане замикання = ×{r:.3f} '
      f'(було б ×2 при ДОДАТКОВОМУ проході)')
NMV = ['⛔ sub93 (опори обрізані)', '🎯 опори з ПОВНОГО',
       '🎯 опори з ПОВНОГО + β·D']
print(f'\n{"варіант":>28s} {"MSE":>13s} {"× до sub93":>11s}')
print('─' * 56)
b = E12[:, 0].mean()
for k in range(3):
    print(f'{NMV[k]:>28s} {E12[:, k].mean():13.6e} {b/E12[:, k].mean():11.4f}')

print('\n🎯 ВИРОК')
g = b / E12[:, 2].mean()
if mx >= 0.01:
    print('   ⛔ ПЛОМБА ПРИДАТНОСТІ ПРОВАЛЕНА — опори розходяться, гілка закрита')
elif g > 1.003:
    print(f'   🟢 точність ×{g:.4f} ⟹ бал 1.928e-07 → {1.928e-07/g:.4e}')
    print(f'   💰 доплата ×{r:.3f} до замикання; ГЕЙТ: чи лишається множник 0.1')
else:
    print(f'   ⛔ виграш ×{g:.4f} — замалий, щоб платити ×{r:.3f} за замикання')
print(f'\n📊 БАЛ: sub87 1.954e-07 · sub92 1.946e-07 · sub93 1.928e-07')
