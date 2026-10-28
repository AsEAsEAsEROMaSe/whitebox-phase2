"""НЕЗМІЩЕНА КОНСТРУКЦІЯ: ЛИШЕ ТОЧНІ КОНТРОЛЬНІ ЗМІННІ. (С85, вивід з нуля)

## ТЕОРЕМА, ЩО ВИЗНАЧАЄ ВСЕ
Контрольна змінна `g` знімає `ρ²` дисперсії ЛИШЕ якщо `E[g]` відоме ТОЧНО.
🚨 Замінити невідоме `E[g]` вибірковою оцінкою НЕМОЖЛИВО: `ḡ − ĝ ≡ 0`.
🚨 Крос-фітинг (оцінити `E[g]` на іншій частині вибірки) ТЕЖ не рятує —
   виводиться аналітично: при частці `α` на основну оцінку
   `MSE = (V/αN)(1 − ρ²(1−α))`, похідна `−(1−ρ²)/α² < 0` ⟹ оптимум `α→1`,
   тобто «не поправляти взагалі». **Ділення вибірки коштує більше, ніж дає.**
⟹ 🔑 **НЕЗМІЩЕНА СХЕМА МОЖЕ ВИКОРИСТОВУВАТИ ЛИШЕ АНАЛІТИЧНО ТОЧНІ СЕРЕДНІ.**

## КАТАЛОГ ТОЧНИХ СЕРЕДНІХ ДЛЯ НАШОЇ МЕРЕЖІ (усе, що взагалі існує)
Шар 1 — вхід ГАУСІВ, тому ВСЕ точно:
  `E[h₁]=0` · `Cov(h₁)=W₁ᵀW₁` · `s₁=‖W₁[:,i]‖` · `P₁=P(h₁>0)=½` ·
  `m₁=E[ReLU]=σ/√(2π)` · `q₁=E[a₁²]=σ²/2`
Шар 2 — точні лише ДРУГІ моменти:
  `E[a₁a₁ᵀ]` = arc-cosine kernel (Cho&Saul), ТОЧНО:
     `(σᵢσⱼ/2π)(sin θ + (π−θ)cos θ)`, `cos θ = (W₁ᵀW₁)ᵢⱼ/(σᵢσⱼ)`
  ⟹ `Cov(h₂) = W₂ᵀ E[a₁a₁ᵀ] W₂` ТОЧНО ⟹ **`s₂` ТОЧНЕ**
  ⛔ але `m₂ = E[ReLU(h₂)]` вимагає РОЗПОДІЛУ `h₂`, який не гаусів ⟹ НІ.
Шар 3+ — нічого точного: `Cov(a₂)` вже потребує розподілу.
⟹ 🎯 **`G_exact` = {усе шару 1} ∪ {`s₂`}. Більше НІЧОГО не існує.**

## ЩО МІРЯЄМО І ЯКИЙ ПОРІГ
Наші канали на шарах 1..8 знімають `1 − 6.733e-03/1.114e-02 = 39.6%`
дисперсії — і платять `b² = 5.054e-07` (21.4% балу).
Незміщена схема лишає лише `G_exact`, знімає менше, але `b = 0`.
📐 **ПОРІГ**: незміщена виграє, якщо `ρ²_exact > 1 − (V/N + b²)·N/V₀`
   `= 1 − 2.363e-06·3625/1.114e-02 = **0.231**`.
⟹ якщо шари 1–2 самі дають ≥23.1% — переходимо на незміщену конструкцію.

Варіанти: `KSIG ∈ {0,1,2,4,8}` × `npair × {1.0, 1.5, 2.5}` ⟹ фіт `V/N + b²`
для КОЖНОГО KSIG ⟹ крива «дисперсія знята vs зміщення додане».
⛔ПЛОМБА: KSIG=8 мусить відтворити `b²≈5.05e-07` із `floorcheck.py`;
          KSIG=0 мусить дати `b²≈0` (чистий МК незміщений за побудовою).

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: `b²(KSIG)` буде ≈0 для KSIG≤2 і різко зросте на
   3..8 (бо саме там замикання перестає бути точним). Частка дисперсії від
   шарів 1–2 — головна невідома; моя оцінка 15–20%, тобто ТРОХИ НИЖЧЕ порогу
   0.231. Шанс, що незміщена конструкція виграє: ~35%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/exactcv.py [cuda:N] [NETS] [NDRAW]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        W_, GATE_N, INV_SQRT_2PI, PILOT_PAIRS)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 5
L_ = 32
W_BLEND, ALPHA = 0.0883, -0.00751
KSIGS = [0, 1, 2, 4, 8]
KS = [1.0, 1.5, 2.5]
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

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
print(f'{len(FS)} мереж × {NDRAW} жеребів × {len(KSIGS)} KSIG × {len(KS)} '
      f'масштабів, {DEV}\n', flush=True)


def exact_layer2_s(wc, dev):
    """🎯 ТОЧНЕ `s₂` через arc-cosine kernel — без жодного наближення."""
    w1 = wc[0].double()
    G = w1.T @ w1                                   # (W₁ᵀW₁)
    sg = G.diagonal().clamp_min(1e-30).sqrt()
    cs = (G / torch.outer(sg, sg)).clamp(-1.0, 1.0)
    th = torch.arccos(cs)
    K = (torch.outer(sg, sg) / (2.0 * math.pi)) * (
        torch.sin(th) + (math.pi - th) * cs)         # E[a₁a₁ᵀ] ТОЧНО
    m1 = sg * INV_SQRT_2PI                           # E[a₁] ТОЧНО
    C1 = K - torch.outer(m1, m1)                     # Cov(a₁) ТОЧНО
    w2 = wc[1].double()
    cov2 = w2.T @ (C1 @ w2)
    return cov2.diagonal().clamp_min(1e-30).sqrt(), K, m1


def run(wc, npair, dev, seed, cl, ksig, s2ex):
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
        if i <= ksig:
            # 🎯 опора `s` для передактивації ПІСЛЯ `wc[1]` — ТОЧНА
            #    (arc-cosine), не з Мелера. 🚨 індекс `i` — номер МАТРИЦІ:
            #    `s_cl[i-1]` є `s` після `wc[i]`, тож точний варіант стоїть
            #    при i == 1, а не i == 2 (перша версія плутала об'єкти і
            #    показала фальшиві 18.4% розбіжності).
            s_ref = s2ex if i == 1 else s_cl[i-1]
            xm = x.double().sum(0) / den
            xq = x.double().pow(2).sum(0) / den
            v = (v + (SP.LAM_S[i-1] * _phi(mh/sh)) * (sh - s_ref)
                 + SP.LAM_A[i-1] * (xm - m_cl[i-1])
                 + SP.LAM_Q[i-1] * (xq - q_cl[i-1])
                 + SP.LAM_G[i-1] * (gate - P_cl[i-1]))
        if i == depth - 1:
            last = x.double().sum(0) / den
    ours = last - v
    return ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)


N = len(FS)
MSE = np.zeros((len(KSIGS), len(KS), N))
NP = np.zeros((len(KS), N))
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, np0, _ = cut_budget91(W, DEV)
        cl = closure91(wc, 8, DEV, kdeep=len(wc))
        s2ex, _, _ = exact_layer2_s(wc, DEV)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        # ⛔ПЛОМБА придатності точного s₂: наскільки він розходиться з Мелером
        if n == 0:
            rel = float((s2ex - cl[0][0]).abs().mean() / cl[0][0].abs().mean())
            print(f'⛔ПЛОМБА: |s(arc-cos) − s(Мелер)|/|s| після wc[1] = '
                  f'{rel:.3e}  (очікуємо ~ρ² ≈ 2e-03; якщо ≫ — щось не те)\n',
                  flush=True)
        for j, k in enumerate(KS):
            npk = max(256, int(round(np0 * k)))
            NP[j, n] = npk
            for a, ks in enumerate(KSIGS):
                acc = 0.0
                for d in range(NDRAW):
                    pr = run(wc, npk, DEV, 71_000_000 + 733*n + d, cl, ks, s2ex)
                    acc += float(((pr - T1) * (pr - T2)).mean()) / NDRAW
                MSE[a, j, n] = acc
        print(f'  … {n+1}/{N} seed {sd0} npair₀={np0} · ' +
              ' '.join(f'K{ks}={MSE[a,0,n]:.2e}' for a, ks in enumerate(KSIGS)),
              flush=True)
    del W, wc
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/exactcv.npz',
         MSE=MSE, NP=NP, KSIGS=np.array(KSIGS), KS=np.array(KS))

xs = 1.0 / NP.mean(1)
A = np.vstack([xs, np.ones_like(xs)]).T
Nb = NP.mean(1)[0]
c1B = 0.0817 / Nb
c0B = 0.08791 - 0.0817 + 0.0134
Nfl = (0.1 - c0B) / c1B
print(f'\n{"KSIG":>5s} {"V":>12s} {"b²":>12s} {"знято дисп.":>12s} '
      f'{"MSE(підл.)":>12s} {"БАЛ":>12s}')
print('─' * 72)
V0 = None
rows = []
for a, ks in enumerate(KSIGS):
    y = MSE[a].mean(1)
    V, b2 = np.linalg.lstsq(A, y, rcond=None)[0]
    if V0 is None:
        V0 = V
    cut = 1.0 - V / V0
    mse_fl = V / Nfl + max(b2, 0.0)
    rows.append((ks, V, b2, cut, mse_fl, mse_fl * 0.1))
    print(f'{ks:5d} {V:12.5e} {b2:12.5e} {cut*100:11.2f}% '
          f'{mse_fl:12.5e} {mse_fl*0.1:12.5e}')
print('─' * 72)
best = min(rows, key=lambda r: r[5])
cur = [r for r in rows if r[0] == 8][0]
print(f'🎯 НАЙКРАЩИЙ KSIG = {best[0]} ⟹ бал {best[5]:.5e}  '
      f'(поточний KSIG=8: {cur[5]:.5e}) ⟹ ×{cur[5]/best[5]:.4f}')
print(f'📐 поріг незміщеності: шари 1–2 мусять знімати >23.1% дисперсії; '
      f'фактично KSIG=2 знімає {rows[2][3]*100:.2f}%')
