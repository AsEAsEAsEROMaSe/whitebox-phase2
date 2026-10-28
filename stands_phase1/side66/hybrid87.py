"""ГІБРИД: ЛАНЦЮГ РЕКУРСУЄ `m`, А `σ`,`γ` БЕРЕ З НАШИХ ЖЕ МК-СЕМПЛІВ. (С87)

## ЗВІДКИ — ЛАНЦЮЖОК ФАКТІВ, ЯКИЙ ПРИВІВ СЮДИ
1. `ladder87.py`: `L10` (рекурсія `m` при правильних `σ`,`γ`) = **5.377e-08**
   проти лідерових 5.211e-08 — метод відтворено.
2. ТЗ: `σ` треба з точністю **1%**, `γ` вистачить **3%**.
3. `mehlerk87.py`: ⛔`K=3,4,5,6` ряду Мелера НЕ ДАЮТЬ НІЧОГО (масштаб 0.8725 →
   0.8726, похибка 0.9431% → 0.9421%; контроль зі зміненим знаком теж нуль)
   ⟹ обрив ряду НЕ винен, і виправити `σ` зсередини замикання НЕМА ЧИМ.
4. 🚨 ПОПРАВКА ДО МОГО Ж ПОПЕРЕДНЬОГО ВИСНОВКУ: похибка `σ` НЕ росте
   геометрично — вона **НАСИЧУЄТЬСЯ**:
   `0.10% → 1.99% → 5.38% → 8.41% → 11.09% → 13.76% → 14.87%`
   приріст за 5 шарів: ×20 → ×2.7 → ×1.56 → ×1.32 → ×1.24 → ×1.08. Це АТРАКТОР,
   а не вибух. «×1.17 за шар» було моєю екстраполяцією по двох кінцях.

## 🔑 ІДЕЯ
`σ` з рекурсії безнадійна (14.9%). Але **ми вже маємо МК-семпли** — у сабміті
`p = x @ w_cut[i]` рахується на КОЖНОМУ шарі. Вибіркові моменти дають:
```
віднос. похибка σ̂ = 1/√(2N)  = 0.91% при N=6000·2   ⟹ РІВНО ТЗ (1%)
віднос. похибка γ̂₃ = √(6/N)  = 2.2%                 ⟹ у межах ТЗ (3%)
```
⟹ 🎯 **ПІДСТАВИТИ `σ̂`,`γ̂` У РЕКУРСІЮ ЗАМИКАННЯ.**
🚨 ЦЕ НЕ ТЕ, ЩО МИ РОБИМО ЗАРАЗ. Зараз `sh − s_cl` — КОНТРОЛЬНА ЗМІННА, а
замикання йде своїм шляхом із накопиченою похибкою 14.9%. Тут вибіркова `σ`
входить У САМУ РЕКУРСІЮ і обриває накопичення.

## 💰 ЦІНА — ПРАКТИЧНО НУЛЬОВА, УСЕ ВЖЕ Є В САБМІТІ
* семпли `p` на кожному шарі — вже рахуються;
* 4 редукції на шар (`mh,e2,e3,e4`) — уже робляться на шарах 1..8, треба на 32;
* `C`-рекурсія 2 матмули `256³`/шар = 2.15e9 = 0.79% бюджету — вже платиться
  за глибоке замикання в `sub97`.
⟹ додаткова ціна ≈ 24 шари × 4 редукції ≈ 100 викликів `fnp` ≈ 0.05% бюджету.

## 🎯 ЩО МІРЯЄМО
Ланцюг, де `σ`,`γ` — з ВИБІРКИ розміру `N`, `m` рекурсується. `N` = 3000, 6000,
12000, 24000 (наш сабміт має 2×n_pairs ≈ 6000..12000).

⛔ПЛОМБИ:
 H1 при `N → ∞` мусить збігтися з `L10` = 0.0605% (`ladder87.py`).
 H2 при `N` малому мусить бути ГІРШЕ за базу — бо шум `σ̂` входить у рекурсію
    і може підсилитись. Якщо цього НЕ видно — семпли не входять насправді.
 H3 ⛔КОНТРОЛЬ: `σ̂` від ЧУЖОЇ мережі мусить дати катастрофу.
 H4 незалежні семпли для `σ̂` і для істини — інакше міряємо самофіт
    ([[finding-nact-channel-survives-but-selffit-eats-84-percent]]).

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) `N=∞` відтворить 0.06% (пломба H1). Шанс 85%.
 (б) 🎯При `N=12000` вийде **0.15–0.35%** ⟹ бал 1.4e-07…7.7e-07. Тобто
     гранично з `sub97` (1.873e-07). Шанс, що ПОБ'Є `sub97`: **40%**.
 (в) ⚠️ГОЛОВНИЙ РИЗИК: шум `σ̂` входить у рекурсію 32 рази і накопичується
     як √32 ≈ 5.7× ⟹ ефективна похибка `σ` стане 5%, а ТЗ дає при 3% вже
     ×176. Шанс цього: **50%**. Якщо так — гілка мертва, і це буде ясно одразу.
 (г) Якщо (в) справдиться, наступний хід — усадка `σ̂` до `s_cl` (бленд шумної
     незміщеної та гладкої зміщеної), як уже робиться для δs у сабміті.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/hybrid87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N
BATCH = 1 << 12
L_, W_ = 32, 256
NSAMP = (6000, 9000, 12000, 16000, 20000)
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def edge_relu(m, s, g3, g4):
    t = m / s
    P, f = Phi(t), phi(t)
    return s * ((t * P + f) + (g3 / 6.0) * (-t * f)
                + (g4 / 24.0) * ((t * t - 1.0) * f)
                + (g3 * g3 / 72.0) * ((t ** 4 - 6.0 * t * t + 3.0) * f))


def edge_relu2(m, s, g3, g4):
    t = m / s
    P, f = Phi(t), phi(t)
    s2 = s * s
    return ((m * m + s2) * P + m * s * f
            + (g3 / 3.0) * s2 * f - (g4 / 12.0) * s2 * t * f
            + (g3 * g3 / 36.0) * s2 * (3.0 * t - t ** 3) * f)


def chain(W, samp_s=None, samp_g=None, shrink=0.0, s_cl_out=None):
    """`samp_s[i]`,`samp_g[i]` — ВИБІРКОВІ `σ`,`γ` шару `i` (або None).

    `shrink` — усадка до замикання: `σ = (1−w)·σ̂ + w·σ_замик`."""
    dev = W[0].device
    n = W[0].shape[1]
    zero = torch.zeros(n, dtype=torch.float64, device=dev)
    C = W[0].T @ W[0]
    m = zero.clone()
    for i in range(1, len(W) + 1):
        s_cl = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        if s_cl_out is not None:
            s_cl_out.append(s_cl.clone())
        s = s_cl if samp_s is None else (
            (1.0 - shrink) * samp_s[i - 1] + shrink * s_cl)
        g3, g4 = samp_g[i - 1] if samp_g is not None else (zero, zero)
        m_a = edge_relu(m, s, g3, g4)
        if i == len(W):
            return m_a
        t = m / s
        P, f = Phi(t), phi(t)
        C_a = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        d = edge_relu2(m, s, g3, g4) - m_a * m_a
        C_a = C_a + torch.diag(torch.clamp(d, min=1e-30) - torch.diagonal(C_a))
        m = m_a @ W[i]
        C = W[i].T @ (C_a @ W[i])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · істина 2^{LOG2N} · N={NSAMP} · {DEV}\n', flush=True)

TAGS = (['БАЗА (замикання)', 'ORC σ,γ (L10)']
        + [f'вибірка N={q}' for q in NSAMP]
        + [f'МК N={q}' for q in NSAMP]
        + [f'БЛЕНД N={q}' for q in NSAMP]
        + ['⛔ЧУЖА σ̂,γ̂'])
E = {t: np.zeros(len(FS)) for t in TAGS}
R = {t: np.zeros(len(FS)) for t in TAGS}
SERR = {q: np.zeros(len(FS)) for q in NSAMP}
ALIEN = None
BLEND_H = {q: [] for q in NSAMP}
BLEND_M = {q: [] for q in NSAMP}
TRUS, NRMS = [], []


def mc_estimate(W32, nsmp, seed):
    """ПРЯМА МК-оцінка `E[a_last]` на тих самих семплах (антитетика)."""
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
    half = nsmp // 2
    d = 0
    while d < half:
        b = min(BATCH, half - d)
        z = torch.randn(b, W_, device=DEV, generator=g)
        a = torch.cat([z, -z], 0)
        for q in range(L_):
            a = torch.relu(a @ W32[q])
        acc += a.sum(0, dtype=torch.float64)
        d += b
    return acc / float(2 * half)


def moments_from(W32, nsmp, seed):
    """ВИБІРКОВІ `σ̂`,`γ̂` на кожному шарі з `nsmp` семплів (антитетика, як у нас)."""
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    acc = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
    half = nsmp // 2
    d = 0
    while d < half:
        b = min(BATCH, half - d)
        z = torch.randn(b, W_, device=DEV, generator=g)
        a = torch.cat([z, -z], 0)
        for q in range(L_):
            h = a @ W32[q]
            hd = h.double()
            p = hd
            for kk in range(4):
                acc[q, kk] += p.sum(0)
                p = p * hd
            a = torch.relu(h)
        d += b
    acc /= float(2 * half)
    ss, gg = [], []
    for q in range(L_):
        m1, m2, m3, m4 = acc[q, 0], acc[q, 1], acc[q, 2], acc[q, 3]
        c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
        c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        c4 = (m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
              - 3.0 * c2 * c2)
        sq = torch.sqrt(c2)
        ss.append(sq)
        gg.append((c3 / sq ** 3, c4 / sq ** 4))
    return ss, gg


for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        # ⛔H4: істина — з НЕЗАЛЕЖНОГО потоку (сід інший, ніж у вибірок)
        MOM = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(90000 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]
                hd = h.double()
                p = hd
                for kk in range(4):
                    MOM[q, kk] += p.sum(0)
                    p = p * hd
                a = torch.relu(h)
            acc += a.sum(0, dtype=torch.float64)
            d += b
        MOM /= NS
        tru = acc / NS
        nrm = float(torch.sqrt((tru ** 2).mean()))
        OS, OG = [], []
        for q in range(L_):
            m1, m2, m3, m4 = MOM[q, 0], MOM[q, 1], MOM[q, 2], MOM[q, 3]
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            c4 = (m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
                  - 3.0 * c2 * c2)
            sq = torch.sqrt(c2)
            OS.append(sq)
            OG.append((c3 / sq ** 3, c4 / sq ** 4))

        def ev(tag, pr):
            E[tag][n] = float(torch.sqrt(((pr - tru) ** 2).mean())) / nrm
            R[tag][n] = float((pr * tru).sum() / (tru * tru).sum())

        ev('БАЗА (замикання)', chain(W))
        ev('ORC σ,γ (L10)', chain(W, samp_s=OS, samp_g=OG))
        SS = {}
        for q in NSAMP:
            ss, gg = moments_from(W32, q, 500 + 7 * n + q)
            SS[q] = (ss, gg)
            SERR[q][n] = float(np.mean([
                float(torch.sqrt(((ss[j] - OS[j]) ** 2).mean()
                                 / (OS[j] * OS[j]).mean())) for j in range(L_)]))
            ev(f'вибірка N={q}', chain(W, samp_s=ss, samp_g=gg))
        # 🔑 БЛЕНД: гібрид і ПРЯМА МК-оцінка на ТИХ САМИХ семплах.
        #    Вони корельовані, тож вага береться LOO по мережах (нижче).
        for q in NSAMP:
            mc = mc_estimate(W32, q, 500 + 7 * n + q)
            ev(f'МК N={q}', mc)
            hy = chain(W, samp_s=SS[q][0], samp_g=SS[q][1])
            BLEND_H[q].append(hy)
            BLEND_M[q].append(mc)
            ev(f'БЛЕНД N={q}', 0.5 * hy + 0.5 * mc)
        if ALIEN is None:
            ALIEN = SS[NSAMP[2]]
            ev('⛔ЧУЖА σ̂,γ̂', chain(W))
        else:
            ev('⛔ЧУЖА σ̂,γ̂', chain(W, samp_s=ALIEN[0], samp_g=ALIEN[1]))
        TRUS.append(tru)
        NRMS.append(nrm)
    print(f'  net{n} · база {E["БАЗА (замикання)"][n]:.4%} · '
          + ' · '.join(f'N{q} {E[f"вибірка N={q}"][n]:.4%}' for q in NSAMP),
          flush=True)


def loo_resid(err, rat):
    out = np.zeros_like(err)
    for i in range(len(err)):
        msk = np.arange(len(err)) != i
        out[i] = math.sqrt(max(err[i] ** 2 - (rat[i] - 1.0) ** 2
                               + (rat[i] - rat[msk].mean()) ** 2, 1e-30))
    return out


print('\n══ ТОЧНІСТЬ ВИБІРКОВОЇ `σ̂` (середнє по шарах) проти теорії 1/√(2N) ══')
for q in NSAMP:
    print(f'  N={q:6d}: зміряно {SERR[q].mean():.4%} · теорія '
          f'{1.0 / math.sqrt(2.0 * q):.4%}')

print('\n══ РЕЗУЛЬТАТ ══')
print('варіант                сира     після α    MSE        бал(×0.1)  vs sub97')
for t in TAGS:
    rs = loo_resid(E[t], R[t])
    mse = 1.698e-05 * (rs.mean() / 0.00941) ** 2
    print(f'{t:22s} {E[t].mean():7.4%} {rs.mean():7.4%}  {mse:.3e}  '
          f'{mse * 0.1:.3e}  ×{1.873e-07 / (mse * 0.1):7.3f}')
print(f'{"ЛІДЕР":22s} {0.0521 / 100:7.4%} {0.0521 / 100:7.4%}  {5.211e-08:.3e}  '
      f'{5.102e-08:.3e}  ×{1.873e-07 / 5.102e-08:7.3f}')
# 🔑 ЧЕСНИЙ КОШТОРИС: множник НЕ підлога, якщо `N` великий
print('\n══ 🎯 ЧЕСНИЙ БАЛ (множник = max(0.1, C/B), C = прогін + замикання + residual) ══')
print('N        похибка   MSE        бюджет   множник   БАЛ         vs sub97')
BUD = 2.72e11
for q in NSAMP:
    for tag, pref in (('вибірка', 'гібрид'), ('БЛЕНД', 'бленд ')):
        t = f'{tag} N={q}'
        rs = loo_resid(E[t], R[t])
        mse = 1.698e-05 * (rs.mean() / 0.00941) ** 2
        # прогін `q` семплів крізь ОБРІЗАНУ мережу (~0.8) + C-рекурсія + residual
        C_ = q * 2.0 * L_ * W_ * W_ * 0.8 + 2.0 * L_ * W_ ** 3 + 0.015 * BUD
        u = C_ / BUD
        mult = max(0.1, u)
        print(f'{pref} {q:6d} {rs.mean():7.4%} {mse:.3e}  {u:6.2%}  {mult:6.3f}  '
              f'{mse * mult:.3e}  ×{1.873e-07 / (mse * mult):6.3f}')
print('\n🎯 ГЕЙТ: «vs sub97» > 1.0 ⟹ гібрид Б\'Є наш рекорд 1.873e-07.')
