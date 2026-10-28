"""ЕДЖВОРТ-3 УСЕРЕДИНІ ЗАМИКАННЯ (а не у воротах) — С82

## ЗВІДКИ ПИТАННЯ
С81 закрила канал ВОРІТ: там Еджворт-3 з істинним `γ₁` різав похибку `q`
удвічі (0.774→0.385%), але аргумент `t` приходив ІЗ ЗАМИКАННЯ, і його власна
похибка з'їдала весь виграш ([[finding-gate-channel-needs-q-four-times-more-accurate]]).
У самому ж замиканні `t` не успадковується, а БУДУЄТЬСЯ на кожному шарі —
тож перешкода може не діяти. Це єдина гілка, яку С81 відкрила й не закрила.

## 🔑 ФОРМУЛИ (виведені вручну, перевірені двома границями)
Замикання вважає передактивацію `p` гаусовою `(mp, s)`. Насправді вона
скошена. Еджворт-3 править щільність: `f(x)=φ(x)[1+γ₁H₃(x)/6]`, `H₃=x³−3x`,
`x=(p−mp)/s`. Інтеграли беруться ТОЧНО через `H₃φ = −φ'''`:

    M1 = E[relu p]  = mp·Φ + s·φ − (γ₁/6)·mp·φ
    M2 = E[relu²p]  = (mp²+s²)·Φ + mp·s·φ + (γ₁/3)·s²·φ
    M3 = E[relu³p]  = (mp³+3mp·s²)·Φ + s(mp²+2s²)·φ + γ₁·s³·Φ

⛔ ПЛОМБА ГРАНИЦЬ (t→+∞, ReLU не ріже): M1→mp ✅, M2→mp²+s² ✅,
   M3→mp³+3mp·s²+γ₁s³ ✅ — рівно означення `γ₁=μ₃/s³`. Друга: t→−∞ ⟹ усе 0 ✅.
⛔ ПЛОМБА КОДУ: при `γ₁≡0` замикання мусить дати РІВНО нинішнє.

Ціна: усе поелементне, O(n) поверх наявної рекурсії `C` (O(n²)) ⟹ додаткові
флопи ≈0.1% — а вони БЕЗКОШТОВНІ, бо бюджет під підлогою множника.

## ЗВІДКИ БРАТИ `γ₁` (тут і вся гілка)
`M3` дає третій момент АКТИВАЦІЙ задарма: `μ₃(a) = M3 − 3·M1·M2 + 2·M1³`.
Далі для наступного шару `p' = Σ_j W_j a_j` — і тут єдине наближення:
    `μ₃(p') ≈ Σ_j W_j³ μ₃(a_j)`   (діагональ = незалежність активацій).
🚨 Саме цей шлях уже мірявся ОКРЕМО і промахувався у **26×**
([[finding-edgeworth-correction-is-the-live-lever]]) — але мірявся як
ПЕРЕДБАЧЕННЯ `μ₃`, а не за впливом на `m`. Похибка може бути СИСТЕМАТИЧНОЮ
(один множник), тому міряю ще й `κ`-сітку.

## ВАРІАНТИ (24 мережі, істина 1 048 576 семплів)
  0 ⛔ гаусове (нинішнє)          — пломба
  1 🎯 істинний `γ₁` ззовні       — СТЕЛЯ гілки
  2 🟢 рекурсивний діагональний   — реалістично, БЕЗКОШТОВНО
  3 🟢 рекурсивний × `κ`          — сітка калібрування (чи похибка стала)

Міряю rel-похибку `m` (годує μ-якір), `s` (годує σ-канал), `q` (ворота) і
самого `γ₁` — усе проти істини, по шарах 1..31.

📣 ПЕРЕДБАЧЕННЯ-1 (до пробного) БУЛО: істинний `γ₁` заріже похибку `m` у
   1.5…3×, рекурсивний дасть <10%. **ОБИДВА ПУНКТИ СПРОСТОВАНО** на 3 мережах:
   істинний `γ₁` зробив `m` ГІРШЕ (0.973×), рекурсивний із промахом 98% —
   КРАЩЕ (1.089× при κ=4). Виграв лише другий момент: `s` 5.137→4.400%.

📣 ПЕРЕДБАЧЕННЯ-2 (це прогін, 24 мережі): гаусове замикання тримається на
   ВЗАЄМНОМУ ГАСІННІ двох хиб — Мелер K=2 у `C` і гаусова щільність у
   моментах. Чекаю: варіант «лише M2» (дисперсія) з ІСТИННИМ `γ₁` буде
   найкращим і для `m`, і для `s`; «лише M1» — найгіршим; а вигода κ=4
   виявиться підгонкою, що НЕ переживе 24 мережі. Якщо ж «лише M2» дасть
   >1.10× по `s` — гілка жива й іде в end-to-end (`side66/clsend.py`).
   ⚠️ Навіть тоді лишається вирок С76 [[finding-lambda-absorbs-any-mechanism-you-fix]]:
   λ всмоктують хибу механіки, тож остаточне слово — за MSE, не за цим стендом.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/clsedge.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts            # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
L_, W_, KS = 32, 256, 31
NPAIR, CH = 524_288, 32768
I2PI = 0.3989422804014327
KGRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
# 🔑 РОЗЩЕПЛЕННЯ ПОПРАВКИ. Пробний прогін (3 мережі) показав ПАРАДОКС:
#    ІСТИННИЙ `γ₁` робить `m` ГІРШЕ (0.973×), але `s` КРАЩЕ (1.167×), а
#    фальшивий рекурсивний (промах 98%!) із `κ=4` робить `m` кращим (1.089×).
#    ⟹ поправки до M1 і до M2 треба міряти ОКРЕМО: можливо, корисна лише
#    одна з них, а разом вони ламають баланс двох хиб гаусового замикання
#    (той самий механізм, що вбив ворота в С81 — див. [[finding-gate-channel-...]]).
SPEC = ([('⛔ 0 · гаусове (нинішнє)', None, 0.0, (1, 1, 1)),
         ('🎯 1 · іст. γ₁, ПОВНА', 'T', 1.0, (1, 1, 1)),
         ('🎯 2 · іст. γ₁, лише M1 (середнє)', 'T', 1.0, (1, 0, 0)),
         ('🎯 3 · іст. γ₁, лише M2 (дисперсія)', 'T', 1.0, (0, 1, 0))]
        + [(f'🟢 · рекурс. × κ={k:g}', None, k, (1, 1, 1)) for k in KGRID]
        + [(f'🟠 · рекурс. лише M2, κ={k:g}', None, k, (0, 1, 1))
           for k in (2.0, 4.0)])
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _phi(x):
    return torch.exp(-0.5 * x * x) * I2PI


def _Phi(x):
    return 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def closure_edge(wc, ksig, dev, gam_ext=None, kappa=1.0, use=(1, 1, 1)):
    """Замикання Мелера K=2 з ЕДЖВОРТ-3 поправкою першого трьох моментів.

    `gam_ext=None, kappa=0` ⟹ РІВНО `closure_parts` із `base_real` (пломба).
    `gam_ext` — список істинних `γ₁` передактивацій шарів 1..ksig (стеля).
    Інакше `γ₁` будується РЕКУРСИВНО з `μ₃` активацій попереднього шару.
    """
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    mu3 = torch.zeros(n0, dtype=torch.float64, device=dev)   # μ₃ ВХОДУ шару
    s_o, m_o, q_o, g_o = [], [], [], []
    for i in range(ksig + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        # 🔑 `γ₁` передактивації. Шар 0: вхід — білий гаусів ⟹ скосу НЕМА.
        if i == 0:
            gam = torch.zeros_like(s)
        elif gam_ext is not None:
            gam = gam_ext[i - 1]
        else:
            gam = kappa * ((w ** 3) * mu3[:, None]).sum(0) / s ** 3
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        s2 = s * s
        M1 = mp * Ph + s * ph - use[0] * (gam / 6.0) * mp * ph
        M2 = (mp * mp + s2) * Ph + mp * s * ph + use[1] * (gam / 3.0) * s2 * ph
        M3 = ((mp ** 3 + 3.0 * mp * s2) * Ph
              + s * (mp * mp + 2.0 * s2) * ph + use[2] * gam * s2 * s * Ph)
        d = M2 - M1 * M1
        mu3 = M3 - 3.0 * M1 * M2 + 2.0 * M1 ** 3
        m = M1
        C = (cp * torch.outer(Ph, Ph)
             + (0.5 * cp * cp) * torch.outer(ph / s, ph / s))
        C = C + torch.diag(d - C.diagonal())
        if i >= 1:
            s_o.append(s)
            m_o.append(m)
            q_o.append(Ph + ph * gam * (t * t - 1.0) / 6.0)
            g_o.append(gam)
    return s_o, m_o, q_o, g_o


def moments(wc, dev, seed, npair):
    """Істинні `q`, `m_relu`, `s`, `γ₁` передактивацій шарів 1..KS."""
    ns = [w.shape[1] for w in wc]
    acc = [[torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
           for _ in range(5)]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npair:
        k = min(CH, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            p = (x @ w).double()
            acc[0][i] += (p > 0).double().sum(0)
            p2 = p * p
            acc[1][i] += p.sum(0)
            acc[2][i] += p2.sum(0)
            acc[3][i] += (p2 * p).sum(0)
            x = torch.relu(p).float()
            acc[4][i] += x.double().sum(0)          # ⟵ середнє ПІСЛЯ ReLU
        done += k
    n = 2.0 * done
    out = []
    for i in range(1, KS + 1):
        m1, m2, m3 = acc[1][i] / n, acc[2][i] / n, acc[3][i] / n
        s = (m2 - m1 * m1).clamp_min(1e-300).sqrt()
        c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        out.append((acc[0][i] / n, acc[4][i] / n, s, c3 / s ** 3))
    return out


def rel(pred, tru, lo=0, hi=KS):
    num = sum(float(((pred[i] - tru[i]) ** 2).sum()) for i in range(lo, hi))
    den = sum(float((tru[i] ** 2).sum()) for i in range(lo, hi))
    return np.sqrt(num / max(den, 1e-300))


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]

NM = [s[0] for s in SPEC]
E = [[[] for _ in NM] for _ in range(4)]        # m · s · q · γ₁
SEAL, PROF = [], [[] for _ in NM]
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, _ = cut_and_budget(W, DEV)
        MT = moments(wc, DEV, 830_000 + 7919 * n, NPAIR)
        q_t = [MT[i][0] for i in range(KS)]
        m_t = [MT[i][1] for i in range(KS)]
        s_t = [MT[i][2] for i in range(KS)]
        g_t = [MT[i][3] for i in range(KS)]
        # ⛔ пломба: нинішнє замикання проти власної реалізації при γ₁≡0
        s_r, m_r = closure_parts(wc, KS, DEV)
        for v, (_, src, kap, use) in enumerate(SPEC):
            s_c, m_c, q_c, g_c = closure_edge(
                wc, KS, DEV, gam_ext=(g_t if src == 'T' else None),
                kappa=kap, use=use)
            if v == 0:
                SEAL.append(max(rel(m_c, m_r), rel(s_c, s_r)))
            for j, (P, T) in enumerate(((m_c, m_t), (s_c, s_t),
                                        (q_c, q_t), (g_c, g_t))):
                E[j][v].append(rel(P, T))
            PROF[v].append([rel(m_c, m_t, k, k + 1) for k in range(KS)])
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}', flush=True)

print(f'\n{NETS} мереж · істина {2*NPAIR} семплів · шари 1..{KS}')
print(f'⛔ ПЛОМБА «γ₁≡0 ≡ нинішнє замикання»: {max(SEAL):.3e} '
      f'{"✅" if max(SEAL) < 1e-12 else "🚨 РОЗБІЖНІСТЬ"}')
print(f'\n{"варіант":32s} {"похибка m":>11s} {"вигр.":>7s} {"похибка s":>11s} '
      f'{"похибка q":>11s} {"похибка γ₁":>11s}')
print('-' * 92)
b = float(np.mean(E[0][0]))
for v, nm in enumerate(NM):
    a = [float(np.mean(E[j][v])) for j in range(4)]
    print(f'{nm:32s} {100*a[0]:10.4f}% {b/a[0]:6.3f}× {100*a[1]:10.4f}% '
          f'{100*a[2]:10.4f}% {100*a[3]:10.2f}%')

best = 1 + int(np.argmin([float(np.mean(E[0][v])) for v in range(1, len(NM))]))
print(f'\nПРОФІЛЬ ПОХИБКИ `m` ПО ГЛИБИНІ (шар: гаусове → {NM[best]})')
p0, p1 = np.mean(PROF[0], 0), np.mean(PROF[best], 0)
for k in range(0, KS, 3):
    print(f'  шар {k+1:2d}: {100*p0[k]:8.3f}% → {100*p1[k]:8.3f}% '
          f'({p0[k]/max(p1[k],1e-30):5.3f}×)')
print('\n🎯 ВОРОТА для порівняння (С81): похибка q 0.887% (нинішня) → 0.385% '
      'з істинним γ₁ ⟹ ×1.1296 у каналі, але реалістично ×1.0025 ⟹ мертво.')
print('📊 БАЛ 1.954e-07. Тут вирок ЛИШЕ про точність замикання; кінцевий — '
      'end-to-end, бо λ всмоктують хибу механіки (С76).')
