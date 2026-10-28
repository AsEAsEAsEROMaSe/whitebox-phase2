"""ЧИ ШКОДИТЬ `γ` НАСПРАВДІ — АБО ЦЕ БУВ АРТЕФАКТ ВІДСУТНЬОЇ α? (С87)

## ЗВІДКИ — ПІДОЗРА, ЩО ВИНИКЛА СЬОГОДНІ
`ladder87` записав: `L2b` (наше замикання + ОРАКУЛЬНІ `γ₃,γ₄` узгоджено) =
1.0031% проти бази 0.9416% ⟹ висновок «ФОРМА БЕЗ МАСШТАБУ ВІД'ЄМНА».
🚨 АЛЕ сьогодні `layerk87` показав РІВНО таку саму помилку на `L6`: без α вона
виглядала ГІРШОЮ за базу (1.2631% проти 0.9449%), а ПІСЛЯ α — нейтральною
(×0.959). Причина: α знімає масштабну складову, яку поправка зсуває.
`γ`-поправка зсуває масштаб так само: `E[ReLU] = s·(G₀ + γ₃/6·G₃ + γ₄/24·G₄)`.
⟹ вирок «форма шкодить» міг бути винесений ПРОТИ КАЛІЧЕНОЇ БАЗИ.

## 🎯 ЧОМУ САМЕ `γ` ВАРТА ПЕРЕВІРКИ (єдина м'яка вимога)
```
ТЗ із ladder87:  σ на 3% ⟹ ×176 ГІРШЕ  ·  γ на 3% ⟹ БЕЗКАРНО
```
А `κ₃` МИ ВМІЄМО РАХУВАТИ: С86 (`finding-jjk-term...`) дала корел **+0.9282**
і масштаб **0.9359 ± 0.0213 (2.3%)** через член `(j,j,k)`, де вага стоїть у
ПАРНОМУ степені `W²` і знаки не скорочуються.
⟹ 2.3% точності при вимозі 3% — ЄДИНА величина, де опора вже є.

## ⛔ЧОМУ ЦЕ НЕ ПОВТОР ЗАКРИТОГО
Закрито «Еджворт ×3» (С49/С73) і «форма без масштабу» — але ОБИДВА виміряні
БЕЗ α-калібровки. Закон, здобутий сьогодні:
**α-КАЛІБРОВКА МУСИТЬ СТОЯТИ В БАЗІ ЗАВЖДИ** ⟹ ці вироки треба перевиносити.

## ⛔ПЛОМБИ
 P1 база мусить дати 0.9416% / MSE 7.84e-05 — інакше не той об'єкт.
 P2 `L5` (оракул m,σ,γ) мусить бути НАЙКРАЩОЮ — інакше баг.
 P3 α-калібровка LOO ПО МЕРЕЖАХ (не по всіх одразу) — інакше витік.
 P4 ⛔КОНТРОЛЬ: `γ` ЧУЖОЇ мережі. Якщо працює не гірше — сигнал не в `γ`.
 P5 ⛔КОНТРОЛЬ ЗНАКУ: `γ → −γ` мусить ПОГІРШИТИ вдвічі сильніше.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) БЕЗ α `γ` погіршить (відтворить старе 1.0031%). Шанс 85%.
 (б) ПІСЛЯ α `γ` стане НЕЙТРАЛЬНОЮ або кращою (×≥1.0). Шанс **55%** — саме
     так поводилась `L6`.
 (в) ПІСЛЯ α `γ` дасть ×≥1.3. Шанс **20%** — це вже був би реальний важіль.
 (г) ⛔чужа `γ` буде помітно гіршою за свою. Шанс 80%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/gamalpha87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)
SUB97 = 1.873e-06


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
    return ((m * m + s2) * P + m * s * f + (g3 / 3.0) * s2 * f
            - (g4 / 12.0) * s2 * t * f
            + (g3 * g3 / 36.0) * s2 * (3.0 * t - t ** 3) * f)


def chain(W, orc_g=None, orc_ms=None, coherent=True, gscale=1.0):
    """`orc_g` — форма (γ₃,γ₄) по шарах; `gscale` масштабує її (для ⛔контролю)."""
    n = W[0].shape[1]
    zero = torch.zeros(n, dtype=torch.float64, device=W[0].device)
    C = W[0].T @ W[0]
    m = zero.clone()
    for i in range(1, len(W) + 1):
        s_own = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        if orc_ms is not None:
            m_u, s = orc_ms[i - 1]
        else:
            m_u, s = m, s_own
        if orc_g is not None:
            g3, g4 = orc_g[i - 1]
            g3, g4 = g3 * gscale, g4 * gscale
        else:
            g3, g4 = zero, zero
        m_a = edge_relu(m_u, s, g3, g4)
        if i == len(W):
            return m_a
        t = m_u / s
        P, f = Phi(t), phi(t)
        Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        q2 = (edge_relu2(m_u, s, g3, g4) if coherent
              else (m_u * m_u + s * s) * P + (m_u * s) * f)
        Ca = Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30)
                             - torch.diagonal(Ca))
        m = m_a @ W[i]
        C = W[i].T @ (Ca @ W[i])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} семплів · {DEV}\n', flush=True)

TAGS = ('B0 база (без γ)', 'G1 +γ лише в m', 'G2 +γ УЗГОДЖЕНО',
        'G3 +γ₃ тільки (γ₄=0)', '⛔G4 γ ЧУЖОЇ мережі', '⛔G5 γ → −γ',
        'L4 оракул m,σ', 'L5 оракул m,σ,γ')
E = {t: np.zeros(len(FS)) for t in TAGS}
MS = {t: np.zeros(len(FS)) for t in TAGS}
PT = {t: np.zeros(len(FS)) for t in TAGS}
PP = {t: np.zeros(len(FS)) for t in TAGS}
TT = np.zeros(len(FS))
NVAR = np.zeros(len(FS))
NRM2 = np.zeros(len(FS))
STASH = []


def load_w(i):
    sd = int(np.load(FS[i])['seed'])
    w32 = [torch.tensor(np.asarray(WZ[have[sd]][q], np.float32), device=DEV)
           for q in range(L_)]
    return w32, [w.double() for w in w32]


for n in range(len(FS)):
    W32, W = load_w(n)
    with torch.no_grad():
        MOM = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        acch = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(3300 + n)
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
            r = a.sum(0, dtype=torch.float64)
            acc += r
            if d < NS // 2:
                acch += r
            d += b
        tru = acc / NS
        NRM2[n] = float((tru ** 2).mean())
        NVAR[n] = float((((acch / (NS // 2)) - tru) ** 2).mean())
        MOM /= NS
        G, MSv = [], []
        for q in range(L_):
            m1, m2, m3, m4 = MOM[q, 0], MOM[q, 1], MOM[q, 2], MOM[q, 3]
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            c4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4 - 3.0 * c2 * c2
            s = torch.sqrt(c2)
            G.append((c3 / s ** 3, c4 / s ** 4))
            MSv.append((m1.clone(), s))
        STASH.append(([a.cpu() for a, _ in G], [b.cpu() for _, b in G],
                      [a.cpu() for a, _ in MSv], [b.cpu() for _, b in MSv],
                      tru.cpu()))
    print(f'  МК net{n}', flush=True)

print('\n══ ПРОХІД 2 ══', flush=True)
for n in range(len(FS)):
    W32, W = load_w(n)
    with torch.no_grad():
        g3s, g4s, m1s, ss, trc = STASH[n]
        G = [(g3s[q].to(DEV), g4s[q].to(DEV)) for q in range(L_)]
        MSv = [(m1s[q].to(DEV), ss[q].to(DEV)) for q in range(L_)]
        tru = trc.to(DEV)
        no = (n + 1) % len(FS)
        Go = [(STASH[no][0][q].to(DEV), STASH[no][1][q].to(DEV))
              for q in range(L_)]
        Z = [(torch.zeros_like(G[q][0]), torch.zeros_like(G[q][1]))
             for q in range(L_)]
        runs = {
            'B0 база (без γ)': chain(W),
            'G1 +γ лише в m': chain(W, orc_g=G, coherent=False),
            'G2 +γ УЗГОДЖЕНО': chain(W, orc_g=G, coherent=True),
            'G3 +γ₃ тільки (γ₄=0)': chain(
                W, orc_g=[(G[q][0], Z[q][1]) for q in range(L_)]),
            '⛔G4 γ ЧУЖОЇ мережі': chain(W, orc_g=Go),
            '⛔G5 γ → −γ': chain(W, orc_g=G, gscale=-1.0),
            'L4 оракул m,σ': chain(W, orc_ms=MSv),
            'L5 оракул m,σ,γ': chain(W, orc_g=G, orc_ms=MSv),
        }
        TT[n] = float((tru * tru).sum())
        for t, pr in runs.items():
            MS[t][n] = float(((pr - tru) ** 2).mean())
            E[t][n] = math.sqrt(MS[t][n] / NRM2[n])
            PT[t][n] = float((pr * tru).sum())
            PP[t][n] = float((pr * pr).sum())
    print(f'  net{n} · B0 {E["B0 база (без γ)"][n]:.4%} · '
          f'G2 {E["G2 +γ УЗГОДЖЕНО"][n]:.4%}', flush=True)


def alpha_mse(t):
    """⛔P3: глобальний множник, підібраний LOO ПО МЕРЕЖАХ (як `_ALPHA`)."""
    out = np.zeros(len(FS))
    a_n = PT[t] / np.maximum(PP[t], 1e-30)
    for i in range(len(FS)):
        msk = np.arange(len(FS)) != i
        a = a_n[msk].mean() if len(FS) > 1 else a_n[i]
        out[i] = (a * a * PP[t][i] - 2.0 * a * PT[t][i] + TT[i]) / W_
    return out.mean()


print(f'\n{"═" * 74}\n⛔ПЛОМБИ')
b0 = MS['B0 база (без γ)'].mean() - NVAR.mean()
print(f' P1 база {E["B0 база (без γ)"].mean():.4%} · MSE {b0:.3e} '
      f'(очік. 0.9416% / 7.84e-05) '
      f'{"✅" if abs(E["B0 база (без γ)"].mean() - 0.009416) < 0.0008 else "⛔"}')
print(f' P4 шум істини {NVAR.mean():.3e} = {NVAR.mean() / b0:.2%} бази')

b0a = alpha_mse('B0 база (без γ)') - NVAR.mean()
print(f'\n{"═" * 74}')
print('🎯 ЧИ ДАЄ `γ` ЩОСЬ ПІСЛЯ α — ГОЛОВНА ТАБЛИЦЯ')
print(f'{"сходинка":22s} {"віднос.":>8s} {"БЕЗ α":>10s} {"ПІСЛЯ α":>10s} '
      f'{"vs базаБЕЗα":>11s} {"vs базаПІСЛЯα":>13s}')
for t in TAGS:
    raw = max(MS[t].mean() - NVAR.mean(), NVAR.mean() * 0.2)
    aft = max(alpha_mse(t) - NVAR.mean(), NVAR.mean() * 0.2)
    print(f'{t:22s} {E[t].mean():7.4%} {raw:10.3e} {aft:10.3e} '
          f'{"×%.3f" % (b0 / raw):>11s} {"×%.3f" % (b0a / aft):>13s}')
print(f'{"🏆 sub97":22s} {0.001456:7.4%} {SUB97:10.3e} {SUB97:10.3e}')

g2 = alpha_mse('G2 +γ УЗГОДЖЕНО') - NVAR.mean()
g4c = alpha_mse('⛔G4 γ ЧУЖОЇ мережі') - NVAR.mean()
print(f'\n📣 ВИРОК: `γ` ПІСЛЯ α дає ×{b0a / g2:.3f} проти бази')
print(f'   ⛔чужа γ ×{b0a / g4c:.3f} — своя мусить бути ПОМІТНО кращою')
if b0a / g2 >= 1.3 and (b0a / g2) > 1.3 * (b0a / g4c):
    print('   🟢 ФОРМА ПРАЦЮЄ ПІСЛЯ α — старий вирок «форма від\'ємна» був')
    print('      артефактом каліченої бази. Наступний крок: γ₃ з (j,j,k).')
else:
    print('   ⛔ форма не дає нічого і після α ⟹ старий вирок ПІДТВЕРДЖЕНО')
