"""ЧИ МОЖНА ВИПРАВИТИ `σ` КОНСТАНТАМИ З БАНКУ — І ЩО ЦЕ ДАЄ. (С87)

## ЗВІДКИ
`side66/ladder87.py` встановив три речі:
1. `L10` (рекурсія `m` при правильних `σ`,`γ`) = **5.377e-08** проти лідерових
   5.211e-08 — метод відтворено.
2. ТЗ на точність: `γ` можна псувати на 3% майже безкарно (**×1.2**), а `σ`
   на 3% — катастрофа (**×176**). ⟹ уся вимога на `σ`.
3. РОЗКЛАД похибки `σ` нашого ланцюга:
```
шар:        1      4      7     13     19     25     31
масштаб r 1.0000 0.9962 0.9847 0.9523 0.9259 0.8935 0.8724   ← σ ЗАНИЖЕНА
розкид r  0.0001 0.0008 0.0019 0.0088 0.0096 0.0142 0.0118   ← СТАБІЛЬНИЙ!
σ без масшт 0.099% 0.781% 2.137% 5.203% 6.958% 8.433% 8.720%
```
🔑 З 14.887% похибки `σ` **12.07% — це СПІЛЬНИЙ МАСШТАБ**, і він стабільний по
мережах (розкид 1.2%) ⟹ лікується 32 КОНСТАНТАМИ з банку, безкоштовно.
Лишається 8.72% розкиду.

## 🎯 ЩО МІРЯЄМО
| варіант | що робить |
|---|---|
| BASE | наш ланцюг як є |
| SCAL | `σ ← σ / r̄_шар`, `r̄` — LOO-константа ПО МЕРЕЖАХ |
| SCAL+γ | те саме + ОРАКУЛЬНА форма, УЗГОДЖЕНО |
| ORC+γ | `L10` — стеля (оракульна `σ`) |

⚠️ЧОМУ ЦЕ НЕ ТЕ САМЕ, ЩО ТЗ `E10`: там `σ` псувалася ВИПАДКОВИМ шумом 10%, а
наша похибка СТРУКТУРОВАНА і частково скорочується з похибкою `m` (доказ:
`L6` — оракульна `σ` при нашому `m` дає 1.24%, ГІРШЕ за базу 0.94%).
⟹ вирок може дати лише прямий вимір.

⛔ПЛОМБИ:
 Q1 `r̄` рахується LOO ПО МЕРЕЖАХ — інакше це підгін під власну відповідь.
 Q2 BASE мусить відтворити 0.9416% (`ladder87.py`).
 Q3 ⛔КОНТРОЛЬ: `r̄` від ПЕРЕМІШАНИХ шарів мусить ЗІПСУВАТИ (якщо ні — константи
    ловлять не структуру, а спільний зсув).
 Q4 після SCAL масштаб `r` мусить стати ≈1.0 — інакше корекція не спрацювала.

📣 ПЕРЕДБАЧЕННЯ:
 (а) SCAL сам по собі ПОГІРШИТЬ або дасть нуль — бо це рівно `L6` у слабшій
     формі (виправляємо `σ`, ламаючи скорочення з `m`). Шанс 65%.
 (б) SCAL+γ дасть 0.25–0.5%: краще за нашу базу 0.94%, але далеко від
     лідерових 0.052%. Шанс 45%.
 (в) 🎯ГОЛОВНЕ: якщо SCAL+γ ≤ 0.312% — детермінований шлях б'є `sub97` і гілка
     оживає. Шанс **30%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/selfsig87.py [cuda:N] [NETS] [LOG2N]
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


def chain(W, orc_g=None, orc_s=None, scal=None, trace=None):
    """Ланцюг Мелера. `orc_s` — оракульна σ; `scal[i]` — множник корекції σ."""
    dev = W[0].device
    n = W[0].shape[1]
    zero = torch.zeros(n, dtype=torch.float64, device=dev)
    C = W[0].T @ W[0]
    m = zero.clone()
    for i in range(1, len(W) + 1):
        s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        if orc_s is not None:
            s = orc_s[i - 1]
        elif scal is not None:
            s = s / scal[i - 1]
        g3, g4 = orc_g[i - 1] if orc_g is not None else (zero, zero)
        if trace is not None:
            so = trace['orc'][i - 1]
            trace['r'].append(float((s * so).sum() / (so * so).sum()))
            trace['e'].append(float(torch.sqrt(((s - so) ** 2).mean()
                                               / (so * so).mean())))
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
print(f'{len(FS)} мереж · 2^{LOG2N} семплів · {DEV}\n', flush=True)

ORC_S, ORC_G, TRU, NRM, SEEDS = [], [], [], [], []
RMAT = np.zeros((len(FS), L_))

# ── прохід 1: МК-оракули і масштаби `r` базового ланцюга ──
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    SEEDS.append(sd0)
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        MOM = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(6600 + n)
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
        os_, og_ = [], []
        for q in range(L_):
            m1, m2, m3, m4 = MOM[q, 0], MOM[q, 1], MOM[q, 2], MOM[q, 3]
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            c4 = (m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
                  - 3.0 * c2 * c2)
            sq = torch.sqrt(c2)
            os_.append(sq)
            og_.append((c3 / sq ** 3, c4 / sq ** 4))
        ORC_S.append(os_)
        ORC_G.append(og_)
        TRU.append(tru)
        NRM.append(float(torch.sqrt((tru ** 2).mean())))
        tr = {'orc': os_, 'r': [], 'e': []}
        chain(W, trace=tr)
        RMAT[n] = np.array(tr['r'])
    print(f'  net{n} (seed {sd0}) оракул готовий · r[31]={RMAT[n, 31]:.4f}',
          flush=True)

# ── 🔑 ІТЕРАТИВНА КАЛІБРОВКА: `sc` — НЕРУХОМА ТОЧКА, а не одне ділення ──
# ПЕРША СПРОБА ПРОВАЛИЛАСЬ І ЦЕ БУЛА МОЯ ПОМИЛКА: `r̄` міряний на
# НЕСКОРИГОВАНОМУ ланцюгу, а застосований до скоригованого ⟹ додатний зворотний
# зв'язок (σ↑ ⟹ C_a↑ ⟹ σ наступного шару↑ ⟹ ...) і масштаб поїхав 1.0→3.39.
# Правильно: шукати `sc`, при якому СКОРИГОВАНИЙ ланцюг має `r = 1`.
# LOO тримається ЧЕРЕЗ УСІ ітерації — інакше це підгін під власну відповідь.
WCACHE = []
for n in range(len(FS)):
    WCACHE.append([torch.tensor(np.asarray(WZ[have[SEEDS[n]]][q], np.float32),
                                device=DEV).double() for q in range(L_)])

SC = np.ones((len(FS), L_))
for it in range(8):
    RIT = np.zeros((len(FS), L_))
    with torch.no_grad():
        for n in range(len(FS)):
            tr = {'orc': ORC_S[n], 'r': [], 'e': []}
            chain(WCACHE[n], scal=torch.tensor(SC[n], dtype=torch.float64,
                                               device=DEV), trace=tr)
            RIT[n] = np.array(tr['r'])
    NEW = np.zeros_like(SC)
    for i in range(len(FS)):
        msk = np.arange(len(FS)) != i
        NEW[i] = SC[i] * RIT[msk].mean(0)      # LOO на КОЖНІЙ ітерації
    SC = NEW
    print(f'  калібровка it{it}: r[31] = {RIT[:, 31].mean():.5f} · '
          f'sc[31] = {SC[:, 31].mean():.5f}', flush=True)
RBAR = SC
print(f'\n⛔Q1 `sc` LOO по мережах · sc[шари 1,8,16,24,32] = '
      + ' '.join(f'{RBAR[:, q].mean():.4f}' for q in (0, 7, 15, 23, 31)))

TAGS = ('BASE', 'SCAL', 'SCAL+γ', 'ORC+γ (L10)', '⛔SCAL перемішані шари')
E = {t: np.zeros(len(FS)) for t in TAGS}
R = {t: np.zeros(len(FS)) for t in TAGS}
RES_R = np.zeros((len(FS), L_))

for n, f in enumerate(FS):
    W = WCACHE[n]
    sc = torch.tensor(RBAR[n], dtype=torch.float64, device=DEV)
    shuf = torch.tensor(RBAR[n][::-1].copy(), dtype=torch.float64, device=DEV)
    with torch.no_grad():
        tr = {'orc': ORC_S[n], 'r': [], 'e': []}
        runs = {
            'BASE': chain(W),
            'SCAL': chain(W, scal=sc, trace=tr),
            'SCAL+γ': chain(W, orc_g=ORC_G[n], scal=sc),
            'ORC+γ (L10)': chain(W, orc_g=ORC_G[n], orc_s=ORC_S[n]),
            '⛔SCAL перемішані шари': chain(W, scal=shuf),
        }
        RES_R[n] = np.array(tr['r'])
        for t, pr in runs.items():
            E[t][n] = float(torch.sqrt(((pr - TRU[n]) ** 2).mean())) / NRM[n]
            R[t][n] = float((pr * TRU[n]).sum() / (TRU[n] * TRU[n]).sum())


def loo_resid(err, rat):
    out = np.zeros_like(err)
    for i in range(len(err)):
        msk = np.arange(len(err)) != i
        out[i] = math.sqrt(max(err[i] ** 2 - (rat[i] - 1.0) ** 2
                               + (rat[i] - rat[msk].mean()) ** 2, 1e-30))
    return out


print(f'\n⛔Q4 масштаб `r` ПІСЛЯ корекції (мусить бути ≈1.0): '
      + ' '.join(f'{RES_R[:, q].mean():.4f}' for q in (0, 7, 15, 23, 31)))
print('\n══ РЕЗУЛЬТАТ ══')
print('варіант                 сира     після α    MSE        бал(×0.1)   vs sub97')
for t in TAGS:
    rs = loo_resid(E[t], R[t])
    mse = 1.698e-05 * (rs.mean() / 0.00941) ** 2
    print(f'{t:22s} {E[t].mean():7.4%} {rs.mean():7.4%}  {mse:.3e}  '
          f'{mse * 0.1:.3e}   ×{1.873e-07 / (mse * 0.1):6.3f}')
print(f'{"ЛІДЕР 322059":22s} {0.0521 / 100:7.4%} {0.0521 / 100:7.4%}  '
      f'{5.211e-08:.3e}  {5.102e-08:.3e}   ×{1.873e-07 / 5.102e-08:6.3f}')
print('\n🎯 ГЕЙТ: «бал(×0.1)» < 1.873e-07 ⟹ детермінований шлях Б\'Є `sub97`.')
