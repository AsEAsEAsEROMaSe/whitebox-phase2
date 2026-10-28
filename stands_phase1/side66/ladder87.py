"""ДРАБИНА ТОЧНОСТІ: ЩО САМЕ ТРЕБА ЗНАТИ, ЩОБ ДІЙТИ ДО 0.053% ЛІДЕРА. (С87)

## ЗАВДАННЯ ЮЗЕРА
«Копай метод лідера. В лідера може бути й інший метод — згадай, який у нього
насправді.»

## 🔑 ЩО ПРО ЛІДЕРА ВІДОМО ТВЕРДО (сабміт 322059, joe_wanza)
```
F = 1.300e7 флопів · R = 2647 мс · C/B = 97.32% · MSE = 5.211e-08
ALL-LAYERS MSE 0.7537 · GROWTH L0→L31 0× ⟹ проміжні шари НЕ передбачає (як і ми)
IQR 3.37e-8…5.94e-8 (×1.76) · best 2.03e-8 · worst 1.32e-7
```
🚨 ВИПРАВЛЕННЯ МОГО ПОПЕРЕДНЬОГО ПРОЧИТАННЯ: «1.24× діагонального замикання» —
це про ЦІНУ (1.3e7 = 1.24 × 1.05e7 флопів), а НЕ про точність.

## 📐 ЩО ФЛОПИ ДОЗВОЛЯЮТЬ, А ЩО ЗАБОРОНЯЮТЬ (арифметика, не здогад)
```
матвек 256×256              1.31e5  ⟹ F дозволяє ~100 штук = 3.1 на шар
матмул 256³ (одна C=WᵀCW)   3.36e7  ⟹ ДОРОЖЧЕ ЗА ВЕСЬ ЙОГО БЮДЖЕТ
повний прогін 1 зразка      4.19e6  ⟹ 3.1 прогони ⟹ МК НЕМАЄ
```
⟹ 🔑 **ЛІДЕР ВЕДЕ 3–4 ВЕКТОРИ І ЖОДНОЇ МАТРИЦІ.** Це або `(m, v, κ₃)`, або
`(m, v)` + квадратура на 3 вузлах.
⟹ 🔑 2647 мс / 8192 нейронів = **323 мкс/нейрон** ⟹ Python-цикл із важким
усередині (квадратура / розв'язувач / спецфункції поелементно).

## 📐 ПЕРЕКЛАД ЙОГО MSE У ВІДНОСНУ ПОХИБКУ
Наше чисте замикання: 0.941% ⟺ MSE 1.698e-05. MSE ∝ похибка² ⟹
    `0.941% × √(5.211e-08 / 1.698e-05)` = **0.0521%**
⟹ 🎯 **ЦІЛЬ ДРАБИНИ: знайти, який НАБІР ЗНАНЬ дає 0.052% у ЛАНЦЮГУ.**
Це відсіє клас методів, бо кожна сходинка каже, що ОБОВ'ЯЗКОВО мусить бути.

## 🎯 СХОДИНКИ (усі детерміновані; «оракул» = з МК тієї самої мережі)
| # | що знає | чим це є для лідера |
|---|---|---|
| L0 | лише `(m, v)`, `Var(p')=Σ W² v` — БЕЗ `C` | найдешевше, що влазить у 1.3e7 |
| L1 | наше замикання: Мелер K=2 + повна `C` | наша база, 165× дорожча за його `F` |
| L2 | L1 + ОРАКУЛЬНІ `γ₃,γ₄` понейронно | чи вистачить самої ФОРМИ |
| L3 | L0 + ОРАКУЛЬНІ `γ₃,γ₄` | 🔑 чи можна БЕЗ `C` взагалі |
| L4 | ОРАКУЛЬНІ `m,σ` на кожному шарі, гаус | внесок самої `C`, ізольовано |
| L5 | ОРАКУЛЬНІ `m,σ,γ₃,γ₄` | 🎯 СТЕЛЯ чотирьох моментів |

⛔ПЛОМБИ:
 P1 L1 мусить дати 0.94% — це вже двічі зміряно (`onefac2.py`, банк 0.9645%).
 P2 L5 мусить бути НАЙКРАЩОЮ сходинкою (більше знань — менша похибка);
    порушення = баг.
 P3 шум істини: піврозмір МК.
 P4 L4 і L5 на ОДНОМУ шарі відтворюють `perneuron.py` (гаус 0.19%, 4 моменти
    0.0147%) — якщо ні, прилад міряє не той об'єкт.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) L5 ≈ 0.05–0.09% ⟹ ЗБІЖИТЬСЯ з лідером 0.052%. Шанс **60%**. Якщо так —
     він веде саме 4 понейронні моменти, і питання зводиться до «як дешево».
 (б) L3 (без `C`, але з точною формою) буде НАБАГАТО гіршою за L2 — бо `Var(p)`
     без кореляцій має похибку 4–10% (`onefac.py`). Шанс 70%.
     ⚠️Якщо L3 виявиться доброю — це переворот: тоді `C` не потрібна ВЗАГАЛІ,
     і 1.3e7 флопів лідера пояснюються повністю.
 (в) L2 ≈ 0.07–0.15%: форма поверх нашої `C` дає більшу частину виграшу.
 (г) L0 буде катастрофою (≥3%) — діагональ без кореляцій.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/ladder87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
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
    """`E[ReLU]` через `G_n(t) = (−1)^(n−2)φ^(n−2)(−t)`; `g3=g4=0` ⟹ гаус."""
    t = m / s
    P, f = Phi(t), phi(t)
    G0 = t * P + f
    G3 = -t * f
    G4 = (t * t - 1.0) * f
    G6 = (t ** 4 - 6.0 * t * t + 3.0) * f
    return s * (G0 + (g3 / 6.0) * G3 + (g4 / 24.0) * G4 + (g3 * g3 / 72.0) * G6)


def edge_relu2(m, s, g3, g4):
    """`E[ReLU(X)²]` з Еджвортом — ВИВЕДЕНО, не вгадано.

    `H_n = ∫_{-t}^∞ (t+z)² He_n(z)φ(z)dz`; двічі інтегруємо частинами, обидва
    позаінтегральні члени зникають (`f(−t)=f'(−t)=0`, бо `f=(t+z)²`), лишається
        `H_n = 2·He_{n−3}(−t)·φ(t)`  для `n ≥ 3`
    ⟹ `(γ₃/6)H_3 = (γ₃/3)φ` і `(γ₄/24)H_4 = −(γ₄/12)tφ` — РІВНО `_C3=1/3`,
    `_C4=1/12` із сабміта ✅ (незалежне підтвердження тієї формули)."""
    t = m / s
    P, f = Phi(t), phi(t)
    s2 = s * s
    return ((m * m + s2) * P + m * s * f
            + (g3 / 3.0) * s2 * f - (g4 / 12.0) * s2 * t * f
            + (g3 * g3 / 36.0) * s2 * (3.0 * t - t ** 3) * f)


def relu_var_gauss(m, s, g3=None, g4=None):
    """`(E[ReLU], Var(ReLU))` для діагонального ланцюга."""
    z = torch.zeros_like(s) if g3 is None else g3
    z4 = torch.zeros_like(s) if g4 is None else g4
    m1 = edge_relu(m, s, z, z4)
    m2 = edge_relu2(m, s, z, z4)
    return m1, torch.clamp(m2 - m1 * m1, min=1e-30)


def chain(W, mode, orc_g=None, orc_ms=None, coherent=False, trace=None,
          only=None):
    """mode: 'diag' — без `C`; 'mehler' — наше замикання.

    `orc_g[i] = (γ₃,γ₄)` передактивації шару `i` (оракул) ⟹ форма.
    `orc_ms[i] = (m,σ)` передактивації шару `i` (оракул) ⟹ масштаб."""
    dev = W[0].device
    n = W[0].shape[1]
    zero = torch.zeros(n, dtype=torch.float64, device=dev)
    if mode == 'diag':
        v = torch.diagonal(W[0].T @ W[0]).clone()
    else:
        C = W[0].T @ W[0]
    m = zero.clone()
    for i in range(1, len(W) + 1):
        s_own = torch.sqrt(torch.clamp(v if mode == 'diag'
                                       else torch.diagonal(C), min=1e-30))
        if orc_ms is not None:
            m_o, s_o = orc_ms[i - 1]
            # 🔑 РОЗДІЛЕННЯ: `only='s'` бере з оракула ЛИШЕ масштаб розкиду,
            #    `only='m'` — ЛИШЕ середнє. Так видно, ЯКА з двох величин несе
            #    похибку ланцюга (закон «спершу еталон, тоді прилад»).
            m_u = m if only == 's' else m_o
            s = s_own if only == 'm' else s_o
        else:
            s, m_u = s_own, m
        g3, g4 = orc_g[i - 1] if orc_g is not None else (zero, zero)
        if trace is not None:                 # 🔑 ДЕ НАРОДЖУЄТЬСЯ ПОХИБКА σ
            mo, so = trace['orc'][i - 1]
            trace['m'].append(float(torch.sqrt(((m_u - mo) ** 2).mean()
                                               / (mo * mo).mean())))
            trace['s'].append(float(torch.sqrt(((s - so) ** 2).mean()
                                               / (so * so).mean())))
            # 🔑 РОЗКЛАД похибки σ: СПІЛЬНИЙ МАСШТАБ проти РОЗКИДУ.
            #    Масштаб лікується ОДНІЄЮ константою на шар (безкоштовно),
            #    розкид — ні. Від цього залежить, чи гілка взагалі жива.
            rr = float((s * so).sum() / (so * so).sum())
            trace['r'].append(rr)
            trace['res'].append(float(torch.sqrt((((s / rr) - so) ** 2).mean()
                                                 / (so * so).mean())))
        m_a = edge_relu(m_u, s, g3, g4)
        if i == len(W):
            return m_a
        if mode == 'diag':
            _, v_a = relu_var_gauss(m_u, s, *((g3, g4) if coherent else (None, None)))
            v = (W[i] * W[i]).T @ v_a
            m = m_a @ W[i]
        else:
            t = m_u / s
            P, f = Phi(t), phi(t)
            C_a = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
            # 🔑 УЗГОДЖЕНІСТЬ: якщо `m_a` виправлене Еджвортом, то й `E[ReLU²]`
            #    мусить бути виправлене ТИМИ САМИМИ `γ` — інакше діагональ `C_a`
            #    спотворюється сильніше, ніж лікується середнє.
            q2 = (edge_relu2(m_u, s, g3, g4) if coherent
                  else (m_u * m_u + s * s) * P + (m_u * s) * f)
            d = q2 - m_a * m_a
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

TAGS = ('L0 діаг(m,v)', 'L1 наше(Мелер+C)', 'L2 наше+γ лише в m',
        'L2b наше+γ УЗГОДЖЕНО', 'L3 діаг+γ лише в m', 'L3b діаг+γ УЗГОДЖЕНО',
        'L4 оракул m,σ гаус', 'L5 оракул m,σ,γ',
        'L6 оракул ЛИШЕ σ', 'L7 оракул ЛИШЕ m',
        'L8 діаг+оракул σ', 'L10 оракул σ,γ · m СВОЄ',
        'E1 σ,γ псовані 1%', 'E3 σ,γ псовані 3%', 'E10 σ,γ псовані 10%',
        'S3 лише σ псов 3%', 'G3 лише γ псов 3%')
E = {t: np.zeros(len(FS)) for t in TAGS}
R = {t: np.zeros(len(FS)) for t in TAGS}
NOISE = np.zeros(len(FS))
TR_M = np.zeros((len(FS), L_))
TR_T = np.zeros((len(FS), L_))
TR_R = np.zeros((len(FS), L_))
TR_RS = np.zeros((len(FS), L_))
TR_S = np.zeros((len(FS), L_))
STEP_G = np.zeros(len(FS))
STEP_4 = np.zeros(len(FS))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        MOM = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        acch = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(5500 + n)
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
        nrm = float(torch.sqrt((tru ** 2).mean()))
        NOISE[n] = float(torch.sqrt((((acch / (NS // 2)) - tru) ** 2).mean())) / nrm

        MOM /= NS
        ORC_G, ORC_MS = [], []
        for q in range(L_):
            m1, m2, m3, m4 = MOM[q, 0], MOM[q, 1], MOM[q, 2], MOM[q, 3]
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            c4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4 - 3.0 * c2 * c2
            s = torch.sqrt(c2)
            ORC_G.append((c3 / s ** 3, c4 / s ** 4))
            ORC_MS.append((m1.clone(), s))

        def SP(lst, eps, sd):
            """Псує кожен вектор МУЛЬТИПЛІКАТИВНИМ шумом `eps` (фіксований сід)."""
            gg = torch.Generator(device=DEV)
            gg.manual_seed(1000 * sd + n)
            out = []
            for aa, bb in lst:
                nz = 1.0 + eps * torch.randn(aa.shape, device=DEV, dtype=aa.dtype,
                                             generator=gg)
                nz2 = 1.0 + eps * torch.randn(bb.shape, device=DEV, dtype=bb.dtype,
                                              generator=gg)
                out.append((aa * nz, bb * nz2))
            return out

        tr = {'orc': ORC_MS, 'm': [], 's': [], 'r': [], 'res': []}
        chain(W, 'mehler', trace=tr)
        TR_M[n] = np.array(tr['m'])
        TR_S[n] = np.array(tr['s'])
        TR_T[n] = np.array([float(torch.abs(mo / so).mean())
                            for mo, so in ORC_MS])
        TR_R[n] = np.array(tr['r'])
        TR_RS[n] = np.array(tr['res'])
        runs = {
            'L0 діаг(m,v)': chain(W, 'diag'),
            'L1 наше(Мелер+C)': chain(W, 'mehler'),
            'L2 наше+γ лише в m': chain(W, 'mehler', orc_g=ORC_G),
            'L2b наше+γ УЗГОДЖЕНО': chain(W, 'mehler', orc_g=ORC_G, coherent=True),
            'L3 діаг+γ лише в m': chain(W, 'diag', orc_g=ORC_G),
            'L3b діаг+γ УЗГОДЖЕНО': chain(W, 'diag', orc_g=ORC_G, coherent=True),
            'L4 оракул m,σ гаус': chain(W, 'mehler', orc_ms=ORC_MS),
            'L5 оракул m,σ,γ': chain(W, 'mehler', orc_g=ORC_G, orc_ms=ORC_MS),
            'L6 оракул ЛИШЕ σ': chain(W, 'mehler', orc_ms=ORC_MS, only='s'),
            'L7 оракул ЛИШЕ m': chain(W, 'mehler', orc_ms=ORC_MS, only='m'),
            # 🔑 L8: чи вистачить ВЕКТОРА σ без матриці `C` — це гіпотеза
            #    «лідер веде 3 вектори» у чистому вигляді.
            'L8 діаг+оракул σ': chain(W, 'diag', orc_ms=ORC_MS, only='s'),
            # 🔑 L10: САМОУЗГОДЖЕНІСТЬ — чи вийде ланцюг на точне `m` САМ,
            #    якщо дати йому правильні σ і форму (m рекурсується своє).
            'L10 оракул σ,γ · m СВОЄ': chain(W, 'mehler', orc_g=ORC_G,
                                             orc_ms=ORC_MS, only='s',
                                             coherent=True),
            # 🎯 ТЗ НА ТОЧНІСТЬ: скільки похибки у σ і γ ще терпить L10.
            #    Без цього «метод відтворено» — порожні слова: треба знати,
            #    яку точність ОПОРИ доведеться купити (ЦІННІСТЬ РЕГРЕСОРА =
            #    min(СИГНАЛ, ТОЧНІСТЬ ОПОРИ)).
            'E1 σ,γ псовані 1%': chain(W, 'mehler', orc_g=SP(ORC_G, 0.01, 1),
                                       orc_ms=SP(ORC_MS, 0.01, 2), only='s',
                                       coherent=True),
            'E3 σ,γ псовані 3%': chain(W, 'mehler', orc_g=SP(ORC_G, 0.03, 3),
                                       orc_ms=SP(ORC_MS, 0.03, 4), only='s',
                                       coherent=True),
            'E10 σ,γ псовані 10%': chain(W, 'mehler', orc_g=SP(ORC_G, 0.10, 5),
                                         orc_ms=SP(ORC_MS, 0.10, 6), only='s',
                                         coherent=True),
            'S3 лише σ псов 3%': chain(W, 'mehler', orc_g=ORC_G,
                                       orc_ms=SP(ORC_MS, 0.03, 7), only='s',
                                       coherent=True),
            'G3 лише γ псов 3%': chain(W, 'mehler', orc_g=SP(ORC_G, 0.03, 8),
                                       orc_ms=ORC_MS, only='s', coherent=True),
        }
        for t, pr in runs.items():
            E[t][n] = float(torch.sqrt(((pr - tru) ** 2).mean())) / nrm
            R[t][n] = float((pr * tru).sum() / (tru * tru).sum())
        # ⛔P4: ОДИН крок на останньому шарі (звірка з `perneuron.py`)
        mL, sL = ORC_MS[L_ - 1]
        g3L, g4L = ORC_G[L_ - 1]
        STEP_G[n] = float(torch.sqrt(((edge_relu(mL, sL, torch.zeros_like(sL),
                                                 torch.zeros_like(sL))
                                       - tru) ** 2).mean())) / nrm
        STEP_4[n] = float(torch.sqrt(((edge_relu(mL, sL, g3L, g4L)
                                       - tru) ** 2).mean())) / nrm
    print(f'  net{n} · ' + ' · '.join(f'{t.split()[0]} {E[t][n]:.4%}'
                                      for t in TAGS), flush=True)


def loo_resid(err, rat):
    out = np.zeros_like(err)
    for i in range(len(err)):
        msk = np.arange(len(err)) != i
        out[i] = math.sqrt(max(err[i] ** 2 - (rat[i] - 1.0) ** 2
                               + (rat[i] - rat[msk].mean()) ** 2, 1e-30))
    return out


print(f'\n⛔P3 шум істини (півМК): {NOISE.mean():.4%}')
print(f'⛔P4 ОДИН крок, шар 31 (звірка з perneuron.py: гаус 0.1898%, '
      f'4 моменти 0.0147%):\n   гаус {STEP_G.mean():.4%} · '
      f'4 моменти {STEP_4.mean():.4%} · ×{(STEP_G / STEP_4).mean() ** 2:.2f} по MSE')

TARGET = 0.0521
print(f'\n══ ДРАБИНА (ціль лідера = {TARGET:.4f}% віднос. похибки) ══')
print('сходинка              сира      після α    у MSE-одиницях   до лідера')
for t in TAGS:
    rs = loo_resid(E[t], R[t])
    mse = 1.698e-05 * (rs.mean() / 0.00941) ** 2
    print(f'{t:20s}  {E[t].mean():7.4%}  {rs.mean():7.4%}   {mse:.3e}     '
          f'×{mse / 5.211e-08:8.1f}')
print('\n══ ДЕ НАРОДЖУЄТЬСЯ ПОХИБКА: наш ланцюг проти оракула, ПО ШАРАХ ══')
print('шар : ' + ' '.join(f'{q+1:6d}' for q in range(0, L_, 3)))
print('t=m/σ (оракул): ' + ' '.join(f'{TR_T[:, q].mean():5.2f}' for q in range(0, L_, 3)))
print('m   : ' + ' '.join(f'{TR_M[:, q].mean():6.3%}' for q in range(0, L_, 3)))
print('σ   : ' + ' '.join(f'{TR_S[:, q].mean():6.3%}' for q in range(0, L_, 3)))
print('\n🔑 РОЗКЛАД ПОХИБКИ σ: чи це СПІЛЬНИЙ МАСШТАБ (лікується константою)?')
print('масшт r  : ' + ' '.join(f'{TR_R[:, q].mean():6.4f}' for q in range(0, L_, 3)))
print('розкид r : ' + ' '.join(f'{TR_R[:, q].std():6.4f}' for q in range(0, L_, 3)))
print('σ БЕЗ масшт: ' + ' '.join(f'{TR_RS[:, q].mean():5.3%}' for q in range(0, L_, 3)))

print(f'{"ЛІДЕР 322059":20s}  {TARGET / 100:7.4%}  {TARGET / 100:7.4%}   '
      f'{5.211e-08:.3e}     ×     1.0')
